#!/usr/bin/env python
"""CLI orchestration: universe -> data -> factors -> normalize -> composite ->
screen -> quintile backtest -> validate -> DuckDB -> dashboard export.

Usage (from the repo root):

    python run.py screen                 # latest ranked screen
    python run.py backtest               # every freq in config's backtest.frequencies_to_test
    python run.py backtest --freq monthly     # (or restrict to just one)
    python run.py backtest --freq quarterly
    python run.py decompose              # FF5+UMD alpha decomposition (--freq, default monthly)
    python run.py optimize               # CAPM betas + mean-variance optimizer over the screen
    python run.py multiasset             # cross-asset (equity/bond/commodity ETF) allocation
    python run.py dividend               # dividend-income screen (yield + payout sustainability)
    python run.py export                 # write outputs/*.parquet for the dashboard
    python run.py reports                # run the named SQL analytics in sql/queries.sql

Each subcommand loads ``config.yaml``, does its work, and writes results into
the DuckDB store at ``output.duckdb_path`` (config-driven, default
``outputs/screener.duckdb``) so the dashboards and SQL layer can read them.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from screener.capm import estimate_betas_panel
from screener.config import Config, load_config
from screener.data.edgar import EdgarProvider
from screener.data.fmp import FMPProvider
from screener.data.french import get_ff_factors
from screener.data.yf import YFinanceProvider
from screener.db import connect, write_table
from screener.dividend import build_dividend_screen
from screener.factors import SLEEVES, compute_factors, pit_fundamentals, price_asof
from screener.normalize import composite, inverse_vol_weights, sector_neutral_z
from screener.optimize import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_stats,
)
from screener.pit_membership import all_time_tickers, constituents_as_of, get_sp500_changes
from screener.portfolio import assign_quantiles, concentration
from screener.portfolio import weights as portfolio_weights
from screener.report import export_for_dashboard, run_reports
from screener.screen import apply_hard_filters, build_screen
from screener.universe import get_sp500
from screener.validate import (
    ff_decompose,
    ic_decay,
    ic_summary,
    information_coefficient,
    max_drawdown,
    net_of_cost,
    quintile_returns,
    sharpe,
    turnover,
    weighted_quintile_returns,
)


def _month_ends(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    rule = "ME" if freq == "monthly" else "QE"
    return pd.date_range(start, end, freq=rule)


def _load_universe_and_data(
    cfg: Config, max_names=None, start=None, end=None, only=None, extra_tickers=None
):
    """Load universe, point-in-time fundamentals, and adjusted prices.

    Fails CLOSED: a factor screen/backtest is meaningless without point-in-time
    fundamentals, so a provider that cannot supply them (or supplies none) aborts
    the run rather than silently continuing on an empty frame.

    ``extra_tickers`` (optional) are pulled alongside the current constituent
    list without being added to ``uni``'s metadata — used by the backtest to
    fetch data for historically-removed S&P 500 names so point-in-time
    membership can be applied per rebalance date (see pit_membership.py).
    """
    start = start or cfg.backtest.start
    end = end or cfg.backtest.end
    uni = get_sp500()
    if only:  # explicit ticker subset (FMP free tier only serves fundamentals for some names)
        uni = uni[uni["ticker"].isin(only)].reset_index(drop=True)
    if max_names:
        uni = uni.head(max_names)  # bound the universe to stay under FMP free-tier daily call cap
    tickers = uni["ticker"].tolist()
    if extra_tickers:
        seen = set(tickers)
        tickers = tickers + [t for t in extra_tickers if t not in seen]
    provider = _make_provider(cfg)
    try:
        fund = provider.get_fundamentals(tickers)
    except NotImplementedError:
        sys.exit(f"[run.py] provider '{cfg.data.provider}' has no point-in-time fundamentals. "
                 f"Set data.provider to 'edgar' or 'fmp' in config.yaml.")
    prices = provider.get_prices(tickers, start, end)
    if fund is None or fund.empty:
        sys.exit("[run.py] no point-in-time fundamentals were returned — aborting rather than "
                 "producing a factor screen with no fundamentals. Check the provider/universe.")
    return uni, fund, prices


def _make_provider(cfg: Config):
    if cfg.data.provider == "edgar":
        return EdgarProvider(contact_email=cfg.data.edgar_contact, cache_dir=cfg.data.cache_dir)
    if cfg.data.provider == "yfinance":
        return YFinanceProvider()
    return FMPProvider(api_key_env=cfg.data.fmp_api_key_env, cache_dir=cfg.data.cache_dir)


def _composite_at(fund, prices, uni, cfg: Config, as_of, sleeve_weights: dict[str, float]):
    """One cross-section: raw factors -> sector-neutral z per sleeve -> composite score.

    Returns ``(raw, comp, fallback_pct, z_by_sleeve)`` — the per-sleeve z-scores are
    kept (not just folded into the composite) so callers can persist factor
    *attribution*: which sleeve drove a name's rank, not just its final score.
    """
    uni_idx = uni.set_index("ticker")
    raw = compute_factors(fund, prices, as_of, cfg.quality_gates, sector=uni_idx["sector"])
    z_by_sleeve = {}
    fallback_pct = {}
    for sleeve, cols in SLEEVES.items():
        sleeve_vals = raw[cols].mean(axis=1, skipna=True)  # equal-weight within the sleeve
        hierarchy = [h.model_dump() for h in cfg.normalization.hierarchy]
        z, pct = sector_neutral_z(sleeve_vals, uni_idx, hierarchy, cfg.normalization.clip)
        z_by_sleeve[sleeve] = z
        fallback_pct[sleeve] = pct
    comp = composite(z_by_sleeve, sleeve_weights)
    return raw, comp, fallback_pct, z_by_sleeve


def cmd_screen(cfg: Config, con, max_names=None, only=None) -> None:
    # A screen is "as of now", so use a recent price window (covers 12-1 momentum).
    # FMP's free tier only serves the latest ~5 quarters of fundamentals, so a screen
    # dated in the config's historical backtest range would be gated out entirely.
    today = pd.Timestamp.today().normalize()
    start = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    uni, fund, prices = _load_universe_and_data(
        cfg, max_names, start, today.strftime("%Y-%m-%d"), only)
    as_of = prices.index.max() if not prices.empty else today
    equal_w = {"value": 1 / 3, "quality": 1 / 3, "momentum": 1 / 3}
    raw, comp, fallback_pct, z_by_sleeve = _composite_at(fund, prices, uni, cfg, as_of, equal_w)

    screen_df = build_screen(comp, raw, cfg.portfolio.hard_filters,
                              cfg.portfolio.select, cfg.portfolio.top_n)
    w = portfolio_weights(screen_df.index, cfg.portfolio.weighting, market_cap=raw["market_cap"])
    conc = concentration(w)

    out = screen_df.reset_index().rename(columns={"index": "ticker"})
    out.insert(0, "as_of", as_of)
    write_table(con, "screen", out)
    write_table(con, "universe", uni)

    # Attribution: which sleeve drove the rank (z_value/z_quality/z_momentum),
    # plus decision-driving valuation metrics derived from the raw factor ratios
    # (P/E, P/B, D/E are the inverse/negation of the "higher = better" factor
    # columns already computed in factors.py — no new data pulls needed).
    panel = raw.reset_index().rename(columns={"index": "ticker"})
    panel["as_of"] = as_of
    panel["composite"] = comp.reindex(raw.index).values
    for sleeve, z in z_by_sleeve.items():
        panel[f"z_{sleeve}"] = z.reindex(raw.index).values
    # raw is still indexed by ticker but panel has been reset to a plain range
    # index, so these must be assigned by position (.values), not by label —
    # otherwise pandas aligns on index and every value silently becomes NaN.
    panel["pe"] = ((1.0 / raw["ep"]).where(raw["ep"] > 0)).values
    panel["pb"] = ((1.0 / raw["bp"]).where(raw["bp"] > 0)).values
    panel["debt_to_equity"] = (-raw["de_inv"]).values
    write_table(con, "factor_panel", panel)

    print(f"Screen as of {as_of.date()}: {len(screen_df)} names")
    print(screen_df.head(15).to_string())
    print(f"\nNormalization fallback %: {fallback_pct}")
    print(f"Concentration: HHI={conc['hhi']:.3f}  top10%={conc['top_10pct_weight']:.1%}  "
          f"effective_n={conc['effective_n']:.1f}")


def _trailing_vols(prices: pd.DataFrame, as_of, lookback_months: int = 12) -> pd.Series:
    """Trailing realized volatility (std of monthly returns) per ticker, using
    only price history STRICTLY BEFORE ``as_of`` -- for inverse-vol weighting
    within a quintile bucket. Monthly (not daily) returns, matching the
    convention already used for covariance in optimize.py/cmd_optimize."""
    as_of = pd.Timestamp(as_of)
    window = prices[(prices.index >= as_of - pd.DateOffset(months=lookback_months))
                     & (prices.index < as_of)]
    if len(window) < 2:
        return pd.Series(dtype=float)
    monthly = window.resample("ME").last()
    return monthly.pct_change().dropna(how="all").std()


def cmd_backtest(cfg: Config, con, freq: str | None = None, max_names=None) -> None:
    """Run the backtest for one frequency, or (if ``freq`` is omitted) every
    frequency in ``cfg.backtest.frequencies_to_test`` — each write MERGES by
    ``freq`` rather than clobbering the other frequency's previously-stored rows.
    """
    freqs = [freq] if freq else cfg.backtest.frequencies_to_test
    for f in freqs:
        _run_one_backtest(cfg, con, f, max_names)


def _run_one_backtest(cfg: Config, con, freq: str, max_names=None) -> None:
    # Point-in-time membership: reconstruct historical S&P 500 constituents
    # (Wikipedia's changes table) so each rebalance only scores/holds names
    # that were ACTUALLY index members on that date, mitigating (not fully
    # eliminating -- see pit_membership.py) survivorship bias. This requires
    # pulling data for some names no longer in today's list.
    current = get_sp500()
    if max_names:
        current = current.head(max_names)
    current_tickers = set(current["ticker"])
    changes = get_sp500_changes()
    extra = sorted(all_time_tickers(current_tickers, changes, since=cfg.backtest.start)
                    - current_tickers)
    print(f"[{freq}] Point-in-time membership: {len(current_tickers)} current constituents + "
          f"{len(extra)} historically-removed/renamed names pulled for the {cfg.backtest.start}"
          f"..{cfg.backtest.end} window (some may lack EDGAR/yfinance data post-delisting).")
    uni, fund, prices = _load_universe_and_data(cfg, max_names, extra_tickers=extra)
    dates = _month_ends(cfg.backtest.start, cfg.backtest.end, freq)
    dates = [d for d in dates if d <= prices.index.max()] if not prices.empty else []
    if len(dates) < 3:
        print(f"[{freq}] Not enough price history for a backtest yet "
              f"(need >= 3 rebalance dates).")
        return

    equal_w = {"value": 1 / 3, "quality": 1 / 3, "momentum": 1 / 3}

    # --- Pass 1: raw factors + per-sleeve z-scores + forward returns at every
    # date. Sleeve weighting hasn't been decided yet -- z_by_sleeve doesn't
    # depend on it (composite() is what applies weights), so it's safe to
    # compute up front and blend afterward with whatever weights pass 2 picks.
    raw_hist, z_hist, fallback_hist, fwd_returns = {}, {}, {}, {}
    for i, dt in enumerate(dates[:-1]):
        raw, _, fallback_pct, z_by_sleeve = _composite_at(fund, prices, uni, cfg, dt, equal_w)
        raw_hist[dt] = raw
        z_hist[dt] = z_by_sleeve
        fallback_hist[dt] = fallback_pct
        px_now = prices[prices.index <= dt].iloc[-1] if (prices.index <= dt).any() else None
        nxt = dates[i + 1]
        px_next = prices[prices.index <= nxt].iloc[-1] if (prices.index <= nxt).any() else None
        if px_now is not None and px_next is not None:
            fwd_returns[dt] = (px_next / px_now - 1.0)

    fwd_df = pd.DataFrame(fwd_returns).T

    # --- Each sleeve's OWN realized Q5-Q1 spread per date (the "sleeve
    # long-short return series" inverse_vol_weights needs), built by reusing
    # quintile_returns() on that sleeve's own z-scores -- not reimplemented.
    sleeve_names = list(SLEEVES.keys())
    sleeve_return_cols = {}
    for sleeve in sleeve_names:
        sleeve_scores = pd.DataFrame({dt: z_hist[dt][sleeve] for dt in z_hist}).T
        sr = quintile_returns(sleeve_scores, fwd_df, cfg.portfolio.n_quantiles)
        sleeve_return_cols[sleeve] = sr["spread"] if "spread" in sr else pd.Series(dtype=float)
    sleeve_returns_df = pd.DataFrame(sleeve_return_cols).reindex(fwd_df.index).fillna(0.0)

    # --- Pass 2: decide each date's sleeve weights from ONLY prior dates'
    # realized sleeve returns (inverse_vol_weights itself slices strictly
    # before as_of_row, so passing the whole series is look-ahead-safe), then
    # blend the composite, apply hard filters + point-in-time membership.
    hf = cfg.portfolio.hard_filters
    scores, market_cap_hist, vols_hist = {}, {}, {}
    for i, dt in enumerate(sleeve_returns_df.index):
        sleeve_w = inverse_vol_weights(
            sleeve_returns_df, lookback=cfg.weighting.vol_lookback_months,
            max_weight=cfg.weighting.max_sleeve_weight,
            burn_in=cfg.weighting.burn_in_months, as_of_row=i)
        raw, z_by_sleeve = raw_hist[dt], z_hist[dt]
        comp = composite(z_by_sleeve, sleeve_w, min_sleeves=2)
        eligible = apply_hard_filters(raw, hf)
        members = constituents_as_of(dt, current_tickers, changes) & set(eligible)
        scores[dt] = comp.reindex(comp.index.intersection(members))
        market_cap_hist[dt] = raw["market_cap"]
        vols_hist[dt] = _trailing_vols(prices, dt, cfg.weighting.vol_lookback_months)

    scores_df = pd.DataFrame(scores).T
    fwd_df = fwd_df.reindex(scores_df.index)
    market_cap_df = pd.DataFrame(market_cap_hist).T.reindex(scores_df.index)
    vols_df = pd.DataFrame(vols_hist).T.reindex(scores_df.index)

    ic = information_coefficient(scores_df, fwd_df)
    ic_stats = ic_summary(ic)
    method = cfg.portfolio.weighting
    qr = (quintile_returns(scores_df, fwd_df, cfg.portfolio.n_quantiles) if method == "equal"
          else weighted_quintile_returns(scores_df, fwd_df, method, market_cap_df, vols_df,
                                          cfg.portfolio.n_quantiles))

    long_rows = []
    for dt, row in qr.iterrows():
        for q in range(1, cfg.portfolio.n_quantiles + 1):
            col = f"Q{q}"
            if col in row and pd.notna(row[col]):
                long_rows.append({"freq": freq, "date": dt, "quantile": q, "ret": row[col]})
    _merge_write_by_freq(con, "backtest_quintiles", pd.DataFrame(long_rows), freq)

    # IC decay: how fast the signal's predictive power fades at longer horizons
    # (lag 0 = same-period IC already computed above; reused as the curve's start).
    decay = ic_decay(scores_df, fwd_df, lags=(1, 3, 5))
    decay_rows = [{"freq": freq, "lag": 0, "ic": ic_stats["mean_ic"]}]
    decay_rows += [{"freq": freq, "lag": int(k.split("_")[1]), "ic": v} for k, v in decay.items()]
    _merge_write_by_freq(con, "ic_decay", pd.DataFrame(decay_rows).sort_values("lag"), freq)

    # Normalization fallback %, by date and sleeve (the screen already prints
    # this for the latest date; the backtest now persists it for every date).
    fb_rows = [{"freq": freq, "date": dt, "sleeve": sleeve, "level": level, "pct": pct}
               for dt, sleeves in fallback_hist.items()
               for sleeve, levels in sleeves.items()
               for level, pct in levels.items()]
    _merge_write_by_freq(con, "normalization_fallback", pd.DataFrame(fb_rows), freq)

    # --- Turnover, net-of-cost, RF-Sharpe, Max Drawdown, persisted as one
    # summary row per frequency. Turnover is measured on the Q5 (top-quintile)
    # holdings-weight panel, using the SAME weighting scheme as the quintile
    # returns above, so turnover and the returns it costs are consistent.
    spread = qr["spread"] if "spread" in qr else pd.Series(dtype=float)
    holdings = {}
    for dt in scores_df.index:
        s = scores_df.loc[dt].dropna()
        if len(s) < cfg.portfolio.n_quantiles * 2:
            continue
        q5 = assign_quantiles(s, cfg.portfolio.n_quantiles)
        q5_members = q5[q5 == cfg.portfolio.n_quantiles].index
        mc = market_cap_df.loc[dt] if dt in market_cap_df.index else None
        vv = vols_df.loc[dt] if dt in vols_df.index else None
        holdings[dt] = portfolio_weights(q5_members, method, market_cap=mc, vols=vv)
    holdings_df = pd.DataFrame(holdings).T.reindex(scores_df.index).fillna(0.0)
    tn = turnover(holdings_df, ppy=12 if freq == "monthly" else 4)

    ff = get_ff_factors(start=cfg.backtest.start, end=cfg.backtest.end)
    ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp("M")
    spread_idx = spread.index.to_period("M").to_timestamp("M") if len(spread) else spread.index
    spread_m = pd.Series(spread.values, index=spread_idx)
    rf = ff["RF"].reindex(spread_m.index)

    ppy = 12 if freq == "monthly" else 4
    gross_sharpe = sharpe(spread, rf=None, ppy=ppy)  # self-financing -> no Rf
    q5_returns = qr["Q5"] if "Q5" in qr else pd.Series(dtype=float)
    if len(q5_returns):
        q5_idx = q5_returns.index.to_period("M").to_timestamp("M")
    else:
        q5_idx = q5_returns.index
    q5_m = pd.Series(q5_returns.values, index=q5_idx)
    q5_gross_sharpe = sharpe(q5_m, rf=rf.reindex(q5_m.index), ppy=ppy)
    mdd = max_drawdown(spread)

    cost_bps = cfg.costs.commission_bps + cfg.costs.spread_bps + cfg.costs.short_rebate_bps
    net_spread = net_of_cost(spread, tn["per_rebalance_two_way"], cost_bps)
    net_sharpe = sharpe(net_spread, rf=None, ppy=ppy)

    summary = pd.DataFrame([{
        "freq": freq, "weighting": method, "n_rebalances": len(scores_df),
        "gross_spread_mean": float(spread.mean()) if len(spread) else float("nan"),
        "net_spread_mean": float(net_spread.mean()) if len(net_spread) else float("nan"),
        "gross_sharpe": gross_sharpe, "net_sharpe": net_sharpe,
        "q5_gross_sharpe": q5_gross_sharpe, "max_drawdown": mdd,
        "turnover_per_rebalance": tn["per_rebalance_two_way"],
        "turnover_annualized": tn["annualized_two_way"],
        "mean_ic": ic_stats["mean_ic"], "ic_t_stat_hac": ic_stats["t_stat_hac"],
    }])
    _merge_write_by_freq(con, "backtest_summary", summary, freq)

    print(f"[{freq}] {len(scores_df)} rebalances, weighting={method}")
    print(f"[{freq}] IC summary: {ic_stats}")
    print(f"[{freq}] IC decay: {decay}")
    if len(spread):
        print(f"[{freq}] Q5-Q1 gross spread mean: {spread.mean():.4f}  "
              f"net: {net_spread.mean():.4f}  (n={spread.notna().sum()})")
    print(f"[{freq}] Sharpe: gross={gross_sharpe:.2f}  net={net_sharpe:.2f}  "
          f"Q5(excess)={q5_gross_sharpe:.2f}  MaxDrawdown={mdd:.2%}")
    print(f"[{freq}] Turnover: {tn['per_rebalance_two_way']:.2%}/rebalance, "
          f"{tn['annualized_two_way']:.2%} annualized")


def cmd_decompose(cfg: Config, con, freq: str = "monthly") -> None:
    bt = None
    if _table_exists(con, "backtest_quintiles"):
        bt = con.execute('SELECT * FROM "backtest_quintiles"').df()
    if bt is None or bt.empty:
        print("Run `python run.py backtest` first — no backtest_quintiles table found.")
        return
    if "freq" in bt.columns:
        bt = bt[bt["freq"] == freq]
        if bt.empty:
            print(f"No backtest_quintiles rows for freq={freq!r} — "
                  f"run `python run.py backtest --freq {freq}` first.")
            return
    piv = bt.pivot_table(index="date", columns="quantile", values="ret")
    spread = piv[piv.columns.max()] - piv[piv.columns.min()]
    spread.index = pd.to_datetime(spread.index)

    ff = get_ff_factors(start=cfg.backtest.start, end=cfg.backtest.end)
    ann = 12 if freq == "monthly" else 4
    period = "M" if freq == "monthly" else "Q"
    ff.index = pd.to_datetime(ff.index).to_period(period).to_timestamp(period)
    spread.index = spread.index.to_period(period).to_timestamp(period)
    if freq != "monthly":  # FF factors are monthly; resample to the backtest's own frequency
        ff = ff.resample("QE").apply(lambda x: (1 + x).prod() - 1)

    result = ff_decompose(spread, ff, is_long_short=True, ppy=ann)
    rows = [{"freq": freq, "metric": "alpha_period", "value": result["alpha_monthly"]},
            {"freq": freq, "metric": "alpha_annualized", "value": result["alpha_annualized"]},
            {"freq": freq, "metric": "alpha_t_nw", "value": result["alpha_t_nw"]},
            {"freq": freq, "metric": "r_squared", "value": result["r_squared"]},
            {"freq": freq, "metric": "n", "value": result["n"]}]
    rows += [{"freq": freq, "metric": f"{k}_loading", "value": v}
              for k, v in result["loadings"].items()]
    _merge_write_by_freq(con, "ff_decomposition", pd.DataFrame(rows), freq)
    print(f"[{freq}] FF5+UMD decomposition of Q5-Q1 (n={result['n']}):")
    print(f"  alpha (annualized) = {result['alpha_annualized']:.4f}  "
          f"(NW t = {result['alpha_t_nw']:.2f})")
    print(f"  R^2 = {result['r_squared']:.3f}")
    print(f"  loadings: {result['loadings']}")


def cmd_optimize(cfg: Config, con) -> None:
    """CAPM betas + a long-only mean-variance optimizer over the screened names.

    The max-Sharpe (tangency) portfolio here IS the CAPM "market portfolio" —
    the theory's efficient-frontier tangency point — so the betas and the
    optimized weights are two views of one model, not two separate features.
    """
    if not _table_exists(con, "screen"):
        print("Run `python run.py screen` first — no screen table found.")
        return
    tickers = con.execute('SELECT ticker FROM "screen"').df()["ticker"].tolist()
    if len(tickers) < 3:
        print("Need at least 3 screened names to optimize a portfolio.")
        return

    provider = _make_provider(cfg)
    prices = provider.get_prices(tickers, cfg.backtest.start, cfg.backtest.end)
    prices = prices.dropna(axis=1, how="all")
    monthly = prices.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    rets = rets.dropna(axis=1, thresh=24)  # need enough history for a stable covariance estimate
    if rets.shape[1] < 3:
        print("Not enough overlapping price history across screened names to optimize.")
        return

    ff = get_ff_factors(start=cfg.backtest.start, end=cfg.backtest.end)
    ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp("M")
    rets.index = pd.to_datetime(rets.index).to_period("M").to_timestamp("M")

    betas = estimate_betas_panel(rets, ff)
    betas.index.name = "ticker"
    write_table(con, "capm_betas", betas.reset_index())

    mean_a, cov_a = rets.mean() * 12, rets.cov() * 12  # annualize monthly mean/cov
    rf_a = float(ff["RF"].reindex(rets.index).mean()) * 12

    w_sharpe = max_sharpe_weights(mean_a, cov_a, rf=rf_a)
    w_minvar = min_variance_weights(cov_a)
    stats_sharpe = portfolio_stats(w_sharpe, mean_a, cov_a, rf=rf_a)
    stats_minvar = portfolio_stats(w_minvar, mean_a, cov_a, rf=rf_a)

    weight_rows = (
        [{"method": "max_sharpe", "ticker": t, "weight": w} for t, w in w_sharpe.items()]
        + [{"method": "min_variance", "ticker": t, "weight": w} for t, w in w_minvar.items()]
    )
    write_table(con, "optimal_weights", pd.DataFrame(weight_rows))
    write_table(con, "efficient_frontier", efficient_frontier(mean_a, cov_a, n_points=25))

    print(f"CAPM: {rets.shape[1]} names, {len(rets)} months. "
          f"Average beta = {betas['beta'].mean():.2f} "
          f"(a broad, diversified basket should sit near 1.0).")
    print(f"Max-Sharpe portfolio: return={stats_sharpe['expected_return']:.1%} "
          f"vol={stats_sharpe['volatility']:.1%} Sharpe={stats_sharpe['sharpe']:.2f}")
    print(f"Min-variance portfolio: return={stats_minvar['expected_return']:.1%} "
          f"vol={stats_minvar['volatility']:.1%} Sharpe={stats_minvar['sharpe']:.2f}")
    print("Top max-Sharpe weights:")
    print(w_sharpe.sort_values(ascending=False).head(5).to_string())


def cmd_multiasset(cfg: Config, con) -> None:
    """Mean-variance / CAPM allocation across asset classes via liquid ETF proxies.

    Equities (SPY), bonds, and commodities enter as ETFs — the equity factor
    screen doesn't apply to bonds/commodities and free per-security data for them
    doesn't exist, so this asset-allocation view is where they legitimately live.
    """
    ac = cfg.asset_classes
    tickers = ac.all_tickers()
    labels = ({ac.equity_proxy: "Equity"}
              | {t: "Bond" for t in ac.bond_etfs}
              | {t: "Commodity" for t in ac.commodity_etfs})

    prices = YFinanceProvider(cfg.data.cache_dir).get_prices(
        tickers, cfg.backtest.start, cfg.backtest.end).dropna(axis=1, how="all")
    monthly = prices.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all").dropna(axis=1, thresh=24)
    if rets.shape[1] < 3:
        print("Not enough overlapping ETF history to build a multi-asset allocation.")
        return

    ff = get_ff_factors(start=cfg.backtest.start, end=cfg.backtest.end)
    ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp("M")
    rets.index = pd.to_datetime(rets.index).to_period("M").to_timestamp("M")

    betas = estimate_betas_panel(rets, ff)
    betas["asset_class"] = betas.index.map(labels)
    betas.index.name = "ticker"
    write_table(con, "multiasset_betas", betas.reset_index())

    mean_a, cov_a = rets.mean() * 12, rets.cov() * 12
    rf_a = float(ff["RF"].reindex(rets.index).mean()) * 12
    w_sharpe = max_sharpe_weights(mean_a, cov_a, rf=rf_a)
    stats = portfolio_stats(w_sharpe, mean_a, cov_a, rf=rf_a)

    alloc = pd.DataFrame({"ticker": w_sharpe.index, "weight": w_sharpe.values})
    alloc["asset_class"] = alloc["ticker"].map(labels)
    write_table(con, "multiasset_weights", alloc)
    write_table(con, "multiasset_frontier", efficient_frontier(mean_a, cov_a, n_points=25))

    by_class = alloc.groupby("asset_class")["weight"].sum().sort_values(ascending=False)
    print(f"Multi-asset max-Sharpe allocation ({rets.shape[1]} ETFs, {len(rets)} months): "
          f"return={stats['expected_return']:.1%} vol={stats['volatility']:.1%} "
          f"Sharpe={stats['sharpe']:.2f}")
    print("By asset class:")
    print(by_class.to_string())


def cmd_dividend(cfg: Config, con, max_names=None) -> None:
    """Dividend-income screen: trailing yield gated on payout sustainability."""
    today = pd.Timestamp.today().normalize()
    start = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    uni, fund, prices = _load_universe_and_data(
        cfg, max_names, start, today.strftime("%Y-%m-%d"))
    if fund.empty or prices.empty:
        print("No fundamentals/prices available for the dividend screen.")
        return

    as_of = prices.index.max()
    pit = pit_fundamentals(fund, as_of)
    price = price_asof(prices, as_of)
    tickers = pit.index.intersection(price.index)
    divs = YFinanceProvider(cfg.data.cache_dir).get_dividends(
        list(tickers), as_of=as_of).set_index("ticker")

    dividends_ttm = divs["dividends_ttm"].reindex(tickers)
    regular_dividends_ttm = divs["regular_dividends_ttm"].reindex(tickers)
    # Actual shares outstanding receive the cash dividend; the diluted
    # weighted-average (which includes hypothetical option/RSU dilution) would
    # overstate total cash paid. Same dated-figure-with-fallback as market cap.
    shares = pit.loc[tickers, "shares_outstanding"].where(
        pit.loc[tickers, "shares_outstanding"] > 0, pit.loc[tickers, "shares_diluted"])
    net_income = pit.loc[tickers, "net_income_ttm"]
    sector = uni.set_index("ticker")["sector"].reindex(tickers)

    screen = build_dividend_screen(
        dividends_ttm, price.loc[tickers], shares, net_income,
        regular_dividends_ttm=regular_dividends_ttm, sector=sector,
        min_yield=cfg.dividend.min_yield, max_payout=cfg.dividend.max_payout,
        top_n=cfg.dividend.top_n)
    screen.insert(0, "as_of", as_of)
    write_table(con, "dividend_screen", screen)

    print(f"Dividend screen as of {as_of.date()}: {len(screen)} names "
          f"(payout <= {cfg.dividend.max_payout:.0%}, ranked by RECURRING yield)")
    show = screen.head(15).copy()
    reg_pct = (show["regular_dividend_yield"] * 100).round(2).astype(str)
    tot_pct = (show["total_dividend_yield"] * 100).round(2).astype(str)
    show["regular_dividend_yield"] = reg_pct + "%"
    show["total_dividend_yield"] = tot_pct + "%"
    show["payout_ratio"] = (show["payout_ratio"] * 100).round(0).astype(str) + "%"
    cols = ["rank", "ticker", "regular_dividend_yield", "total_dividend_yield",
            "has_special_dividend", "payout_ratio", "sector"]
    print(show[cols].to_string(index=False))


def _table_exists(con, name: str) -> bool:
    try:
        con.execute(f'SELECT 1 FROM "{name}" LIMIT 1')
        return True
    except Exception:
        return False


def _merge_write_by_freq(con, table: str, new_df: pd.DataFrame, freq: str) -> None:
    """Write ``new_df`` to ``table``, replacing only rows for ``freq`` and
    preserving any other frequency's previously-written rows — so running
    `backtest --freq monthly` then `--freq quarterly` doesn't clobber the
    first run (closes the old CREATE-OR-REPLACE-per-call clobbering bug)."""
    if _table_exists(con, table):
        existing = con.execute(f'SELECT * FROM "{table}"').df()
        if "freq" in existing.columns:
            existing = existing[existing["freq"] != freq]
        combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    else:
        combined = new_df
    write_table(con, table, combined)


def cmd_export(cfg: Config, con) -> None:
    written = export_for_dashboard(con, cfg.output.export_dir)
    print(f"Exported {len(written)} tables to {cfg.output.export_dir}/: {written}")


def cmd_reports(cfg: Config, con) -> None:
    """Run the named analytical queries in ``sql/queries.sql`` and print each."""
    reports = run_reports(con)
    if not reports:
        print("No query results — build the tables first (`python run.py screen`, `backtest`).")
        return
    for name, df in reports.items():
        print(f"\n=== {name} ({len(df)} rows) ===")
        print(df.to_string(index=False) if not df.empty else "  (no rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["screen", "backtest", "decompose", "optimize", "multiasset", "dividend",
                 "export", "reports"])
    parser.add_argument(
        "--freq", choices=["monthly", "quarterly"], default=None,
        help="backtest/decompose: restrict to one frequency. Omitted for `backtest` -> runs "
             "every frequency in config's backtest.frequencies_to_test; omitted for `decompose` "
             "-> defaults to monthly.")
    parser.add_argument("--max-names", type=int, default=None,
                        help="cap universe size (FMP free tier: ~250 calls/day, 4 per name)")
    parser.add_argument("--tickers", default=None,
                        help="comma-separated ticker subset (e.g. accessible FMP-free names)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    only = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    cfg = load_config(args.config)
    con = connect(cfg.output.duckdb_path)

    if args.command == "screen":
        cmd_screen(cfg, con, args.max_names, only)
    elif args.command == "backtest":
        cmd_backtest(cfg, con, args.freq, args.max_names)
    elif args.command == "decompose":
        cmd_decompose(cfg, con, args.freq or "monthly")
    elif args.command == "optimize":
        cmd_optimize(cfg, con)
    elif args.command == "multiasset":
        cmd_multiasset(cfg, con)
    elif args.command == "dividend":
        cmd_dividend(cfg, con, args.max_names)
    elif args.command == "export":
        cmd_export(cfg, con)
    elif args.command == "reports":
        cmd_reports(cfg, con)


if __name__ == "__main__":
    main()
