#!/usr/bin/env python
"""CLI orchestration: universe -> data -> factors -> normalize -> composite ->
screen -> quintile backtest -> validate -> DuckDB -> dashboard export.

Usage (from the repo root):

    python run.py screen                 # latest ranked screen
    python run.py backtest --freq monthly
    python run.py backtest --freq quarterly
    python run.py decompose              # FF5+UMD alpha decomposition
    python run.py optimize               # CAPM betas + mean-variance optimizer over the screen
    python run.py multiasset             # cross-asset (equity/bond/commodity ETF) allocation
    python run.py dividend               # dividend-income screen (yield + payout sustainability)
    python run.py export                 # write outputs/*.parquet for the dashboard

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
from screener.normalize import composite, sector_neutral_z
from screener.optimize import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_stats,
)
from screener.portfolio import concentration
from screener.portfolio import weights as portfolio_weights
from screener.report import export_for_dashboard
from screener.screen import build_screen
from screener.universe import get_sp500
from screener.validate import (
    ff_decompose,
    ic_decay,
    ic_summary,
    information_coefficient,
    quintile_returns,
)


def _month_ends(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    rule = "ME" if freq == "monthly" else "QE"
    return pd.date_range(start, end, freq=rule)


def _load_universe_and_data(cfg: Config, max_names=None, start=None, end=None, only=None):
    start = start or cfg.backtest.start
    end = end or cfg.backtest.end
    uni = get_sp500()
    if only:  # explicit ticker subset (FMP free tier only serves fundamentals for some names)
        uni = uni[uni["ticker"].isin(only)].reset_index(drop=True)
    if max_names:
        uni = uni.head(max_names)  # bound the universe to stay under FMP free-tier daily call cap
    tickers = uni["ticker"].tolist()
    provider = _make_provider(cfg)
    try:
        fund = provider.get_fundamentals(tickers)
        prices = provider.get_prices(tickers, start, end)
    except RuntimeError as e:
        print(f"[run.py] provider unavailable ({e}); using yfinance for prices only "
              f"(fundamentals need FMP/EDGAR for point-in-time correctness).", file=sys.stderr)
        prices = YFinanceProvider().get_prices(tickers, start, end)
        fund = pd.DataFrame(columns=["ticker", "period_end", "filing_date"])
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
    raw = compute_factors(fund, prices, as_of, cfg.quality_gates)
    uni_idx = uni.set_index("ticker")
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


def cmd_backtest(cfg: Config, con, freq: str, max_names=None) -> None:
    uni, fund, prices = _load_universe_and_data(cfg, max_names)
    dates = _month_ends(cfg.backtest.start, cfg.backtest.end, freq)
    dates = [d for d in dates if d <= prices.index.max()] if not prices.empty else []
    if len(dates) < 3:
        print("Not enough price history for a backtest yet (need >= 3 rebalance dates).")
        return

    equal_w = {"value": 1 / 3, "quality": 1 / 3, "momentum": 1 / 3}
    scores, fwd_returns = {}, {}
    for i, dt in enumerate(dates[:-1]):
        _, comp, _, _ = _composite_at(fund, prices, uni, cfg, dt, equal_w)
        scores[dt] = comp
        px_now = prices[prices.index <= dt].iloc[-1] if (prices.index <= dt).any() else None
        nxt = dates[i + 1]
        px_next = prices[prices.index <= nxt].iloc[-1] if (prices.index <= nxt).any() else None
        if px_now is not None and px_next is not None:
            fwd_returns[dt] = (px_next / px_now - 1.0)

    scores_df = pd.DataFrame(scores).T
    fwd_df = pd.DataFrame(fwd_returns).T.reindex(scores_df.index)

    ic = information_coefficient(scores_df, fwd_df)
    ic_stats = ic_summary(ic)
    qr = quintile_returns(scores_df, fwd_df, cfg.portfolio.n_quantiles)

    long_rows = []
    for dt, row in qr.iterrows():
        for q in range(1, cfg.portfolio.n_quantiles + 1):
            col = f"Q{q}"
            if col in row and pd.notna(row[col]):
                long_rows.append({"date": dt, "quantile": q, "ret": row[col]})
    write_table(con, "backtest_quintiles", pd.DataFrame(long_rows))

    # IC decay: how fast the signal's predictive power fades at longer horizons
    # (lag 0 = same-period IC already computed above; reused as the curve's start).
    decay = ic_decay(scores_df, fwd_df, lags=(1, 3, 5))
    decay_rows = [{"lag": 0, "ic": ic_stats["mean_ic"]}]
    decay_rows += [{"lag": int(k.split("_")[1]), "ic": v} for k, v in decay.items()]
    write_table(con, "ic_decay", pd.DataFrame(decay_rows).sort_values("lag"))

    print(f"Backtest ({freq}): {len(scores_df)} rebalances")
    print(f"IC summary: {ic_stats}")
    print(f"IC decay: {decay}")
    if "spread" in qr:
        print(f"Q5-Q1 mean: {qr['spread'].mean():.4f}  (n={qr['spread'].notna().sum()})")


def cmd_decompose(cfg: Config, con) -> None:
    bt = None
    if _table_exists(con, "backtest_quintiles"):
        bt = con.execute('SELECT * FROM "backtest_quintiles"').df()
    if bt is None or bt.empty:
        print("Run `python run.py backtest` first — no backtest_quintiles table found.")
        return
    piv = bt.pivot_table(index="date", columns="quantile", values="ret")
    spread = piv[piv.columns.max()] - piv[piv.columns.min()]
    spread.index = pd.to_datetime(spread.index)

    ff = get_ff_factors(start=cfg.backtest.start, end=cfg.backtest.end)
    ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp("M")
    spread.index = spread.index.to_period("M").to_timestamp("M")

    result = ff_decompose(spread, ff, is_long_short=True)
    rows = [{"metric": "alpha_monthly", "value": result["alpha_monthly"]},
            {"metric": "alpha_annualized", "value": result["alpha_annualized"]},
            {"metric": "alpha_t_nw", "value": result["alpha_t_nw"]},
            {"metric": "r_squared", "value": result["r_squared"]},
            {"metric": "n", "value": result["n"]}]
    rows += [{"metric": f"{k}_loading", "value": v} for k, v in result["loadings"].items()]
    write_table(con, "ff_decomposition", pd.DataFrame(rows))
    print(f"FF5+UMD decomposition of Q5-Q1 (n={result['n']}):")
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
    divs = YFinanceProvider(cfg.data.cache_dir).get_dividends(list(tickers)).set_index("ticker")

    dividends_ttm = divs["dividends_ttm"].reindex(tickers)
    shares = pit.loc[tickers, "shares_diluted"]
    net_income = pit.loc[tickers, "net_income_ttm"]
    sector = uni.set_index("ticker")["sector"].reindex(tickers)

    screen = build_dividend_screen(
        dividends_ttm, price.loc[tickers], shares, net_income, sector=sector,
        min_yield=cfg.dividend.min_yield, max_payout=cfg.dividend.max_payout,
        top_n=cfg.dividend.top_n)
    screen.insert(0, "as_of", as_of)
    write_table(con, "dividend_screen", screen)

    print(f"Dividend screen as of {as_of.date()}: {len(screen)} names "
          f"(payout <= {cfg.dividend.max_payout:.0%}, ranked by yield)")
    show = screen.head(15).copy()
    show["dividend_yield"] = (show["dividend_yield"] * 100).round(2).astype(str) + "%"
    show["payout_ratio"] = (show["payout_ratio"] * 100).round(0).astype(str) + "%"
    cols = ["rank", "ticker", "dividend_yield", "payout_ratio", "sector"]
    print(show[cols].to_string(index=False))


def _table_exists(con, name: str) -> bool:
    try:
        con.execute(f'SELECT 1 FROM "{name}" LIMIT 1')
        return True
    except Exception:
        return False


def cmd_export(cfg: Config, con) -> None:
    written = export_for_dashboard(con, cfg.output.export_dir)
    print(f"Exported {len(written)} tables to {cfg.output.export_dir}/: {written}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["screen", "backtest", "decompose", "optimize", "multiasset", "dividend", "export"])
    parser.add_argument("--freq", choices=["monthly", "quarterly"], default="monthly")
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
        cmd_decompose(cfg, con)
    elif args.command == "optimize":
        cmd_optimize(cfg, con)
    elif args.command == "multiasset":
        cmd_multiasset(cfg, con)
    elif args.command == "dividend":
        cmd_dividend(cfg, con, args.max_names)
    elif args.command == "export":
        cmd_export(cfg, con)


if __name__ == "__main__":
    main()
