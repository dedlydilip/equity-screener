"""Streamlit dashboard (primary) for the equity screener.

Reads the parquet exports written by ``screener.report.export_for_dashboard`` and
shows the ranked screen + sector exposure, the quintile backtest, and the
Fama-French decomposition. Run from the repo root:

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

from screener.factors import FACTORS  # noqa: E402
from screener.validate import max_drawdown, sharpe  # noqa: E402

# Raw factor column -> plain-English label, for the correlation heatmap and any
# other place a reader needs to map jargon ("ev_ebitda_inv") to meaning.
FACTOR_LABELS = {
    "ep": "Earnings yield", "bp": "Book/Price", "fcf_yield": "FCF yield",
    "ev_ebitda_inv": "EV/EBITDA (inv)", "roe": "ROE", "roic": "ROIC",
    "gross_profitability": "Gross profitability", "de_inv": "Leverage (inv)",
    "momentum": "12-1 momentum",
}


def load(name: str):
    p = OUT / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


st.set_page_config(page_title="Equity Screener", layout="wide")
st.title("Institutional Factor Screener — S&P 500")
st.caption("Value / Quality / Momentum · MAD sector-neutral z · FF5+UMD-validated")

screen = load("screen")
bt = load("backtest_quintiles")
summary = load("backtest_summary")
fallback = load("normalization_fallback")
ffd = load("ff_decomposition")
uni = load("universe")
panel = load("factor_panel")
decay = load("ic_decay")
betas = load("capm_betas")
opt_weights = load("optimal_weights")
frontier = load("efficient_frontier")
ma_weights = load("multiasset_weights")
ma_frontier = load("multiasset_frontier")
ma_betas = load("multiasset_betas")
div_screen = load("dividend_screen")

if screen is None:
    st.warning("No outputs yet. Build them first: `python run.py screen`, "
               "then `backtest` / `decompose`.")
    st.stop()

# backtest_quintiles/ff_decomposition/etc. can hold rows for more than one
# rebalance frequency at once (Stage 3's merge-by-freq write) -> one selector,
# shared by the Backtest and Factor decomposition tabs below.
has_freq = bt is not None and "freq" in bt.columns
# .dropna() defends against stray legacy rows written before the `freq`
# column existed (mixing real freq strings with NaN would crash sorted()).
freq_options = sorted(bt["freq"].dropna().unique().tolist()) if has_freq else []
chosen_freq = st.sidebar.selectbox("Backtest frequency", freq_options) if freq_options else None

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Screen", "Backtest", "Factor decomposition", "Portfolio / CAPM",
     "Multi-asset", "Dividend income"])

with tab1:
    st.subheader("Ranked screen")
    df = screen.copy()
    if uni is not None:
        df = df.merge(uni[["ticker", "security", "sector"]], on="ticker", how="left")
        options = ["All", *sorted(df["sector"].dropna().unique().tolist())]
        pick = st.selectbox("Filter by sector", options)
        if pick != "All":
            df = df[df["sector"] == pick]
    st.dataframe(df.sort_values("rank"), use_container_width=True, hide_index=True)

    if uni is not None and len(df):
        st.subheader("Sector allocation")
        sector_counts = df["sector"].value_counts().reset_index()
        sector_counts.columns = ["sector", "count"]
        chart = alt.Chart(sector_counts).mark_bar().encode(
            x=alt.X("count:Q", title="Names in screen"),
            y=alt.Y("sector:N", sort="-x", title=None),
            tooltip=["sector", "count"],
        )
        st.altair_chart(chart, use_container_width=True)

    if panel is not None:
        st.subheader("Why does this stock rank here?")
        screened_tickers = sorted(df["ticker"].unique().tolist())
        pick_t = st.selectbox("Select a screened name", screened_tickers)
        row = panel[panel["ticker"] == pick_t]
        if not row.empty:
            r = row.iloc[0]
            name = df.loc[df["ticker"] == pick_t, "security"]
            st.markdown(f"**{name.iloc[0] if len(name) else pick_t}** ({pick_t})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P/E", f"{r['pe']:.1f}" if pd.notna(r["pe"]) else "n/a")
            c2.metric("P/B", f"{r['pb']:.2f}" if pd.notna(r["pb"]) else "n/a")
            c3.metric("ROE", f"{r['roe'] * 100:.1f}%" if pd.notna(r["roe"]) else "n/a")
            c4.metric("Debt/Equity", f"{r['debt_to_equity']:.2f}x"
                      if pd.notna(r["debt_to_equity"]) else "n/a")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("ROIC", f"{r['roic'] * 100:.1f}%" if pd.notna(r["roic"]) else "n/a")
            c6.metric("Gross profitability", f"{r['gross_profitability'] * 100:.1f}%"
                      if pd.notna(r["gross_profitability"]) else "n/a")
            c7.metric("FCF yield", f"{r['fcf_yield'] * 100:.1f}%"
                      if pd.notna(r["fcf_yield"]) else "n/a")
            c8.metric("Market cap", f"${r['market_cap'] / 1e9:.1f}B"
                      if pd.notna(r["market_cap"]) else "n/a")

            st.caption("Factor sub-scores (z vs. sector/industry peers; 0 = peer median, "
                       "clipped ±5)")
            sub = pd.DataFrame({
                "sleeve": ["Value", "Quality", "Momentum"],
                "z": [r.get("z_value"), r.get("z_quality"), r.get("z_momentum")],
            })
            bar = alt.Chart(sub).mark_bar().encode(
                x=alt.X("z:Q", title="z-score", scale=alt.Scale(domain=[-5, 5])),
                y=alt.Y("sleeve:N", title=None),
                color=alt.condition(alt.datum.z > 0, alt.value("#2a9d8f"), alt.value("#e76f51")),
                tooltip=["sleeve", "z"],
            )
            st.altair_chart(bar, use_container_width=True)

        st.subheader("Average factor exposure of the screen")
        screened_panel = panel[panel["ticker"].isin(screened_tickers)]
        avg = screened_panel[["z_value", "z_quality", "z_momentum"]].mean().reset_index()
        avg.columns = ["sleeve", "avg_z"]
        avg["sleeve"] = avg["sleeve"].str.replace("z_", "").str.title()
        exp_chart = alt.Chart(avg).mark_bar().encode(
            x=alt.X("avg_z:Q", title="Average z-score across the screen"),
            y=alt.Y("sleeve:N", title=None),
            tooltip=["sleeve", "avg_z"],
        )
        st.altair_chart(exp_chart, use_container_width=True)

        st.subheader("Factor correlation")
        st.caption("Correlation across all eligible names (not just the screen) — "
                   "checks the sleeves aren't secretly measuring the same thing.")
        cols = [c for c in FACTORS if c in panel.columns]
        corr = panel[cols].corr()
        corr_long = corr.reset_index().melt(id_vars="index", var_name="col", value_name="corr")
        corr_long.columns = ["row", "col", "corr"]
        corr_long["row"] = corr_long["row"].map(FACTOR_LABELS).fillna(corr_long["row"])
        corr_long["col"] = corr_long["col"].map(FACTOR_LABELS).fillna(corr_long["col"])
        heat = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X("col:N", title=None), y=alt.Y("row:N", title=None),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["row", "col", alt.Tooltip("corr:Q", format=".2f")],
        )
        st.altair_chart(heat, use_container_width=True)

with tab2:
    if bt is None or chosen_freq is None:
        st.info("No backtest output yet — run `python run.py backtest`.")
    else:
        bt_f = bt[bt["freq"] == chosen_freq] if "freq" in bt.columns else bt
        st.subheader(f"Quintile backtest ({chosen_freq})")
        piv = bt_f.pivot_table(index="date", columns="quantile", values="ret").sort_index()
        cum = (1 + piv).cumprod()
        cum.columns = [f"Q{int(c)}" for c in cum.columns]
        st.line_chart(cum)

        sm = summary[summary["freq"] == chosen_freq] if summary is not None else None
        if sm is not None and len(sm):
            row = sm.iloc[0]
            st.caption(f"Weighting inside each quintile: **{row.get('weighting', 'equal')}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Q5−Q1 gross / net", f"{row['gross_spread_mean'] * 100:.2f}% / "
                     f"{row['net_spread_mean'] * 100:.2f}%")
            c2.metric("Sharpe gross / net", f"{row['gross_sharpe']:.2f} / {row['net_sharpe']:.2f}")
            c3.metric("Max Drawdown", f"{row['max_drawdown'] * 100:.1f}%")
            c4.metric("Turnover (ann.)", f"{row['turnover_annualized'] * 100:.0f}%")
            st.caption(f"Net figures subtract transaction costs (commission + spread + "
                      f"short rebate) scaled by realized turnover — net is always ≤ "
                      f"gross. Mean IC = {row['mean_ic']:.4f} "
                      f"(Newey-West t = {row['ic_t_stat_hac']:.2f}).")
        else:
            # Fall back to a quick recompute if backtest_summary hasn't been exported yet
            spread = piv[piv.columns.max()] - piv[piv.columns.min()]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Q5−Q1 (mean, {chosen_freq})", f"{spread.mean() * 100:.2f}%")
            c2.metric("Q5−Q1 Sharpe (gross)", f"{sharpe(spread):.2f}")
            c3.metric("Q5−Q1 Max Drawdown", f"{max_drawdown(spread) * 100:.1f}%")

        if decay is not None:
            decay_f = decay[decay["freq"] == chosen_freq] if "freq" in decay.columns else decay
            st.subheader("IC decay")
            st.caption("Mean Information Coefficient at increasing forward lags — how fast "
                       "the signal's predictive power fades.")
            decay_chart = alt.Chart(decay_f).mark_line(point=True).encode(
                x=alt.X("lag:O", title="Lag (rebalances ahead)"),
                y=alt.Y("ic:Q", title="Mean IC"),
                tooltip=["lag", alt.Tooltip("ic:Q", format=".4f")],
            )
            st.altair_chart(decay_chart, use_container_width=True)

        if fallback is not None:
            has_fb_freq = "freq" in fallback.columns
            fb_f = fallback[fallback["freq"] == chosen_freq] if has_fb_freq else fallback
            st.subheader("Sector-neutral normalization fallback (mean % across the backtest)")
            st.caption("Which hierarchy level each sleeve typically resolved at — industry group "
                      "(finest, most neutral), sector, or cross-sectional (last resort).")
            fb_avg = fb_f.groupby(["sleeve", "level"], as_index=False)["pct"].mean()
            fb_chart = alt.Chart(fb_avg).mark_bar().encode(
                x=alt.X("pct:Q", title="Mean % of names resolved at this level"),
                y=alt.Y("sleeve:N", title=None),
                color=alt.Color("level:N", title="Hierarchy level"),
                tooltip=["sleeve", "level", alt.Tooltip("pct:Q", format=".1f")],
            )
            st.altair_chart(fb_chart, use_container_width=True)

with tab3:
    ffd_f = (ffd[ffd["freq"] == chosen_freq] if ffd is not None and "freq" in ffd.columns
             and chosen_freq is not None else ffd)
    if ffd_f is None or not len(ffd_f):
        st.info("No decomposition yet — run `python run.py decompose`.")
    else:
        st.subheader(f"Fama-French 5 + Momentum decomposition ({chosen_freq or ''} Q5−Q1 "
                     f"spread, self-financing)")
        st.dataframe(ffd_f.drop(columns=["freq"], errors="ignore"),
                     use_container_width=True, hide_index=True)
        st.caption("Positive alpha after FF5 + UMD ⇒ a premium not explained by the known factors.")

with tab4:
    if betas is None or opt_weights is None:
        st.info("No optimizer output yet — run `python run.py optimize` "
                "(needs `python run.py screen` first).")
    else:
        st.subheader("Security Market Line")
        st.caption("Each screened name's beta vs. its realized mean excess return. The line is "
                   "the CAPM prediction (E[R]−Rf = β × market premium); points above it earned "
                   "more than their systematic risk alone would predict (positive alpha).")
        b = betas.dropna(subset=["beta", "mean_excess_return_annualized"])
        if len(b):
            mkt_premium = float(b["mkt_excess_return_annualized"].iloc[0])
            beta_range = pd.DataFrame({
                "beta": [b["beta"].min(), b["beta"].max()],
            })
            beta_range["sml"] = beta_range["beta"] * mkt_premium
            points = alt.Chart(b).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("beta:Q", title="Beta"),
                y=alt.Y("mean_excess_return_annualized:Q", title="Mean excess return (annualized)",
                        axis=alt.Axis(format="%")),
                color=alt.Color("alpha_annualized:Q", title="Alpha",
                                 scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=["ticker", alt.Tooltip("beta:Q", format=".2f"),
                         alt.Tooltip("mean_excess_return_annualized:Q", format=".1%"),
                         alt.Tooltip("alpha_annualized:Q", format=".1%", title="alpha")],
            )
            line = alt.Chart(beta_range).mark_line(color="gray", strokeDash=[4, 4]).encode(
                x="beta:Q", y="sml:Q",
            )
            st.altair_chart(points + line, use_container_width=True)
            st.caption(f"Average beta = {b['beta'].mean():.2f}. Market risk premium used for the "
                      f"line = {mkt_premium:.1%}/yr (from the Ken French library, same window).")

        st.subheader("Efficient frontier")
        if frontier is not None and len(frontier) and len(opt_weights):
            frontier_chart = alt.Chart(frontier).mark_line(point=True, color="steelblue").encode(
                x=alt.X("volatility:Q", title="Volatility (annualized)", axis=alt.Axis(format="%")),
                y=alt.Y("target_return:Q", title="Expected return (annualized)",
                        axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("volatility:Q", format=".1%"),
                         alt.Tooltip("target_return:Q", format=".1%")],
            )
            st.altair_chart(frontier_chart, use_container_width=True)
            st.caption("Long-only minimum-variance frontier over the screened names (estimation-"
                      "error sensitive — small-sample means/covariances, not a forecast).")

        st.subheader("Optimal portfolio weights")
        method = st.radio("Method", ["max_sharpe", "min_variance"], horizontal=True,
                          format_func=lambda m: "Max Sharpe (CAPM market portfolio)"
                          if m == "max_sharpe" else "Minimum variance")
        w = opt_weights[opt_weights["method"] == method].sort_values("weight", ascending=False)
        w = w[w["weight"] > 0.005]  # hide near-zero long-only weights for readability
        weight_chart = alt.Chart(w).mark_bar().encode(
            x=alt.X("weight:Q", title="Weight", axis=alt.Axis(format="%")),
            y=alt.Y("ticker:N", sort="-x", title=None),
            tooltip=["ticker", alt.Tooltip("weight:Q", format=".1%")],
        )
        st.altair_chart(weight_chart, use_container_width=True)

with tab5:
    if ma_weights is None:
        st.info("No multi-asset output yet — run `python run.py multiasset`.")
    else:
        st.subheader("Cross-asset allocation (equity / bond / commodity ETF proxies)")
        st.caption("Bonds and commodities enter as liquid ETFs, not the equity factor screen — "
                   "the factor engine (P/E, ROE, GICS z-scores) has no meaning for them, and free "
                   "per-security data doesn't exist. This is the asset-allocation view.")
        by_class = ma_weights.groupby("asset_class", as_index=False)["weight"].sum()
        donut = alt.Chart(by_class).mark_arc(innerRadius=60).encode(
            theta=alt.Theta("weight:Q"),
            color=alt.Color("asset_class:N", title="Asset class"),
            tooltip=["asset_class", alt.Tooltip("weight:Q", format=".1%")],
        )
        st.altair_chart(donut, use_container_width=True)

        st.caption("Max-Sharpe weights by ETF:")
        w = ma_weights[ma_weights["weight"] > 0.005].sort_values("weight", ascending=False)
        etf_chart = alt.Chart(w).mark_bar().encode(
            x=alt.X("weight:Q", title="Weight", axis=alt.Axis(format="%")),
            y=alt.Y("ticker:N", sort="-x", title=None),
            color=alt.Color("asset_class:N", title="Asset class"),
            tooltip=["ticker", "asset_class", alt.Tooltip("weight:Q", format=".1%")],
        )
        st.altair_chart(etf_chart, use_container_width=True)

        if ma_frontier is not None and len(ma_frontier):
            st.subheader("Cross-asset efficient frontier")
            ma_front_chart = alt.Chart(ma_frontier).mark_line(point=True, color="steelblue").encode(
                x=alt.X("volatility:Q", title="Volatility (annualized)", axis=alt.Axis(format="%")),
                y=alt.Y("target_return:Q", title="Expected return (annualized)",
                        axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("volatility:Q", format=".1%"),
                         alt.Tooltip("target_return:Q", format=".1%")],
            )
            st.altair_chart(ma_front_chart, use_container_width=True)

with tab6:
    if div_screen is None:
        st.info("No dividend screen yet — run `python run.py dividend`.")
    else:
        st.subheader("Dividend-income screen")
        st.caption("Ranked by RECURRING (regular) trailing-12m dividend yield — special/variable "
                   "payouts (e.g. a one-off large annual dividend) are excluded from the ranking "
                   "yield but shown in the total, flagged via 'has_special_dividend'. Gated on "
                   "payout sustainability (total dividends / earnings) — a high yield funded by "
                   "~all of earnings is a cut waiting to happen, so those are dropped.")
        show = div_screen.copy()
        show["regular_dividend_yield"] = (show["regular_dividend_yield"] * 100).round(2)
        show["total_dividend_yield"] = (show["total_dividend_yield"] * 100).round(2)
        show["payout_ratio"] = (show["payout_ratio"] * 100).round(0)
        cols = ["rank", "ticker", "regular_dividend_yield", "total_dividend_yield",
                "has_special_dividend", "payout_ratio", "sector", "weight"]
        st.dataframe(show[[c for c in cols if c in show.columns]],
                     use_container_width=True, hide_index=True)

        st.subheader("Yield vs. payout ratio")
        st.caption("Top-right = high recurring yield but stretched payout (riskier); "
                   "left = safer coverage. The gate already removed payout > ceiling.")
        scatter = alt.Chart(div_screen).mark_circle(size=90, opacity=0.7).encode(
            x=alt.X("payout_ratio:Q", title="Payout ratio", axis=alt.Axis(format="%")),
            y=alt.Y("regular_dividend_yield:Q", title="Regular dividend yield",
                    axis=alt.Axis(format="%")),
            color=alt.Color("sector:N", title="Sector") if "sector" in div_screen.columns
            else alt.value("#2a9d8f"),
            tooltip=["ticker", alt.Tooltip("regular_dividend_yield:Q", format=".2%"),
                     alt.Tooltip("total_dividend_yield:Q", format=".2%"),
                     alt.Tooltip("payout_ratio:Q", format=".0%")],
        )
        st.altair_chart(scatter, use_container_width=True)
