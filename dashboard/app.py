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
ffd = load("ff_decomposition")
uni = load("universe")
panel = load("factor_panel")
decay = load("ic_decay")

if screen is None:
    st.warning("No outputs yet. Build them first: `python run.py screen`, "
               "then `backtest` / `decompose`.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Screen", "Backtest", "Factor decomposition"])

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
    if bt is None:
        st.info("No backtest output yet — run `python run.py backtest`.")
    else:
        st.subheader("Quintile backtest")
        piv = bt.pivot_table(index="date", columns="quantile", values="ret").sort_index()
        cum = (1 + piv).cumprod()
        cum.columns = [f"Q{int(c)}" for c in cum.columns]
        st.line_chart(cum)
        spread = piv[piv.columns.max()] - piv[piv.columns.min()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Q5−Q1 (mean, monthly)", f"{spread.mean() * 100:.2f}%")
        c2.metric("Q5−Q1 Sharpe", f"{sharpe(spread):.2f}")
        c3.metric("Q5−Q1 Max Drawdown", f"{max_drawdown(spread) * 100:.1f}%")

        if decay is not None:
            st.subheader("IC decay")
            st.caption("Mean Information Coefficient at increasing forward lags — how fast "
                       "the signal's predictive power fades.")
            decay_chart = alt.Chart(decay).mark_line(point=True).encode(
                x=alt.X("lag:O", title="Lag (rebalances ahead)"),
                y=alt.Y("ic:Q", title="Mean IC"),
                tooltip=["lag", alt.Tooltip("ic:Q", format=".4f")],
            )
            st.altair_chart(decay_chart, use_container_width=True)

with tab3:
    if ffd is None:
        st.info("No decomposition yet — run `python run.py decompose`.")
    else:
        st.subheader("Fama-French 5 + Momentum decomposition (Q5−Q1 spread, self-financing)")
        st.dataframe(ffd, use_container_width=True, hide_index=True)
        st.caption("Positive alpha after FF5 + UMD ⇒ a premium not explained by the known factors.")
