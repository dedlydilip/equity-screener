"""Streamlit dashboard (primary) for the equity screener.

Reads the parquet exports written by ``screener.report.export_for_dashboard`` and
shows the ranked screen + sector exposure, the quintile backtest, and the
Fama-French decomposition. Run from the repo root:

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

from screener.validate import max_drawdown, sharpe  # noqa: E402


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

with tab3:
    if ffd is None:
        st.info("No decomposition yet — run `python run.py decompose`.")
    else:
        st.subheader("Fama-French 5 + Momentum decomposition (Q5−Q1 spread, self-financing)")
        st.dataframe(ffd, use_container_width=True, hide_index=True)
        st.caption("Positive alpha after FF5 + UMD ⇒ a premium not explained by the known factors.")
