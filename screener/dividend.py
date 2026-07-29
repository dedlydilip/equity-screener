"""Dividend-income screen: yield + payout sustainability.

An income investor wants a high yield that is also *sustainable* — a high yield
paid out of nearly all (or more than all) of earnings is a cut waiting to happen.
So this ranks on trailing dividend yield but gates on the payout ratio
(dividends / net income) computed from the EDGAR fundamentals already pulled for
the factor engine — no extra fundamentals source needed.

Yield here is trailing-twelve-month cash dividends / current price: a
backward-looking income measure, not a forward dividend forecast.
"""

from __future__ import annotations

import pandas as pd


def dividend_yield(dividends_ttm: pd.Series, price: pd.Series) -> pd.Series:
    """Trailing-12m dividends per share / current price."""
    return (dividends_ttm / price.replace(0, pd.NA)).astype(float)


def payout_ratio(
    dividends_ttm: pd.Series, shares: pd.Series, net_income_ttm: pd.Series
) -> pd.Series:
    """Total dividends paid / net income. NaN when earnings are non-positive
    (a payout ratio off negative or zero earnings is meaningless, not "safe")."""
    total_dividends = dividends_ttm * shares
    return (total_dividends / net_income_ttm.where(net_income_ttm > 0)).astype(float)


def build_dividend_screen(
    dividends_ttm: pd.Series, price: pd.Series, shares: pd.Series, net_income_ttm: pd.Series,
    sector: pd.Series | None = None, min_yield: float = 0.0, max_payout: float = 0.90,
    top_n: int = 30,
) -> pd.DataFrame:
    """Rank by yield among names passing the sustainability (payout) gate.

    Returns columns: ``ticker, dividend_yield, payout_ratio, [sector,] rank`` —
    income-weighted (weight proportional to yield) so higher yielders carry more
    of the income portfolio.
    """
    df = pd.DataFrame({
        "dividend_yield": dividend_yield(dividends_ttm, price),
        "payout_ratio": payout_ratio(dividends_ttm, shares, net_income_ttm),
    })
    if sector is not None:
        df["sector"] = sector.reindex(df.index)

    eligible = df[
        (df["dividend_yield"] > min_yield)
        & (df["payout_ratio"] > 0)
        & (df["payout_ratio"] <= max_payout)
    ].copy()
    eligible = eligible.sort_values("dividend_yield", ascending=False).head(top_n)
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    total_yield = eligible["dividend_yield"].sum()
    eligible["weight"] = (eligible["dividend_yield"] / total_yield) if total_yield else 0.0
    return eligible.reset_index().rename(columns={"index": "ticker"})
