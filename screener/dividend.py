"""Dividend-income screen: recurring yield + payout sustainability.

An income investor wants a high yield that is also *recurring* and
*sustainable*. Two distinct adjustments are made to the naive
"trailing-cash-dividends / price" figure:

  1. **Regular vs. special.** Trailing cash dividends can include a one-off
     special/variable payout (e.g. Progressive's large annual variable
     dividend on top of its small regular quarterly one), which inflates the
     trailing yield relative to the income an investor should actually expect
     next year. `data/yf.get_dividends` splits payments into
     ``regular_dividends_ttm`` / ``special_dividends_ttm`` via a documented
     heuristic (a payment > N times the trailing median is "special"). The
     screen ranks on the **regular** yield — the recurring, expectable income
     — while still reporting the total (incl. specials) alongside for context.
  2. **Payout sustainability.** A high yield paid out of nearly all (or more
     than all) of earnings is a cut waiting to happen, so names are gated on
     the payout ratio (total dividends / net income, from the EDGAR
     fundamentals already pulled for the factor engine) — this intentionally
     uses TOTAL dividends, since a special payout is still real cash leaving
     the business regardless of whether it recurs.

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
    regular_dividends_ttm: pd.Series | None = None, sector: pd.Series | None = None,
    min_yield: float = 0.0, max_payout: float = 0.90, top_n: int = 30,
) -> pd.DataFrame:
    """Rank by RECURRING (regular) yield among names passing the payout gate.

    Returns columns: ``ticker, regular_dividend_yield, total_dividend_yield,
    has_special_dividend, payout_ratio, [sector,] rank, weight`` — income-weighted
    (weight proportional to the regular yield) so higher recurring-yielders carry
    more of the income portfolio. If ``regular_dividends_ttm`` is not supplied,
    the regular and total yields are identical (no regular/special split available).
    """
    total_yield_s = dividend_yield(dividends_ttm, price)
    reg_ttm = regular_dividends_ttm if regular_dividends_ttm is not None else dividends_ttm
    regular_yield_s = dividend_yield(reg_ttm, price)

    df = pd.DataFrame({
        "regular_dividend_yield": regular_yield_s,
        "total_dividend_yield": total_yield_s,
        "has_special_dividend": (dividends_ttm - reg_ttm).fillna(0) > 1e-9,
        "payout_ratio": payout_ratio(dividends_ttm, shares, net_income_ttm),
    })
    if sector is not None:
        df["sector"] = sector.reindex(df.index)

    eligible = df[
        (df["regular_dividend_yield"] > min_yield)
        & (df["payout_ratio"] > 0)
        & (df["payout_ratio"] <= max_payout)
    ].copy()
    eligible = eligible.sort_values("regular_dividend_yield", ascending=False).head(top_n)
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    total = eligible["regular_dividend_yield"].sum()
    eligible["weight"] = (eligible["regular_dividend_yield"] / total) if total else 0.0
    return eligible.reset_index().rename(columns={"index": "ticker"})
