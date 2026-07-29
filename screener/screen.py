"""Hard filters + the final ranked screen.

Applies config hard filters (min market cap, positive earnings, liquidity proxy)
to the factor cross-section, then ranks the survivors by composite score and
returns the top decile / top-N.
"""

from __future__ import annotations

import pandas as pd

from .portfolio import select


def apply_hard_filters(factors_df: pd.DataFrame, filters) -> pd.Index:
    """Return the tickers passing the hard filters."""
    mask = pd.Series(True, index=factors_df.index)
    if "market_cap" in factors_df:
        mask &= factors_df["market_cap"] >= filters.min_market_cap
    if filters.positive_earnings and "ep" in factors_df:
        # ep is NaN'd for non-positive-earnings / out-of-bounds P/E, so ep > 0 encodes it.
        mask &= factors_df["ep"] > 0
    return factors_df.index[mask.fillna(False)]


def build_screen(
    composite_scores: pd.Series, factors_df: pd.DataFrame, filters, method: str = "top_decile",
    top_n: int = 50,
) -> pd.DataFrame:
    """Ranked screen: filter -> rank by composite score -> select top decile/N."""
    eligible = apply_hard_filters(factors_df, filters)
    scores = composite_scores.reindex(eligible).dropna()
    selected = select(scores, method, top_n)
    out = pd.DataFrame({"score": scores.reindex(selected)}).sort_values("score", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out
