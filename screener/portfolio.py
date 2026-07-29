"""Portfolio construction: quantile assignment, name selection, weighting, and
concentration diagnostics.

Weighting inside the selected names is config-driven (equal / market_cap /
inverse_vol). Concentration (HHI + top-10% weight + effective N) answers the
PM's question: "is the top quantile just a handful of names?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_quantiles(scores: pd.Series, n: int = 5) -> pd.Series:
    """Assign 1..n by composite score (n = highest score = best). Ties broken by rank."""
    s = scores.dropna()
    if s.empty:
        return pd.Series(index=scores.index, dtype="float")
    labels = pd.qcut(s.rank(method="first"), n, labels=list(range(1, n + 1)))
    return pd.Series(labels, index=s.index).reindex(scores.index)


def select(scores: pd.Series, method: str = "top_decile", top_n: int = 50) -> pd.Index:
    """Names selected for the screen: top decile (10%) or top-N by score."""
    s = scores.dropna().sort_values(ascending=False)
    k = min(top_n, len(s)) if method == "top_n" else max(1, round(len(s) * 0.10))
    return s.head(k).index


def weights(
    names: pd.Index,
    method: str = "equal",
    market_cap: pd.Series | None = None,
    vols: pd.Series | None = None,
) -> pd.Series:
    """Portfolio weights within the selected names (sum to 1)."""
    names = pd.Index(names)
    if method == "market_cap":
        if market_cap is None:
            raise ValueError("market_cap required for market_cap weighting")
        w = market_cap.reindex(names).clip(lower=0).fillna(0.0)
    elif method == "inverse_vol":
        if vols is None:
            raise ValueError("vols required for inverse_vol weighting")
        w = (1.0 / vols.reindex(names).replace(0, np.nan))
        w = w.fillna(w.mean())
    else:  # equal
        w = pd.Series(1.0, index=names)
    total = float(w.sum())
    return (w / total) if total else pd.Series(1.0 / len(names), index=names)


def concentration(w: pd.Series) -> dict:
    """HHI, top-10%-of-names weight, and effective N (higher HHI = more concentrated)."""
    w = w.dropna()
    if w.empty:
        return {"hhi": float("nan"), "top_10pct_weight": float("nan"), "effective_n": float("nan")}
    hhi = float((w**2).sum())
    k = max(1, round(len(w) * 0.10))
    top = float(w.sort_values(ascending=False).head(k).sum())
    eff_n = 1.0 / hhi if hhi else float("nan")
    return {"hhi": hhi, "top_10pct_weight": top, "effective_n": eff_n}
