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
    """Assign 1..n by composite score (n = highest score = best). Ties broken by rank.

    Returns all-NaN when the cross-section can't support n buckets -- fewer
    names than buckets silently produced non-contiguous labels (3 names into
    n=5 gave [1, 3, 5], i.e. empty buckets and a single-name "top quintile"),
    and fewer than n distinct scores means the split is decided by column order
    rather than by signal.
    """
    s = scores.dropna()
    if len(s) < n or s.nunique() < n:
        return pd.Series(np.nan, index=scores.index, dtype="float")
    labels = pd.qcut(s.rank(method="first"), n, labels=list(range(1, n + 1)))
    return pd.Series(labels, index=s.index).reindex(scores.index)


def select(scores: pd.Series, method: str = "top_decile", top_n: int = 50) -> pd.Index:
    """Names selected for the screen: top decile (10%) or top-N by score."""
    if method not in ("top_decile", "top_n"):
        raise ValueError(f"unknown selection method {method!r}; use 'top_decile' or 'top_n'")
    s = scores.dropna().sort_values(ascending=False)
    k = min(top_n, len(s)) if method == "top_n" else max(1, round(len(s) * 0.10))
    return s.head(k).index


def weights(
    names: pd.Index,
    method: str = "equal",
    market_cap: pd.Series | None = None,
    vols: pd.Series | None = None,
) -> pd.Series:
    """Portfolio weights within the selected names (sum to 1).

    Raises rather than silently degrading when the panel a scheme needs is
    entirely missing: an all-NaN market-cap panel used to fall through to equal
    weights, so a cap-weighted backtest could quietly become equal-weighted.
    """
    names = pd.Index(names)
    if not len(names):
        return pd.Series(dtype=float)
    if method == "market_cap":
        if market_cap is None:
            raise ValueError("market_cap required for market_cap weighting")
        w = market_cap.reindex(names).clip(lower=0)
        if not (w > 0).any():
            raise ValueError("market_cap weighting requested but no name has a positive cap")
        w = w.fillna(0.0)
    elif method == "inverse_vol":
        if vols is None:
            raise ValueError("vols required for inverse_vol weighting")
        v = vols.reindex(names).astype(float)
        positive = v[np.isfinite(v) & (v > 0)]
        if positive.empty:
            raise ValueError("inverse_vol weighting requested but no name has a positive vol")
        # Floor volatility before inverting so one stale/halted near-constant
        # price series can't take essentially the whole bucket (an unfloored
        # 1e-8 vol produced a 99.99998% single-name weight). Anchoring the floor
        # to a quarter of the cross-sectional median bounds the inverse-vol
        # ratio between any two names at ~4x the median's.
        floor = max(float(positive.median()) * 0.25, 1e-12)
        w = 1.0 / v.clip(lower=floor)
        w = w.fillna(float(w.median()))   # unmeasurable -> neutral, not best
    elif method == "equal":
        w = pd.Series(1.0, index=names)
    else:
        raise ValueError(f"unknown weighting method {method!r}")
    total = float(w.sum())
    return (w / total) if total else pd.Series(1.0 / len(names), index=names)


def concentration(w: pd.Series) -> dict:
    """HHI, top-10%-of-names weight, and effective N (higher HHI = more concentrated).

    Weights are renormalized to sum to 1 first: HHI and effective N are only
    interpretable on a normalized book, and feeding an unnormalized vector in
    produced nonsense like "effective N = 0.125 names".
    """
    w = w.dropna()
    if w.empty:
        return {"hhi": float("nan"), "top_10pct_weight": float("nan"), "effective_n": float("nan")}
    total = float(w.sum())
    if total <= 0:
        return {"hhi": float("nan"), "top_10pct_weight": float("nan"), "effective_n": float("nan")}
    w = w / total
    hhi = float((w**2).sum())
    k = max(1, round(len(w) * 0.10))
    top = float(w.sort_values(ascending=False).head(k).sum())
    eff_n = 1.0 / hhi if hhi else float("nan")
    return {"hhi": hhi, "top_10pct_weight": top, "effective_n": eff_n}
