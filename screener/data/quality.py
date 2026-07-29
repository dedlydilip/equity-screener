"""Data-quality gate — runs BEFORE factor normalization.

Stops one bad data point (a P/E of 9,999, a sign-flipped EV/EBITDA) from
poisoning a whole group's median/MAD. Pure functions so they're trivially
unit-testable.

Layers:
  1. Hard bounds        -> values outside a sane range become NaN.
  2. Cross-sectional    -> values > n MAD from the cross-sectional median are
     winsorization         clipped to the boundary.
  3. Negative-denominator rules (per factor) so distressed firms don't look
     "cheap":
       * Book-to-Price  -> NaN when book value < 0
       * EV/EBITDA      -> NaN when EBITDA < 0
       * FCF yield      -> keep negatives (they correctly sort to the bottom)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of sigma for a normal


def mad(x: pd.Series) -> float:
    """Median Absolute Deviation (raw, unscaled)."""
    x = x.dropna().astype(float)
    if x.empty:
        return float("nan")
    return float((x - x.median()).abs().median())


def winsorize_cross_sectional(s: pd.Series, n_mad: float) -> pd.Series:
    """Clip values beyond ``n_mad`` scaled-MADs of the cross-sectional median."""
    s = s.astype(float)
    med = s.median()
    m = mad(s)
    if not np.isfinite(m) or m == 0:
        return s
    scale = MAD_SCALE * m
    return s.clip(lower=med - n_mad * scale, upper=med + n_mad * scale)


def nan_out_of_bounds(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Set values outside [lo, hi] to NaN (hard sanity bounds)."""
    s = s.astype(float)
    return s.where((s >= lo) & (s <= hi))


def mask_negative_denominator(value: pd.Series, denominator: pd.Series) -> pd.Series:
    """NaN out a ratio wherever its denominator is negative (or zero)."""
    return value.where(denominator > 0)
