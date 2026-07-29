"""MAD hierarchy fallback, clipping, and inverse-vol sleeve weighting."""

import numpy as np
import pandas as pd
import pytest

from screener.normalize import (
    _cap_and_redistribute,
    composite,
    inverse_vol_weights,
    robust_z_in_group,
    sector_neutral_z,
)

HIER = [
    {"level": "industry_group", "min_n": 20},
    {"level": "sector", "min_n": 20},
    {"level": "cross_sectional"},
]


def _synthetic_universe(n=120):
    tickers = [f"T{i}" for i in range(n)]
    # 6 industry groups of 20 (all pass min_n=20); 3 sectors of 40 (also pass)
    ig = [f"IG{i % 6}" for i in range(n)]
    sec = [f"SEC{i % 3}" for i in range(n)]
    return pd.DataFrame({"industry_group": ig, "sector": sec}, index=tickers)


def test_hierarchy_resolves_at_finest_level_when_groups_are_large_enough():
    uni = _synthetic_universe(120)
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(size=120), index=uni.index)
    z, pct = sector_neutral_z(values, uni, HIER, clip=5.0)
    assert pct["industry_group"] == 100.0
    assert pct["sector"] == 0.0
    assert pct["cross_sectional"] == 0.0
    assert z.notna().all()


def test_hierarchy_falls_back_to_sector_when_industry_groups_too_small():
    n = 90
    tickers = [f"T{i}" for i in range(n)]
    # industry groups of 10 (< min_n=20) -> must fall back to sector (groups of 30, >= 20)
    ig = [f"IG{i % 9}" for i in range(n)]
    sec = [f"SEC{i % 3}" for i in range(n)]
    uni = pd.DataFrame({"industry_group": ig, "sector": sec}, index=tickers)
    values = pd.Series(np.random.default_rng(1).normal(size=n), index=tickers)
    z, pct = sector_neutral_z(values, uni, HIER, clip=5.0)
    assert pct["industry_group"] == 0.0
    assert pct["sector"] == 100.0


def test_hierarchy_falls_back_to_cross_sectional_as_last_resort():
    n = 10  # too small for either group-level (min_n=20)
    tickers = [f"T{i}" for i in range(n)]
    uni = pd.DataFrame({"industry_group": ["IGx"] * n, "sector": ["SECx"] * n}, index=tickers)
    values = pd.Series(np.random.default_rng(2).normal(size=n), index=tickers)
    z, pct = sector_neutral_z(values, uni, HIER, clip=5.0)
    assert pct["cross_sectional"] == 100.0


def test_extreme_outlier_is_clipped_to_the_configured_bound():
    uni = _synthetic_universe(120)
    values = pd.Series(np.random.default_rng(0).normal(size=120), index=uni.index)
    values.iloc[0] = 1e6  # an absurd outlier
    z, _ = sector_neutral_z(values, uni, HIER, clip=5.0)
    assert z.iloc[0] == 5.0
    assert z.max() <= 5.0 and z.min() >= -5.0


def test_robust_z_in_group_is_zero_at_the_group_median():
    groups = pd.Series(["A"] * 5 + ["B"] * 5)
    values = pd.Series([1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
    z = robust_z_in_group(values, groups)
    assert np.isclose(z.iloc[2], 0.0)  # median of group A
    assert np.isclose(z.iloc[7], 0.0)  # median of group B


def test_composite_is_weight_normalized():
    idx = ["A", "B"]
    z = {"value": pd.Series([1.0, -1.0], index=idx), "quality": pd.Series([2.0, 2.0], index=idx)}
    out = composite(z, {"value": 1.0, "quality": 1.0})
    assert np.isclose(out["A"], 1.5)
    assert np.isclose(out["B"], 0.5)


def test_burn_in_gives_equal_weights_before_lookback_window_exists():
    sleeves = pd.DataFrame(np.random.default_rng(0).normal(0, 0.02, size=(6, 3)),
                            columns=["value", "quality", "momentum"])
    w = inverse_vol_weights(sleeves, lookback=12, max_weight=0.5, burn_in=12, as_of_row=5)
    assert all(np.isclose(v, 1 / 3) for v in w.values())


def test_after_burn_in_low_vol_sleeve_gets_more_weight():
    n = 24
    rng = np.random.default_rng(0)
    sleeves = pd.DataFrame({
        "value": rng.normal(0, 0.005, n),      # low vol
        "quality": rng.normal(0, 0.05, n),     # high vol
        "momentum": rng.normal(0, 0.02, n),
    })
    w = inverse_vol_weights(sleeves, lookback=12, max_weight=0.9, burn_in=12, as_of_row=20)
    assert w["value"] > w["quality"]


def test_cap_and_redistribute_enforces_max_weight():
    w = pd.Series({"value": 0.8, "quality": 0.15, "momentum": 0.05})
    out = _cap_and_redistribute(w, cap=0.5)
    assert out["value"] == pytest.approx(0.5, abs=1e-6)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)
    assert out["quality"] > 0.15  # received redistributed excess
