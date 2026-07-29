import numpy as np
import pandas as pd

from screener.data.quality import (
    mad,
    mask_negative_denominator,
    nan_out_of_bounds,
    winsorize_cross_sectional,
)


def test_mad_of_constant_series_is_zero():
    assert mad(pd.Series([5.0] * 10)) == 0.0


def test_winsorize_clips_a_single_outlier():
    s = pd.Series([1.0, 1.1, 0.9, 1.05, 0.95, 500.0])  # last value is a wild outlier
    out = winsorize_cross_sectional(s, n_mad=10.0)
    assert out.iloc[-1] < 500.0
    assert out.iloc[:-1].equals(s.iloc[:-1])  # well-behaved values untouched


def test_winsorize_is_noop_when_mad_is_zero():
    s = pd.Series([2.0, 2.0, 2.0, 2.0])
    out = winsorize_cross_sectional(s, n_mad=10.0)
    assert out.equals(s)


def test_nan_out_of_bounds():
    s = pd.Series([-5.0, 0.0, 10.0, 501.0, 250.0])
    out = nan_out_of_bounds(s, lo=0.0, hi=500.0)
    assert out.tolist()[:2] == [np.nan, 0.0] or (np.isnan(out.iloc[0]) and out.iloc[1] == 0.0)
    assert out.iloc[2] == 10.0
    assert np.isnan(out.iloc[3])
    assert out.iloc[4] == 250.0


def test_mask_negative_denominator_nans_only_negative_side():
    value = pd.Series([0.4, 0.4, 0.4])
    denom = pd.Series([200.0, -50.0, 0.0])
    out = mask_negative_denominator(value, denom)
    assert out.iloc[0] == 0.4
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])  # zero denominator also masked (not > 0)
