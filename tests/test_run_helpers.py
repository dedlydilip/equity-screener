"""Pipeline helpers in ``run.py`` that carry real methodology.

These exist because two frequency-conversion bugs shipped inside the backtest
loop and silently corrupted published figures: a month-denominated config window
consumed as ``iloc`` row counts, and a monthly risk-free rate charged against a
quarterly return. Both now live in named, tested helpers rather than inline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import _ff_at_freq, _months_per_period, _months_to_rows  # noqa: E402


def test_months_per_period():
    assert _months_per_period("monthly") == 1
    assert _months_per_period("quarterly") == 3


def test_months_to_rows_converts_a_12_month_window_at_each_frequency():
    """The config asks for a 12-MONTH window. Monthly that is 12 rows; quarterly
    it is 4 rows. Passing 12 straight through gave quarterly a 36-month window."""
    assert _months_to_rows(12, "monthly") == 12
    assert _months_to_rows(12, "quarterly") == 4


def test_months_to_rows_never_returns_zero():
    # A window shorter than one period must still leave at least one row, or
    # inverse_vol_weights would slice an empty window and emit NaN weights.
    assert _months_to_rows(1, "quarterly") == 1
    assert _months_to_rows(0, "monthly") == 1


def test_ff_at_freq_leaves_monthly_factors_on_month_ends():
    idx = pd.period_range("2020-01", periods=6, freq="M")
    ff = pd.DataFrame({"RF": 0.001, "Mkt_RF": 0.02}, index=idx)
    out = _ff_at_freq(ff, "monthly")
    assert len(out) == 6
    assert out["RF"].iloc[0] == pytest.approx(0.001)
    assert out.index[0] == pd.Timestamp("2020-01-31")


def test_ff_at_freq_compounds_the_risk_free_rate_for_a_quarterly_backtest():
    """A quarterly leg earns a 3-month return, so it must be charged the
    COMPOUNDED 3-month Rf -- not the raw 1-month rate, which under-subtracts."""
    idx = pd.period_range("2020-01", periods=6, freq="M")
    ff = pd.DataFrame({"RF": 0.001, "Mkt_RF": 0.02}, index=idx)
    out = _ff_at_freq(ff, "quarterly")
    assert len(out) == 2  # six months -> two quarters
    expected = (1.001**3) - 1
    assert out["RF"].iloc[0] == pytest.approx(expected, rel=1e-12)
    assert out["RF"].iloc[0] > 0.001  # strictly larger than the monthly rate
    assert list(out.index) == [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")]


def test_ff_at_freq_quarterly_index_aligns_with_quarter_end_rebalance_dates():
    """The compounded factors must land on the same timestamps the quarterly
    backtest indexes its spread by, or reindex() silently drops every period."""
    idx = pd.period_range("2020-01", periods=12, freq="M")
    ff = pd.DataFrame({"RF": 0.001}, index=idx)
    out = _ff_at_freq(ff, "quarterly")
    rebalances = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"])
    aligned = out["RF"].reindex(rebalances.to_period("Q").to_timestamp("Q"))
    assert aligned.notna().all()
