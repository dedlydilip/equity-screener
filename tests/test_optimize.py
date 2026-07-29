"""Mean-variance optimizer: closed-form sanity checks on small synthetic cases."""

import numpy as np
import pandas as pd
import pytest

from screener.optimize import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_stats,
)


def test_min_variance_matches_two_asset_closed_form():
    # Two UNCORRELATED assets: closed-form min-var weight_i is proportional to 1/var_i.
    idx = ["LOW_VAR", "HIGH_VAR"]
    var1, var2 = 0.01, 0.04
    cov = pd.DataFrame([[var1, 0.0], [0.0, var2]], index=idx, columns=idx)

    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()
    expected_low = (1 / var1) / (1 / var1 + 1 / var2)
    assert w["LOW_VAR"] == pytest.approx(expected_low, abs=0.02)
    assert w["LOW_VAR"] > w["HIGH_VAR"]


def test_max_sharpe_prefers_the_better_risk_adjusted_asset():
    idx = ["GOOD", "BAD"]
    mean = pd.Series({"GOOD": 0.15, "BAD": 0.05}, index=idx)  # higher return
    cov = pd.DataFrame([[0.02, 0.0], [0.0, 0.06]], index=idx, columns=idx)  # AND lower risk

    w = max_sharpe_weights(mean, cov, rf=0.02)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()
    assert w["GOOD"] > w["BAD"]


def test_portfolio_stats_matches_hand_calculation():
    idx = ["A", "B"]
    w = pd.Series({"A": 0.5, "B": 0.5}, index=idx)
    mean = pd.Series({"A": 0.10, "B": 0.20}, index=idx)
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=idx, columns=idx)

    stats = portfolio_stats(w, mean, cov, rf=0.02)
    assert stats["expected_return"] == pytest.approx(0.15, abs=1e-9)
    expected_vol = np.sqrt(0.25 * 0.04 + 0.25 * 0.09)
    assert stats["volatility"] == pytest.approx(expected_vol, abs=1e-9)
    assert stats["sharpe"] == pytest.approx((0.15 - 0.02) / expected_vol, abs=1e-9)


def test_efficient_frontier_is_monotonically_non_decreasing_in_risk():
    idx = ["A", "B", "C"]
    mean = pd.Series({"A": 0.05, "B": 0.10, "C": 0.15}, index=idx)
    cov = pd.DataFrame(np.diag([0.01, 0.02, 0.05]), index=idx, columns=idx)

    frontier = efficient_frontier(mean, cov, n_points=10)
    assert len(frontier) >= 3
    vols = frontier.sort_values("target_return")["volatility"].values
    # The efficient (upper) branch only: risk must not decrease as target return
    # rises. The inefficient lower branch is dropped by efficient_frontier().
    assert all(vols[i + 1] >= vols[i] - 1e-6 for i in range(len(vols) - 1))
    # The lowest-risk point returned IS the global-minimum-variance point.
    assert vols[0] == pytest.approx(frontier["volatility"].min(), abs=1e-9)


def test_weights_are_long_only_even_with_negative_expected_returns():
    idx = ["A", "B"]
    mean = pd.Series({"A": -0.05, "B": 0.10}, index=idx)
    cov = pd.DataFrame([[0.02, 0.0], [0.0, 0.02]], index=idx, columns=idx)
    w = max_sharpe_weights(mean, cov, rf=0.0)
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
