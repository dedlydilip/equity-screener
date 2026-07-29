"""CAPM beta/alpha recovery on synthetic data with a known ground truth."""

import numpy as np
import pandas as pd
import pytest

from screener.capm import estimate_beta_alpha, estimate_betas_panel


@pytest.fixture
def ff_panel():
    rng = np.random.default_rng(7)
    months = pd.date_range("2018-01-31", periods=72, freq="ME")
    ff = pd.DataFrame({"Mkt_RF": rng.normal(0.006, 0.04, 72)}, index=months)
    ff["RF"] = 0.002
    return ff, rng


def test_recovers_known_beta_and_alpha(ff_panel):
    ff, rng = ff_panel
    true_beta, true_alpha = 1.4, 0.004
    noise = rng.normal(0, 0.01, len(ff))
    r = true_alpha + true_beta * ff["Mkt_RF"].values + ff["RF"].values + noise
    returns = pd.Series(r, index=ff.index)

    result = estimate_beta_alpha(returns, ff)
    assert result["beta"] == pytest.approx(true_beta, abs=0.1)
    assert result["alpha_monthly"] == pytest.approx(true_alpha, abs=0.003)
    assert result["r_squared"] > 0.7
    assert result["n"] == 72
    # SML inputs: mean excess return ~= alpha + beta * market excess return (OLS identity).
    implied = result["alpha_annualized"] + result["beta"] * result["mkt_excess_return_annualized"]
    assert result["mean_excess_return_annualized"] == pytest.approx(implied, abs=1e-6)


def test_defensive_market_has_beta_below_one(ff_panel):
    ff, rng = ff_panel
    noise = rng.normal(0, 0.01, len(ff))
    r = 0.003 + 0.4 * ff["Mkt_RF"].values + ff["RF"].values + noise
    returns = pd.Series(r, index=ff.index)
    result = estimate_beta_alpha(returns, ff)
    assert result["beta"] < 1.0


def test_panel_ranks_betas_correctly(ff_panel):
    ff, rng = ff_panel
    tickers = {"LOW_BETA": 0.3, "MARKET_BETA": 1.0, "HIGH_BETA": 1.8}
    data = {}
    for name, beta in tickers.items():
        noise = rng.normal(0, 0.01, len(ff))
        data[name] = 0.002 + beta * ff["Mkt_RF"].values + ff["RF"].values + noise
    returns = pd.DataFrame(data, index=ff.index)

    panel = estimate_betas_panel(returns, ff)
    assert panel.loc["LOW_BETA", "beta"] < panel.loc["MARKET_BETA", "beta"]
    assert panel.loc["MARKET_BETA", "beta"] < panel.loc["HIGH_BETA", "beta"]


def test_too_few_observations_returns_nan_not_error():
    ff = pd.DataFrame({"Mkt_RF": [0.01, 0.02, -0.01], "RF": [0.001] * 3},
                       index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    returns = pd.Series([0.01, 0.02, 0.0], index=ff.index)
    result = estimate_beta_alpha(returns, ff)
    assert np.isnan(result["beta"])
    assert result["n"] == 3
