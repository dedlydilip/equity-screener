"""IC, quintile backtest, and the Fama-French decomposition — including the
subtle Rf-handling distinction: the self-financing L/S spread is regressed RAW
(no Rf subtracted), while a single leg is regressed as excess return (R - Rf)."""

import numpy as np
import pandas as pd
import pytest

from screener.validate import (
    FF_COLS,
    ff_decompose,
    ic_summary,
    information_coefficient,
    max_drawdown,
    net_of_cost,
    quintile_returns,
    sharpe,
    turnover,
    weighted_quintile_returns,
)


@pytest.fixture
def signal_panel():
    rng = np.random.default_rng(1)
    months = pd.date_range("2019-01-31", periods=60, freq="ME")
    stocks = [f"S{i}" for i in range(100)]
    scores = pd.DataFrame(rng.normal(size=(60, 100)), index=months, columns=stocks)
    z = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1), axis=0)
    noise = rng.normal(0, 0.05, size=(60, 100))
    fwd = pd.DataFrame(0.03 * z.values + noise, index=months, columns=stocks)
    return scores, fwd


def test_positive_signal_produces_positive_ic(signal_panel):
    scores, fwd = signal_panel
    ic = information_coefficient(scores, fwd)
    s = ic_summary(ic)
    assert s["mean_ic"] > 0.3
    assert s["n"] == 60


def test_quintile_backtest_is_monotonic_q5_beats_q1(signal_panel):
    scores, fwd = signal_panel
    qr = quintile_returns(scores, fwd)
    assert qr["Q5"].mean() > qr["Q3"].mean() > qr["Q1"].mean()
    assert qr["spread"].mean() > 0


def test_sharpe_subtracts_risk_free_rate():
    """The rf branch must actually be exercised: the same return series scored
    against a non-zero Rf has to produce a strictly smaller Sharpe."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    r = pd.Series(0.01 + rng.normal(0, 0.02, 24), index=idx)
    rf = pd.Series(0.004, index=idx)
    gross, excess = sharpe(r, rf=None), sharpe(r, rf=rf)
    assert excess < gross
    # Subtracting a CONSTANT Rf shifts the mean but not the vol, so the excess
    # Sharpe is exactly (mean - rf) / sd * sqrt(12) -- pin it numerically.
    expected = (r.mean() - 0.004) / r.std(ddof=1) * np.sqrt(12)
    assert excess == pytest.approx(expected, rel=1e-12)
    # a genuinely-zero-vol excess-return series is undefined (0/0) -> NaN, not inf
    assert np.isnan(sharpe(pd.Series(0.0, index=idx)))


def test_max_drawdown_is_negative_for_a_declining_series():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    r = pd.Series([0.05, -0.10, -0.10, 0.02, 0.03, 0.01], index=idx)
    # Exact, not just sign: the peak is 1.05 after month 1 and the trough is
    # 1.05*0.9*0.9 = 0.8505, so the drawdown is 0.8505/1.05 - 1 = -19%.
    assert max_drawdown(r) == pytest.approx(-0.19, abs=1e-9)


def test_max_drawdown_counts_a_loss_in_the_very_first_period():
    """The equity curve must be seeded at 1.0. Without that seed cummax[0] equals
    curve[0], so an opening loss reports a drawdown of exactly zero -- while the
    identical loss one period later reports correctly. That incoherence silently
    understated the published backtest drawdowns."""
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    loss_first = max_drawdown(pd.Series([-0.20, 0.0, 0.0], index=idx))
    loss_second = max_drawdown(pd.Series([0.0, -0.20, 0.0], index=idx))
    assert loss_first == pytest.approx(-0.20, abs=1e-9)
    assert loss_first == pytest.approx(loss_second, abs=1e-9)


def test_max_drawdown_is_zero_for_a_monotonically_rising_series():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    assert max_drawdown(pd.Series([0.01, 0.02, 0.01, 0.03], index=idx)) == pytest.approx(0.0)


def test_max_drawdown_of_an_empty_series_is_nan():
    assert np.isnan(max_drawdown(pd.Series(dtype=float)))


def test_turnover_is_zero_for_static_holdings():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    static = pd.DataFrame(0.25, index=idx, columns=list("ABCD"))
    t = turnover(static)
    assert t["per_rebalance_two_way"] == pytest.approx(0.0, abs=1e-9)


def test_turnover_is_nonzero_for_a_full_rotation():
    # Complete portfolio swap each rebalance: A/B -> C/D -> A/B ...
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    h = pd.DataFrame(0.0, index=idx, columns=list("ABCD"))
    h.loc[idx[0::2], ["A", "B"]] = 0.5
    h.loc[idx[1::2], ["C", "D"]] = 0.5
    t = turnover(h, ppy=12)
    # Every rebalance fully exits the old names and enters new ones -> one-way
    # turnover = 1.0 (sum|delta|/2 = 2.0/2), two-way = 2.0.
    assert t["per_rebalance_two_way"] == pytest.approx(2.0, abs=1e-9)
    assert t["annualized_two_way"] == pytest.approx(24.0, abs=1e-9)


def test_net_of_cost_subtracts_a_larger_drag_for_higher_turnover():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    gross = pd.Series(0.02, index=idx)
    low_turn_net = net_of_cost(gross, turnover_per_rebalance=0.2, cost_bps_per_side=15)
    high_turn_net = net_of_cost(gross, turnover_per_rebalance=1.0, cost_bps_per_side=15)
    assert (low_turn_net < gross).all()
    assert (high_turn_net < low_turn_net).all()  # more turnover -> more cost drag
    assert net_of_cost(gross, 0.0, 15).equals(gross)  # zero turnover -> zero cost


def test_weighted_quintile_returns_market_cap_weights_are_actually_cap_weighted(signal_panel):
    """Pin the ARITHMETIC, not just 'differs from equal-weight' -- inverse-cap or
    random weights would also differ."""
    scores, fwd = signal_panel
    tickers = scores.columns
    mc_row = pd.Series(np.arange(1, len(tickers) + 1, dtype=float), index=tickers)
    market_cap = pd.DataFrame([mc_row] * len(scores), index=scores.index)

    weighted = weighted_quintile_returns(scores, fwd, method="market_cap", market_cap=market_cap)

    # Recompute Q5 for the first date by hand from the cap weights.
    dt = scores.index[0]
    s = scores.loc[dt].dropna()
    q = pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    members = q[q == 5].index
    w = mc_row.reindex(members) / mc_row.reindex(members).sum()
    expected = float((fwd.loc[dt].reindex(members) * w).sum())
    assert weighted.loc[dt, "Q5"] == pytest.approx(expected, rel=1e-12)


def test_equal_weighted_variant_matches_the_plain_quintile_returns(signal_panel):
    """method='equal' must reduce EXACTLY to quintile_returns. These previously
    disagreed whenever a forward return was missing, because the weighted
    version summed weight*return with the weights still summing to 1 -- silently
    scoring an uncovered name as a 0% return and halving the spread."""
    scores, fwd = signal_panel
    eq = quintile_returns(scores, fwd)
    wt = weighted_quintile_returns(scores, fwd, method="equal")
    pd.testing.assert_frame_equal(eq, wt, check_exact=False, rtol=1e-12)


def test_missing_forward_returns_do_not_become_zero_percent_observations():
    idx = [pd.Timestamp("2020-01-31")]
    cols = [f"S{i}" for i in range(10)]
    scores = pd.DataFrame([[float(i) for i in range(10)]], index=idx, columns=cols)
    fwd = pd.DataFrame([[np.nan] * 10], index=idx, columns=cols)
    fwd.iloc[0, 0], fwd.iloc[0, 9] = 0.10, 0.20   # only the extremes are covered

    for frame in (quintile_returns(scores, fwd),
                  weighted_quintile_returns(scores, fwd, method="equal")):
        assert frame["Q1"].iloc[0] == pytest.approx(0.10)
        assert frame["Q5"].iloc[0] == pytest.approx(0.20)
        assert frame["spread"].iloc[0] == pytest.approx(0.10)   # not halved to 0.05
        assert np.isnan(frame["Q3"].iloc[0])                    # uncovered -> NaN, not 0.0


def test_a_constant_signal_produces_no_quintile_observation():
    """rank(method='first') breaks ties by column order, so a zero-information
    (all-tied) score vector used to manufacture a large Q5-Q1 spread out of
    ticker ordering alone."""
    idx = [pd.Timestamp("2020-01-31")]
    cols = [f"S{i}" for i in range(10)]
    scores = pd.DataFrame([[1.0] * 10], index=idx, columns=cols)
    fwd = pd.DataFrame([[i / 100 for i in range(10)]], index=idx, columns=cols)
    assert quintile_returns(scores, fwd).empty
    assert weighted_quintile_returns(scores, fwd, method="equal").empty


def test_ic_summary_reports_both_iid_and_hac_t_stats(signal_panel):
    scores, fwd = signal_panel
    ic = information_coefficient(scores, fwd)
    s = ic_summary(ic)
    assert "t_stat" in s and "t_stat_hac" in s
    assert np.isfinite(s["t_stat"])
    assert np.isfinite(s["t_stat_hac"])
    # For a strong, fairly stable signal the two needn't match, but both
    # should agree on the (positive) sign of the effect.
    assert (s["t_stat"] > 0) == (s["t_stat_hac"] > 0)


def test_ff_decompose_recovers_known_alpha_and_loading():
    rng = np.random.default_rng(2)
    months = pd.date_range("2018-01-31", periods=72, freq="ME")
    ff = pd.DataFrame(rng.normal(0, 0.02, size=(72, 6)), index=months, columns=FF_COLS)
    ff["RF"] = 0.003
    true_alpha, true_hml_beta = 0.005, 0.8
    signal = true_alpha + true_hml_beta * ff["HML"].values + rng.normal(0, 0.005, 72)
    ls = pd.Series(signal, index=months)

    result = ff_decompose(ls, ff, is_long_short=True)
    assert result["alpha_monthly"] == pytest.approx(true_alpha, abs=0.003)
    assert result["loadings"]["HML"] == pytest.approx(true_hml_beta, abs=0.15)
    assert result["r_squared"] > 0.7


def test_ff_decompose_rf_handling_differs_between_spread_and_single_leg():
    """The core reviewer-flagged correction: a self-financing spread must NOT have
    Rf subtracted, but a single long-only leg must. The two alphas must differ by
    exactly the risk-free rate."""
    rng = np.random.default_rng(3)
    months = pd.date_range("2018-01-31", periods=72, freq="ME")
    ff = pd.DataFrame(rng.normal(0, 0.02, size=(72, 6)), index=months, columns=FF_COLS)
    ff["RF"] = 0.003
    ret = pd.Series(0.005 + 0.5 * ff["HML"].values + rng.normal(0, 0.005, 72), index=months)

    as_spread = ff_decompose(ret, ff, is_long_short=True)   # no Rf subtracted
    as_leg = ff_decompose(ret, ff, is_long_short=False)     # Rf subtracted from y first

    diff = as_spread["alpha_monthly"] - as_leg["alpha_monthly"]
    assert diff == pytest.approx(0.003, abs=1e-9)  # exactly the constant RF


def test_ff_decompose_ppy_controls_annualization():
    rng = np.random.default_rng(4)
    months = pd.date_range("2018-01-31", periods=40, freq="ME")
    ff = pd.DataFrame(rng.normal(0, 0.02, size=(40, 6)), index=months, columns=FF_COLS)
    ff["RF"] = 0.001
    ls = pd.Series(0.01 + rng.normal(0, 0.005, 40), index=months)
    monthly = ff_decompose(ls, ff, is_long_short=True, ppy=12)
    quarterly = ff_decompose(ls, ff, is_long_short=True, ppy=4)
    assert monthly["alpha_monthly"] == pytest.approx(quarterly["alpha_monthly"])  # same regression
    assert monthly["alpha_annualized"] == pytest.approx(monthly["alpha_monthly"] * 12)
    assert quarterly["alpha_annualized"] == pytest.approx(quarterly["alpha_monthly"] * 4)
