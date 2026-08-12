import numpy as np
import pandas as pd
import pytest

from screener.portfolio import assign_quantiles, concentration, select, weights


def test_quantiles_are_evenly_sized_and_ordered():
    scores = pd.Series(np.arange(100.0), index=[f"T{i}" for i in range(100)])
    q = assign_quantiles(scores, 5)
    assert q.value_counts().sort_index().tolist() == [20] * 5
    assert int(q["T99"]) == 5   # highest score -> best (Q5)
    assert int(q["T0"]) == 1    # lowest score -> worst (Q1)


def test_top_decile_selects_the_highest_10_percent():
    scores = pd.Series(np.arange(100.0), index=[f"T{i}" for i in range(100)])
    sel = select(scores, "top_decile")
    assert len(sel) == 10
    assert "T99" in sel and "T89" not in sel


def test_weighting_schemes_sum_to_one_and_favor_the_right_names():
    names = pd.Index([f"T{i}" for i in range(10)])
    mc = pd.Series(np.arange(1, 101.0, 10), index=names)
    vols = pd.Series(np.linspace(0.01, 0.05, 10), index=names)

    we = weights(names, "equal")
    wm = weights(names, "market_cap", market_cap=mc)
    wv = weights(names, "inverse_vol", vols=vols)

    for w in (we, wm, wv):
        assert np.isclose(w.sum(), 1.0)
    assert wm.idxmax() == mc.idxmax()
    assert wv.idxmax() == vols.idxmin()  # lowest vol gets the most weight


def test_concentration_flags_a_degenerate_book():
    diversified = pd.Series(0.1, index=[f"T{i}" for i in range(10)])
    concentrated = pd.Series([0.9, 0.05, 0.03, 0.02], index=list("abcd"))
    d, c = concentration(diversified), concentration(concentrated)
    assert d["hhi"] < c["hhi"]
    assert d["effective_n"] > c["effective_n"]
    assert c["top_10pct_weight"] >= 0.9


def test_weights_raise_rather_than_silently_equal_weighting_an_empty_panel():
    """An all-NaN market-cap panel used to fall through to equal weights, so a
    cap-weighted backtest could quietly become equal-weighted."""
    names = pd.Index(["A", "B"])
    with pytest.raises(ValueError, match="positive cap"):
        weights(names, "market_cap", market_cap=pd.Series([np.nan, np.nan], index=names))
    with pytest.raises(ValueError, match="positive vol"):
        weights(names, "inverse_vol", vols=pd.Series([np.nan, np.nan], index=names))


def test_inverse_vol_weights_are_floored_so_one_stale_name_cannot_take_the_book():
    """A halted or stale price series has a near-zero realized vol. Unfloored,
    1/vol handed it 99.99998% of the bucket."""
    names = pd.Index(["STALE"] + [f"N{i}" for i in range(9)])
    vols = pd.Series([1e-8] + [0.05] * 9, index=names)
    w = weights(names, "inverse_vol", vols=vols)
    assert w["STALE"] > w["N0"]          # still ranked as the lower-risk name
    assert w["STALE"] < 0.5              # but nowhere near the whole book
    assert w.sum() == pytest.approx(1.0)


def test_a_missing_volatility_does_not_outrank_a_measured_one():
    names = pd.Index(["NODATA", "LOWVOL", "HIGHVOL"])
    w = weights(names, "inverse_vol", vols=pd.Series([np.nan, 0.05, 0.10], index=names))
    assert w["LOWVOL"] > w["NODATA"]     # measured low risk beats unknown
    assert w.sum() == pytest.approx(1.0)


def test_weights_of_no_names_is_empty_not_a_zero_division():
    assert weights(pd.Index([]), "equal").empty


def test_unknown_weighting_and_selection_methods_raise():
    with pytest.raises(ValueError, match="unknown weighting method"):
        weights(pd.Index(["A"]), "not_a_method")
    with pytest.raises(ValueError, match="unknown selection method"):
        select(pd.Series({"A": 1.0}), method="top_ten")


def test_concentration_normalizes_before_reporting_effective_n():
    """Unnormalized weights previously yielded 'effective N = 0.125 names'."""
    out = concentration(pd.Series([2.0, 2.0]))
    assert out["hhi"] == pytest.approx(0.5)
    assert out["effective_n"] == pytest.approx(2.0)


def test_assign_quantiles_returns_nan_when_there_are_fewer_names_than_buckets():
    """3 names into n=5 previously produced non-contiguous labels [1, 3, 5]."""
    assert assign_quantiles(pd.Series([1.0, 2.0, 3.0]), n=5).isna().all()


def test_assign_quantiles_returns_nan_for_a_constant_score_vector():
    assert assign_quantiles(pd.Series([1.0] * 20), n=5).isna().all()
