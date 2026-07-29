import numpy as np
import pandas as pd

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
