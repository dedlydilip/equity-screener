"""First coverage for modules the audit found had none at all.

`screen.py` gates every screen, `report.py` IS the export/reports subcommands,
`cache.py` decides whether stale data is silently reused, `gics.py` underpins the
whole sector-neutral hierarchy, and `french.py`'s rescale sets the units of every
alpha in the project -- none of them had a single test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.config import HardFilters
from screener.data.cache import Cache, version_hash
from screener.data.french import tidy_ff
from screener.db import connect, write_table
from screener.gics import industry_group_for
from screener.report import export_for_dashboard, run_reports
from screener.screen import apply_hard_filters, build_screen
from screener.validate import ic_decay


# --------------------------------------------------------------------------- screen
def _factors(**over) -> pd.DataFrame:
    base = pd.DataFrame({
        "market_cap": [5e9, 5e8, 5e9, 5e9],      # B is below a 1e9 floor
        "ep": [0.05, 0.05, -0.01, np.nan],        # C loses money, D is unknown
    }, index=["A", "B", "C", "D"])
    for k, v in over.items():
        base[k] = v
    return base


def test_hard_filters_drop_small_caps_loss_makers_and_unknowns():
    keep = apply_hard_filters(_factors(), HardFilters(min_market_cap=1e9, positive_earnings=True))
    assert list(keep) == ["A"]


def test_positive_earnings_filter_can_be_switched_off():
    """With the earnings gate off only the market-cap floor applies, so the
    loss-maker and the unknown-earnings name both come back."""
    keep = apply_hard_filters(_factors(), HardFilters(min_market_cap=1e9, positive_earnings=False))
    assert set(keep) == {"A", "C", "D"}     # B still excluded on market cap


def test_build_screen_ranks_survivors_by_score_descending():
    scores = pd.Series({"A": 1.0, "B": 9.9, "C": 5.0, "D": 3.0})
    out = build_screen(scores, _factors(), HardFilters(min_market_cap=1e9, positive_earnings=True),
                       method="top_n", top_n=10)
    assert list(out.index) == ["A"]         # only A survives the gates
    assert out["rank"].tolist() == [1]


def test_build_screen_is_empty_when_everything_is_filtered_out():
    scores = pd.Series({"B": 9.9})
    out = build_screen(scores, _factors(), HardFilters(min_market_cap=1e9, positive_earnings=True),
                       method="top_n", top_n=10)
    assert out.empty


# --------------------------------------------------------------------------- cache
def test_cache_roundtrip(tmp_path):
    c = Cache(str(tmp_path), version="v1")
    assert c.get("k") is None
    c.put("k", pd.DataFrame({"x": [1, 2]}))
    assert c.get("k")["x"].tolist() == [1, 2]


def test_a_different_version_does_not_see_the_old_entry(tmp_path):
    """The whole point of versioning: changing the extraction logic (and hence
    the version) must invalidate, or the stale frame is returned forever and a
    newly-added column silently comes back all-NaN."""
    Cache(str(tmp_path), version="v1").put("edgar_fund_AAPL", pd.DataFrame({"x": [1]}))
    assert Cache(str(tmp_path), version="v1").get("edgar_fund_AAPL") is not None
    assert Cache(str(tmp_path), version="v2").get("edgar_fund_AAPL") is None


def test_expired_entries_are_ignored(tmp_path):
    import os
    import time
    c = Cache(str(tmp_path), version="v1", ttl_days=1.0)
    c.put("k", pd.DataFrame({"x": [1]}))
    assert c.get("k") is not None
    old = time.time() - 3 * 86400
    os.utime(c._path("k"), (old, old))
    assert c.get("k") is None


def test_version_hash_is_stable_and_sensitive():
    a = version_hash({"revenue": ["Revenues"]}, ["Assets"])
    assert a == version_hash({"revenue": ["Revenues"]}, ["Assets"])       # deterministic
    assert a != version_hash({"revenue": ["RevenueNet"]}, ["Assets"])     # tag change -> new key


# --------------------------------------------------------------------------- gics
def test_industry_group_maps_a_known_sub_industry():
    assert industry_group_for("Semiconductors") == "Semiconductors & Semiconductor Equipment"


def test_industry_group_is_none_for_an_unknown_sub_industry():
    assert industry_group_for("Not A Real Sub-Industry") is None


# --------------------------------------------------------------------------- french
def _raw_french():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    ff5 = pd.DataFrame({"Mkt-RF": [1.0, 2.0, -1.0], "SMB": [0.5, 0.5, 0.5],
                        "HML": [0.1, 0.1, 0.1], "RMW": [0.2, 0.2, 0.2],
                        "CMA": [0.3, 0.3, 0.3], "RF": [0.1, 0.1, 0.1]}, index=idx)
    mom = pd.DataFrame({"Mom   ": [0.4, 0.4, 0.4]}, index=idx)
    return ff5, mom


def test_tidy_ff_converts_percent_to_decimal_and_renames():
    out = tidy_ff(*_raw_french())
    assert set(["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"]) <= set(out.columns)
    assert out["Mkt_RF"].iloc[0] == pytest.approx(0.01)   # 1.0 percent -> 0.01
    assert out["RF"].iloc[0] == pytest.approx(0.001)
    assert out.index[0] == pd.Timestamp("2020-01-31")     # month-end timestamps


def test_tidy_ff_raises_if_an_expected_column_is_missing():
    ff5, mom = _raw_french()
    with pytest.raises(ValueError, match="missing expected columns"):
        tidy_ff(ff5.drop(columns=["RMW"]), mom)


# --------------------------------------------------------------------------- report
def test_export_writes_a_parquet_per_existing_table(tmp_path):
    con = connect(str(tmp_path / "t.duckdb"))
    write_table(con, "screen", pd.DataFrame({
        "as_of": [pd.Timestamp("2024-01-31")], "ticker": ["A"], "rank": [1], "score": [1.0]}))
    written = export_for_dashboard(con, str(tmp_path / "out"))
    assert any(p.endswith("screen.parquet") for p in written)
    assert pd.read_parquet(tmp_path / "out" / "screen.parquet")["ticker"].tolist() == ["A"]


def test_run_reports_returns_a_frame_per_named_query(tmp_path):
    con = connect(str(tmp_path / "t.duckdb"))
    write_table(con, "screen", pd.DataFrame({
        "as_of": [pd.Timestamp("2024-01-31")] * 2, "ticker": ["A", "B"],
        "rank": [1, 2], "score": [2.0, 1.0]}))
    out = run_reports(con)
    assert "top_screen" in out
    assert out["top_screen"]["ticker"].tolist() == ["A", "B"]


# --------------------------------------------------------------------------- ic_decay
def test_ic_decay_falls_off_as_the_signal_ages():
    """A signal that predicts t+1 strongly and t+5 weakly must show decay."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=40, freq="ME")
    cols = [f"S{i}" for i in range(50)]
    scores = pd.DataFrame(rng.normal(size=(40, 50)), index=dates, columns=cols)
    # next-period return tracks the score; later periods are pure noise
    panel = pd.DataFrame(rng.normal(0, 1.0, size=(40, 50)), index=dates, columns=cols)
    panel.iloc[1:] = 0.9 * scores.iloc[:-1].values + 0.1 * panel.iloc[1:].values

    out = ic_decay(scores, panel, lags=(1, 3, 5))
    assert set(out) == {"lag_1", "lag_3", "lag_5"}
    assert out["lag_1"] > 0.5
    assert out["lag_1"] > out["lag_3"]
    assert abs(out["lag_5"]) < 0.2


def test_ic_decay_returns_nan_when_the_horizon_exceeds_the_panel():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    cols = [f"S{i}" for i in range(10)]
    scores = pd.DataFrame(1.0, index=dates, columns=cols)
    panel = pd.DataFrame(1.0, index=dates, columns=cols)
    assert np.isnan(ic_decay(scores, panel, lags=(10,))["lag_10"])
