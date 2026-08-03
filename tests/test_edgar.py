"""EDGAR quarterly reconstruction — pure logic, no network."""

import pandas as pd

from screener.data.edgar import EdgarProvider


def _p(start, end, filed, val, form="10-Q"):
    return {"form": form, "start": start, "end": end, "filed": filed, "val": val}


def test_direct_three_month_points_pass_through(tmp_path):
    prov = EdgarProvider(cache_dir=str(tmp_path))
    pts = [
        _p("2023-01-01", "2023-03-31", "2023-04-30", 100),
        _p("2023-04-01", "2023-06-30", "2023-07-30", 110),
    ]
    out = prov._quarterly_flow(pts)
    assert out[pd.Timestamp("2023-03-31")][0] == 100
    assert out[pd.Timestamp("2023-06-30")][0] == 110


def test_ytd_ladder_is_differenced_into_clean_quarters(tmp_path):
    prov = EdgarProvider(cache_dir=str(tmp_path))
    # Cash-flow style: only cumulative year-to-date figures are filed.
    pts = [
        _p("2023-01-01", "2023-03-31", "2023-04-30", 100),          # Q1 YTD (also 3-month)
        _p("2023-01-01", "2023-06-30", "2023-07-30", 250),          # 6-month YTD
        _p("2023-01-01", "2023-09-30", "2023-10-30", 400),          # 9-month YTD
        _p("2023-01-01", "2023-12-31", "2024-02-15", 600, "10-K"),  # full year
    ]
    out = prov._quarterly_flow(pts)
    assert out[pd.Timestamp("2023-03-31")][0] == 100          # Q1
    assert out[pd.Timestamp("2023-06-30")][0] == 150          # 250 - 100
    assert out[pd.Timestamp("2023-09-30")][0] == 150          # 400 - 250
    assert out[pd.Timestamp("2023-12-31")][0] == 200          # 600 - 400 (Q4 from 10-K)


def test_filing_date_is_carried_through(tmp_path):
    prov = EdgarProvider(cache_dir=str(tmp_path))
    out = prov._quarterly_flow([_p("2023-01-01", "2023-03-31", "2023-04-30", 100)])
    assert out[pd.Timestamp("2023-03-31")][1] == pd.Timestamp("2023-04-30")


def test_original_filing_wins_over_later_restatement(tmp_path):
    prov = EdgarProvider(cache_dir=str(tmp_path))
    pts = [
        _p("2023-01-01", "2023-03-31", "2023-04-30", 100),   # original
        _p("2023-01-01", "2023-03-31", "2024-01-15", 105),   # later restatement
    ]
    out = prov._quarterly_flow(pts)
    assert out[pd.Timestamp("2023-03-31")] == (100, pd.Timestamp("2023-04-30"))
    # NOTE: this is a deliberate PiT choice, not an oversight (see docstring on
    # _quarterly_flow / plan Stage 2 M5) — an investor trading between the two
    # filing dates only ever saw the as-first-reported 100, so using it (dated to
    # its original filing) avoids leaking a later data revision into the past.
    # The live/current screen re-pulling EDGAR after 2024-01-15 would naturally
    # pick up 105 for a NEW as_of, since a fresh companyfacts call is uncached.


def test_shares_outstanding_keyed_by_filing_not_period_end(tmp_path):
    """dei:EntityCommonStockSharesOutstanding is a cover-page 'as of' fact whose
    own date does NOT line up with the 10-Q's period end -> it must be matched to
    a row by the filing date it was disclosed alongside, not by its own 'end'."""
    prov = EdgarProvider(cache_dir=str(tmp_path))
    # cover-page date (2023-04-25) sits between the quarter end (2023-03-31) and
    # the filing date (2023-04-30) -- neither of which it equals.
    pts = [{"form": "10-Q", "end": "2023-04-25", "filed": "2023-04-30", "val": 1_000_000}]
    out = prov._shares_outstanding_by_filing(pts)
    assert out[pd.Timestamp("2023-04-30")] == 1_000_000
    assert pd.Timestamp("2023-04-25") not in out
