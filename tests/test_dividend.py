"""Dividend yield / payout ratio math and the income screen's sustainability gate."""

import numpy as np
import pandas as pd

from screener.dividend import build_dividend_screen, dividend_yield, payout_ratio


def test_dividend_yield_is_dividends_over_price():
    div = pd.Series({"A": 2.0, "B": 0.0})
    px = pd.Series({"A": 50.0, "B": 40.0})
    y = dividend_yield(div, px)
    assert y["A"] == 0.04
    assert y["B"] == 0.0


def test_payout_ratio_nans_on_non_positive_earnings():
    div = pd.Series({"PROFIT": 1.0, "LOSS": 1.0})
    shares = pd.Series({"PROFIT": 100.0, "LOSS": 100.0})
    ni = pd.Series({"PROFIT": 200.0, "LOSS": -50.0})
    p = payout_ratio(div, shares, ni)
    assert p["PROFIT"] == 0.5           # 1*100 / 200
    assert np.isnan(p["LOSS"])          # loss-maker: ratio is meaningless, not "safe"


def test_screen_excludes_unsustainable_payers():
    idx = ["SAFE", "STRETCHED", "LOSSMAKER", "LOWYIELD"]
    div = pd.Series(dict(SAFE=3.0, STRETCHED=5.0, LOSSMAKER=4.0, LOWYIELD=1.0), index=idx)
    px = pd.Series(dict(SAFE=100.0, STRETCHED=100.0, LOSSMAKER=100.0, LOWYIELD=100.0), index=idx)
    shares = pd.Series(dict(SAFE=1.0, STRETCHED=1.0, LOSSMAKER=1.0, LOWYIELD=1.0), index=idx)
    # SAFE pays 3 of 10 (30%); STRETCHED pays 5 of 5 (100% > gate); LOSSMAKER loss.
    ni = pd.Series(dict(SAFE=10.0, STRETCHED=5.0, LOSSMAKER=-1.0, LOWYIELD=10.0), index=idx)

    out = build_dividend_screen(div, px, shares, ni, min_yield=0.005, max_payout=0.90, top_n=30)
    tickers = out["ticker"].tolist()
    assert "SAFE" in tickers
    assert "STRETCHED" not in tickers   # payout 100% > 90% gate
    assert "LOSSMAKER" not in tickers   # negative earnings
    assert "LOWYIELD" in tickers        # sustainable, just low-yielding


def test_screen_ranks_by_yield_and_weights_sum_to_one():
    idx = ["HI", "MID", "LO"]
    div = pd.Series({"HI": 6.0, "MID": 4.0, "LO": 2.0}, index=idx)
    px = pd.Series({"HI": 100.0, "MID": 100.0, "LO": 100.0}, index=idx)
    shares = pd.Series({"HI": 1.0, "MID": 1.0, "LO": 1.0}, index=idx)
    ni = pd.Series({"HI": 100.0, "MID": 100.0, "LO": 100.0}, index=idx)

    out = build_dividend_screen(div, px, shares, ni, min_yield=0.0, max_payout=0.90, top_n=30)
    assert out.iloc[0]["ticker"] == "HI"            # highest yield ranks first
    assert out["weight"].sum() == 1.0 or abs(out["weight"].sum() - 1.0) < 1e-9
    assert out.iloc[0]["weight"] > out.iloc[-1]["weight"]  # income-weighted toward higher yield


def test_screen_ranks_by_regular_yield_not_inflated_by_a_special_dividend():
    # PGR-like case: SPECIAL has a huge total (mostly a one-off special payout)
    # but a tiny recurring yield; REGULAR has a smaller total but ALL of it is
    # recurring. Ranking on regular yield must put REGULAR first, not SPECIAL.
    idx = ["SPECIAL", "REGULAR"]
    total_div = pd.Series({"SPECIAL": 14.0, "REGULAR": 3.0}, index=idx)
    # SPECIAL's recurring (non-special) slice of its dividends:
    regular_div = pd.Series({"SPECIAL": 0.4, "REGULAR": 3.0}, index=idx)
    px = pd.Series({"SPECIAL": 200.0, "REGULAR": 100.0}, index=idx)
    shares = pd.Series({"SPECIAL": 1.0, "REGULAR": 1.0}, index=idx)
    ni = pd.Series({"SPECIAL": 100.0, "REGULAR": 100.0}, index=idx)

    out = build_dividend_screen(
        total_div, px, shares, ni, regular_dividends_ttm=regular_div,
        min_yield=0.0, max_payout=0.90, top_n=30)

    assert out.iloc[0]["ticker"] == "REGULAR"  # 3% recurring beats SPECIAL's 0.2% recurring
    special_row = out[out["ticker"] == "SPECIAL"].iloc[0]
    assert bool(special_row["has_special_dividend"])
    assert not bool(out[out["ticker"] == "REGULAR"].iloc[0]["has_special_dividend"])
    # total yield still reflects the full cash paid, even though ranking doesn't
    assert special_row["total_dividend_yield"] > special_row["regular_dividend_yield"]
