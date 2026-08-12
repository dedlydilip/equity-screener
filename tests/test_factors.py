"""Point-in-time correctness and negative-denominator masking — the two properties
that most distinguish an institutional screener from a retail one."""

import numpy as np
import pandas as pd

from screener.factors import compute_factors, momentum_12_1, pit_fundamentals, price_asof
from tests.conftest import make_quarters


def test_price_asof_carries_each_ticker_forward_to_its_own_last_close():
    """Taking the last ROW wholesale gives NaN to any name that didn't trade on
    exactly that date -- and because halts/gaps correlate with distress, that
    silently biases the cross-section."""
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    px = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [20.0, 21.0, np.nan]}, index=idx)
    out = price_asof(px, "2020-03-31")
    assert out["A"] == 12.0
    assert out["B"] == 21.0  # B's own last observation, not NaN


def test_momentum_does_not_divide_by_a_non_positive_prior_price():
    """A zero prior price is bad data; dividing by it yields +inf, which the
    winsorizer would previously have clipped into a plausible-looking return."""
    idx = pd.date_range("2019-01-31", periods=40, freq="ME")
    px = pd.DataFrame({"GOOD": np.linspace(10, 20, 40), "BAD": np.linspace(10, 20, 40)}, index=idx)
    px.loc[px.index[: 40 - 13], "BAD"] = 0.0  # zero price 13 months back
    mom = momentum_12_1(px, px.index[-1])
    assert np.isfinite(mom["GOOD"])
    assert np.isnan(mom["BAD"])


def test_pit_fundamentals_excludes_unfiled_future_quarter(qends, as_of):
    aaa = make_quarters("AAA", qends)
    future_q = pd.DataFrame([{
        "ticker": "AAA", "period_end": pd.Timestamp("2023-09-30"),
        "filing_date": pd.Timestamp("2023-09-30") + pd.Timedelta(days=45),  # filed AFTER as_of
        "revenue": 100, "gross_profit": 30, "operating_income": 15, "pretax_income": 12,
        "tax_expense": 3, "net_income": 999999, "ebitda": 20, "free_cash_flow": 8,
        "book_equity": 200, "total_debt": 50, "cash": 20, "total_assets": 500,
        "shares_diluted": 100,
    }])
    fund = pd.concat([aaa, future_q], ignore_index=True)

    pit = pit_fundamentals(fund, as_of)

    assert pit.loc["AAA", "net_income_ttm"] == 40.0  # 4 * 10, NOT poisoned by the 999999
    # Once that quarter IS filed, it must actually be picked up -- otherwise the
    # assertion above would also pass for a function that ignores late data.
    later = pit_fundamentals(fund, pd.Timestamp("2023-11-30"))
    assert later.loc["AAA", "net_income_ttm"] > 40.0


def test_pit_fundamentals_advances_as_more_quarters_are_known(qends):
    aaa = make_quarters("AAA", qends)
    early = pit_fundamentals(aaa, pd.Timestamp("2022-06-01"))  # only Q1 filed by then (2022-05-15)
    later = pit_fundamentals(aaa, pd.Timestamp("2023-06-30"))  # all 6 quarters filed
    # A "TTM" needs 4 consecutive quarters -> 1 known quarter is NOT a valid TTM
    # (a partial-history figure must never masquerade as a full trailing year).
    assert np.isnan(early.loc["AAA", "net_income_ttm"])
    assert later.loc["AAA", "net_income_ttm"] == 40.0   # TTM = last 4 of 6 known quarters


def test_pit_fundamentals_rejects_a_gap_in_the_quarterly_sequence(qends):
    # Drop the third quarter -> the remaining 4 known quarters (0,1,3,4 by index)
    # span a ~6-month gap, not 4 consecutive ~91-day quarters -> must NOT sum to
    # a fabricated TTM.
    aaa = make_quarters("AAA", qends)
    gapped = aaa.drop(aaa.index[2]).reset_index(drop=True)
    pit = pit_fundamentals(gapped, pd.Timestamp("2023-06-30"))
    assert np.isnan(pit.loc["AAA", "net_income_ttm"])


def test_negative_book_value_nans_bp_but_not_other_factors(qends, synthetic_prices, as_of, gates):
    fund = make_quarters("BBB", qends, book_equity=-100)
    f = compute_factors(fund, synthetic_prices[["BBB"]], as_of, gates)
    assert np.isnan(f.loc["BBB", "bp"])
    assert np.isnan(f.loc["BBB", "roe"])       # also book-denominated
    assert np.isnan(f.loc["BBB", "de_inv"])    # also book-denominated
    assert not np.isnan(f.loc["BBB", "ep"])    # earnings yield unaffected


def test_negative_ebitda_nans_ev_ebitda_only(qends, synthetic_prices, as_of, gates):
    fund = make_quarters("CCC", qends, ebitda=-5)
    f = compute_factors(fund, synthetic_prices[["CCC"]], as_of, gates)
    assert np.isnan(f.loc["CCC", "ev_ebitda_inv"])
    assert not np.isnan(f.loc["CCC", "bp"])
    assert not np.isnan(f.loc["CCC", "roe"])


def test_negative_fcf_is_kept_not_masked(qends, synthetic_prices, as_of, gates):
    fund = make_quarters("AAA", qends, free_cash_flow=-8)
    f = compute_factors(fund, synthetic_prices[["AAA"]], as_of, gates)
    assert f.loc["AAA", "fcf_yield"] < 0  # negative FCF must sort to the bottom, not vanish


def test_ratios_match_hand_calculation(qends, synthetic_prices, as_of, gates):
    fund = make_quarters("AAA", qends)
    f = compute_factors(fund, synthetic_prices[["AAA"]], as_of, gates)
    assert np.isclose(f.loc["AAA", "roe"], 40.0 / 200.0)          # net_income_ttm / book_equity
    assert np.isclose(f.loc["AAA", "de_inv"], -(50.0 / 200.0))    # -total_debt / book_equity
    assert np.isclose(f.loc["AAA", "gross_profitability"], 120.0 / 500.0)  # gp_ttm / assets


def test_market_cap_prefers_dated_shares_outstanding_over_diluted_average(
    qends, synthetic_prices, as_of, gates
):
    fund = make_quarters("AAA", qends, shares_diluted=100, shares_outstanding=150)
    f = compute_factors(fund, synthetic_prices[["AAA"]], as_of, gates)
    price = synthetic_prices["AAA"].asof(as_of)
    assert np.isclose(f.loc["AAA", "market_cap"], price * 150)  # dated figure, not the 100 avg


def test_market_cap_falls_back_to_diluted_shares_when_dated_figure_missing(
    qends, synthetic_prices, as_of, gates
):
    # make_quarters doesn't set shares_outstanding by default -> NaN -> must fall
    # back to the weighted-average diluted count rather than produce NaN mcap.
    fund = make_quarters("AAA", qends, shares_diluted=100)
    f = compute_factors(fund, synthetic_prices[["AAA"]], as_of, gates)
    price = synthetic_prices["AAA"].asof(as_of)
    assert np.isclose(f.loc["AAA", "market_cap"], price * 100)


def test_financial_sector_masks_ev_ebitda_de_and_roic(qends, synthetic_prices, as_of, gates):
    fund = make_quarters("AAA", qends)
    sector = pd.Series({"AAA": "Financials"})
    f = compute_factors(fund, synthetic_prices[["AAA"]], as_of, gates, sector=sector)
    assert np.isnan(f.loc["AAA", "ev_ebitda_inv"])
    assert np.isnan(f.loc["AAA", "de_inv"])
    assert np.isnan(f.loc["AAA", "roic"])
    assert not np.isnan(f.loc["AAA", "roe"])  # ROE and value factors still apply to banks


def test_non_financial_sector_is_not_masked(qends, synthetic_prices, as_of, gates):
    fund = make_quarters("AAA", qends)
    sector = pd.Series({"AAA": "Information Technology"})
    f = compute_factors(fund, synthetic_prices[["AAA"]], as_of, gates, sector=sector)
    assert not np.isnan(f.loc["AAA", "ev_ebitda_inv"])
    assert not np.isnan(f.loc["AAA", "de_inv"])
