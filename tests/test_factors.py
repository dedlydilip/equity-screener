"""Point-in-time correctness and negative-denominator masking — the two properties
that most distinguish an institutional screener from a retail one."""

import numpy as np
import pandas as pd

from screener.factors import compute_factors, pit_fundamentals
from tests.conftest import make_quarters


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
    assert 999999 not in fund[fund["filing_date"] <= as_of]["net_income"].values


def test_pit_fundamentals_advances_as_more_quarters_are_known(qends):
    aaa = make_quarters("AAA", qends)
    early = pit_fundamentals(aaa, pd.Timestamp("2022-06-01"))  # only Q1 filed by then (2022-05-15)
    later = pit_fundamentals(aaa, pd.Timestamp("2023-06-30"))  # all 6 quarters filed
    assert early.loc["AAA", "net_income_ttm"] == 10.0   # only 1 quarter known
    assert later.loc["AAA", "net_income_ttm"] == 40.0   # TTM = last 4 of 6 known quarters


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
