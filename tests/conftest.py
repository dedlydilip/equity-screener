"""Shared fixtures — pure synthetic data, no network access (CI-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.config import QualityGates


@pytest.fixture
def gates() -> QualityGates:
    return QualityGates()


@pytest.fixture
def as_of() -> pd.Timestamp:
    return pd.Timestamp("2023-06-30")


@pytest.fixture
def qends() -> pd.DatetimeIndex:
    return pd.to_datetime(
        ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30"]
    )


def make_quarters(ticker: str, qends: pd.DatetimeIndex, **overrides) -> pd.DataFrame:
    """Six quarters of clean synthetic fundamentals for one ticker, filed 45d after period end."""
    base = dict(
        revenue=100, gross_profit=30, operating_income=15, pretax_income=12,
        tax_expense=3, net_income=10, ebitda=20, free_cash_flow=8,
        book_equity=200, total_debt=50, cash=20, total_assets=500, shares_diluted=100,
    )
    base.update(overrides)
    rows = [
        dict(ticker=ticker, period_end=q, filing_date=q + pd.Timedelta(days=45), **base)
        for q in qends
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-01", "2023-06-30")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(index=idx)
    for t, drift in [("AAA", 1.0), ("BBB", 1.0), ("CCC", 1.0)]:
        prices[t] = np.linspace(4.0 * drift, 5.0 * drift, len(idx)) + rng.normal(0, 0.01, len(idx))
    return prices
