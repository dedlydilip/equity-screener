"""Abstract data-provider interface.

Every provider returns the SAME shapes so the rest of the pipeline is
source-agnostic. Two invariants matter for institutional correctness:

* **prices** are split-and-dividend **ADJUSTED total-return** closes — never raw
  closes (a raw close makes a split look like a -50% momentum crash and a
  dividend payer look like an underperformer).
* **fundamentals** carry the SEC **filing timestamp** so the backtest can enforce
  no-look-ahead: a fundamental row is only usable on/after its ``filing_date``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Interface every data source (FMP, yfinance, ...) must implement."""

    @abstractmethod
    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Adjusted (total-return) daily close prices.

        Returns a DataFrame indexed by date (DatetimeIndex), columns = tickers,
        values = split-and-dividend-adjusted closes. Feeds returns & 12-1 momentum.
        """

    @abstractmethod
    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Point-in-time fundamentals in tidy/long form.

        Required columns:
            ticker, period_end, filing_date  (+ raw statement line items:
            net_income, book_equity, ebitda, enterprise_value, free_cash_flow,
            revenue, gross_profit, total_assets, total_debt, total_equity,
            market_cap, shares, price, ...)

        ``filing_date`` (FMP ``fillingDate`` / ``acceptedDate``) is MANDATORY:
        the screener treats a fundamental as unknown until this date.
        """
