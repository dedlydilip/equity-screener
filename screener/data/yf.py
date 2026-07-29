"""yfinance provider — adjusted (total-return) daily closes. No API key.

``auto_adjust=True`` folds splits AND dividends into the close, giving a
total-return series (the correct input for momentum and backtest P&L).

Fundamentals on yfinance are not point-in-time, so ``get_fundamentals`` is left
unimplemented here: the FMP provider owns the PiT fundamentals path.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .base import DataProvider


class YFinanceProvider(DataProvider):
    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,   # total-return adjusted close
            progress=False,
            group_by="column",
            threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].copy()
        else:  # single ticker
            close = data[["Close"]].copy()
            close.columns = [tickers[0] if isinstance(tickers, list) else tickers]
        close.index = pd.to_datetime(close.index)
        return close.dropna(how="all")

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError(
            "yfinance fundamentals are not point-in-time. Use the FMP provider "
            "(FMP_API_KEY) for the PiT fundamentals path."
        )
