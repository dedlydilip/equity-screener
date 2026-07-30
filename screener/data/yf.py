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
from .cache import Cache


class YFinanceProvider(DataProvider):
    def __init__(self, cache_dir: str = ".cache") -> None:
        self._cache = Cache(cache_dir)

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

    def get_dividends(self, tickers: list[str], as_of=None) -> pd.DataFrame:
        """Trailing-12-month cash dividend per share as of ``as_of`` (default today).

        Returns columns ``ticker, dividends_ttm`` = cash dividends with an ex-date in
        ``(as_of - 1 year, as_of]``. The window is anchored to the as-of date, NOT to
        the stock's last ex-dividend date — anchoring to the last ex-date can capture
        five payments for a name that shifted its pay schedule, overstating the yield.
        Backward-looking income proxy, not a forward dividend forecast.
        """
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
        as_of = as_of.normalize()
        cutoff = as_of - pd.DateOffset(years=1)
        rows = []
        for t in tickers:
            key = f"yf_div_{t}_{as_of.date()}"
            cached = self._cache.get(key)
            if cached is not None:
                rows.append(cached)
                continue
            try:
                div = yf.Ticker(t).dividends
            except Exception:
                div = None
            ttm = 0.0
            if div is not None and len(div):
                div.index = pd.to_datetime(div.index, utc=True).tz_localize(None)
                ttm = float(div[(div.index > cutoff) & (div.index <= as_of)].sum())
            rec = pd.DataFrame([{"ticker": t, "dividends_ttm": ttm}])
            self._cache.put(key, rec)
            rows.append(rec)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError(
            "yfinance fundamentals are not point-in-time. Use the FMP provider "
            "(FMP_API_KEY) for the PiT fundamentals path."
        )
