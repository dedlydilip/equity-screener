"""FMP provider — point-in-time fundamentals + adjusted prices.

Uses FMP's ``/stable`` API (the legacy ``/api/v3`` endpoints were retired in 2025
and now return 403). Fundamentals carry the SEC filing timestamp (``filingDate``)
so the screener never uses a number before it was public. Per-ticker responses
are cached to parquet, so on FMP's rate-limited free tier the dataset builds
*incrementally*: re-run, and only-uncached tickers are fetched.

Set the API key in the env var named by ``config.data.fmp_api_key_env``
(default ``FMP_API_KEY``): ``setx FMP_API_KEY "your_key"`` (then a new terminal).
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .base import DataProvider
from .cache import Cache

BASE = "https://financialmodelingprep.com/stable"
_HEADERS = {"User-Agent": "equity-screener/0.1 (research)"}

# FMP field -> our column name
_INCOME = {
    "revenue": "revenue", "costOfRevenue": "cost_of_revenue", "grossProfit": "gross_profit",
    "operatingIncome": "operating_income", "incomeBeforeTax": "pretax_income",
    "incomeTaxExpense": "tax_expense", "netIncome": "net_income", "ebitda": "ebitda",
    "weightedAverageShsOutDil": "shares_diluted",
}
_BALANCE = {
    "totalStockholdersEquity": "book_equity", "totalDebt": "total_debt",
    "cashAndShortTermInvestments": "cash", "totalAssets": "total_assets",
}
_CASHFLOW = {"freeCashFlow": "free_cash_flow"}


class FMPProvider(DataProvider):
    def __init__(
        self,
        api_key_env: str = "FMP_API_KEY",
        cache_dir: str = ".cache",
        min_interval: float = 0.25,
        period: str = "quarter",
        limit: int = 5,  # FMP free tier caps statement history at 5 periods
    ) -> None:
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"FMP API key not found in env var {api_key_env!r}. Set it with: "
                f'setx {api_key_env} "your_key"  (then open a new terminal).'
            )
        self.cache = Cache(cache_dir)
        self.min_interval = min_interval
        self.period = period
        self.limit = limit
        self._last = 0.0

    def _get(self, path: str, params: dict | None = None):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {**(params or {}), "apikey": self.api_key}
        for attempt in range(4):
            r = requests.get(f"{BASE}/{path}", params=params, headers=_HEADERS, timeout=30)
            self._last = time.time()
            if r.status_code == 429:  # rate limited -> exponential backoff
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"FMP rate-limited repeatedly on {path}")

    # ---- prices -------------------------------------------------------------
    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for t in tickers:
            key = f"fmp_px_{t}_{start}_{end}"
            cached = self.cache.get(key)
            if cached is not None:
                cols[t] = cached["adj_close"]
                continue
            data = self._get(
                "historical-price-eod/dividend-adjusted",
                {"symbol": t, "from": start, "to": end},
            )
            hist = data if isinstance(data, list) else []
            if not hist:
                continue
            s = pd.DataFrame(hist)[["date", "adjClose"]].rename(columns={"adjClose": "adj_close"})
            s["date"] = pd.to_datetime(s["date"])
            s = s.set_index("date").sort_index()
            self.cache.put(key, s)
            cols[t] = s["adj_close"]
        return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()

    # ---- fundamentals (point-in-time) --------------------------------------
    def _statement(self, path: str, ticker: str, fields: dict, tag: str) -> pd.DataFrame:
        rows = self._get(path, {"symbol": ticker, "period": self.period, "limit": self.limit})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        keep = ["date", "filingDate", *[c for c in fields if c in df.columns]]
        keep = [c for c in keep if c in df.columns]
        return df[keep].rename(
            columns={"date": "period_end", "filingDate": f"filing_date_{tag}", **fields}
        )

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        for t in tickers:
            key = f"fmp_fund_{t}_{self.period}_{self.limit}"
            cached = self.cache.get(key)
            if cached is not None:
                frames.append(cached)
                continue
            inc = self._statement("income-statement", t, _INCOME, "inc")
            bal = self._statement("balance-sheet-statement", t, _BALANCE, "bal")
            cf = self._statement("cash-flow-statement", t, _CASHFLOW, "cf")
            if inc.empty:
                continue
            merged = inc
            for other in (bal, cf):
                if not other.empty:
                    merged = merged.merge(other, on="period_end", how="left")
            # Conservative PiT: the row is "known" only once ALL its statements are filed.
            fdates = [c for c in merged.columns if c.startswith("filing_date_")]
            merged["filing_date"] = merged[fdates].max(axis=1)
            merged = merged.drop(columns=fdates)
            merged.insert(0, "ticker", t)
            merged["period_end"] = pd.to_datetime(merged["period_end"])
            merged["filing_date"] = pd.to_datetime(merged["filing_date"])
            self.cache.put(key, merged)
            frames.append(merged)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "period_end"])
