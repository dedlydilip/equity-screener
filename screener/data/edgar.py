"""SEC EDGAR provider — free, full-history, point-in-time fundamentals.

Pulls XBRL ``companyfacts`` from data.sec.gov (no API key; a descriptive
User-Agent with contact info is required by the SEC fair-access policy). Every
data point carries its SEC ``filed`` date, so this is genuinely point-in-time
with decades of history — unlike the FMP free tier (5 quarters, whitelist only).

Flow items (revenue, net income, ...) are reported both quarterly and
year-to-date; we take the direct ~3-month values and reconstruct Q4 from the
10-K annual minus the three interim quarters. Balance-sheet items are
instantaneous levels. EBITDA, free cash flow, and total debt are derived from
their components; anything unavailable is left NaN (the quality gate handles it).

Restatements (deliberate design choice, not a gap): when the same period is
disclosed in more than one filing (e.g. a later 10-K restates a prior quarter's
comparatives), the EARLIEST filing is kept, not the latest. This is the
as-originally-reported figure — the number an investor actually saw at the
time — which is the standard point-in-time convention (mirrors how CRSP/
Compustat PiT datasets default). It also means a restatement's correction
never leaks into the backtest, avoiding any hint of hindsight bias.

EDGAR has no prices, so ``get_prices`` delegates to the (also free, adjusted)
yfinance provider.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .base import DataProvider
from .cache import Cache, version_hash
from .yf import YFinanceProvider

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# our field -> candidate us-gaap tags (first present wins)
_FLOW = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss"],
    "dep_amort": [
        "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
_STOCK = {
    "book_equity": ["StockholdersEquity"],
    "total_assets": ["Assets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt": ["DebtCurrent", "LongTermDebtCurrent"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
# dei: (not us-gaap:) cover-page fact -> dated shares outstanding, see
# _shares_outstanding_by_filing.
_DEI_SHARES_TAGS = ["EntityCommonStockSharesOutstanding"]


class EdgarProvider(DataProvider):
    def __init__(self, contact_email: str = "adilip407@gmail.com", cache_dir: str = ".cache",
                 min_interval: float = 0.12) -> None:
        self.headers = {"User-Agent": f"equity-screener research {contact_email}"}
        # Version the cache on the XBRL tag maps: the cached frame is the OUTPUT
        # of extraction, so editing a tag map must invalidate it. Otherwise the
        # stale frame is returned verbatim, the extractor never re-runs, and any
        # column added later silently comes back all-NaN.
        self.cache = Cache(
            cache_dir,
            version=f"edgar_{version_hash(_FLOW, _STOCK, _DEI_SHARES_TAGS)}",
        )
        self.min_interval = min_interval
        self._last = 0.0
        self._cik: dict[str, str] | None = None
        self._prices = YFinanceProvider()

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return self._prices.get_prices(tickers, start, end)

    def _get(self, url: str):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = requests.get(url, headers=self.headers, timeout=30)
        self._last = time.time()
        r.raise_for_status()
        return r.json()

    def _cik_map(self) -> dict[str, str]:
        if self._cik is None:
            data = self._get(_TICKERS_URL)
            self._cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
        return self._cik

    @staticmethod
    def _usd_points(facts: dict, tags: list[str], namespace: str = "us-gaap") -> list[dict]:
        ns = facts.get("facts", {}).get(namespace, {})
        for tag in tags:
            if tag in ns:
                units = ns[tag].get("units", {})
                for unit_key in ("USD", "USD/shares", "shares"):
                    if unit_key in units:
                        return units[unit_key]
        return []

    @staticmethod
    def _shares_outstanding_by_filing(points: list[dict]) -> dict[pd.Timestamp, float]:
        """Dated shares outstanding, keyed by filing date.

        ``dei:EntityCommonStockSharesOutstanding`` is a cover-page fact "as of" a
        date close to (not equal to) the filing — unlike the balance-sheet items,
        its own ``end`` date won't line up with the 10-Q/10-K period end, so it
        can't be joined the same way. Every 10-Q/10-K cover page carries one, so
        it's matched to a row by that filing's ``filed`` date instead: the market
        cap as of a filing should use the share count *disclosed in that filing*,
        not a stale trailing average.
        """
        out: dict[pd.Timestamp, float] = {}
        for p in points:
            if p.get("form") not in ("10-Q", "10-K") or "val" not in p:
                continue
            filed = pd.Timestamp(p["filed"])
            out[filed] = float(p["val"])
        return out

    @staticmethod
    def _dur_days(p: dict) -> int | None:
        if not p.get("start") or not p.get("end"):
            return None
        return (pd.Timestamp(p["end"]) - pd.Timestamp(p["start"])).days

    def _quarterly_flow(self, points: list[dict]) -> dict[pd.Timestamp, tuple[float, pd.Timestamp]]:
        """Return {period_end: (quarterly_value, filed_date)}.

        Prefers direct ~3-month points; fills any remaining gaps by differencing the
        year-to-date ladder (points sharing a fiscal-year start), which is how
        cash-flow items are reported. A differenced value is only accepted when the
        incremental span is itself ~one quarter, guarding against missing rungs.
        """
        pts = [p for p in points
               if p.get("form") in ("10-Q", "10-K") and self._dur_days(p) is not None]
        out: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
        by_start: dict[pd.Timestamp, list[tuple[pd.Timestamp, float, pd.Timestamp]]] = {}
        for p in pts:
            end = pd.Timestamp(p["end"])
            start = pd.Timestamp(p["start"])
            filed = pd.Timestamp(p["filed"])
            val = float(p["val"])
            if 80 <= self._dur_days(p) <= 100 and (end not in out or filed < out[end][1]):
                out[end] = (val, filed)  # earliest filing kept -> see module docstring
            by_start.setdefault(start, []).append((end, val, filed))

        for start, ladder in by_start.items():
            prev_end, prev_val = None, 0.0
            for end, val, filed in sorted(ladder, key=lambda x: x[0]):
                inc_start = prev_end if prev_end is not None else start
                if 80 <= (end - inc_start).days <= 100 and end not in out:
                    out[end] = (val - prev_val, filed)  # YTD_t - YTD_{t-1} = the quarter
                prev_end, prev_val = end, val
        return out

    def _stock_level(self, points: list[dict]) -> dict[pd.Timestamp, tuple[float, pd.Timestamp]]:
        """Return {period_end: (level, filed_date)} for instantaneous balance-sheet items."""
        out: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
        for p in points:
            if p.get("form") not in ("10-Q", "10-K") or not p.get("end"):
                continue
            end = pd.Timestamp(p["end"])
            filed = pd.Timestamp(p["filed"])
            if end not in out or filed < out[end][1]:
                out[end] = (float(p["val"]), filed)
        return out

    def _facts_to_frame(self, ticker: str, facts: dict) -> pd.DataFrame:
        series: dict[str, dict] = {}
        filed_by_end: dict[pd.Timestamp, pd.Timestamp] = {}
        for field, tags in _FLOW.items():
            q = self._quarterly_flow(self._usd_points(facts, tags))
            series[field] = {end: v for end, (v, _f) in q.items()}
            for end, (_v, f) in q.items():
                filed_by_end[end] = max(filed_by_end.get(end, f), f)
        for field, tags in _STOCK.items():
            lvl = self._stock_level(self._usd_points(facts, tags))
            series[field] = {end: v for end, (v, _f) in lvl.items()}
        shares_by_filing = self._shares_outstanding_by_filing(
            self._usd_points(facts, _DEI_SHARES_TAGS, namespace="dei"))

        ends = sorted(e for e in filed_by_end)  # period ends anchored to the income statement
        if not ends:
            return pd.DataFrame()
        rows = []
        for end in ends:
            filed = filed_by_end[end]
            row = {"ticker": ticker, "period_end": end, "filing_date": filed}
            for field in {**_FLOW, **_STOCK}:
                row[field] = series[field].get(end, float("nan"))
            row["shares_outstanding"] = shares_by_filing.get(filed, float("nan"))
            rows.append(row)
        df = pd.DataFrame(rows)

        # Derived quantities (NaN-safe): EBITDA, free cash flow, total debt.
        df["ebitda"] = df["operating_income"] + df["dep_amort"]
        df["free_cash_flow"] = df["op_cash_flow"] - df["capex"]
        df["total_debt"] = df[["long_term_debt", "short_term_debt"]].sum(axis=1, min_count=1)
        derived = ["dep_amort", "op_cash_flow", "capex", "long_term_debt", "short_term_debt"]
        return df.drop(columns=derived)

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Point-in-time fundamentals for ``tickers``.

        Every reason a name is dropped is counted and reported. Silently
        skipping HTTP failures made a throttled pull indistinguishable from a
        complete one: SEC returns 403/429 under fair-access limits, so a run
        that recovered a tenth of the universe still produced a well-formed
        frame and called itself "the S&P 500 factor screen".
        """
        cikmap = self._cik_map()
        frames = []
        no_cik: list[str] = []
        http_failed: list[str] = []
        no_facts: list[str] = []
        for t in tickers:
            cik = cikmap.get(t)
            if cik is None:
                no_cik.append(t)          # not an SEC filer (often post-acquisition)
                continue
            key = f"edgar_fund_{t}"
            cached = self.cache.get(key)
            if cached is not None:
                frames.append(cached)
                continue
            try:
                facts = self._get(_FACTS_URL.format(cik=cik))
            except requests.HTTPError as e:
                http_failed.append(f"{t} ({getattr(e.response, 'status_code', '?')})")
                continue
            df = self._facts_to_frame(t, facts)
            if df.empty:
                no_facts.append(t)
                continue
            self.cache.put(key, df)
            frames.append(df)

        n = len(tickers)
        for label, dropped in (("no SEC CIK", no_cik), ("HTTP error", http_failed),
                               ("no usable XBRL facts", no_facts)):
            if dropped:
                print(f"[edgar] {len(dropped)}/{n} dropped - {label}: "
                      f"{', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}")
        if http_failed:
            # Distinguish "this company has no data" (a fact about the world)
            # from "we failed to fetch it" (a fact about this run).
            print(f"[edgar] WARNING: {len(http_failed)} ticker(s) failed to download and are "
                  "MISSING from this run, not genuinely absent - re-run to retry.")
        if frames:
            # One decimal, not zero: 501/503 rounds to "100%" at :.0f, which
            # reads as full coverage when two names are actually missing.
            print(f"[edgar] fundamentals recovered for {len(frames)}/{n} tickers "
                  f"({100 * len(frames) / n:.1f}% coverage)")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "period_end"])
