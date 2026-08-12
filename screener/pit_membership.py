"""Point-in-time S&P 500 constituent reconstruction (mitigates survivorship bias).

Historical membership changes are sourced from Wikipedia's "Selected changes to
the list of S&P 500 components" table — the same CC BY-SA page (and same
User-Agent) that ``universe.get_sp500`` already uses for the current list, just
a second table on it. Reconstructing membership on a past date means starting
from TODAY's constituent list and walking backward through every LATER change,
undoing it: a name added after the target date wasn't yet a member; a name
removed after the target date hadn't been removed yet.

Documented, honest limitation: this mitigates but does not fully eliminate
survivorship bias. A removed/acquired constituent's fundamentals are only
recoverable if it's still an SEC-registered filer (EDGAR's CIK lookup simply
returns nothing otherwise), and its price history is only available if
yfinance still serves data up to its delisting date — neither is guaranteed,
especially for older or fully-deregistered removals. Sector-neutral
normalization also still peers each name against the FULL available
cross-section (including names not-yet or no-longer members at that date) —
only *portfolio/backtest membership* is restricted to true point-in-time
constituents, not the normalization peer group (GICS classification itself is
not point-in-time either — see universe.py).
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "equity-screener/0.1 (research project; contact via GitHub)"}
_CACHE = Path(__file__).with_name("data") / "sp500_changes_cache.csv"


def _normalize_ticker_col(s: pd.Series) -> pd.Series:
    # Wikipedia uses "BRK.B"; normalize to the "BRK-B" form yfinance/EDGAR expect
    # (same convention as universe.get_sp500). Keeps missing values as <NA>.
    return s.astype("string").str.replace(".", "-", regex=False).str.strip()


def _parse_changes(html: str) -> pd.DataFrame:
    """Parse the 'changes to the list' table. Separate from the fetch so a
    layout change is distinguishable from a network failure."""
    raw = pd.read_html(io.StringIO(html))[1]
    if raw.shape[1] != 6:
        raise ValueError(
            f"Wikipedia changes table has {raw.shape[1]} columns, expected 6 "
            "(date, added ticker/security, removed ticker/security, reason) - "
            "the positional column assignment below would silently mislabel them."
        )
    raw.columns = ["date", "added_ticker", "added_security",
                   "removed_ticker", "removed_security", "reason"]
    parsed = pd.to_datetime(raw["date"], format="%B %d, %Y", errors="coerce")
    unparseable = int(parsed.isna().sum())
    if unparseable:
        # Dropping these silently means those membership changes are never
        # undone, so every reconstructed historical universe quietly retains
        # names that had actually left the index.
        print(f"[pit_membership] WARNING: {unparseable}/{len(raw)} change rows have an "
              "unparseable date and are excluded; reconstructed historical membership "
              "will be incomplete for the affected dates.")
    return pd.DataFrame({
        "date": parsed,
        "added_ticker": _normalize_ticker_col(raw["added_ticker"]),
        "removed_ticker": _normalize_ticker_col(raw["removed_ticker"]),
    }).dropna(subset=["date"])


def get_sp500_changes(use_cache: bool = True) -> pd.DataFrame:
    """Tidy changes table: columns ``date, added_ticker, removed_ticker``,
    one row per effective-date change event, sorted ascending by date.

    As in :func:`screener.universe.get_sp500`, a fallback to the committed CSV
    is announced and does not rewrite the cache.
    """
    df: pd.DataFrame | None = None
    from_cache = False
    try:
        resp = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        df = _parse_changes(resp.text)
    except Exception as e:  # noqa: BLE001 - network OR parse; both fall back
        if use_cache and _CACHE.exists():
            age_days = (time.time() - _CACHE.stat().st_mtime) / 86400
            print(f"[pit_membership] live fetch failed ({type(e).__name__}: {e}); "
                  f"using the cached changes table, {age_days:.0f} days old.")
            df = pd.read_csv(_CACHE, parse_dates=["date"])
            from_cache = True
        else:
            raise

    if use_cache and not from_cache:
        df.to_csv(_CACHE, index=False)
    return df.sort_values("date").reset_index(drop=True)


def constituents_as_of(
    target_date, current_tickers: set[str], changes: pd.DataFrame
) -> set[str]:
    """Reconstruct S&P 500 membership on ``target_date`` by undoing every
    change strictly AFTER it, walking back from today's constituent list.

    Must undo in REVERSE-chronological order (most recent first): if a ticker
    was replaced twice in succession (e.g. A1->A2 in 2021, then A2->A3 in
    2022), undoing 2021 before 2022 would incorrectly resurrect A2 as a
    permanent member instead of recognizing it only ever held that index slot
    for one year in between.
    """
    target = pd.Timestamp(target_date)
    members = set(current_tickers)
    later = changes[changes["date"] > target].sort_values("date", ascending=False)
    for _, row in later.iterrows():
        added, removed = row["added_ticker"], row["removed_ticker"]
        if pd.notna(added):
            members.discard(added)   # not yet added, as of target_date
        if pd.notna(removed):
            members.add(removed)     # not yet removed, as of target_date
    return members


def all_time_tickers(
    current_tickers: set[str], changes: pd.DataFrame, since=None
) -> set[str]:
    """Union of today's constituents plus every ticker involved in a change
    on/after ``since`` — the universe to pull data for so a backtest can
    apply point-in-time membership across a date window. Changes before
    ``since`` never need to be undone for any target date >= since, so they're
    excluded to avoid pulling data for names irrelevant to the window."""
    relevant = changes if since is None else changes[changes["date"] >= pd.Timestamp(since)]
    tickers = set(current_tickers)
    tickers |= set(relevant["added_ticker"].dropna())
    tickers |= set(relevant["removed_ticker"].dropna())
    return tickers
