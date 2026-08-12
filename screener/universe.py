"""S&P 500 universe + GICS classification.

Current constituents are fetched from Wikipedia (no API key). Using the
**current** membership list is a survivorship-biased universe — a documented
limitation (look-ahead on *fundamentals* is handled separately via SEC filing
dates). A local CSV cache is written and used as an offline fallback.

Returns columns: ``ticker, security, sector, sub_industry, industry_group``.
``industry_group`` is mapped from the GICS sub-industry via :mod:`screener.gics`;
unmapped names are left NA and the normalizer falls back to sector for them.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from .gics import industry_group_for

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia 403s the default urllib user-agent, so fetch with a real UA.
_HEADERS = {"User-Agent": "equity-screener/0.1 (research project; contact via GitHub)"}
_CACHE = Path(__file__).with_name("data") / "sp500_cache.csv"


def _parse_constituents(html: str) -> pd.DataFrame:
    """Parse the constituents table out of the Wikipedia page.

    Deliberately separate from the fetch: wrapping both in one ``except`` made a
    parse break (Wikipedia reordering its tables, or renaming a column) look
    exactly like a network outage, and both silently resolved to a stale CSV.
    """
    raw = pd.read_html(io.StringIO(html))[0]
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Wikipedia constituents table is missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            # Wikipedia uses "BRK.B"; normalize to the "BRK-B" form yfinance expects.
            "ticker": raw["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip(),
            "security": raw["Security"].astype(str),
            "sector": raw["GICS Sector"].astype(str).str.strip(),
            "sub_industry": raw["GICS Sub-Industry"].astype(str).str.strip(),
        }
    )


def get_sp500(use_cache: bool = True) -> pd.DataFrame:
    """Load S&P 500 constituents with GICS sector / sub-industry / industry group.

    On a fetch or parse failure this falls back to the committed CSV snapshot,
    but says so loudly and does NOT rewrite the cache -- overwriting it on the
    fallback path refreshed the file's mtime with its own stale contents,
    destroying the only evidence of how old the universe was.
    """
    df: pd.DataFrame | None = None
    from_cache = False
    try:
        resp = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        df = _parse_constituents(resp.text)
    except Exception as e:  # noqa: BLE001 - network OR parse; both fall back
        if use_cache and _CACHE.exists():
            age_days = (time.time() - _CACHE.stat().st_mtime) / 86400
            print(f"[universe] live fetch failed ({type(e).__name__}: {e}); "
                  f"falling back to the cached constituent list, {age_days:.0f} days old. "
                  "The universe may not reflect current index membership.")
            df = pd.read_csv(_CACHE)
            from_cache = True
        else:
            raise

    df["industry_group"] = df["sub_industry"].map(industry_group_for)

    if use_cache and not from_cache:
        df.to_csv(_CACHE, index=False)

    return df.reset_index(drop=True)


def coverage(df: pd.DataFrame) -> dict:
    """Diagnostics: how many names got an industry-group mapping (rest -> sector)."""
    n = len(df)
    mapped = int(df["industry_group"].notna().sum())
    return {
        "n": int(n),
        "sectors": int(df["sector"].nunique()),
        "industry_groups": int(df["industry_group"].dropna().nunique()),
        "industry_group_mapped_pct": round(100 * mapped / n, 1) if n else 0.0,
    }
