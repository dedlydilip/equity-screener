"""On-disk parquet cache.

Keeps us from re-hitting rate-limited APIs and — critically for FMP's free tier —
lets the fundamentals dataset accrue *incrementally*: run repeatedly, and each
run only fetches tickers not already cached.

**Versioning.** Every entry is namespaced by a ``version`` string supplied by the
caller, which is expected to encode whatever would change the *meaning* of the
cached payload — for EDGAR, a hash of the XBRL tag maps. Without it, editing a
tag map had no effect on any already-cached ticker: the stale frame was returned
verbatim, the extraction code never re-ran, and a column added later came back
silently all-NaN. An optional ``ttl_days`` additionally expires entries by age.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd


def version_hash(*parts: object, length: int = 10) -> str:
    """Short stable digest of anything JSON-serializable — use for cache versions.

    Pass the structures whose change should invalidate the cache (e.g. the tag
    maps an extractor depends on) and the digest changes with them.
    """
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


class Cache:
    def __init__(self, cache_dir: str = ".cache", version: str = "v1",
                 ttl_days: float | None = None) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.ttl_days = ttl_days

    def _path(self, key: str) -> Path:
        safe = f"{self.version}__{key}".replace("/", "_").replace(":", "_")
        return self.dir / f"{safe}.parquet"

    def _fresh(self, p: Path) -> bool:
        if self.ttl_days is None:
            return True
        return (time.time() - p.stat().st_mtime) <= self.ttl_days * 86400

    def get(self, key: str) -> pd.DataFrame | None:
        p = self._path(key)
        if not p.exists() or not self._fresh(p):
            return None
        return pd.read_parquet(p)

    def put(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(key), index=True)
