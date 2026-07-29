"""On-disk parquet cache.

Keeps us from re-hitting rate-limited APIs and — critically for FMP's free tier —
lets the fundamentals dataset accrue *incrementally*: run repeatedly, and each
run only fetches tickers not already cached.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class Cache:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.dir / f"{safe}.parquet"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> pd.DataFrame | None:
        p = self._path(key)
        return pd.read_parquet(p) if p.exists() else None

    def put(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(key), index=True)
