"""DuckDB analytical store.

DuckDB is column-oriented (fast for the panel/analytical queries here) and
embedded/zero-dependency like SQLite. Output DataFrames (universe, factor panel,
screen, backtest quintiles, FF decomposition) are written as tables; the
analytical SQL lives in ``sql/queries.sql``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def connect(path: str = "outputs/screener.duckdb"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def write_table(con, name: str, df: pd.DataFrame) -> None:
    """Create-or-replace a table from a DataFrame (non-default indexes are reset in)."""
    d = df if isinstance(df.index, pd.RangeIndex) else df.reset_index()
    con.register("_df", d)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _df')
    con.unregister("_df")


def query(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def query_file(con, path: str) -> dict[str, pd.DataFrame]:
    """Run a .sql file of named queries delimited by ``-- name: <id>`` and return {id: df}."""
    text = Path(path).read_text(encoding="utf-8")
    out: dict[str, pd.DataFrame] = {}
    name: str | None = None
    buf: list[str] = []

    def flush():
        if name and "".join(buf).strip():
            try:
                out[name] = query(con, "\n".join(buf))
            except Exception:
                pass  # a referenced table may not exist for this run; skip that report

    for line in text.splitlines():
        if line.strip().lower().startswith("-- name:"):
            flush()
            name = line.split(":", 1)[1].strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return out
