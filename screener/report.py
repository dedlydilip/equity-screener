"""Export helpers: push DuckDB tables to parquet/CSV for the dashboards, and run
the named analytical queries in ``sql/queries.sql``.
"""

from __future__ import annotations

from pathlib import Path

from .db import query, query_file

_TABLES = [
    "universe", "screen", "factor_panel", "backtest_quintiles", "ff_decomposition", "ic_decay",
]


def export_for_dashboard(con, export_dir: str = "outputs") -> list[str]:
    """Write each existing table to parquet (dashboards read these). Returns paths written."""
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for t in _TABLES:
        try:
            df = query(con, f'SELECT * FROM "{t}"')
        except Exception:
            continue
        p = out / f"{t}.parquet"
        df.to_parquet(p)
        written.append(str(p))
    return written


def run_reports(con, queries_path: str = "sql/queries.sql") -> dict:
    """Run the analytical queries and return {name: DataFrame}."""
    return query_file(con, queries_path)
