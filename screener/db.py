"""DuckDB analytical store.

DuckDB is column-oriented (fast for the panel/analytical queries here) and
embedded/zero-dependency like SQLite. The canonical table shapes live in
``sql/schema.sql`` and are applied on every :func:`connect` (typed skeleton);
:func:`write_table` then populates those typed tables from the pipeline's output
DataFrames. The analytical queries live in ``sql/queries.sql`` and are run by
:func:`query_file`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def connect(path: str = "outputs/screener.duckdb", schema_path: str | Path | None = _SCHEMA_PATH):
    """Open (creating parent dirs) and apply the declared schema so the typed
    tables exist before anything is written or queried."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    if schema_path and Path(schema_path).exists():
        apply_schema(con, schema_path)
    return con


def apply_schema(con, schema_path: str | Path = _SCHEMA_PATH) -> None:
    """Run ``sql/schema.sql`` (a bundle of ``CREATE TABLE IF NOT EXISTS``), so a
    fresh database opens with the canonical typed tables already present."""
    con.execute(Path(schema_path).read_text(encoding="utf-8"))


def _table_exists(con, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone() is not None


def write_table(con, name: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame. When the table was declared in ``schema.sql`` (the
    normal case), INSERT into the existing typed table so its declared column
    types are preserved rather than re-inferred (``INSERT ... BY NAME`` matches
    columns by name and tolerates ordering). Falls back to create-or-replace for
    ad-hoc tables not in the schema, or frames carrying columns beyond it."""
    d = df if isinstance(df.index, pd.RangeIndex) else df.reset_index()
    con.register("_df", d)
    try:
        if _table_exists(con, name):
            declared = {r[0] for r in con.execute(f'DESCRIBE "{name}"').fetchall()}
            if set(d.columns) <= declared:
                con.execute(f'DELETE FROM "{name}"')
                con.execute(f'INSERT INTO "{name}" BY NAME SELECT * FROM _df')
                return
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _df')
    finally:
        con.unregister("_df")


def query(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def iter_named_queries(path: str | Path) -> list[tuple[str, str]]:
    """Parse a ``.sql`` file of ``-- name: <id>`` blocks into ``[(id, sql), ...]``
    in file order. Shared by :func:`query_file` and the dashboard's SQL tab."""
    text = Path(path).read_text(encoding="utf-8")
    out: list[tuple[str, str]] = []
    name: str | None = None
    buf: list[str] = []

    def flush():
        if name and "".join(buf).strip():
            out.append((name, "\n".join(buf).strip()))

    for line in text.splitlines():
        if line.strip().lower().startswith("-- name:"):
            flush()
            name = line.split(":", 1)[1].strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return out


def _is_missing_table(err: Exception) -> bool:
    """A query referencing a table absent from *this* run (e.g. no dividend
    screen was built) is expected — skip it. A missing *column* or a syntax
    error is a genuine bug and must not be swallowed."""
    return isinstance(err, duckdb.CatalogException) and "does not exist" in str(err)


def query_file(con, path: str) -> dict[str, pd.DataFrame]:
    """Run a ``.sql`` file of named queries and return ``{id: DataFrame}``.

    Only a reference to a legitimately-absent table is skipped; any other error
    (bad column, bad SQL) propagates rather than being silently swallowed."""
    out: dict[str, pd.DataFrame] = {}
    for name, sql in iter_named_queries(path):
        try:
            out[name] = query(con, sql)
        except Exception as e:
            if _is_missing_table(e):
                continue
            raise
    return out
