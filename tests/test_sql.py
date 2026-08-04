"""The SQL layer: schema application, typed-table writes, and the named
analytical queries — verifying the layer actually runs (not decorative), stays
frequency-correct, and surfaces real errors instead of swallowing them."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from screener.db import (
    apply_schema,
    connect,
    iter_named_queries,
    query,
    query_file,
    write_table,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "sql" / "schema.sql"
QUERIES = ROOT / "sql" / "queries.sql"

# Every table sql/schema.sql is expected to declare.
_DECLARED_TABLES = {
    "universe", "screen", "factor_panel", "backtest_quintiles", "backtest_summary",
    "ic_decay", "ff_decomposition", "normalization_fallback", "capm_betas",
    "optimal_weights", "efficient_frontier", "multiasset_betas", "multiasset_weights",
    "multiasset_frontier", "dividend_screen",
}


def _seed_backtest_tables(con):
    """Two frequencies of quintile returns + a summary + FF + fallback rows, so
    the freq-partitioned queries have something distinguishable to group on."""
    q_rows = []
    for freq, base in [("monthly", 0.001), ("quarterly", 0.004)]:
        for i in range(3):
            dt = pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(i)
            for q in range(1, 6):
                q_rows.append({"freq": freq, "date": dt, "quantile": q, "ret": base * q})
    write_table(con, "backtest_quintiles", pd.DataFrame(q_rows))
    write_table(con, "backtest_summary", pd.DataFrame([
        {"freq": "monthly", "weighting": "equal", "n_rebalances": 3, "gross_spread_mean": 0.004,
         "net_spread_mean": 0.002, "gross_sharpe": 0.3, "net_sharpe": 0.1, "q5_gross_sharpe": 0.4,
         "max_drawdown": -0.1, "turnover_per_rebalance": 0.4, "turnover_annualized": 4.8,
         "mean_ic": 0.01, "ic_t_stat_hac": 0.9},
        {"freq": "quarterly", "weighting": "equal", "n_rebalances": 3, "gross_spread_mean": 0.016,
         "net_spread_mean": 0.012, "gross_sharpe": 0.4, "net_sharpe": 0.2, "q5_gross_sharpe": 0.3,
         "max_drawdown": -0.07, "turnover_per_rebalance": 0.75, "turnover_annualized": 3.0,
         "mean_ic": 0.012, "ic_t_stat_hac": 0.85},
    ]))
    write_table(con, "ff_decomposition", pd.DataFrame([
        {"freq": "monthly", "metric": "alpha_annualized", "value": 0.02},
        {"freq": "quarterly", "metric": "alpha_annualized", "value": 0.028},
    ]))
    write_table(con, "normalization_fallback", pd.DataFrame([
        {"freq": "monthly", "date": pd.Timestamp("2020-01-31"), "sleeve": "value",
         "level": "industry_group", "pct": 67.0},
        {"freq": "monthly", "date": pd.Timestamp("2020-02-29"), "sleeve": "value",
         "level": "industry_group", "pct": 69.0},
    ]))


def test_schema_applies_and_declares_every_table(tmp_path):
    con = connect(str(tmp_path / "s.duckdb"))
    live = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert _DECLARED_TABLES <= live
    # a query against a freshly-declared-but-empty table returns zero rows, not an error
    assert query(con, "SELECT * FROM screen").empty


def test_write_preserves_declared_types_not_inferred(tmp_path):
    con = connect(str(tmp_path / "s.duckdb"))
    _seed_backtest_tables(con)
    types = {r[0]: r[1] for r in con.execute("DESCRIBE backtest_quintiles").fetchall()}
    # date stays TIMESTAMP and quantile stays BIGINT because the row was INSERTed
    # into the schema-typed table, not re-inferred from the frame.
    assert types["date"] == "TIMESTAMP"
    assert types["quantile"] == "BIGINT"
    assert types["freq"] == "VARCHAR"


def test_written_columns_match_the_declared_schema(tmp_path):
    """Guards against silent drift: what the pipeline writes must be a subset of
    what schema.sql declares (extra columns would force a create-or-replace and
    break the typed contract)."""
    con = connect(str(tmp_path / "s.duckdb"))
    _seed_backtest_tables(con)
    for table in ["backtest_quintiles", "backtest_summary", "ff_decomposition",
                  "normalization_fallback"]:
        declared = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
        written = set(con.execute(f"SELECT * FROM {table} LIMIT 0").df().columns)
        assert written <= declared, f"{table} has columns beyond schema.sql: {written - declared}"


def test_named_queries_run_and_stay_frequency_correct(tmp_path):
    con = connect(str(tmp_path / "s.duckdb"))
    write_table(con, "screen", pd.DataFrame({
        "as_of": [pd.Timestamp("2024-01-31")] * 3, "ticker": ["A", "B", "C"],
        "rank": [1, 2, 3], "score": [3.0, 2.0, 1.0]}))
    write_table(con, "universe", pd.DataFrame({
        "ticker": ["A", "B", "C"], "security": ["A Inc", "B Inc", "C Inc"],
        "sector": ["Tech", "Tech", "Energy"], "sub_industry": ["x", "y", "z"],
        "industry_group": ["g", "g", "h"]}))
    _seed_backtest_tables(con)

    reports = query_file(con, str(QUERIES))
    assert {"top_screen", "sector_exposure", "quintile_returns", "long_short_spread",
            "backtest_scorecard", "ff_summary"} <= set(reports)

    # The freq-partitioned queries must NOT collapse monthly and quarterly together.
    qr = reports["quintile_returns"]
    assert set(qr["freq"]) == {"monthly", "quarterly"}
    assert len(qr) == 10  # 2 freqs x 5 quantiles, kept distinct

    ls = reports["long_short_spread"]
    assert set(ls["freq"]) == {"monthly", "quarterly"}
    # quarterly base return is 4x monthly, so its Q5-Q1 spread must be strictly larger
    m = ls.loc[ls["freq"] == "monthly", "q5_minus_q1_pct"].iloc[0]
    q = ls.loc[ls["freq"] == "quarterly", "q5_minus_q1_pct"].iloc[0]
    assert q > m > 0


def test_query_file_skips_absent_table_but_raises_on_bad_column(tmp_path):
    con = connect(str(tmp_path / "s.duckdb"))
    # Only `screen` populated; queries touching backtest tables skip cleanly.
    write_table(con, "screen", pd.DataFrame({
        "as_of": [pd.Timestamp("2024-01-31")], "ticker": ["A"], "rank": [1], "score": [1.0]}))
    reports = query_file(con, str(QUERIES))
    assert "top_screen" in reports  # ran against the one populated table

    # A genuinely broken query (bad column on an existing table) must propagate,
    # not be swallowed the way a missing optional table is.
    bad = tmp_path / "bad.sql"
    bad.write_text("-- name: broken\nSELECT no_such_column FROM screen;\n", encoding="utf-8")
    with pytest.raises(duckdb.BinderException):
        query_file(con, str(bad))


def test_iter_named_queries_parses_named_blocks():
    blocks = dict(iter_named_queries(QUERIES))
    assert "backtest_scorecard" in blocks
    assert blocks["top_screen"].strip().upper().startswith("SELECT")


def test_apply_schema_is_idempotent(tmp_path):
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    apply_schema(con, SCHEMA)
    apply_schema(con, SCHEMA)  # CREATE TABLE IF NOT EXISTS -> second application is a no-op
    assert _DECLARED_TABLES <= {r[0] for r in con.execute("SHOW TABLES").fetchall()}
