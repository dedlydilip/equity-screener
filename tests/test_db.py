import pandas as pd

from screener.db import connect, query, write_table


def test_write_table_and_query_roundtrip(tmp_path):
    con = connect(str(tmp_path / "t.duckdb"))
    df = pd.DataFrame({"ticker": ["A", "B"], "score": [1.5, 2.5]})
    write_table(con, "screen", df)
    out = query(con, "SELECT * FROM screen ORDER BY ticker")
    assert out["ticker"].tolist() == ["A", "B"]
    assert out["score"].tolist() == [1.5, 2.5]


def test_write_table_is_replace_not_append(tmp_path):
    con = connect(str(tmp_path / "t.duckdb"))
    write_table(con, "screen", pd.DataFrame({"x": [1, 2, 3]}))
    write_table(con, "screen", pd.DataFrame({"x": [9]}))
    out = query(con, "SELECT * FROM screen")
    assert len(out) == 1
    assert out["x"].iloc[0] == 9
