-- Analytical schema (DuckDB). Tables are created-or-replaced from DataFrames by
-- screener/db.py at run time; these definitions document the expected shapes.

CREATE TABLE IF NOT EXISTS universe (
    ticker          VARCHAR,
    security        VARCHAR,
    sector          VARCHAR,
    sub_industry    VARCHAR,
    industry_group  VARCHAR
);

-- Latest ranked screen (one row per selected name).
CREATE TABLE IF NOT EXISTS screen (
    as_of   DATE,
    rank    INTEGER,
    ticker  VARCHAR,
    score   DOUBLE
);

-- Raw + composite factor panel for the latest cross-section.
CREATE TABLE IF NOT EXISTS factor_panel (
    as_of                DATE,
    ticker               VARCHAR,
    ep                   DOUBLE,
    bp                   DOUBLE,
    fcf_yield            DOUBLE,
    ev_ebitda_inv        DOUBLE,
    roe                  DOUBLE,
    roic                 DOUBLE,
    gross_profitability  DOUBLE,
    de_inv               DOUBLE,
    momentum             DOUBLE,
    composite            DOUBLE,
    market_cap           DOUBLE
);

-- Quintile forward returns over the backtest (one row per date x quantile).
CREATE TABLE IF NOT EXISTS backtest_quintiles (
    date      DATE,
    quantile  INTEGER,
    ret       DOUBLE
);

-- Fama-French 5 + UMD decomposition of the Q5-Q1 spread (tidy key/value).
CREATE TABLE IF NOT EXISTS ff_decomposition (
    metric  VARCHAR,
    value   DOUBLE
);
