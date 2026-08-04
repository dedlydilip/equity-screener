-- Analytical schema (DuckDB) — the canonical, typed table contract.
--
-- Applied by screener/db.py:connect() on every connection (CREATE TABLE IF NOT
-- EXISTS), so a fresh database opens with the typed skeleton already in place and
-- a query against a not-yet-populated table returns zero rows instead of raising.
-- screener/db.py:write_table() then INSERTs each pipeline DataFrame into the
-- matching typed table (INSERT ... BY NAME), preserving these declared types
-- rather than re-inferring them from the frame. tests/test_sql.py enforces that
-- what the pipeline writes conforms to this file, so it can't silently drift.
--
-- Frequency-partitioned tables (backtest_quintiles, backtest_summary, ic_decay,
-- ff_decomposition, normalization_fallback) carry a `freq` column ('monthly' /
-- 'quarterly') so both rebalance frequencies coexist without overwriting each
-- other; always filter or group by `freq` when querying them.

-- S&P 500 universe with GICS classification.
CREATE TABLE IF NOT EXISTS universe (
    ticker          VARCHAR,
    security        VARCHAR,
    sector          VARCHAR,
    sub_industry    VARCHAR,
    industry_group  VARCHAR
);

-- Latest ranked screen (one row per selected name).
CREATE TABLE IF NOT EXISTS screen (
    as_of   TIMESTAMP,
    ticker  VARCHAR,
    rank    BIGINT,
    score   DOUBLE
);

-- Raw + normalized + composite factor panel for the latest cross-section.
CREATE TABLE IF NOT EXISTS factor_panel (
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
    market_cap           DOUBLE,
    as_of                TIMESTAMP,
    composite            DOUBLE,
    z_value              DOUBLE,
    z_quality            DOUBLE,
    z_momentum           DOUBLE,
    pe                   DOUBLE,
    pb                   DOUBLE,
    debt_to_equity       DOUBLE
);

-- Quintile forward returns over the backtest (one row per freq x date x quantile).
CREATE TABLE IF NOT EXISTS backtest_quintiles (
    freq      VARCHAR,
    date      TIMESTAMP,
    quantile  BIGINT,
    ret       DOUBLE
);

-- One headline row per backtest frequency: gross/net spread & Sharpe, drawdown,
-- turnover, and the HAC-adjusted IC t-stat.
CREATE TABLE IF NOT EXISTS backtest_summary (
    freq                    VARCHAR,
    weighting               VARCHAR,
    n_rebalances            BIGINT,
    gross_spread_mean       DOUBLE,
    net_spread_mean         DOUBLE,
    gross_sharpe            DOUBLE,
    net_sharpe              DOUBLE,
    q5_gross_sharpe         DOUBLE,
    max_drawdown            DOUBLE,
    turnover_per_rebalance  DOUBLE,
    turnover_annualized     DOUBLE,
    mean_ic                 DOUBLE,
    ic_t_stat_hac           DOUBLE
);

-- Mean IC at increasing forward lags (factor lifespan), per frequency.
CREATE TABLE IF NOT EXISTS ic_decay (
    freq  VARCHAR,
    lag   BIGINT,
    ic    DOUBLE
);

-- Fama-French 5 + UMD decomposition of the Q5-Q1 spread (tidy key/value), per freq.
CREATE TABLE IF NOT EXISTS ff_decomposition (
    freq    VARCHAR,
    metric  VARCHAR,
    value   DOUBLE
);

-- Sector-neutral normalization: % of names resolved at each hierarchy level, per
-- freq x rebalance date x sleeve (level = industry_group / sector / cross_sectional).
CREATE TABLE IF NOT EXISTS normalization_fallback (
    freq    VARCHAR,
    date    TIMESTAMP,
    sleeve  VARCHAR,
    level   VARCHAR,
    pct     DOUBLE
);

-- Per-security CAPM beta / alpha vs. the market (Newey-West).
CREATE TABLE IF NOT EXISTS capm_betas (
    ticker                         VARCHAR,
    beta                           DOUBLE,
    alpha_monthly                  DOUBLE,
    alpha_annualized               DOUBLE,
    alpha_t_nw                     DOUBLE,
    r_squared                      DOUBLE,
    n                              BIGINT,
    mean_excess_return_annualized  DOUBLE,
    mkt_excess_return_annualized   DOUBLE
);

-- Long-only mean-variance optimal weights (one row per method x ticker).
CREATE TABLE IF NOT EXISTS optimal_weights (
    method  VARCHAR,
    ticker  VARCHAR,
    weight  DOUBLE
);

-- Efficient frontier points for the equity optimizer.
CREATE TABLE IF NOT EXISTS efficient_frontier (
    target_return  DOUBLE,
    volatility     DOUBLE
);

-- Cross-asset (equity/bond/commodity ETF) CAPM betas.
CREATE TABLE IF NOT EXISTS multiasset_betas (
    ticker                         VARCHAR,
    beta                           DOUBLE,
    alpha_monthly                  DOUBLE,
    alpha_annualized               DOUBLE,
    alpha_t_nw                     DOUBLE,
    r_squared                      DOUBLE,
    n                              BIGINT,
    mean_excess_return_annualized  DOUBLE,
    mkt_excess_return_annualized   DOUBLE,
    asset_class                    VARCHAR
);

-- Cross-asset optimal weights (one row per ticker, with its asset class).
CREATE TABLE IF NOT EXISTS multiasset_weights (
    ticker       VARCHAR,
    weight       DOUBLE,
    asset_class  VARCHAR
);

-- Cross-asset efficient frontier points.
CREATE TABLE IF NOT EXISTS multiasset_frontier (
    target_return  DOUBLE,
    volatility     DOUBLE
);

-- Dividend-income screen: recurring vs. total trailing yield + payout sustainability.
CREATE TABLE IF NOT EXISTS dividend_screen (
    as_of                   TIMESTAMP,
    ticker                  VARCHAR,
    rank                    BIGINT,
    regular_dividend_yield  DOUBLE,
    total_dividend_yield    DOUBLE,
    has_special_dividend    BOOLEAN,
    payout_ratio            DOUBLE,
    sector                  VARCHAR,
    weight                  DOUBLE
);
