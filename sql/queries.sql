-- Analytical queries. Each block is delimited by "-- name: <id>" and is run by
-- screener/db.py:query_file, returning {id: DataFrame}.

-- name: top_screen
SELECT rank, ticker, ROUND(score, 3) AS score
FROM screen
ORDER BY rank
LIMIT 25;

-- name: sector_exposure
SELECT u.sector,
       COUNT(*)                                              AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)    AS pct_of_screen
FROM screen s
JOIN universe u ON s.ticker = u.ticker
GROUP BY u.sector
ORDER BY n DESC;

-- name: quintile_returns
SELECT quantile,
       ROUND(AVG(ret) * 100, 3)  AS avg_monthly_ret_pct,
       COUNT(*)                  AS n_months
FROM backtest_quintiles
GROUP BY quantile
ORDER BY quantile;

-- name: long_short_spread
SELECT ROUND((AVG(CASE WHEN quantile = 5 THEN ret END)
            - AVG(CASE WHEN quantile = 1 THEN ret END)) * 100, 3) AS q5_minus_q1_pct
FROM backtest_quintiles;

-- name: ff_summary
SELECT metric, ROUND(value, 4) AS value
FROM ff_decomposition;
