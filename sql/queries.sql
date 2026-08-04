-- Analytical queries. Each block is delimited by "-- name: <id>" and is run by
-- screener/db.py:query_file, returning {id: DataFrame}. `python run.py reports`
-- prints them; the dashboard's "SQL" tab runs them live and shows this text.
--
-- Frequency-partitioned tables carry a `freq` column, so any query over the
-- backtest tables groups/filters by `freq` — otherwise monthly and quarterly
-- rows get silently averaged together.

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
SELECT freq,
       quantile,
       ROUND(AVG(ret) * 100, 3)  AS avg_ret_pct,
       COUNT(*)                  AS n_periods
FROM backtest_quintiles
GROUP BY freq, quantile
ORDER BY freq, quantile;

-- name: long_short_spread
SELECT freq,
       ROUND((AVG(CASE WHEN quantile = 5 THEN ret END)
            - AVG(CASE WHEN quantile = 1 THEN ret END)) * 100, 3) AS q5_minus_q1_pct
FROM backtest_quintiles
GROUP BY freq
ORDER BY freq;

-- name: backtest_scorecard
SELECT freq,
       weighting,
       n_rebalances,
       ROUND(gross_spread_mean * 100, 3)   AS gross_spread_pct,
       ROUND(net_spread_mean * 100, 3)     AS net_spread_pct,
       ROUND(gross_sharpe, 2)              AS gross_sharpe,
       ROUND(net_sharpe, 2)                AS net_sharpe,
       ROUND(max_drawdown * 100, 1)        AS max_drawdown_pct,
       ROUND(turnover_annualized * 100, 0) AS turnover_ann_pct,
       ROUND(ic_t_stat_hac, 2)             AS ic_t_hac
FROM backtest_summary
ORDER BY freq;

-- name: normalization_fallback_avg
SELECT freq,
       sleeve,
       level,
       ROUND(AVG(pct), 1)  AS mean_pct
FROM normalization_fallback
GROUP BY freq, sleeve, level
ORDER BY freq, sleeve, level;

-- name: ff_summary
SELECT freq, metric, ROUND(value, 4) AS value
FROM ff_decomposition
ORDER BY freq, metric;
