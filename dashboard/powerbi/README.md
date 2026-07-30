# Power BI dashboard (secondary "enterprise BI" artifact)

The **primary** dashboard is Streamlit (`dashboard/app.py`) — browser-viewable and
free to host on Streamlit Cloud. Power BI is included as a second artifact to show
BI-tool fluency (relevant to your accounting-internship Power BI experience).

## Build it from the parquet exports
1. Run the pipeline so `outputs/*.parquet` exist — one command at a time:
   ```bash
   python run.py screen
   python run.py backtest --freq monthly
   python run.py decompose
   python run.py export
   ```
2. In **Power BI Desktop**: *Get Data → Parquet* (or *Folder* → `outputs/`) and load
   `screen`, `backtest_quintiles`, `factor_panel`, `ff_decomposition`, `universe`.
3. Model: relate `screen.ticker` and `factor_panel.ticker` to `universe.ticker`.

## Suggested pages
- **Screen** — a table of the ranked names (rank, ticker, security, score) with a
  sector slicer; a bar of sector exposure (count by `universe.sector`).
- **Backtest** — a line chart of cumulative quintile returns (from `backtest_quintiles`,
  running-total of `ret` by `quantile` over `date`); cards for Q5−Q1 Sharpe and Max Drawdown.
- **Factor decomposition** — cards/table from `ff_decomposition` (alpha, t-stat, R²) and a
  bar of the FF/UMD loadings.

## Commit
Save the `.pbix` here and add 2–3 screenshots (PNG) so GitHub visitors can see it without
Power BI Desktop.
