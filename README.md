# Institutional Factor Screener — S&P 500

A factor-based equity screener built like the systematic-equity desks at MSCI, S&P, or AQR: a
**Value / Quality / Momentum** composite, sector-neutral cross-sectional scoring, and a
**Fama-French 5 + Momentum decomposition** that turns the screen into a research engine — proving
whether any outperformance is a genuine premium or just repackaged HML/RMW/CMA/UMD.

[![tests](https://github.com/dedlydilip/equity-screener/actions/workflows/test.yml/badge.svg)](https://github.com/dedlydilip/equity-screener/actions/workflows/test.yml)

## Why this exists

Most retail "stock screeners" rank on trailing ratios computed off today's numbers with no
regard for when those numbers actually became public, and no answer to "is this premium already
explained by a known factor?" This project is built to institutional standards instead:

- **No look-ahead.** Fundamentals are gated by their SEC **filing date** (FMP `fillingDate`), not
  the fiscal period end — a Q1 number isn't usable until it was actually filed.
- **Sector-neutral, robust normalization.** MAD (median absolute deviation) z-scores, computed
  within a finest-first GICS hierarchy (industry group → sector → cross-sectional), so a screen
  isn't secretly just "buy tech."
- **Honest about distress.** A negative book value doesn't make Book/Price look "cheap"; negative
  EBITDA doesn't make EV/EBITDA look "cheap." Negative FCF yield is kept — it correctly sorts to
  the bottom.
- **The FF decomposition uses the right benchmark mechanics.** The long-short spread is
  self-financing, so it's regressed on the **raw** return difference (no risk-free rate
  subtracted); a single long-only leg is regressed as an **excess** return (R − Rf). Getting this
  backwards silently overstates or understates alpha.

## Architecture

```
config.yaml          pre-registered, Pydantic-validated run configuration
screener/
  universe.py         S&P 500 constituents + GICS sector / industry-group classification
  data/
    fmp.py            point-in-time fundamentals (SEC filing-date gated) + prices
    yf.py             adjusted total-return price fallback (no API key)
    french.py          Fama-French 5 + momentum factor returns (Ken French Data Library)
    quality.py        data-quality gate: hard bounds, MAD winsorization, negative-denominator masks
    cache.py          incremental parquet cache (works within FMP's free-tier rate limit)
  factors.py          raw value / quality / momentum factors from PiT fundamentals + prices
  normalize.py        MAD sector-neutral z-scoring (hierarchy fallback) + inverse-vol composite
  portfolio.py         quantile assignment, equal / market-cap / inverse-vol weighting, concentration
  screen.py           hard filters + the ranked screen
  validate.py         IC, quintile backtest, Sharpe, Max Drawdown, turnover, FF5+UMD decomposition
  db.py, report.py    DuckDB analytical store + parquet export for the dashboards
sql/                  DuckDB analytical queries (sector exposure, quintile returns, ...)
dashboard/app.py      Streamlit dashboard (primary) — screen, backtest, factor decomposition
tests/                pytest suite (fixtures only — no live API calls in CI)
run.py                CLI: screen | backtest | decompose | export
```

## Methodology

**Factors** (all oriented so higher = better):

| Sleeve | Signals |
|---|---|
| Value | Earnings yield, Book/Price, FCF yield, EV/EBITDA (inverted) |
| Quality | ROE, ROIC, Gross Profitability, Debt/Equity (inverted) |
| Momentum | 12-1 month total return |

**Normalization:** robust z-score `(x − median) / (1.4826 × MAD)` within a group, clipped at ±5
MAD. Groups follow a finest-first fallback hierarchy — GICS industry group (min 20 names) → GICS
sector (min 20) → the full cross-section as a last resort — so a small industry group doesn't get
a noisy median. The fallback split is reported with every run.

**Composite:** sleeve z-scores are blended with inverse-volatility weights computed from each
sleeve's own trailing 12-month long-short return series, capped at 50% per sleeve (redistributed
if exceeded), with a **12-month equal-weight burn-in** at the start of the backtest to avoid using
a lookback window that doesn't exist yet.

**Validation:** Information Coefficient (Spearman) with decay at 1/3/5-month lags; a quintile
backtest (Sharpe over the Ken French risk-free rate, Max Drawdown of the Q5−Q1 spread); turnover
per-rebalance and annualized; net-of-cost returns (5 bps commission + 10 bps spread + 25 bps short
rebate, per side); and the FF5+UMD alpha decomposition described above, with Newey-West
standard errors.

## Data

- **Universe:** current S&P 500 constituents (Wikipedia). Using today's membership list means the
  backtest has **survivorship bias on the constituent list** — a documented, honest limitation.
  Fundamental look-ahead is a separate, solved problem (SEC filing dates).
- **Fundamentals:** [Financial Modeling Prep](https://financialmodelingprep.com) (free tier),
  point-in-time via `fillingDate`.
- **Prices:** FMP adjusted closes, falling back to `yfinance` (`auto_adjust=True`) if FMP is
  unavailable — both are split-and-dividend-adjusted total-return series, never raw closes.
- **Factor returns:** the Ken French Data Library, via `pandas_datareader` (free, no key).

## Running it

```bash
pip install -e ".[dev]"
setx FMP_API_KEY "your_key"      # free tier at financialmodelingprep.com; open a NEW terminal after

python run.py screen                    # latest ranked screen + concentration diagnostics
python run.py backtest --freq monthly   # IC, quintile spread, turnover
python run.py decompose                 # FF5+UMD alpha decomposition of the Q5-Q1 spread
python run.py export                    # write outputs/*.parquet for the dashboard

streamlit run dashboard/app.py          # browse the results
```

Tests (fixtures only, no network — this is what CI runs):

```bash
pytest -q
```

### Live demo on the FMP free tier

FMP's free tier serves point-in-time fundamentals for only a whitelist of large-cap
names (and the most recent ~5 quarters). `demo_tickers.txt` is a ready-made subset of 32
free-accessible S&P 500 names spanning six sectors, so the pipeline runs end-to-end on real data:

```bash
python run.py screen --tickers "$(cat demo_tickers.txt)"
python run.py export
```

Example real output (screen as of 2026-07):

| rank | ticker | sector | composite |
|---|---|---|---|
| 1 | AMD | Information Technology | 1.48 |
| 2 | C | Financials | 1.11 |
| 3 | GOOGL | Communication Services | 1.05 |

The **full 500-name universe and the historical backtest/decomposition** require a paid FMP tier
(or another point-in-time fundamentals source) — see Limitations. All the methodology is validated
on synthetic fixtures in CI regardless.

## Limitations

- **Data access, not methodology, is the binding constraint on the free tier.** FMP's free plan
  serves point-in-time fundamentals only for a whitelist of large caps and only the most recent
  ~5 quarters. That's enough for a live *current* screen on the mega-caps (see the demo above), but
  **the decade-long quintile backtest and FF5+UMD decomposition need deeper history across the full
  universe — i.e. a paid FMP tier or an equivalent PiT source.** The backtest/decomposition code is
  fully implemented and unit-tested; it's gated on data, not correctness.
- **Survivorship on the constituent list** — the screen uses today's S&P 500 membership, not the
  historical roster at each rebalance date. A future extension would source point-in-time index
  membership; for now this is a documented trade-off, not a hidden flaw.
- **MAD can be noisy in small groups** — mitigated by the finest-first fallback hierarchy with a
  reported fallback percentage, but not eliminated.
- **Transaction costs are modeled, not simulated** — commission, spread, and short-rebate are
  applied as flat per-side bps, not a market-impact model.
- **This is a screener and factor-research engine, not a portfolio optimizer** — position sizing
  is rule-based (equal / market-cap / inverse-vol), not mean-variance optimized.

## Author

Built by Dilip Amaranarayana as a portfolio project demonstrating factor-investing methodology,
point-in-time data discipline, and full-stack analytics engineering (Python, SQL/DuckDB,
Streamlit/Power BI).
