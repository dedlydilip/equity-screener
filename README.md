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

- **No look-ahead.** Fundamentals are gated by their SEC **filing date** (the XBRL `filed`
  timestamp), not the fiscal period end — a Q1 number isn't usable until it was actually filed.
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
    edgar.py          SEC EDGAR XBRL point-in-time fundamentals (default; free, full history)
    fmp.py            alternative point-in-time fundamentals provider (FMP /stable API)
    yf.py             adjusted total-return prices (no API key)
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

Everything runs on **free** sources — no API key required for the default path:

- **Fundamentals (default): SEC EDGAR XBRL** (`data.sec.gov`). Full multi-decade history, genuinely
  point-in-time via each fact's SEC `filed` date, for every US filer — no key, no whitelist. Flow
  items are reconstructed into clean quarters (direct 3-month values, with the year-to-date ladder
  differenced to fill gaps such as cash-flow statements); EBITDA / FCF / total debt are derived from
  their components. FMP (`data.provider: fmp`) is supported as an alternative but its free tier is
  limited to ~5 quarters and a whitelist of names.
- **Prices:** `yfinance` (`auto_adjust=True`) — split-and-dividend-adjusted total-return closes,
  never raw closes.
- **Factor returns:** the Ken French Data Library, via `pandas_datareader`.
- **Universe:** current S&P 500 constituents (Wikipedia). Using today's membership list means the
  backtest has **survivorship bias on the constituent list** — a documented limitation (fundamental
  look-ahead is separately eliminated via filing dates).

## Running it

No API key needed — the default provider is SEC EDGAR.

```bash
pip install -e ".[dev]"

python run.py screen                    # latest ranked screen + concentration diagnostics
python run.py backtest --freq monthly   # IC, quintile spread, turnover
python run.py decompose                 # FF5+UMD alpha decomposition of the Q5-Q1 spread
python run.py export                    # write outputs/*.parquet for the dashboard

streamlit run dashboard/app.py          # browse the results  (pip install -e ".[dashboard]")
```

`--max-names N` bounds the universe (handy for a quick run); omit it for the full S&P 500.

Tests (fixtures only, no network — this is what CI runs):

```bash
pytest -q
```

## Results (real data)

Full S&P 500 universe, monthly 2015–2024, on real EDGAR fundamentals + yfinance total-return prices
(`python run.py screen`, then `backtest` / `decompose`):

| metric | value |
|---|---|
| Screen (latest) | 42 names, effective N ≈ 42 (HHI 0.024) — diversified across sectors |
| Sector-neutral normalization | **67% of names scored within their GICS industry group**, 30% sector, 3% cross-sectional |
| Rebalances | 118 (monthly) |
| Mean Information Coefficient | +0.004 (t ≈ 0.6) |
| Q5−Q1 spread | +20 bps/month |
| FF5+UMD alpha (annualized) | +2.3%, **Newey-West t ≈ 1.0** |
| Loadings | small negative SMB (large-cap tilt), mild positive HML |

The honest read: across the full liquid large-cap universe the equal-weighted V/Q/M composite shows a
small positive spread, but the alpha is **not statistically distinguishable from zero** after FF5+UMD
— the realistic finding that simple composites are largely arbitraged out of large caps, and exactly
what the decomposition exists to reveal. The genuine strengths on display are the point-in-time
discipline, the **sector-neutral normalization actually resolving most names within their industry
group**, and honest reporting rather than an overfit backtest. Natural next steps: point-in-time
index membership (removes survivorship bias) and inverse-vol sleeve weighting in the composite.

## Limitations

- **Survivorship on the constituent list** — the screen uses today's S&P 500 membership, not the
  historical roster at each rebalance date, so delisted/removed names are absent from the backtest.
  This is the main remaining bias; a future extension would source point-in-time index membership.
  (Fundamental look-ahead is *not* a problem here — it's eliminated via SEC filing dates.)
- **EDGAR coverage is best for recent years** — derived quantities (EBITDA, FCF, total debt) depend
  on XBRL tags that are sparser in older filings, so those factors carry more NaNs pre-2015. The
  quality gate handles missing values (a factor simply doesn't contribute for that name).
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
