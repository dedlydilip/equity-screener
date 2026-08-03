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

- **Fundamental look-ahead controlled.** Fundamentals are gated by their SEC **filing date** (the
  XBRL `filed` timestamp), *strictly before* the rebalance date — a Q1 number isn't usable until it
  was actually filed.
- **Point-in-time constituent membership (backtest).** Historical S&P 500 membership is reconstructed
  from Wikipedia's dated changes table, so each monthly rebalance only scores/holds names that were
  *actually* index members on that date — not today's list applied retroactively. This mitigates, but
  doesn't fully eliminate, survivorship bias: a removed/acquired constituent's data is only recoverable
  if it's still an SEC filer and yfinance still serves its pre-delisting prices, which isn't always
  true. One **residual** look-ahead remains and is disclosed, not hidden: GICS sector/industry
  classification is **today's**, applied to all history (no free source provides classification
  *as of* a historical date). See Limitations.
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
  pit_membership.py   point-in-time historical constituent reconstruction (backtest only)
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
  capm.py             per-security CAPM beta / alpha vs. the market (Newey-West)
  optimize.py         long-only mean-variance: max-Sharpe, min-variance, efficient frontier (SLSQP)
  dividend.py         dividend-income screen: trailing yield gated on payout sustainability
  db.py, report.py    DuckDB analytical store + parquet export for the dashboards
sql/                  DuckDB analytical queries (sector exposure, quintile returns, ...)
dashboard/app.py      Streamlit dashboard (primary) — 6 tabs (screen, backtest, decomposition,
                      portfolio/CAPM, multi-asset, dividend income)
tests/                pytest suite (fixtures only — no live API calls in CI)
run.py                CLI: screen | backtest | decompose | optimize | multiasset | dividend | export
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
a noisy median. The fallback split (% of names resolved at each level) is reported for the screen.

**Composite:** the three sleeve z-scores are combined into one score as the **weight-normalized mean
of the sleeves a name actually has** — a missing sleeve (e.g. momentum before 12 months of history
exists) is renormalized away rather than treated as a neutral 0, and a name with fewer than two
present sleeves is dropped rather than ranked on too little signal. Sleeves are blended with
**inverse-volatility weighting**: each sleeve's own trailing 12-month realized long-short return
volatility sets its weight (a steadier sleeve gets more say), capped at 50% with any excess
redistributed proportionally, and defaulting to equal weights during a 12-month no-look-ahead burn-in
(before a full trailing window of sleeve returns exists). Wired into the backtest itself, not just
unit-tested: at each rebalance date the sleeve weights are computed only from sleeve returns realized
*strictly before* that date (`normalize.inverse_vol_weights`).

**Validation:** Information Coefficient (Spearman), reported with **both an i.i.d. and a Newey-West
HAC-adjusted t-statistic** (IC is autocorrelated across overlapping rebalances, so HAC is the
defensible figure), plus decay at 1/3/5-period lags; a quintile backtest (Q1..Q5 forward returns,
weighted by the configured scheme — equal / market-cap / inverse-vol) with the Q5−Q1 spread's
**gross and net-of-cost Sharpe**, **Max Drawdown**, and **per-rebalance and annualized turnover**;
and the **FF5 + UMD alpha decomposition** (Newey-West standard errors) — the headline research layer.
Transaction costs (commission + spread + short rebate, applied per-side and scaled by realized
turnover) are subtracted from the gross spread to get the net figures reported above — net Sharpe is
never allowed to exceed gross by construction. **Monthly and quarterly backtests are run and persisted
side by side** (a `freq` column on every table, so re-running one frequency never overwrites the
other's rows), and the sector-neutral fallback percentage is persisted per rebalance date, not just
printed once for the latest screen.

**Portfolio construction & CAPM:** each screened name gets a CAPM beta and alpha (regressing its
excess return on the market factor, Newey-West errors), shown on a Security Market Line. A long-only
mean-variance optimizer then produces the **maximum-Sharpe (tangency)** and **minimum-variance**
portfolios plus the **efficient frontier** — the tangency portfolio *is* the CAPM market portfolio,
so the betas and the optimized weights are two views of one model. This is an **in-sample,
descriptive** view: it optimizes over today's screened names using their full-history sample
mean/covariance, so the reported Sharpe is not an out-of-sample tradeable result (see Limitations).

**Multi-asset allocation:** the same optimizer runs across asset classes represented by liquid ETFs —
equities (SPY), bonds (AGG/TLT/IEF/LQD/HYG/TIP), and commodities (GLD/SLV/USO/DBC/DBA/CPER). Bonds and
commodities have no equity-style fundamentals and no free per-security data, so they enter here as
ETF proxies, never through the factor screen.

**Dividend income:** a separate screen ranking on trailing-12-month **recurring (regular)** dividend
yield but gated on **payout sustainability** (total dividends / earnings, from the same EDGAR data) —
a high yield funded by nearly all of earnings is a cut waiting to happen, so those are dropped. A
one-off **special/variable** payment (e.g. Progressive's large annual variable dividend on top of its
small regular one) is separated from the regular yield via a documented heuristic (a payment > 2x the
trailing median is "special"), so the headline yield reflects recurring income, not a one-time payout —
the total (incl. specials) is still reported alongside for context.

## Roadmap (not yet built)

Being explicit so the claims above are not overstated:

- **SQL as a first-class analytical layer** — `sql/queries.sql` exists but isn't yet executed by the
  pipeline or surfaced as a dashboard tab.
- **A reproducibility manifest** (git revision, config hash, data-snapshot dates, provider versions)
  written alongside every run's outputs.
- **An analysis notebook** (`notebooks/analysis.ipynb`) walking through the FF decomposition and
  factor-correlation heatmap outside the dashboard.
- **A built Power BI `.pbix`** — the data layer is exported and ready; the file itself needs Power BI
  Desktop, which this environment doesn't have.

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
- **Universe:** current S&P 500 constituents (Wikipedia) for the live screen; the **backtest**
  reconstructs point-in-time historical membership from Wikipedia's dated changes table
  (`screener/pit_membership.py`) so each rebalance only scores names that were actually index members
  on that date — this mitigates, but doesn't fully eliminate, survivorship bias (see Limitations).

## Running it

No API key needed — the default provider is SEC EDGAR.

```bash
pip install -e ".[dev]"

python run.py screen                    # latest ranked screen + concentration diagnostics
python run.py backtest                  # IC, quintile spread, costs/turnover — every configured freq
python run.py backtest --freq monthly   # (or restrict to one frequency)
python run.py decompose --freq monthly  # FF5+UMD alpha decomposition of the Q5-Q1 spread (per freq)
python run.py optimize                  # CAPM betas + mean-variance portfolio over the screen
python run.py multiasset                # cross-asset (equity/bond/commodity ETF) allocation
python run.py dividend                  # dividend-income screen (yield + payout sustainability)
python run.py export                    # write outputs/*.parquet for the dashboard

streamlit run dashboard/app.py          # browse the results  (pip install -e ".[dashboard]")
```

`--max-names N` bounds the universe (handy for a quick run); omit it for the full S&P 500.

Tests (fixtures only, no network — this is what CI runs):

```bash
pytest -q
```

## Results (real data)

Full S&P 500 universe (point-in-time membership: 503 current + 242 historically-removed/renamed
names pulled for the window), **2015-01 → 2024-12**, on real EDGAR fundamentals + yfinance
total-return prices, both rebalance frequencies persisted side by side in the same run
(`python run.py screen`, then `backtest` / `decompose`, no `--freq`):

| metric | monthly (118 rebalances) | quarterly (38 rebalances) |
|---|---|---|
| Mean IC (HAC t-stat) | +0.0067 (t = 0.84) | +0.0120 (t = 0.89) |
| Q5−Q1 spread, gross / net of costs | +0.15% / **−0.01%** per period | +0.57% / **+0.27%** per period |
| Sharpe, gross / net of costs | 0.31 / **−0.03** | 0.39 / **0.18** |
| Max Drawdown (Q5−Q1) | −13.5% | −6.7% |
| Turnover, per-rebalance / annualized | 40.0% / **480%/yr** | 75.2% / **301%/yr** |
| FF5+UMD alpha (annualized, NW t) | +1.9% (t = 0.92) | +2.8% (t = 1.54) |

Net figures subtract commission + spread + short-rebate costs (40 bps/side, `config.yaml`) scaled by
the realized turnover shown alongside.

Latest screen: 41 names, effective N ≈ 41 (HHI 0.024, top-10% weight 9.8%) — diversified across
sectors. Sector-neutral normalization resolved **67–68% of value/quality scores and 67% of momentum
scores within their GICS industry group**, ~28–33% at the sector level, and under 4% at the full
cross-section — the hierarchy is doing its job on the full universe rather than falling back to a
noisy broad median.

**The honest read — this is the finding the pipeline exists to surface, not paper over:** neither
frequency's signal is statistically distinguishable from zero (HAC IC t-stats of 0.84 and 0.89; FF5+UMD
alpha NW t-stats of 0.92 and 1.54 are all well under the ~2.0 conventional bar). And **turnover is
what actually decides whether the weak edge is investable**: at monthly rebalancing the composite
churns 480%/year, which is expensive enough that transaction costs erase the entire gross spread and
flip net Sharpe negative (0.31 → −0.03). At quarterly rebalancing turnover drops to 301%/year and
roughly half the gross edge survives net of costs (0.39 → 0.18), with a shallower drawdown too
(−6.7% vs. −13.5%). A simple equal/inverse-vol-weighted Value/Quality/Momentum composite over the full
liquid large-cap S&P 500 has a real but weak, cost-sensitive edge that is not tradeable at monthly
frequency and only marginally so at quarterly — a realistic result, not an overfit backtest.

## Limitations

- **Survivorship bias is mitigated, not eliminated.** The backtest reconstructs historical S&P 500
  membership from Wikipedia's changes table (406 change events, 1976–present), so each rebalance
  correctly excludes not-yet-added names and includes names not-yet-removed. The residual gap: a
  removed/acquired constituent's fundamentals are only recoverable if it's still an SEC-registered
  filer (an acquirer that deregistered its target stops appearing in EDGAR's filer list entirely —
  in a spot-check of 11 historically-removed names, 4 were still recoverable, 7 were not, e.g. Aetna
  and Alexion post-acquisition), and its price history is only available if yfinance still serves data
  up to its delisting date, which it often doesn't. So some historically-real constituents are still
  silently absent — an improvement over using today's list, not a complete fix.
- **Classification look-ahead (permanent, not fixable for free)** — GICS sector/industry-group labels
  are today's, applied to all history. No free source maps a company to its GICS sector *as of* a
  historical date (that mapping is proprietary S&P/Capital IQ or CRSP/Compustat `HGIC` data); only the
  classification *taxonomy's* evolution is publicly documented, not the company-level assignments. This
  is disclosed, not a work-in-progress. (Fundamental look-ahead *is* controlled — SEC filing dates,
  strictly before the trade date — and restatements deliberately use the as-originally-filed figure,
  never a later correction, avoiding any hindsight leakage; see `data/edgar.py`.)
- **Market cap uses point-in-time shares outstanding** (EDGAR's `dei:EntityCommonStockSharesOutstanding`
  cover-page fact, matched to each filing's date), falling back to the weighted-average diluted share
  count only when that dated figure is unavailable.
- **The CAPM/optimizer results are in-sample and descriptive** — full-history sample mean/covariance
  over today's screened names; not an out-of-sample tradeable strategy.
- **EDGAR coverage is best for recent years** — derived quantities (EBITDA, FCF, total debt) depend
  on XBRL tags that are sparser in older filings, so those factors carry more NaNs pre-2015. The
  quality gate handles missing values (a factor simply doesn't contribute for that name).
- **MAD can be noisy in small groups** — mitigated by the finest-first fallback hierarchy with a
  reported fallback percentage, but not eliminated.
- **Transaction costs are modeled, not simulated** — commission, spread, and short-rebate are
  applied as flat per-side bps, not a market-impact model.
- **Mean-variance is estimation-error sensitive** — the optimizer uses plain sample means and
  covariances, so it produces concentrated, sometimes corner solutions (e.g. the multi-asset run
  can allocate ~0% to bonds when their sample risk-adjusted return over the window is weak). That is
  a well-known property of unconstrained Markowitz, not a bug; robust/shrinkage covariance (e.g.
  Ledoit-Wolf) and per-asset-class weight floors are the natural next step.
- **Bonds & commodities are ETF proxies, not individual securities** — an asset-allocation view, not
  security-level fixed-income or commodity analytics (which need paid data feeds). The CAPM betas
  for them are descriptive.
- **CAPM betas are single-factor and descriptive** — useful for the SML view and as optimizer inputs,
  not a standalone trading signal.

## Author

Built by Dilip Amaranarayana as a portfolio project demonstrating factor-investing methodology,
point-in-time data discipline, portfolio construction (CAPM / mean-variance), and full-stack
analytics engineering (Python, SQL/DuckDB, Streamlit/Power BI).
