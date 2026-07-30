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
  was actually filed. Two **residual** look-aheads are known and disclosed, not hidden: the universe
  uses **current** S&P 500 membership and **current** GICS classification applied to all history
  (survivorship + classification bias). See Limitations.
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
present sleeves is dropped rather than ranked on too little signal. The current pipeline uses
**equal sleeve weights**; an inverse-volatility sleeve-weighting scheme (with a 12-month burn-in and a
50% cap) is implemented and unit-tested in `normalize.py` but is **not yet wired into the backtest**
(see Roadmap).

**Validation:** Information Coefficient (Spearman) with decay at 1/3/5-month lags; a quintile
backtest (Q1..Q5 forward returns and the Q5−Q1 spread; the dashboard shows the spread's Sharpe and
Max Drawdown); and the **FF5 + UMD alpha decomposition** (Newey-West standard errors) — the headline
research layer. Transaction costs, turnover, net-of-cost returns, a risk-free-rate Sharpe inside the
backtest, and a monthly-vs-quarterly comparison exist as functions/config but are **not yet wired
into `run.py`** (see Roadmap); today's IC t-stat is an i.i.d. approximation, not HAC.

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

**Dividend income:** a separate screen ranking on trailing-12-month dividend yield but gated on
**payout sustainability** (dividends / earnings, from the same EDGAR data) — a high yield funded by
nearly all of earnings is a cut waiting to happen, so those are dropped. Trailing yield currently
includes one-off **special/variable** dividends (e.g. Progressive's large annual variable payout),
which overstate recurring income; separating regular from special dividends is on the Roadmap.

## Roadmap (implemented as tested functions/config, but NOT yet wired into the pipeline)

Being explicit so the claims above are not overstated — these are built and unit-tested at the
function level, and wiring them into `run.py`/the backtest is in progress:

- **Inverse-volatility sleeve weighting + 12-month burn-in + 50% cap** (`normalize.inverse_vol_weights`).
- **Transaction costs, turnover, net-of-cost returns, risk-free-rate Sharpe, Max-Drawdown** reported
  and persisted *inside* the backtest (`validate.turnover/net_of_cost/sharpe/max_drawdown`).
- **Monthly-vs-quarterly** rebalance comparison in one run.
- **HAC (Newey-West) IC t-statistic**; per-date fallback-% persistence.
- **Dated shares-outstanding** for market cap (currently weighted-average diluted shares); strict
  four-quarter TTM; financial-sector-specific factor handling; effective-dated GICS/index membership.
- **SQL** surfaced in the dashboard; a reproducibility **manifest**; an analysis **notebook**; a
  built **Power BI `.pbix`**.

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

> **Being regenerated.** The figures below predate a round of correctness fixes (strictly-before
> filing gate, composite renormalization, dividend TTM window) and a set of data-correctness changes
> in progress; they will be re-run and re-published from a reproducibility manifest. Treat them as
> indicative, not final.

Full S&P 500 universe, monthly backtest **2015-01 → 2024-10 (118 rebalances)**, on real EDGAR
fundamentals + yfinance total-return prices (`python run.py screen`, then `backtest` / `decompose`):

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
- **Classification look-ahead** — GICS sector/industry-group labels are today's, applied to all
  history, so the sector-neutral normalization uses classifications a historical investor didn't have.
  Both this and survivorship need effective-dated index membership + classification (Roadmap); until
  then they are disclosed, not fixed. (Fundamental look-ahead *is* controlled — SEC filing dates,
  strictly before the trade date.)
- **Market cap uses weighted-average diluted shares** (the EDGAR income-statement share count), not
  point-in-time shares outstanding — a modest distortion to every price-based ratio, being replaced
  with dated `EntityCommonStockSharesOutstanding` (Roadmap).
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
