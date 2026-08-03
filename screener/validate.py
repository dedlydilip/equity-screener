"""Validation — the factor research engine layer.

From a panel of cross-sectional composite scores and the forward returns that
follow each rebalance:
  * Information Coefficient (Spearman) with IC-IR and a t-stat, plus IC decay at
    forward lags.
  * A quintile backtest: Q1..Q5 forward returns, the Q5-Q1 spread, Sharpe
    (excess over the French RF for long-only legs; the self-financing L/S uses
    the raw spread), and the Max Drawdown of Q5-Q1.
  * Turnover (per-rebalance and annualized) and net-of-cost returns.
  * A Fama-French 5-factor + momentum decomposition of the Q5-Q1 spread: the
    spread is regressed **RAW** (self-financing -> Rf is NOT subtracted); single
    legs are regressed as excess (R - Rf). Reports alpha (Newey-West t),
    factor loadings, and R^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .portfolio import weights as _portfolio_weights

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

FF_COLS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"]


def information_coefficient(scores: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Per-date Spearman rank correlation between score and next-period return."""
    ics = {}
    for dt in scores.index:
        if dt not in fwd_returns.index:
            continue
        pair = pd.concat([scores.loc[dt], fwd_returns.loc[dt]], axis=1).dropna()
        if len(pair) < 5:
            continue
        ics[dt] = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).correlation
    return pd.Series(ics).dropna()


def ic_summary(ic: pd.Series, nw_lags: int = 6) -> dict:
    """Mean IC, IC-IR, and BOTH an i.i.d. t-stat (``t_stat``, kept for reference)
    and a Newey-West HAC-adjusted t-stat (``t_stat_hac``) for the mean IC being
    non-zero. IC series are autocorrelated (overlapping signal information
    across nearby rebalances), so the i.i.d. ``IR * sqrt(n)`` formula understates
    the true standard error; ``t_stat_hac`` is the more defensible figure."""
    n = len(ic)
    mean, std = ic.mean(), ic.std(ddof=1)
    ir = mean / std if std else np.nan
    t_hac = np.nan
    if sm is not None and n > nw_lags + 1:
        X = np.ones((n, 1))
        res = sm.OLS(ic.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        t_hac = float(res.tvalues[0])
    return {"mean_ic": float(mean), "ic_ir": float(ir),
            "t_stat": float(ir * np.sqrt(n)) if n else np.nan,
            "t_stat_hac": t_hac, "n": int(n)}


def ic_decay(scores: pd.DataFrame, returns_panel: pd.DataFrame, lags=(1, 3, 5)) -> dict:
    """Mean IC of score_t vs the return realised `lag` periods later (factor lifespan)."""
    dates = list(returns_panel.index)
    pos = {d: i for i, d in enumerate(dates)}
    out = {}
    for lag in lags:
        vals = []
        for dt in scores.index:
            i = pos.get(dt)
            if i is None or i + lag >= len(dates):
                continue
            pair = pd.concat([scores.loc[dt], returns_panel.iloc[i + lag]], axis=1).dropna()
            if len(pair) < 5:
                continue
            vals.append(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).correlation)
        out[f"lag_{lag}"] = float(np.nanmean(vals)) if vals else np.nan
    return out


def quintile_returns(scores: pd.DataFrame, fwd_returns: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Equal-weighted forward return of each score quintile, per date; + Q5-Q1 spread."""
    rows = {}
    for dt in scores.index:
        if dt not in fwd_returns.index:
            continue
        s = scores.loc[dt].dropna()
        if len(s) < n * 2:
            continue
        q = pd.qcut(s.rank(method="first"), n, labels=list(range(1, n + 1)))
        rows[dt] = fwd_returns.loc[dt].reindex(s.index).groupby(q, observed=True).mean()
    df = pd.DataFrame(rows).T
    df.columns = [f"Q{int(c)}" for c in df.columns]
    if f"Q{n}" in df and "Q1" in df:
        df["spread"] = df[f"Q{n}"] - df["Q1"]
    return df


def weighted_quintile_returns(
    scores: pd.DataFrame, fwd_returns: pd.DataFrame, method: str = "equal",
    market_cap: pd.DataFrame | None = None, vols: pd.DataFrame | None = None, n: int = 5,
) -> pd.DataFrame:
    """Same as :func:`quintile_returns`, but each bucket's return is a WEIGHTED
    average (via :func:`screener.portfolio.weights` -- reused, not reimplemented)
    instead of a naive equal-weighted mean.

    ``market_cap``/``vols`` are optional (dates x tickers) panels required for
    ``method="market_cap"``/``"inverse_vol"`` respectively (unused, and may be
    ``None``, for ``method="equal"``).
    """
    rows = {}
    for dt in scores.index:
        if dt not in fwd_returns.index:
            continue
        s = scores.loc[dt].dropna()
        if len(s) < n * 2:
            continue
        q = pd.qcut(s.rank(method="first"), n, labels=list(range(1, n + 1)))
        fwd = fwd_returns.loc[dt].reindex(s.index)
        mc = market_cap.loc[dt] if market_cap is not None and dt in market_cap.index else None
        vv = vols.loc[dt] if vols is not None and dt in vols.index else None
        row = {}
        for bucket in range(1, n + 1):
            members = q[q == bucket].index
            w = _portfolio_weights(members, method=method, market_cap=mc, vols=vv)
            row[bucket] = float((fwd.reindex(members) * w.reindex(members)).sum())
        rows[dt] = pd.Series(row)
    df = pd.DataFrame(rows).T
    df.columns = [f"Q{int(c)}" for c in df.columns]
    if f"Q{n}" in df and "Q1" in df:
        df["spread"] = df[f"Q{n}"] - df["Q1"]
    return df


def sharpe(returns: pd.Series, rf: pd.Series | None = None, ppy: int = 12) -> float:
    r = returns.dropna()
    if rf is not None:
        r = (r - rf.reindex(r.index)).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd else np.nan


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns.dropna()).cumprod()
    if curve.empty:
        return np.nan
    return float((curve / curve.cummax() - 1).min())


def turnover(holdings: pd.DataFrame, ppy: int = 12) -> dict:
    """Two-way turnover from a holdings-weight panel (dates x names)."""
    h = holdings.fillna(0.0)
    one_way = (h.diff().abs().sum(axis=1) / 2.0).iloc[1:].mean()
    two_way = float(one_way * 2)
    return {"per_rebalance_two_way": two_way, "annualized_two_way": two_way * ppy}


def net_of_cost(
    returns: pd.Series, turnover_per_rebalance: float, cost_bps_per_side: float
) -> pd.Series:
    """Subtract round-trip cost (2 sides x one-way turnover x bps) each period."""
    cost = turnover_per_rebalance * (cost_bps_per_side / 1e4)  # two-way turnover -> both sides
    return returns - cost


def ff_decompose(ls_returns: pd.Series, ff: pd.DataFrame, is_long_short: bool = True,
                 nw_lags: int = 6, ppy: int = 12) -> dict:
    """Regress on FF5 + UMD. Long-short -> RAW spread (no Rf); single leg -> excess (R - Rf).

    ``ppy`` (periods per year) annualizes alpha — 12 for a monthly backtest
    (the default, matching every existing call site), 4 for quarterly.
    """
    if sm is None:
        raise RuntimeError("statsmodels is required for the FF decomposition")
    df = pd.concat([ls_returns.rename("y"), ff[FF_COLS + ["RF"]]], axis=1).dropna()
    y = df["y"] if is_long_short else df["y"] - df["RF"]
    X = sm.add_constant(df[FF_COLS])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return {
        "alpha_monthly": float(res.params["const"]),
        "alpha_annualized": float(res.params["const"] * ppy),
        "alpha_t_nw": float(res.tvalues["const"]),
        "loadings": {c: float(res.params[c]) for c in FF_COLS},
        "r_squared": float(res.rsquared),
        "n": int(res.nobs),
    }
