"""Long-only mean-variance portfolio optimization (Markowitz).

Solved with SLSQP (``scipy.optimize``, already a project dependency) — a
long-only, budget-constrained mean-variance problem is a small, well-behaved
quadratic program, no need for a dedicated QP solver.

Two canonical points on the efficient frontier:
  * **Minimum variance** — the least-risk portfolio, ignoring expected return.
  * **Maximum Sharpe (tangency)** — where the Capital Market Line touches the
    frontier. This *is* the CAPM "market portfolio": the theory says every
    rational investor holds some mix of the risk-free asset and this portfolio.

``mean``/``cov`` are expected to be in the same units (both annualized, or both
monthly) — callers (``run.py``) annualize before calling in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve(objective, n: int) -> np.ndarray:
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"optimizer failed to converge: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Long-only minimum-variance weights."""
    sigma = cov.values
    w = _solve(lambda x: x @ sigma @ x, len(cov))
    return pd.Series(w, index=cov.index)


def max_sharpe_weights(mean: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Long-only maximum-Sharpe (tangency) weights — the CAPM 'market portfolio'."""
    mu, sigma = mean.values, cov.values

    def neg_sharpe(w):
        ret = w @ mu - rf
        vol = np.sqrt(max(w @ sigma @ w, 1e-12))
        return -ret / vol

    w = _solve(neg_sharpe, len(mean))
    return pd.Series(w, index=mean.index)


def portfolio_stats(w: pd.Series, mean: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> dict:
    """Expected return, volatility, and Sharpe ratio of a weight vector."""
    ret = float(w @ mean)
    vol = float(np.sqrt(w @ cov @ w))
    return {"expected_return": ret, "volatility": vol,
            "sharpe": (ret - rf) / vol if vol else float("nan")}


def efficient_frontier(mean: pd.Series, cov: pd.DataFrame, n_points: int = 20) -> pd.DataFrame:
    """Long-only efficient frontier: minimum variance at each of a grid of target returns.

    The Markowitz "bullet" has two branches meeting at the global-minimum-variance
    (GMV) point: an inefficient lower branch (risk falls as target return rises)
    and the efficient upper branch (risk rises with target return). Only the
    upper branch is the *efficient* frontier — a point on the lower branch is
    always dominated by the GMV point (same or less risk, more return) — so
    everything below the GMV point's return is dropped before returning.
    """
    mu, sigma = mean.values, cov.values
    n = len(mean)
    targets = np.linspace(float(mean.min()), float(mean.max()), n_points)
    bounds = [(0.0, 1.0)] * n
    rows = []
    for t in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=t: w @ mu - t},
        ]
        res = minimize(lambda w: w @ sigma @ w, np.full(n, 1.0 / n), method="SLSQP",
                        bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            continue
        w = np.clip(res.x, 0.0, None)
        w = w / w.sum() if w.sum() else w
        rows.append({"target_return": t, "volatility": float(np.sqrt(w @ sigma @ w))})
    curve = pd.DataFrame(rows).sort_values("target_return").reset_index(drop=True)
    if curve.empty:
        return curve
    gmv_idx = curve["volatility"].idxmin()
    return curve.iloc[gmv_idx:].reset_index(drop=True)
