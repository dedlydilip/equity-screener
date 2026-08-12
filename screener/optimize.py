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


def _aligned(mean: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Align ``mean`` to ``cov`` BY LABEL and validate the covariance.

    ``.values`` was previously taken from each independently, so a ``mean``
    ordered differently from ``cov`` silently produced wrong weights carrying
    correct-looking labels.
    """
    if not mean.index.equals(cov.index):
        missing = mean.index.difference(cov.index)
        if len(missing):
            raise ValueError(f"mean has assets absent from cov: {list(missing)[:5]}")
        mean = mean.reindex(cov.index)
    if not cov.index.equals(cov.columns):
        raise ValueError("cov must be square with matching index and columns")
    sigma = cov.values
    variances = np.diag(sigma)
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(mean.values)):
        raise ValueError("mean/cov contain non-finite values")
    if np.any(variances <= 0):
        bad = [str(a) for a, v in zip(cov.index, variances, strict=True) if v <= 0]
        raise ValueError(f"assets with non-positive variance cannot be optimized: {bad[:5]}")
    return mean.values, sigma, cov.index


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Long-only minimum-variance weights."""
    _, sigma, idx = _aligned(pd.Series(0.0, index=cov.index), cov)
    w = _solve(lambda x: x @ sigma @ x, len(idx))
    return pd.Series(w, index=idx)


def max_sharpe_weights(mean: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Long-only maximum-Sharpe (tangency) weights — the CAPM 'market portfolio'.

    Raises when no asset's expected return clears ``rf``: maximizing
    ``(r - rf)/vol`` over an all-negative numerator inverts the objective and
    selects the HIGHEST-volatility asset (it minimizes the magnitude of a
    negative Sharpe), which silently returns the worst portfolio as the "best".
    """
    mu, sigma, idx = _aligned(mean, cov)
    if not np.any(mu > rf):
        raise ValueError(
            f"no asset's expected return exceeds rf={rf:.4f}; a max-Sharpe portfolio "
            "is not defined here (cash dominates every long-only mix)"
        )

    def neg_sharpe(w):
        ret = w @ mu - rf
        vol = np.sqrt(w @ sigma @ w)
        return -ret / vol

    w = _solve(neg_sharpe, len(idx))
    return pd.Series(w, index=idx)


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

    Each point records the return the solved portfolio ACHIEVES, not the target
    it was asked for: the weights are clipped and renormalized after solving, so
    the two differ and plotting the requested target against the realized
    volatility would mislabel the curve.
    """
    mu, sigma, idx = _aligned(mean, cov)
    n = len(idx)
    targets = np.linspace(float(np.min(mu)), float(np.max(mu)), n_points)
    bounds = [(0.0, 1.0)] * n
    rows, failed = [], 0
    for t in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=t: w @ mu - t},
        ]
        res = minimize(lambda w: w @ sigma @ w, np.full(n, 1.0 / n), method="SLSQP",
                        bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            failed += 1
            continue
        w = np.clip(res.x, 0.0, None)
        w = w / w.sum() if w.sum() else w
        rows.append({"target_return": float(w @ mu),   # ACHIEVED, not requested
                     "volatility": float(np.sqrt(w @ sigma @ w))})
    if failed:
        print(f"[optimize] efficient_frontier: {failed}/{n_points} target returns "
              f"did not converge and were dropped")
    if not rows:
        raise RuntimeError(
            f"efficient_frontier: all {n_points} target returns failed to converge"
        )
    curve = pd.DataFrame(rows).sort_values("target_return").reset_index(drop=True)
    gmv_idx = int(curve["volatility"].idxmin())
    return curve.iloc[gmv_idx:].reset_index(drop=True)
