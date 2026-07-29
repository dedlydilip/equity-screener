"""CAPM: per-security beta and alpha vs. the market factor.

Each security's excess return (R - Rf) is regressed on the market excess return
(``Mkt_RF`` from the same Ken French library used by ``validate.ff_decompose``)
to get its **beta** (systematic-risk exposure — the CAPM's only priced risk) and
**alpha** (Newey-West t-stat) — the return CAPM leaves unexplained. Same
regression pattern as the FF decomposition, just the one-factor special case.

The mean-variance tangency portfolio computed in ``optimize.py`` from these
same securities *is* the CAPM "market portfolio" — the point where the Capital
Market Line touches the efficient frontier. Betas here and weights there are
two views of the same model, not two unrelated features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

_EMPTY = {"beta": np.nan, "alpha_monthly": np.nan, "alpha_annualized": np.nan,
          "alpha_t_nw": np.nan, "r_squared": np.nan, "n": 0}


def estimate_beta_alpha(returns: pd.Series, ff: pd.DataFrame, nw_lags: int = 6) -> dict:
    """CAPM regression of one security's excess return on the market excess return.

    ``ff`` must have ``Mkt_RF`` and ``RF`` columns (as returned by
    ``data.french.get_ff_factors``), indexed on the same dates as ``returns``.
    """
    if sm is None:
        raise RuntimeError("statsmodels is required for CAPM estimation")
    df = pd.concat([returns.rename("r"), ff[["Mkt_RF", "RF"]]], axis=1).dropna()
    if len(df) < 6:  # too few observations for a meaningful regression
        return dict(_EMPTY, n=len(df))
    y = df["r"] - df["RF"]
    X = sm.add_constant(df["Mkt_RF"])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return {
        "beta": float(res.params["Mkt_RF"]),
        "alpha_monthly": float(res.params["const"]),
        "alpha_annualized": float(res.params["const"] * 12),
        "alpha_t_nw": float(res.tvalues["const"]),
        "r_squared": float(res.rsquared),
        "n": int(res.nobs),
    }


def estimate_betas_panel(returns: pd.DataFrame, ff: pd.DataFrame, nw_lags: int = 6) -> pd.DataFrame:
    """CAPM beta/alpha/R^2 for every column (ticker) in ``returns``. Index = ticker."""
    rows = {t: estimate_beta_alpha(returns[t], ff, nw_lags) for t in returns.columns}
    return pd.DataFrame.from_dict(rows, orient="index")
