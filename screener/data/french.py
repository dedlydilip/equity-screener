"""Fama-French factor returns from the Ken French Data Library (free, no key).

Monthly FF5 (Mkt-RF, SMB, HML, RMW, CMA), the momentum factor (UMD), and the
risk-free rate RF. Used by validate.py for the alpha decomposition and for
excess-return Sharpe. French reports percent; we convert to decimals.
"""

from __future__ import annotations

import pandas as pd
import pandas_datareader.data as web


def tidy_ff(ff5: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    """Join, rename, rescale and re-index the two raw French tables.

    Split out from the network call so it is unit-testable: a missing or
    duplicated ``/100`` here would silently rescale every alpha and Sharpe in
    the project by a factor of 100, and a missed rename surfaces only much later
    as a ``KeyError`` on ``FF_COLS``.
    """
    df = ff5.join(mom)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Mkt-RF": "Mkt_RF", "Mom": "UMD"})
    missing = {"Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"} - set(df.columns)
    if missing:
        raise ValueError(f"French data is missing expected columns: {sorted(missing)}")
    df = df / 100.0  # percent -> decimal

    # famafrench returns a monthly PeriodIndex; make it a month-end timestamp.
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="end").normalize()
    df.index.name = "date"
    return df


def get_ff_factors(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Monthly factor returns indexed by month-end.

    Columns: Mkt_RF, SMB, HML, RMW, CMA, RF, UMD (all decimals).
    """
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start, end)[0]
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start, end)[0]
    return tidy_ff(ff5, mom)
