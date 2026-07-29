"""Raw factor construction (value / quality / momentum) as of a rebalance date.

Point-in-time: only fundamentals whose ``filing_date <= as_of`` are used; flow
items are TTM (sum of the last four such quarters), stock items are the latest.
Ratios combine those with the price as of the date; momentum is the 12-1 total
return (skip the most recent month). Every factor is oriented so **higher =
better**, so they can be z-scored and blended directly.

Negative-denominator ratios are NaN'd (a distressed firm must not screen as
"cheap"): B/P dropped when book < 0, EV/EBITDA dropped when EBITDA < 0; negative
FCF yield is kept (it correctly sorts to the bottom). Each factor is winsorized
cross-sectionally to tame extremes before normalization.
"""

from __future__ import annotations

import pandas as pd

from .data.quality import mask_negative_denominator, winsorize_cross_sectional

FLOW = ["revenue", "gross_profit", "operating_income", "pretax_income", "tax_expense",
        "net_income", "ebitda", "free_cash_flow"]
STOCK = ["book_equity", "total_debt", "cash", "total_assets", "shares_diluted"]
FACTORS = ["ep", "bp", "fcf_yield", "ev_ebitda_inv", "roe", "roic", "gross_profitability",
           "de_inv", "momentum"]
SLEEVES = {
    "value": ["ep", "bp", "fcf_yield", "ev_ebitda_inv"],
    "quality": ["roe", "roic", "gross_profitability", "de_inv"],
    "momentum": ["momentum"],
}


def pit_fundamentals(fund: pd.DataFrame, as_of) -> pd.DataFrame:
    """Per ticker: TTM sums of flow items + latest stock items, using only filings
    known by ``as_of`` (filing_date <= as_of)."""
    as_of = pd.Timestamp(as_of)
    known = fund[fund["filing_date"] <= as_of].sort_values(["ticker", "period_end"])
    out = {}
    for tkr, g in known.groupby("ticker"):
        last4 = g.tail(4)
        rec = {f"{c}_ttm": last4[c].sum(min_count=1) for c in FLOW if c in g}
        latest = g.iloc[-1]
        rec.update({c: latest[c] for c in STOCK if c in g})
        out[tkr] = rec
    return pd.DataFrame.from_dict(out, orient="index")


def price_asof(prices: pd.DataFrame, as_of) -> pd.Series:
    """Last available close on/before ``as_of`` for each ticker."""
    px = prices[prices.index <= pd.Timestamp(as_of)]
    return px.iloc[-1] if len(px) else pd.Series(dtype=float)


def momentum_12_1(prices: pd.DataFrame, as_of) -> pd.Series:
    """12-1 month total return: price at ~1m ago / price at ~13m ago - 1."""
    as_of = pd.Timestamp(as_of)
    recent = price_asof(prices, as_of - pd.DateOffset(months=1))
    old = price_asof(prices, as_of - pd.DateOffset(months=13))
    if recent.empty or old.empty:
        return pd.Series(dtype=float)
    return recent / old - 1.0


def compute_factors(fund: pd.DataFrame, prices: pd.DataFrame, as_of, gates) -> pd.DataFrame:
    """Cross-sectional raw factors (ticker x factor) as of ``as_of``."""
    pit = pit_fundamentals(fund, as_of)
    price = price_asof(prices, as_of)
    tickers = pit.index.intersection(price.index)
    pit = pit.loc[tickers]
    price = price.loc[tickers].astype(float)

    g = pit.reindex(columns=[f"{c}_ttm" for c in FLOW] + STOCK)
    ni, ebitda, gp = g["net_income_ttm"], g["ebitda_ttm"], g["gross_profit_ttm"]
    fcf, opinc = g["free_cash_flow_ttm"], g["operating_income_ttm"]
    pretax, tax = g["pretax_income_ttm"], g["tax_expense_ttm"]
    book, debt, cash = g["book_equity"], g["total_debt"], g["cash"]
    assets, shares = g["total_assets"], g["shares_diluted"]

    mcap = price * shares
    etr = (tax / pretax).where(pretax > 0).clip(0, 0.5).fillna(0.21)
    ev = mcap + debt - cash
    ic = debt + book - cash

    f = pd.DataFrame(index=tickers)
    # --- Value (higher = cheaper = better) ---
    f["ep"] = ni / mcap
    bp = book / mcap
    f["bp"] = mask_negative_denominator(bp, book) if gates.drop_negative_book_bp else bp
    f["fcf_yield"] = fcf / mcap
    ev_ebitda_inv = (ebitda / ev).where(ev > 0)
    if gates.drop_negative_ebitda_ev:
        ev_ebitda_inv = ev_ebitda_inv.where(ebitda >= 0)
    f["ev_ebitda_inv"] = ev_ebitda_inv
    # --- Quality (higher = better) ---
    f["roe"] = (ni / book).where(book > 0)
    f["roic"] = (opinc * (1 - etr) / ic).where(ic > 0)
    f["gross_profitability"] = (gp / assets).where(assets > 0)
    f["de_inv"] = -(debt / book).where(book > 0)   # lower leverage -> higher (better)
    # --- Momentum ---
    f["momentum"] = momentum_12_1(prices, as_of).reindex(tickers)

    # Quality gate: positive earnings + P/E sanity bounds on E/P
    pe = (mcap / ni).where(ni > 0)
    lo, hi = gates.pe_bounds
    f["ep"] = f["ep"].where((pe > lo) & (pe < hi))

    # Winsorize each factor cross-sectionally (tame extremes pre-normalization)
    for c in FACTORS:
        f[c] = winsorize_cross_sectional(f[c], gates.xs_mad_flag)

    f["market_cap"] = mcap
    return f
