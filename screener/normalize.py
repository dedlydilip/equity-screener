"""Cross-sectional normalization + factor-sleeve weighting (the institutional core).

Cross-sectional (per rebalance date):
  * MAD-robust z-score within a group: (x - median) / (1.4826 * MAD), clipped ±clip.
  * Grouping uses a finest-first fallback HIERARCHY:
    industry_group -> sector -> cross_sectional. A name is z-scored at the finest
    level whose group has >= min_n members; otherwise it rolls UP to a broader,
    larger-n level. The % of names resolved at each level is reported.

Sleeve weighting (across time, for the backtest):
  * inverse-volatility from each sleeve's own trailing RETURN series (not
    cross-sectional dispersion), with a max_weight cap (excess redistributed) and
    a burn-in during which weights are EQUAL — using a trailing window that
    doesn't exist yet would be look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # scales MAD to be a consistent sigma estimate under normality


def mad_z(sample: pd.Series, targets: pd.Series | None = None) -> pd.Series | None:
    """Robust z of ``targets`` against the median/MAD of ``sample``.

    Returns ``None`` when the MAD is zero or non-finite, i.e. the sample carries
    no dispersion to standardize against -- callers treat that as "this group
    can't score these names" and fall back rather than dividing by zero.

    Single source of truth for the MAD z-score: :func:`robust_z_in_group` and
    :func:`sector_neutral_z` both delegate here, so the arithmetic that ships is
    the arithmetic that is unit-tested. They previously carried two independent
    copies of this formula and only the one NOT used by the pipeline was tested.
    """
    if targets is None:
        targets = sample
    med = sample.median()
    m = (sample - med).abs().median()
    if not np.isfinite(m) or m == 0:
        return None
    return (targets - med) / (MAD_SCALE * m)


def robust_z_in_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    """MAD z-score of ``values`` within each label of ``groups``."""
    values = values.astype(float)
    out = pd.Series(np.nan, index=values.index)
    for _g, idx in groups.groupby(groups).groups.items():
        z = mad_z(values.loc[idx])
        if z is None:
            continue
        out.loc[idx] = z
    return out


def sector_neutral_z(
    values: pd.Series, universe: pd.DataFrame, hierarchy: list[dict], clip: float
) -> tuple[pd.Series, dict]:
    """Robust z with the finest-first fallback hierarchy.

    ``universe`` is indexed by ticker with columns matching the non-terminal
    hierarchy levels (e.g. ``industry_group``, ``sector``). ``hierarchy`` is a
    list of ``{"level": name, "min_n": k}`` ending in ``cross_sectional``.
    Returns ``(z_scores, {level: pct_resolved})``.
    """
    values = values.astype(float)
    valid = values.dropna()
    z = pd.Series(np.nan, index=values.index)
    resolved = pd.Series(index=values.index, dtype=object)
    remaining: set = set(valid.index)

    for level in hierarchy:
        name, min_n = level["level"], level.get("min_n", 0)
        if not remaining:
            break
        if name == "cross_sectional":
            labels = pd.Series("ALL", index=valid.index)
        else:
            labels = universe.reindex(valid.index)[name]
        # Group stats use the FULL group membership (not just the unresolved
        # names); we only ASSIGN z to names still unresolved. A group smaller
        # than min_n defers to a broader level.
        for _g, members in labels.groupby(labels).groups.items():
            members = list(members)
            if name != "cross_sectional" and len(members) < min_n:
                continue
            targets = [t for t in members if t in remaining]
            if not targets:
                continue
            # Stats from the FULL group, assigned only to still-unresolved names.
            zz = mad_z(valid.loc[members], valid.loc[targets])
            if zz is None:
                continue
            z.loc[targets] = zz
            resolved.loc[targets] = name
            remaining.difference_update(targets)

    z = z.clip(lower=-clip, upper=clip)
    total = int(resolved.notna().sum())
    pct = {
        h["level"]: (round(100 * int((resolved == h["level"]).sum()) / total, 1) if total else 0.0)
        for h in hierarchy
    }
    return z, pct


def composite(
    z_by_sleeve: dict[str, pd.Series], weights: dict[str, float], min_sleeves: int = 2
) -> pd.Series:
    """Weighted mean of the sleeve z-scores a name actually has.

    A missing sleeve (NaN z — e.g. momentum before 12 months of history exists) is
    NOT treated as a neutral 0: the weights are renormalized across the sleeves that
    are present, so an under-covered name is neither silently biased toward the
    median nor given an artificial score. A name with fewer than ``min_sleeves``
    present sleeves is left NaN (dropped) rather than ranked on too little signal.
    """
    idx = next(iter(z_by_sleeve.values())).index
    num = pd.Series(0.0, index=idx)
    wsum = pd.Series(0.0, index=idx)
    count = pd.Series(0, index=idx)
    for sleeve, z in z_by_sleeve.items():
        w = weights[sleeve]
        zz = z.reindex(idx)
        present = zz.notna()
        num = num.add((w * zz).where(present, 0.0), fill_value=0.0)
        wsum = wsum.add(present.astype(float) * w, fill_value=0.0)
        count = count.add(present.astype(int), fill_value=0)
    out = num / wsum.where(wsum > 0)
    return out.where(count >= min_sleeves)


def _cap_and_redistribute(w: pd.Series, cap: float) -> pd.Series:
    """Cap weights at ``cap`` and redistribute the excess proportionally.

    Raises if the cap is infeasible (``cap * n < 1``): weights must sum to 1, so
    no allocation can respect a cap that low. Previously the redistribution loop
    ran out of headroom, broke, and the final renormalization pushed every
    weight back ABOVE the cap -- silently returning an uncapped blend that
    looked capped.
    """
    w = w.astype(float).copy()
    n = len(w)
    if n and cap * n < 1.0 - 1e-12:
        raise ValueError(
            f"max_sleeve_weight={cap} is infeasible for {n} sleeves: "
            f"{n} x {cap} = {n * cap:.4f} < 1. Use a cap of at least {1.0 / n:.4f}."
        )
    for _ in range(100):
        over = w[w > cap + 1e-12]
        if over.empty:
            break
        excess = float((over - cap).sum())
        w[over.index] = cap
        under = w[w < cap - 1e-12]
        if under.empty:
            break
        w[under.index] = under + excess * under / under.sum()
    return w / w.sum()


def inverse_vol_weights(
    factor_returns: pd.DataFrame, lookback: int, max_weight: float, burn_in: int, as_of_row: int
) -> dict[str, float]:
    """Sleeve weights for the rebalance at index position ``as_of_row``.

    * If fewer than ``burn_in`` prior rows exist -> EQUAL weights (no look-ahead).
    * Else weight_i ~ 1 / vol_i over the trailing ``lookback`` rows, capped at
      ``max_weight`` with the excess redistributed.

    Volatilities are floored before inverting, at half the smallest positive
    volatility observed in the window. That keeps ``1/vol`` from exploding on a
    (near-)zero-vol sleeve while still letting the lowest-risk sleeve rank
    highest -- ``max_weight`` is what actually bounds the concentration. Without
    the floor a zero-vol sleeve was handed the cross-sleeve MEAN inverse-vol,
    ranking it below a measured higher-volatility sleeve: the inverse of intent.

    A sleeve with no measurable volatility at all (an all-NaN window) is
    genuinely unknown rather than riskless, so it takes the neutral median.
    """
    sleeves = list(factor_returns.columns)
    n = len(sleeves)
    if not n:
        return {}
    window = factor_returns.iloc[max(0, as_of_row - lookback):as_of_row]
    # An empty window carries no information (e.g. as_of_row == 0 with burn_in
    # set to 0) -- fall back to equal weights instead of emitting NaN, which
    # would propagate through composite() and void the whole cross-section.
    if as_of_row < burn_in or window.empty:
        return {s: 1.0 / n for s in sleeves}
    vol = window.std()
    measurable = vol[np.isfinite(vol)]
    positive = measurable[measurable > 0]
    if positive.empty:
        return {s: 1.0 / n for s in sleeves}   # nothing measurable to rank on
    floor = max(float(positive.min()) * 0.5, 1e-12)
    inv = 1.0 / measurable.clip(lower=floor)
    inv = inv.reindex(sleeves)
    inv = inv.fillna(float(inv.median()))      # unmeasurable -> neutral
    w = _cap_and_redistribute(inv / inv.sum(), max_weight)
    return {s: float(w[s]) for s in sleeves}
