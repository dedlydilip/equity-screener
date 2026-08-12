"""Typed, validated configuration.

Loads ``config.yaml`` into Pydantic models so every run is *pre-registered* and
invalid settings fail fast at load time rather than silently corrupting results
(e.g. sleeve weights that don't sum to 1.0, or a normalization hierarchy that
never falls back to the cross-sectional bucket).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class UniverseConfig(BaseModel):
    # "fmp" was offered here but never implemented -- get_sp500 always reads
    # Wikipedia. Narrowed to what actually exists rather than advertising a
    # source that silently does nothing.
    name: Literal["sp500"] = "sp500"
    source: Literal["wikipedia"] = "wikipedia"


class DataConfig(BaseModel):
    provider: Literal["fmp", "edgar", "yfinance"] = "edgar"
    fmp_api_key_env: str = "FMP_API_KEY"
    edgar_contact: str = "adilip407@gmail.com"
    cache_dir: str = ".cache"
    # NOTE: `fundamentals_pit` was removed -- point-in-time gating by SEC filing
    # date is unconditional in factors.pit_fundamentals and cannot be turned
    # off, so a flag claiming otherwise was misleading.


class FactorConfig(BaseModel):
    """Descriptor weights within each sleeve.

    Keys are validated against the factor columns the pipeline actually
    computes: the momentum sleeve was configured as ``ret_12_1`` while the
    column is ``momentum``, so the weight could never have been applied to
    anything -- exactly the kind of silent config/code drift this catches.
    """

    value: dict[str, float]
    quality: dict[str, float]
    momentum: dict[str, float]

    @model_validator(mode="after")
    def _keys_match_pipeline_factors(self) -> FactorConfig:
        from .factors import SLEEVES  # local import: factors imports config-free helpers

        for sleeve, expected in SLEEVES.items():
            configured = set(getattr(self, sleeve))
            unknown = configured - set(expected)
            if unknown:
                raise ValueError(
                    f"factors.{sleeve} has descriptors the pipeline does not compute: "
                    f"{sorted(unknown)}; valid options are {sorted(expected)}"
                )
            if any(w < 0 for w in getattr(self, sleeve).values()):
                raise ValueError(f"factors.{sleeve} weights must be non-negative")
            if configured and sum(getattr(self, sleeve).values()) <= 0:
                raise ValueError(f"factors.{sleeve} weights must not sum to zero")
        return self


class SleeveWeights(BaseModel):
    value: float
    quality: float
    momentum: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> SleeveWeights:
        total = self.value + self.quality + self.momentum
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sleeve_weights must sum to 1.0, got {total:.6f}")
        return self


class WeightingConfig(BaseModel):
    method: Literal["inverse_vol", "equal", "custom"] = "inverse_vol"
    vol_lookback_months: int = Field(12, gt=0)
    burn_in_months: int = Field(12, ge=0)
    max_sleeve_weight: float = Field(0.50, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _cap_is_feasible(self) -> WeightingConfig:
        # Weights across the three sleeves must sum to 1, so a cap below 1/3
        # cannot be satisfied. Caught here rather than at runtime, where the
        # redistribution loop used to give up and renormalize every weight back
        # above the cap -- silently returning an uncapped blend.
        n_sleeves = 3
        if self.max_sleeve_weight * n_sleeves < 1.0 - 1e-9:
            raise ValueError(
                f"max_sleeve_weight={self.max_sleeve_weight} is infeasible for {n_sleeves} "
                f"sleeves ({n_sleeves} x {self.max_sleeve_weight:.4f} < 1); "
                f"use at least {1 / n_sleeves:.4f}"
            )
        return self


class HierarchyLevel(BaseModel):
    level: Literal["industry_group", "sector", "cross_sectional"]
    min_n: int = Field(20, gt=0)


class NormalizationConfig(BaseModel):
    method: Literal["mad"] = "mad"
    clip: float = Field(5.0, gt=0.0)
    hierarchy: list[HierarchyLevel]

    @field_validator("hierarchy")
    @classmethod
    def _ends_cross_sectional(cls, v: list[HierarchyLevel]) -> list[HierarchyLevel]:
        if not v or v[-1].level != "cross_sectional":
            raise ValueError("normalization.hierarchy must end with a cross_sectional level")
        return v


class QualityGates(BaseModel):
    # NOTE: `de_min` was removed -- it duplicated the `book_equity > 0` mask that
    # already NaNs D/E in factors.compute_factors, and nothing read it.
    pe_bounds: tuple[float, float] = (0.0, 500.0)
    xs_mad_flag: float = Field(10.0, gt=0.0)
    drop_negative_book_bp: bool = True
    drop_negative_ebitda_ev: bool = True

    @model_validator(mode="after")
    def _pe_bounds_ordered(self) -> QualityGates:
        lo, hi = self.pe_bounds
        if not lo < hi:
            raise ValueError(f"pe_bounds must be (low, high) with low < high, got {self.pe_bounds}")
        return self


class HardFilters(BaseModel):
    min_market_cap: float = 1.0e9
    positive_earnings: bool = True


class PortfolioConfig(BaseModel):
    weighting: Literal["equal", "market_cap", "inverse_vol"] = "equal"
    n_quantiles: int = Field(5, ge=2)
    select: Literal["top_decile", "top_n"] = "top_decile"
    top_n: int = Field(50, gt=0)
    hard_filters: HardFilters = Field(default_factory=HardFilters)


class BacktestConfig(BaseModel):
    # NOTE: `rebalance` was removed -- it was superseded by
    # `frequencies_to_test`, which is what `run.py backtest` actually loops over.
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    frequencies_to_test: list[Literal["monthly", "quarterly"]] = Field(
        default_factory=lambda: ["monthly", "quarterly"]
    )

    @model_validator(mode="after")
    def _dates_ordered_and_frequencies_unique(self) -> BacktestConfig:
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError(f"backtest.start ({self.start}) must precede end ({self.end})")
        if not self.frequencies_to_test:
            raise ValueError("backtest.frequencies_to_test must not be empty")
        if len(set(self.frequencies_to_test)) != len(self.frequencies_to_test):
            raise ValueError(
                f"duplicate entries in frequencies_to_test: {self.frequencies_to_test}"
            )
        return self


class CostsConfig(BaseModel):
    # Units are in the names deliberately. commission/spread are per-side costs
    # charged on each unit of turnover; the short rebate is a borrow rate quoted
    # PER ANNUM by market convention and accrued over the holding period. Leaving
    # that unstated is how it came to be charged per rebalance -- 12x too much at
    # monthly frequency, and the same class of units bug as the months-vs-rows
    # one in the sleeve-weighting window.
    commission_bps: float = Field(5.0, ge=0.0)          # per side, per unit traded
    spread_bps: float = Field(10.0, ge=0.0)             # per side, per unit traded
    short_rebate_bps_annual: float = Field(25.0, ge=0.0)  # per YEAR, on short notional


class OutputConfig(BaseModel):
    duckdb_path: str = "outputs/screener.duckdb"
    export_dir: str = "outputs"


class AssetClasses(BaseModel):
    """Cross-asset building blocks for the multi-asset optimizer.

    Bonds and commodities are represented by liquid, free-to-price ETFs — the
    equity factor engine (P/E, ROE, GICS z-scores) has no meaning for them, and
    per-security fixed-income/commodity data isn't available free, so they enter
    the tool only here, as assets in the mean-variance / CAPM allocation.
    """

    equity_proxy: str = "SPY"
    bond_etfs: list[str] = Field(default_factory=lambda: ["AGG", "TLT", "IEF", "LQD", "HYG", "TIP"])
    commodity_etfs: list[str] = Field(
        default_factory=lambda: ["GLD", "SLV", "USO", "DBC", "DBA", "CPER"]
    )

    def all_tickers(self) -> list[str]:
        return [self.equity_proxy, *self.bond_etfs, *self.commodity_etfs]


class DividendConfig(BaseModel):
    min_yield: float = Field(0.0, ge=0.0)          # keep names yielding at least this (decimal)
    max_payout: float = Field(0.90, gt=0.0)        # sustainability gate: payout ratio ceiling
    top_n: int = Field(30, gt=0)


class Config(BaseModel):
    universe: UniverseConfig
    data: DataConfig
    factors: FactorConfig
    sleeve_weights: SleeveWeights
    weighting: WeightingConfig
    normalization: NormalizationConfig
    quality_gates: QualityGates
    portfolio: PortfolioConfig
    backtest: BacktestConfig
    costs: CostsConfig
    output: OutputConfig
    asset_classes: AssetClasses = Field(default_factory=AssetClasses)
    dividend: DividendConfig = Field(default_factory=DividendConfig)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read and validate ``config.yaml`` into a typed :class:`Config`."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
