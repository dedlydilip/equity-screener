"""Typed, validated configuration.

Loads ``config.yaml`` into Pydantic models so every run is *pre-registered* and
invalid settings fail fast at load time rather than silently corrupting results
(e.g. sleeve weights that don't sum to 1.0, or a normalization hierarchy that
never falls back to the cross-sectional bucket).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class UniverseConfig(BaseModel):
    name: str = "sp500"
    source: Literal["wikipedia", "fmp"] = "wikipedia"


class DataConfig(BaseModel):
    provider: Literal["fmp", "edgar", "yfinance"] = "edgar"
    fmp_api_key_env: str = "FMP_API_KEY"
    edgar_contact: str = "adilip407@gmail.com"
    cache_dir: str = ".cache"
    fundamentals_pit: bool = True


class FactorConfig(BaseModel):
    value: dict[str, float]
    quality: dict[str, float]
    momentum: dict[str, float]


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
    pe_bounds: tuple[float, float] = (0.0, 500.0)
    de_min: float = 0.0
    xs_mad_flag: float = Field(10.0, gt=0.0)
    drop_negative_book_bp: bool = True
    drop_negative_ebitda_ev: bool = True


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
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    rebalance: Literal["monthly", "quarterly"] = "monthly"
    frequencies_to_test: list[Literal["monthly", "quarterly"]] = Field(
        default_factory=lambda: ["monthly", "quarterly"]
    )


class CostsConfig(BaseModel):
    commission_bps: float = Field(5.0, ge=0.0)
    spread_bps: float = Field(10.0, ge=0.0)
    short_rebate_bps: float = Field(25.0, ge=0.0)


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
