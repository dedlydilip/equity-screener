import pytest
import yaml
from pydantic import ValidationError

from screener.config import Config, load_config

CONFIG = "config.yaml"


def test_config_yaml_loads_and_validates():
    cfg = load_config("config.yaml")
    assert cfg.universe.name == "sp500"
    assert cfg.normalization.hierarchy[-1].level == "cross_sectional"


def test_sleeve_weights_must_sum_to_one():
    raw = load_config("config.yaml").model_dump()
    raw["sleeve_weights"] = {"value": 0.5, "quality": 0.5, "momentum": 0.5}  # sums to 1.5
    with pytest.raises(ValidationError):
        Config(**raw)


def test_normalization_hierarchy_must_end_cross_sectional():
    raw = load_config("config.yaml").model_dump()
    raw["normalization"]["hierarchy"] = [{"level": "sector", "min_n": 20}]  # no cross_sectional
    with pytest.raises(ValidationError):
        Config(**raw)


def test_factor_descriptor_keys_must_match_the_pipelines_factor_columns():
    """The momentum sleeve was configured as 'ret_12_1' while the column the
    pipeline computes is 'momentum', so that weight could never be applied to
    anything. Config/code drift like that must fail loudly."""
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cfg["factors"]["momentum"] = {"ret_12_1": 1.0}
    with pytest.raises(ValidationError, match="does not compute"):
        Config(**cfg)


def test_infeasible_max_sleeve_weight_is_rejected_at_load():
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cfg["weighting"]["max_sleeve_weight"] = 0.30   # 3 x 0.30 < 1
    with pytest.raises(ValidationError, match="infeasible"):
        Config(**cfg)


def test_backtest_dates_must_be_ordered():
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cfg["backtest"]["start"], cfg["backtest"]["end"] = "2024-12-31", "2015-01-01"
    with pytest.raises(ValidationError, match="must precede"):
        Config(**cfg)


def test_duplicate_test_frequencies_are_rejected():
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cfg["backtest"]["frequencies_to_test"] = ["monthly", "monthly"]
    with pytest.raises(ValidationError, match="duplicate"):
        Config(**cfg)


def test_pe_bounds_must_be_ordered():
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cfg["quality_gates"]["pe_bounds"] = [500.0, 0.0]
    with pytest.raises(ValidationError, match="low < high"):
        Config(**cfg)
