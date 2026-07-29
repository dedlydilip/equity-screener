import pytest
from pydantic import ValidationError

from screener.config import Config, load_config


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
