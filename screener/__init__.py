"""Institutional-grade factor-based S&P 500 equity screener & factor research engine."""

from .config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config"]
