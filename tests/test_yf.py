"""Pure logic in the yfinance provider — no network calls."""

import pandas as pd
import pytest

from screener.data.yf import YFinanceProvider


def test_classify_flags_a_large_outlier_payment_as_special():
    # PGR-like: three ~$0.10 regular payments + one $13.60 special.
    window = pd.Series([0.10, 0.10, 0.10, 13.60])
    regular, special = YFinanceProvider._classify_regular_special(window)
    assert regular == pytest.approx(0.30)
    assert special == pytest.approx(13.60)


def test_classify_treats_uniform_payments_as_all_regular():
    window = pd.Series([0.50, 0.50, 0.50, 0.50])
    regular, special = YFinanceProvider._classify_regular_special(window)
    assert regular == 2.0
    assert special == 0.0


def test_classify_handles_a_single_payment_as_regular():
    # A lone payment can never exceed `special_multiple` times its own value
    # (the multiple is > 1), so it must never be misclassified as special.
    window = pd.Series([5.0])
    regular, special = YFinanceProvider._classify_regular_special(window)
    assert regular == 5.0
    assert special == 0.0


def test_classify_handles_empty_window():
    regular, special = YFinanceProvider._classify_regular_special(pd.Series(dtype=float))
    assert regular == 0.0
    assert special == 0.0


def test_classify_respects_custom_multiple():
    window = pd.Series([1.0, 1.0, 2.5])  # median 1.0; 2.5x median at multiple=2.0 -> special
    regular, special = YFinanceProvider._classify_regular_special(window, special_multiple=2.0)
    assert special == 2.5
    # with a looser 3x threshold, the same payment no longer qualifies as special
    regular2, special2 = YFinanceProvider._classify_regular_special(window, special_multiple=3.0)
    assert special2 == 0.0
    assert regular2 == 4.5
