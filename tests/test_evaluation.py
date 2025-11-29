from __future__ import annotations

import math

import pandas as pd
import pytest

pytest.importorskip("pmdarima")

from src.models.evaluation import mae, rmse, time_series_train_test_split


def test_time_series_train_test_split_preserves_order_and_sizes() -> None:
    n = 50
    series = pd.Series(range(n), index=pd.date_range("2020-01-01", periods=n, freq="D"))
    train_ratio = 0.8
    test_size = 1 - train_ratio
    train, test = time_series_train_test_split(series, test_size=test_size)

    expected_test_len = max(int(n * test_size), 1)
    expected_train_len = n - expected_test_len

    assert len(train) == expected_train_len
    assert len(test) == expected_test_len
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
    assert train.index.max() < test.index.min()
    assert list(train.index) + list(test.index) == list(series.index)


def test_rmse_and_mae_known_values() -> None:
    y_true = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 1, 4])

    expected_rmse = math.sqrt((0**2 + 1**2 + 1**2) / 3)
    expected_mae = (0 + 1 + 1) / 3

    assert math.isclose(rmse(y_true, y_pred), expected_rmse, rel_tol=1e-9)
    assert math.isclose(mae(y_true, y_pred), expected_mae, rel_tol=1e-9)
