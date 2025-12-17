from __future__ import annotations

import math

import pandas as pd
import pytest

pytest.importorskip("pmdarima")

from src.models.evaluation import (
    mae,
    rmse,
    rolling_origin_arima_evaluation,
    time_series_train_test_split,
)


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


def test_rolling_origin_reuses_selected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = {"select": 0}

    def fake_select(series: pd.Series) -> tuple[int, int, int]:
        call_counter["select"] += 1
        return (1, 0, 0)

    def fake_run(series: pd.Series, order: tuple[int, int, int]) -> dict:
        class DummyForecast:
            def __init__(self, last_value: float):
                self.last_value = last_value

            def summary_frame(self) -> pd.DataFrame:
                return pd.DataFrame({"mean": [self.last_value]})

        class DummyModel:
            def __init__(self, data: pd.Series):
                self.data = data

            def get_forecast(self, steps: int = 1) -> DummyForecast:
                return DummyForecast(float(self.data.iloc[-1]))

            def append(self, new_endog: pd.Series, refit: bool = False) -> "DummyModel":
                return DummyModel(pd.concat([self.data, new_endog]))

        dummy_model = DummyModel(series)

        def forecast(steps: int = 1) -> pd.DataFrame:
            return dummy_model.get_forecast(steps).summary_frame()

        return {"model": dummy_model, "forecast": forecast}

    monkeypatch.setattr("src.models.evaluation.select_arima", fake_select)
    monkeypatch.setattr("src.models.evaluation.run_arima", fake_run)

    series = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10, freq="D"))
    res = rolling_origin_arima_evaluation(series, order=None, test_size=2, forecast_horizon=1)

    assert call_counter["select"] == 1
    assert len(res["predictions"]) == 2
