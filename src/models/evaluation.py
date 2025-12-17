from __future__ import annotations

import numpy as np
import pandas as pd

from .arima import run_arima, select_arima


def time_series_train_test_split(series: pd.Series, test_size: int | float) -> tuple[pd.Series, pd.Series]:
    """
    Chronologically split a series into train and test segments.

    test_size can be an integer count or a float fraction (0,1).
    """
    series = series.dropna()
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as float must be between 0 and 1.")
        test_len = max(int(len(series) * test_size), 1)
    else:
        test_len = int(test_size)

    if test_len <= 0:
        raise ValueError("test_size must be positive.")
    if test_len >= len(series):
        raise ValueError("test_size must be smaller than the length of the series.")

    train = series.iloc[:-test_len]
    test = series.iloc[-test_len:]
    return train, test


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error with alignment on the common index."""
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    if aligned_true.empty:
        raise ValueError("No overlapping index between y_true and y_pred.")
    return float(np.sqrt(np.mean((aligned_true - aligned_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error with alignment on the common index."""
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    if aligned_true.empty:
        raise ValueError("No overlapping index between y_true and y_pred.")
    return float(np.mean(np.abs(aligned_true - aligned_pred)))


def _extract_mean_forecast(forecast_df: pd.DataFrame, offset: int = 0) -> float:
    """Grab the mean forecast value for a given step from a statsmodels summary frame."""
    for col in ("mean", "predicted_mean"):
        if col in forecast_df.columns:
            return float(forecast_df[col].iloc[offset])
    return float(forecast_df.iloc[offset, 0])


def rolling_origin_arima_evaluation(
    series: pd.Series,
    order: tuple[int, int, int] | None = None,
    test_size: int = 30,
    forecast_horizon: int = 1,
) -> dict:
    """
    Perform rolling-origin evaluation with one-step-ahead ARIMA forecasts.

    The ARIMA order can be supplied; otherwise, it is reselected on each iteration.
    """
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    series = series.dropna()
    train, test = time_series_train_test_split(series, test_size=test_size)
    preds: list[float] = []

    for i in range(len(test)):
        history = pd.concat([train, test.iloc[:i]])
        eval_order = order if order is not None else select_arima(history)
        arima_res = run_arima(history, eval_order)
        forecast_df = arima_res["forecast"](steps=forecast_horizon)
        preds.append(_extract_mean_forecast(forecast_df, offset=forecast_horizon - 1))

    predictions = pd.Series(preds, index=test.index, name="Forecast")
    metrics = {
        "rmse": rmse(test, predictions),
        "mae": mae(test, predictions),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),
    }
    return {"metrics": metrics, "predictions": predictions}
