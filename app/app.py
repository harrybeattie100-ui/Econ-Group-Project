from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from src.models.evaluation import rolling_origin_arima_evaluation
from src.models.arima import run_arima, select_arima
from src.models.garch import run_garch
from src.models.ols import diagnostics_ols, run_ols
from src.utils.load_data import load_all_data


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    """Load and cache the FSI dataset."""
    return load_all_data()


def run_model_suite(df: pd.DataFrame, horizon: int) -> Dict[str, object]:
    """Run OLS, ARIMA, and GARCH models on the provided training data."""
    results: Dict[str, object] = {}

    ols_res = run_ols(df)
    results["ols"] = {**ols_res, "diagnostics": diagnostics_ols(ols_res["residuals"])}

    order = select_arima(df["FSI"])
    arima_res = run_arima(df["FSI"], order)
    results["arima"] = {"order": order, "forecast": arima_res["forecast"](horizon)}

    garch_res = run_garch(df["FSI"])
    results["garch"] = {"forecast": garch_res["forecast"](horizon)}

    return results


def main() -> None:
    st.set_page_config(page_title="Crisis Forecaster", layout="wide")
    st.title("Crisis Forecaster prototype")

    data = get_data()
    min_date = data.index.min().date()
    max_date = data.index.max().date()

    st.sidebar.header("Controls")
    view_choice = st.sidebar.radio(
        "Data view", options=["FSI + components", "FSI only"], index=0, horizontal=False
    )
    train_end: date = st.sidebar.slider(
        "Training sample end date",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD",
    )
    horizon = st.sidebar.slider("Forecast horizon (days)", min_value=5, max_value=60, value=30, step=5)
    run_models = st.sidebar.button("Run models")

    train_df = data.loc[:pd.to_datetime(train_end)]
    insufficient_data = len(train_df) < 100

    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None

    results = None
    models_ready = False
    model_error: Exception | None = None

    if run_models and not insufficient_data:
        try:
            results = run_model_suite(train_df, horizon)
            st.session_state["model_results"] = {
                "results": results,
                "train_end": train_end,
                "horizon": horizon,
            }
            models_ready = True
        except Exception as exc:  # pragma: no cover - defensive
            model_error = exc
            st.session_state["model_results"] = None

    cached_results = st.session_state.get("model_results")
    if not models_ready and cached_results:
        if (
            cached_results.get("train_end") == train_end
            and cached_results.get("horizon") == horizon
            and not insufficient_data
        ):
            results = cached_results["results"]
            models_ready = True

    home_tab, data_tab, model_tab, forecast_tab, eval_tab = st.tabs(
        ["Home", "Data explorer", "Model estimation", "Forecasts and diagnostics", "Forecast evaluation"]
    )

    with home_tab:
        st.subheader("Overview")
        st.write(
            "Explore the Financial Stress Index (FSI), fit OLS/ARIMA/GARCH models, "
            "and visualize forecasts. Use the sidebar to choose the training window and "
            "forecast horizon, then navigate the tabs for data, estimation outputs, diagnostics, "
            "and evaluation."
        )
        st.write(
            "The app keeps your selections in the sidebar consistent across tabs so you can move "
            "from data inspection to model estimation and forecast review without losing context."
        )

    with data_tab:
        st.subheader("Data explorer")
        cols_to_show = list(data.columns) if view_choice == "FSI + components" else ["FSI"]
        st.write("Recent observations")
        st.dataframe(data[cols_to_show].tail(10))
        st.line_chart(data[cols_to_show])

    with model_tab:
        st.subheader("Model estimation")
        if insufficient_data:
            st.info("Select a longer sample to run models (need at least 100 daily observations).")
        elif model_error:
            st.error(f"Model estimation failed: {model_error}")
        elif models_ready and results is not None:
            st.write(f"Training sample ends on {train_end} | Forecast horizon: {horizon} days")

            st.markdown("**OLS results**")
            ols = results["ols"]
            st.write(f"R-squared: {ols['r2']:.3f}")
            st.dataframe(ols["params"].to_frame("coef"))
            st.write("Diagnostics")
            st.dataframe(pd.Series(ols["diagnostics"]).to_frame("value"))

            st.markdown("**ARIMA selection**")
            st.write(f"Selected order: {results['arima']['order']}")

            st.markdown("**GARCH**")
            st.write("Volatility forecast prepared.")
        else:
            st.info("Use the sidebar to run models and view estimation outputs.")

    with forecast_tab:
        st.subheader("Forecasts and diagnostics")
        if not models_ready or results is None:
            st.info("Run models from the sidebar to view forecasts and diagnostics.")
        else:
            history = train_df["FSI"].iloc[-200:]
            arima_forecast = results["arima"]["forecast"]
            forecast_mean = arima_forecast["mean"]
            combined = pd.concat([history, forecast_mean.rename("forecast")])
            st.line_chart(combined)

            ci_df = arima_forecast[["mean_ci_lower", "mean_ci_upper"]]
            ci_df.columns = ["lower", "upper"]
            st.area_chart(ci_df)

            garch_forecast = results["garch"]["forecast"]
            garch_forecast.index.name = "step"
            st.line_chart(garch_forecast)

    with eval_tab:
        st.subheader("Forecast evaluation")
        st.write(
            "Out-of-sample performance for the ARIMA model on the Financial Stress Index. "
            "Choose a training ratio and run the evaluation to see forecast accuracy on the held-out window."
        )
        train_ratio = st.slider("Training ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
        if st.button("Run evaluation"):
            try:
                eval_df = get_data()
                fsi_series = eval_df["FSI"]
                test_fraction = 1 - train_ratio
                eval_res = rolling_origin_arima_evaluation(
                    fsi_series, order=None, test_size=test_fraction, forecast_horizon=1
                )
                metrics = eval_res["metrics"]
                preds = eval_res["predictions"]
                actual = fsi_series.loc[preds.index]
                actual_aligned, preds_aligned = actual.align(preds, join="inner")
                if actual_aligned.empty:
                    st.warning("No overlapping dates between actual and predicted series.")
                else:
                    col1, col2 = st.columns(2)
                    col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                    col2.metric("MAE", f"{metrics['mae']:.4f}")

                    train_end_dt = pd.to_datetime(metrics["train_end"]).date()
                    test_start_dt = pd.to_datetime(metrics["test_start"]).date()
                    test_end_dt = pd.to_datetime(metrics["test_end"]).date()
                    st.write(
                        f"Train end: {train_end_dt} | Test window: {test_start_dt} to {test_end_dt}"
                    )

                    overlay_df = pd.DataFrame(
                        {"Actual FSI": actual_aligned, "Predicted FSI": preds_aligned}
                    )
                    st.line_chart(overlay_df)

                    errors = (actual_aligned - preds_aligned).rename("Forecast error")
                    st.line_chart(errors.to_frame())

                    st.dataframe(
                        pd.DataFrame(
                            {"Actual": actual_aligned, "Predicted": preds_aligned}
                        ).tail(10)
                    )
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
