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
    """Load and cache the FSI dataset (prefer cached CSVs to avoid network flakes)."""
    return load_all_data(prefer_cached=True)


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

    home_tab, data_tab, model_tab, forecast_tab, eval_tab, sim_tab = st.tabs(
        [
            "Home",
            "Data explorer",
            "Model estimation",
            "Forecasts and diagnostics",
            "Forecast evaluation",
            "Crisis simulation",
        ]
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
        max_test_points = 365  # cap to keep evaluation responsive
        if st.button("Run evaluation", key="run_eval"):
            try:
                eval_df = get_data()
                fsi_series = eval_df["FSI"]
                test_fraction = 1 - train_ratio
                requested_test_points = max(int(len(fsi_series) * test_fraction), 1)
                test_points = min(requested_test_points, max_test_points)
                order_hint = (1, 0, 0)  # fast default to avoid slow auto_arima during evaluation

                status = st.empty()
                status.info(
                    f"Running rolling evaluation on {test_points} days "
                    f"(train ratio {train_ratio:.2f}, ARIMA order {order_hint})."
                )
                if requested_test_points > max_test_points:
                    st.info(
                        f"Requested test window ({requested_test_points} days) capped at {max_test_points} "
                        "to keep the evaluation quick."
                    )
                with st.spinner("Running rolling evaluation (ARIMA)…"):
                    eval_res = rolling_origin_arima_evaluation(
                        fsi_series, order=order_hint, test_size=test_points, forecast_horizon=1
                    )
                metrics = eval_res["metrics"]
                preds = eval_res["predictions"]
                actual = fsi_series.loc[preds.index]
                status.success("Evaluation complete.")
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

    with sim_tab:
        st.subheader("Crisis simulation")
        st.write(
            "Simulate simple shock scenarios to VIX or credit spreads and view their impact on the "
            "FSI forecast path. Shocks are applied as constant shifts using OLS coefficients."
        )

        scenario_options = [
            "None",
            "VIX shock plus 25 percent",
            "VIX shock plus 50 percent",
            "Credit spread shock plus 50bp",
            "Credit spread shock plus 100bp",
        ]
        scenario_choice = st.selectbox("Shock scenario", options=scenario_options, index=0)
        sim_horizon = st.slider(
            "Scenario horizon (days)", min_value=5, max_value=60, value=horizon, step=5
        )
        if st.button("Run simulation"):
            if insufficient_data:
                st.info("Select a longer sample to run simulations (need at least 100 daily observations).")
            else:
                try:
                    sim_train = train_df
                    ols_params = None
                    if models_ready and results is not None:
                        ols_params = results["ols"]["params"]
                    if ols_params is None:
                        ols_params = run_ols(sim_train)["params"]

                    vix_beta = float(ols_params.get("VIX", 0.0))
                    spread_beta = float(ols_params.get("Spread", 0.0))
                    latest_obs = sim_train.iloc[-1]
                    current_vix = float(latest_obs["VIX"])
                    current_spread = float(latest_obs["Spread"])

                    shock_shift = 0.0
                    shock_desc = "No shock applied."

                    if scenario_choice == "VIX shock plus 25 percent":
                        shock_size = 0.25 * current_vix
                        shock_shift = vix_beta * shock_size
                        shock_desc = (
                            f"VIX increased by 25% of current level ({shock_size:.2f}); "
                            f"FSI shift = beta_VIX ({vix_beta:.4f}) * shock."
                        )
                    elif scenario_choice == "VIX shock plus 50 percent":
                        shock_size = 0.50 * current_vix
                        shock_shift = vix_beta * shock_size
                        shock_desc = (
                            f"VIX increased by 50% of current level ({shock_size:.2f}); "
                            f"FSI shift = beta_VIX ({vix_beta:.4f}) * shock."
                        )
                    elif scenario_choice == "Credit spread shock plus 50bp":
                        shock_size = 0.50  # 50 basis points expressed in spread units
                        shock_shift = spread_beta * shock_size
                        shock_desc = (
                            f"Credit spread increased by 50bp ({shock_size:.2f}); "
                            f"FSI shift = beta_Spread ({spread_beta:.4f}) * shock."
                        )
                    elif scenario_choice == "Credit spread shock plus 100bp":
                        shock_size = 1.00  # 100 basis points expressed in spread units
                        shock_shift = spread_beta * shock_size
                        shock_desc = (
                            f"Credit spread increased by 100bp ({shock_size:.2f}); "
                            f"FSI shift = beta_Spread ({spread_beta:.4f}) * shock."
                        )

                    arima_order = None
                    if models_ready and results is not None:
                        arima_order = results["arima"]["order"]
                    if arima_order is None:
                        arima_order = select_arima(sim_train["FSI"])

                    arima_res = run_arima(sim_train["FSI"], arima_order)
                    forecast_df = arima_res["forecast"](sim_horizon)
                    baseline_mean = forecast_df["mean"].copy()
                    if not isinstance(baseline_mean.index, pd.DatetimeIndex):
                        future_index = pd.date_range(
                            sim_train.index.max() + pd.Timedelta(days=1), periods=sim_horizon, freq="D"
                        )
                        baseline_mean.index = future_index

                    shocked_mean = baseline_mean + shock_shift

                    history = sim_train["FSI"].iloc[-200:].rename("Historical FSI")
                    future_paths = pd.DataFrame(
                        {"Baseline forecast": baseline_mean, "Shock scenario": shocked_mean}
                    )
                    combined_paths = pd.concat([history, future_paths], axis=1)
                    st.line_chart(combined_paths)

                    st.write(shock_desc)
                    st.write(
                        f"Simulation horizon: {sim_horizon} days. Training data through {train_end} "
                        f"| Current VIX {current_vix:.2f}, Spread {current_spread:.2f}."
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    st.error(f"Simulation failed: {exc}")


if __name__ == "__main__":
    main()
