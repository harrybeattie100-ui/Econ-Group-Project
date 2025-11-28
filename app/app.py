from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

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

    st.header("1) Data overview")
    cols_to_show = list(data.columns) if view_choice == "FSI + components" else ["FSI"]
    st.write("Recent observations")
    st.dataframe(data[cols_to_show].tail(10))
    st.line_chart(data[cols_to_show])

    st.header("2) Model estimation")
    train_df = data.loc[:pd.to_datetime(train_end)]
    if len(train_df) < 100:
        st.info("Select a longer sample to run models (need at least 100 daily observations).")
        return

    if run_models:
        try:
            results = run_model_suite(train_df, horizon)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Model estimation failed: {exc}")
            return

        st.subheader("OLS results")
        ols = results["ols"]
        st.write(f"R-squared: {ols['r2']:.3f}")
        st.dataframe(ols["params"].to_frame("coef"))
        st.write("Diagnostics")
        st.dataframe(pd.Series(ols["diagnostics"]).to_frame("value"))

        st.subheader("ARIMA selection")
        st.write(f"Selected order: {results['arima']['order']}")

        st.subheader("GARCH")
        st.write("Volatility forecast prepared.")

        st.header("3) Forecasts")
        history = train_df["FSI"].iloc[-200:]
        arima_forecast = results["arima"]["forecast"]
        forecast_mean = arima_forecast["mean"]
        combined = pd.concat([history, forecast_mean.rename("forecast")])
        st.line_chart(combined)

        # Confidence interval for ARIMA
        ci_df = arima_forecast[["mean_ci_lower", "mean_ci_upper"]]
        ci_df.columns = ["lower", "upper"]
        st.area_chart(ci_df)

        garch_forecast = results["garch"]["forecast"]
        garch_forecast.index.name = "step"
        st.line_chart(garch_forecast)
    else:
        st.info("Use the sidebar to run models and view forecasts.")


if __name__ == "__main__":
    main()
