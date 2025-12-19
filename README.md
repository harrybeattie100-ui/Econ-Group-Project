# Crisis Forecaster — FIN41660 Financial Econometrics (AY 2025/2026)

## Overview
This repository contains the group project for FIN41660 Financial Econometrics at UCD (Academic Year 2025/2026). The goal is to construct and forecast a Financial Stress Index (FSI) using OLS, ARIMA, and GARCH models, and to surface results through a Streamlit dashboard and a reproducible standalone script.

## Data
- **Financial Stress Index (FSI):** Composite index built from three components:
  - VIX (equity implied volatility)
  - High yield credit spread (FRED series)
  - Bank risk proxy (liquid financial ETF prices)
- The data pipeline assembles these series, aligns them on a daily frequency, z-scores the components, and averages them to form the FSI. Cleaned CSVs are stored under `data/`.
- The loader now reuses cached CSVs in `data/` by default to keep the app working offline/without re-downloading. Pass `refresh=True` to `load_all_data` to force a fresh pull.

## Repository structure
- `src/utils/`: Data loaders and FSI construction (`load_all_data`).
- `src/models/`: Model code for OLS, ARIMA, GARCH, plus evaluation utilities.
- `app/`: Streamlit dashboard with tabs for data exploration, model estimation, forecasts/diagnostics, evaluation, and crisis simulation.
- `forecasting_script.py`: Standalone pipeline to run models, export forecasts/metrics, and save figures.
- `report/`: Generated outputs (CSV forecasts, metrics, and figures).
- `tests/`: Basic pytest suite for evaluation utilities and data loading.
- `data/`: Derived CSV inputs (small, repo-safe).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
- **Streamlit app:** `streamlit run app/app.py`
- **Standalone script:** `python3 forecasting_script.py`
- **Tests:** `pytest`

## Outputs
- Forecast CSVs and metrics are written to `report/` (e.g., ARIMA and GARCH forecasts, `forecast_metrics.csv`).
- Figures saved by the standalone script: `report/fsi_series.png`, `report/arima_forecast.png`, `report/garch_vol_forecast.png`.
- The crisis simulation tab in the app shifts ARIMA forecasts using OLS betas under simple VIX or credit spread shocks.

## Brief vs implementation
- Implements OLS, ARIMA, and GARCH modeling of the FSI.
- Forecast accuracy evaluated via RMSE and MAE (see `report/forecast_metrics.csv` and evaluation tab).
- Standalone script mirrors app behavior for reproducibility.
- Crisis simulation tab illustrates policy or market shocks through VIX/spread shifts applied to ARIMA forecasts using OLS betas.
