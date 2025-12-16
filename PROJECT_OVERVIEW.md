# Crisis Forecaster — Single-File Summary

This file is an AI-friendly overview of the entire project. It explains the goal, data, core code, how to run things, and where outputs live.

## Goal
- Build a daily Financial Stress Index (FSI) from market risk proxies.
- Fit baseline econometric models (OLS, ARIMA, GARCH), evaluate forecasts, and visualize results via Streamlit and a batch script.

## Data pipeline (src/utils/load_data.py)
- Inputs (downloaded when needed):
  - VIX from Yahoo Finance (`load_vix`).
  - High-yield credit spread from FRED (`BAMLH0A0HYM2`, `load_credit_spread`).
  - Bank risk proxy from Yahoo Finance (`load_bank_cds`, fallback tickers EUFN→KBE→XLF).
- Steps:
  1) Daily frequency with forward fill; align overlapping window.
  2) Z-score each component; FSI = average of z-scores.
  3) Saved CSVs: `data/vix.csv`, `data/credit_spread.csv`, `data/bank_cds.csv`, `data/fsi.csv`.

## Models (src/models/)
- OLS (`ols.py`): FSI ~ VIX + Spread + CDS; returns model, residuals, R², params; diagnostics (Ljung–Box, ARCH LM, Jarque–Bera, skew, kurtosis).
- ARIMA (`arima.py`): `select_arima` uses pmdarima auto_arima (p,d,q up to 3,2,3); `run_arima` fits statsmodels ARIMA and exposes `forecast(steps)`.
- GARCH (`garch.py`): arch `arch_model` mean=0, GARCH(1,1); `forecast(horizon)` returns future volatilities.
- Evaluation (`evaluation.py`): time-series train/test split; RMSE/MAE with index alignment; rolling-origin ARIMA evaluation with optional order reselection.

## Streamlit app (app/app.py)
- Tabs: Home, Data explorer, Model estimation, Forecasts & diagnostics, Forecast evaluation, Crisis simulation.
- Sidebar: choose view, training end date, forecast horizon, run models.
- Shows OLS results/diagnostics, ARIMA selection, GARCH vol forecast, forecast charts with CI, evaluation metrics (RMSE/MAE), and shock scenarios (VIX/spread shifts via OLS betas).

## Batch script (forecasting_script.py)
- Reads `data/fsi.csv`; runs OLS, ARIMA, GARCH; evaluates ARIMA; saves outputs to `report/`.
- Figures: `fsi_series.png`, `arima_forecast.png`, `garch_vol_forecast.png`, `ols_residual_acf_pacf.png`, `fsi_decomposition.png` (stacked monthly contributions).
- CSVs: `arima_forecast.csv`, `garch_vol_forecast.csv`, `forecast_metrics.csv`; OLS summary: `ols_summary.txt`.

## Key outputs (report/)
- Latest generated artifacts live in `report/` after running the batch script or the app.
- Data lives in `data/` (small, included).

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python3 forecasting_script.py          # batch pipeline → report/*
streamlit run app/app.py               # interactive app
pytest                                 # tests (may skip if data download blocked)
```

## Tests (tests/)
- `test_load_data.py`: shape/content sanity of loaders (skips gracefully if downloads fail).
- `test_evaluation.py`: train/test split behavior; RMSE/MAE correctness.

## Packaging helper
- `make_submission_zip.py` bundles code, requirements, tests, data CSVs, and report assets into `Crisis_Forecaster_Submission.zip`.
