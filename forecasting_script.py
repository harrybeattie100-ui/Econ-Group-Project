import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.append(os.path.abspath("."))

from src.models.evaluation import rolling_origin_arima_evaluation
from src.models.arima import run_arima, select_arima
from src.models.garch import run_garch
from src.models.ols import diagnostics_ols, run_ols

matplotlib.use("Agg")


"""
Standalone pipeline for the FIN41660 Crisis Forecaster project.

Runs OLS, ARIMA, and GARCH on the FSI, writes forecasts/metrics, and saves
figures for the written report.
"""

REPORT_DIR = Path("report")
DATA_PATH = Path("data") / "fsi.csv"
FORECAST_HORIZON = 30


def _ensure_datetime_index(series: pd.Series, start: pd.Timestamp, periods: int) -> pd.Series:
    """Guarantee a datetime index for forecast series when statsmodels returns an integer index."""
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    future_index = pd.date_range(start + pd.Timedelta(days=1), periods=periods, freq="D")
    series.index = future_index
    return series


def save_figures(
    df: pd.DataFrame,
    arima_forecast: pd.DataFrame | None,
    garch_forecast: pd.DataFrame | None,
    garch_model,
    residuals: pd.Series | None = None,
) -> None:
    """Persist basic figures supporting the project report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["FSI"].plot(ax=ax, label="FSI")
        ax.set_title("Financial Stress Index (full history)")
        ax.set_ylabel("Index level")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "fsi_series.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"FSI figure failed: {exc}")

    try:
        if arima_forecast is not None:
            history = df["FSI"].iloc[-200:]
            forecast_mean = arima_forecast["mean"].copy()
            forecast_mean = _ensure_datetime_index(forecast_mean, df.index.max(), len(forecast_mean))
            arima_forecast = arima_forecast.copy()
            arima_forecast.index = forecast_mean.index

            fig, ax = plt.subplots(figsize=(8, 4))
            history.plot(ax=ax, label="Historical FSI")
            forecast_mean.plot(ax=ax, label="ARIMA forecast")
            ax.fill_between(
                arima_forecast.index,
                arima_forecast["mean_ci_lower"],
                arima_forecast["mean_ci_upper"],
                alpha=0.2,
                label="95% CI",
            )
            ax.set_title("ARIMA forecast with 95% confidence interval")
            ax.legend()
            fig.tight_layout()
            fig.savefig(REPORT_DIR / "arima_forecast.png")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ARIMA figure failed: {exc}")

    try:
        if garch_forecast is not None and garch_model is not None:
            cond_vol = pd.Series(garch_model.conditional_volatility, index=df.index, name="Cond vol")
            forecast_vol = garch_forecast["volatility"].copy()
            if not isinstance(forecast_vol.index, pd.DatetimeIndex):
                future_index = pd.date_range(
                    df.index.max() + pd.Timedelta(days=1), periods=len(forecast_vol), freq="D"
                )
                forecast_vol.index = future_index

            fig, ax = plt.subplots(figsize=(8, 4))
            cond_vol.iloc[-250:].plot(ax=ax, label="In-sample volatility")
            forecast_vol.plot(ax=ax, label="Forecast volatility")
            ax.set_title("GARCH conditional volatility and forecast")
            ax.set_ylabel("Volatility")
            ax.legend()
            fig.tight_layout()
            fig.savefig(REPORT_DIR / "garch_vol_forecast.png")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"GARCH figure failed: {exc}")

    try:
        if residuals is not None and not residuals.empty:
            clean_resid = residuals.dropna()
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            plot_acf(clean_resid, lags=24, ax=axes[0])
            axes[0].set_title("OLS residual ACF")
            plot_pacf(clean_resid, lags=24, ax=axes[1], method="ywm")
            axes[1].set_title("OLS residual PACF")
            fig.tight_layout()
            fig.savefig(REPORT_DIR / "ols_residual_acf_pacf.png")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"OLS residual ACF/PACF figure failed: {exc}")

    try:
        # Decompose the FSI into equal-weighted z-score contributions for the three components.
        zscores = df[["VIX", "Spread", "CDS"]].apply(lambda s: (s - s.mean()) / s.std())
        contrib_df = pd.DataFrame(
            {
                "VIX": zscores["VIX"] / 3,
                "Credit spread": zscores["Spread"] / 3,
                "Bank risk": zscores["CDS"] / 3,
                "FSI": df["FSI"],
            }
        )
        # Focus on recent history and aggregate to weekly to reduce noise.
        plot_df = contrib_df.loc[contrib_df.index >= "2020-01-01"].resample("W").mean().dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {
            "VIX": "#4c78a8",
            "Credit spread": "#9c755f",
            "Bank risk": "#6b6ecf",
        }
        ax.axhline(0, color="#777777", linewidth=1, linestyle="--")

        width = 5  # days, for weekly bars
        vix_vals = plot_df["VIX"]
        spread_vals = plot_df["Credit spread"]
        bank_vals = plot_df["Bank risk"]

        ax.bar(plot_df.index, vix_vals, width=width, label="VIX", color=colors["VIX"])
        ax.bar(
            plot_df.index,
            spread_vals,
            width=width,
            bottom=vix_vals,
            label="Credit spread",
            color=colors["Credit spread"],
        )
        ax.bar(
            plot_df.index,
            bank_vals,
            width=width,
            bottom=vix_vals + spread_vals,
            label="Bank risk",
            color=colors["Bank risk"],
        )

        ax.set_title("FSI component contributions")
        ax.set_ylabel("Contribution to FSI")
        ax.tick_params(labelsize=9)
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
        legend.set_frame_on(False)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "fsi_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"FSI decomposition figure failed: {exc}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/fsi.csv not found. Generate it with load_all_data first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").dropna()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    arima_forecast: pd.DataFrame | None = None
    garch_forecast: pd.DataFrame | None = None
    garch_model = None
    ols_residuals: pd.Series | None = None

    # OLS
    ols_summary_path = REPORT_DIR / "ols_summary.txt"
    try:
        ols_res = run_ols(df)
        ols_residuals = ols_res["residuals"]
        diag = diagnostics_ols(ols_res["residuals"])
        with open(ols_summary_path, "w") as fh:
            fh.write(ols_res["model"].summary().as_text())
            fh.write("\n\nDiagnostics:\n")
            for k, v in diag.items():
                fh.write(f"{k}: {v}\n")
        print(f"OLS completed. R2={ols_res['r2']:.3f}. Summary saved to {ols_summary_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"OLS estimation failed: {exc}")

    # ARIMA
    arima_forecast_path = REPORT_DIR / "arima_forecast.csv"
    try:
        order = select_arima(df["FSI"])
        arima_res = run_arima(df["FSI"], order)
        arima_forecast = arima_res["forecast"](FORECAST_HORIZON)
        arima_forecast.to_csv(arima_forecast_path, index_label="Date")
        print(f"ARIMA order {order} forecast saved to {arima_forecast_path}")

        eval_res = rolling_origin_arima_evaluation(
            df["FSI"], order=order, test_size=FORECAST_HORIZON, forecast_horizon=1
        )
        metrics = eval_res["metrics"]
        metrics_df = pd.DataFrame(
            [
                {
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "train_end_date": metrics["train_end"].date(),
                    "test_start_date": metrics["test_start"].date(),
                    "test_end_date": metrics["test_end"].date(),
                }
            ]
        )
        metrics_path = REPORT_DIR / "forecast_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"ARIMA evaluation metrics saved to {metrics_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ARIMA estimation failed: {exc}")

    # GARCH
    garch_forecast_path = REPORT_DIR / "garch_vol_forecast.csv"
    try:
        garch_res = run_garch(df["FSI"])
        garch_forecast = garch_res["forecast"](FORECAST_HORIZON)
        garch_model = garch_res["model"]
        garch_forecast.to_csv(garch_forecast_path)
        print(f"GARCH volatility forecast saved to {garch_forecast_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"GARCH estimation failed: {exc}")

    save_figures(df, arima_forecast, garch_forecast, garch_model, residuals=ols_residuals)


if __name__ == "__main__":
    main()
