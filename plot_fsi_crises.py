"""
Generate a publication-quality plot of the Financial Stress Index with major crisis periods annotated.

Usage:
    python3 plot_fsi_crises.py  # reads data/fsi.csv and writes report/fsi_with_crises.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data") / "fsi.csv"
REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "fsi_with_crises.png"


def plot_fsi_with_crises(fsi: pd.Series, outfile: Path = OUTPUT_PATH) -> None:
    """Plot FSI with shaded crisis windows and save to outfile."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(fsi.index, fsi.values, label="FSI", color="#1f2d3d", linewidth=1.6)

    crises = [
        ("Global Financial Crisis", "2008-09-01", "2009-06-30", "tab:red", 0.15),
        ("COVID-19", "2020-03-01", "2020-04-30", "tab:orange", 0.12),
        ("Banking stress (SVB)", "2023-03-01", "2023-03-31", "tab:purple", 0.12),
    ]
    for label, start, end, color, alpha in crises:
        ax.axvspan(start, end, color=color, alpha=alpha, label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Financial Stress Index")
    ax.set_title("Financial Stress Index with Major Crisis Periods")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def load_fsi(path: Path = DATA_PATH) -> pd.Series:
    """Load FSI series from CSV with Date index."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df["FSI"].dropna()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/fsi.csv not found. Generate it with load_all_data first.")
    fsi = load_fsi()
    plot_fsi_with_crises(fsi, OUTPUT_PATH)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
