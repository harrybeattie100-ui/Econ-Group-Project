import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_path = repo_root / "data" / "fsi.csv"
    if not data_path.exists():
        raise FileNotFoundError("data/fsi.csv not found. Run src.utils.load_data.load_all_data first.")

    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    print("FSI head:")
    print(df.head())
    print("\nFSI tail:")
    print(df.tail())
    print("\nFSI describe():")
    print(df.describe())

    plt.figure(figsize=(10, 5))
    df["FSI"].plot()
    plt.title("Financial Stress Index")
    plt.tight_layout()
    output_path = repo_root / "report" / "fsi_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
