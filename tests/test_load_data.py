from __future__ import annotations

import pandas as pd
import pytest

from src.utils.load_data import load_all_data


def test_load_all_data_basic_shape() -> None:
    try:
        df = load_all_data(save_csv=False)
    except Exception as exc:  # pragma: no cover - network guard
        pytest.skip(f"Data download unavailable: {exc}")

    cols = set(df.columns)
    assert "VIX" in cols
    assert "FSI" in cols
    assert any(col in cols for col in ["Spread", "CreditSpread"])
    assert any(col in cols for col in ["CDS", "BankRisk"])

    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) > 10

    fsi_clean = df["FSI"].iloc[5:]
    assert not fsi_clean.isna().any()
