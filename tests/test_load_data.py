from __future__ import annotations

import pandas as pd
import pytest

import src.utils.load_data as load_data


def test_load_all_data_basic_shape() -> None:
    try:
        df = load_data.load_all_data(save_csv=False)
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


def test_load_all_data_prefers_cached(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    cached = pd.DataFrame(
        {
            "VIX": [1, 2, 3, 4, 5],
            "Spread": [2, 2, 2, 2, 2],
            "CDS": [3, 3, 3, 3, 3],
            "FSI": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        index=dates,
    )
    cached.index.name = "Date"
    cached_path = tmp_path / "fsi.csv"
    cached.to_csv(cached_path, index_label="Date")

    def _fail_download(*args, **kwargs):
        raise AssertionError("Network loaders should not be called when cached data exist.")

    monkeypatch.setattr(load_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(load_data, "load_vix", _fail_download)
    monkeypatch.setattr(load_data, "load_credit_spread", _fail_download)
    monkeypatch.setattr(load_data, "load_bank_cds", _fail_download)

    df = load_data.load_all_data(prefer_cached=True, save_csv=False)
    pd.testing.assert_index_equal(df.index, cached.index)
    assert set(df.columns) >= {"VIX", "Spread", "CDS", "FSI"}
