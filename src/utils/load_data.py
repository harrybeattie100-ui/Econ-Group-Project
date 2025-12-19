from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "2005-01-01"


def _select_price(data: pd.DataFrame) -> pd.Series:
    """Handle both flat and MultiIndex columns from yfinance downloads."""
    if data.empty:
        raise ValueError("No data returned from download")
    for field in ("Adj Close", "Close"):
        if isinstance(data.columns, pd.MultiIndex):
            if field in data.columns.get_level_values(0):
                series = data.xs(field, level=0, axis=1).squeeze()
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series
        elif field in data.columns:
            series = data[field]
            return series
    raise KeyError("Neither Adj Close nor Close found in downloaded data")


def _to_daily(series: pd.Series) -> pd.Series:
    """Ensure a daily frequency with forward filling for missing days."""
    series = series.sort_index()
    series.index = pd.to_datetime(series.index)
    series.index.name = "Date"
    return series.asfreq("D", method="ffill")


def _write_csv(path: Path, data: pd.Series | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index_label="Date")


def _load_cached_fsi(path: Path | None = None) -> pd.DataFrame | None:
    """
    Load a previously saved FSI CSV if available.

    Returns None when the file is missing or malformed so callers can fall back to downloads.
    """
    csv_path = path or DATA_DIR / "fsi.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    except Exception:
        return None

    expected_cols = {"VIX", "Spread", "CDS", "FSI"}
    if df.empty or not expected_cols.issubset(set(df.columns)):
        return None

    df = df.sort_index()
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(how="all")


def load_vix(start: str = DEFAULT_START, end: str | None = None) -> pd.Series:
    """Download daily VIX data and return a Series named 'VIX' indexed by date."""
    data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    vix = _select_price(data).rename("VIX")
    return _to_daily(vix)


def load_credit_spread(
    start: str = DEFAULT_START, end: str | None = None, fred_code: str = "BAMLH0A0HYM2"
) -> pd.Series:
    """
    Download a high yield credit spread from FRED.

    The default series is ICE BofA US High Yield Index Option-Adjusted Spread (BAMLH0A0HYM2).
    """
    spread = pdr.DataReader(fred_code, "fred", start=start, end=end).squeeze()
    spread.name = "Spread"
    spread.index.name = "Date"
    return _to_daily(spread).ffill()


def load_bank_cds(
    start: str = DEFAULT_START, end: str | None = None, proxies: Iterable[str] | None = None
) -> pd.Series:
    """
    Load a proxy for bank CDS risk from Yahoo Finance.

    Direct CDS tickers are not available on Yahoo Finance, so we use liquid financial ETFs
    as a proxy for bank credit risk. Default order tries European banks (EUFN), then US
    regional banks (KBE), then broad financials (XLF).
    """
    tickers = list(proxies) if proxies is not None else ["EUFN", "KBE", "XLF"]
    last_error: Exception | None = None

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if data.empty:
                continue
            price = _select_price(data).rename("CDS")
            cds = _to_daily(price).ffill()
            cds.attrs["proxy_ticker"] = ticker
            return cds
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

    msg = "Unable to download any proxy for bank CDS risk from Yahoo Finance"
    if last_error:
        msg += f" (last error: {last_error})"
    raise RuntimeError(msg)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(index=series.index, data=0.0)
    return (series - series.mean()) / std


def build_fsi(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Construct the Financial Stress Index from VIX, credit spread, and bank CDS proxy."""
    vix = load_vix(start=start, end=end)
    spread = load_credit_spread(start=start, end=end)
    cds = load_bank_cds(start=start, end=end)

    start_date = max(s.index.min() for s in (vix, spread, cds))
    end_date = min(s.index.max() for s in (vix, spread, cds))
    if start_date >= end_date:
        raise ValueError("Insufficient overlap between series to build FSI")
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")

    aligned = pd.DataFrame(
        {
            "VIX": vix.reindex(full_index).ffill(),
            "Spread": spread.reindex(full_index).ffill(),
            "CDS": cds.reindex(full_index).ffill(),
        }
    )
    zscores = aligned.apply(_zscore)
    aligned["FSI"] = zscores.mean(axis=1)
    aligned.index.name = "Date"
    return aligned


def load_all_data(
    start: str = DEFAULT_START,
    end: str | None = None,
    save_csv: bool = True,
    prefer_cached: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Return the FSI DataFrame and optionally write CSV outputs under data/.

    When prefer_cached=True (default) and a cached CSV exists, use it to avoid network calls.
    Set refresh=True to force rebuilding from downloads even if cached data are present.
    """
    if prefer_cached and not refresh:
        cached = _load_cached_fsi()
        if cached is not None:
            return cached

    fsi = build_fsi(start=start, end=end)
    if save_csv:
        _write_csv(DATA_DIR / "vix.csv", fsi[["VIX"]])
        _write_csv(DATA_DIR / "credit_spread.csv", fsi[["Spread"]])
        _write_csv(DATA_DIR / "bank_cds.csv", fsi[["CDS"]])
        _write_csv(DATA_DIR / "fsi.csv", fsi)
    return fsi


if __name__ == "__main__":
    load_all_data()
