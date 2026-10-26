"""Download validated Yahoo adjusted prices into an external, reproducible cache."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOAD_OPTIONS = {
    "auto_adjust": True,
    "actions": False,
    "interval": "1d",
    "keepna": True,
    "repair": False,
    "progress": False,
    "threads": False,
    "ignore_tz": True,
}


def _check_tickers(tickers: tuple[str, ...]) -> None:
    """Reject empty, repeated, or malformed ticker selections."""
    if not tickers or any(not isinstance(t, str) or not t.strip() for t in tickers):
        raise ValueError("tickers must contain nonempty ticker strings")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be unique")


def validate_prices(prices: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Return finite positive daily prices in ticker order, rejecting gaps without filling.

    This establishes a complete observed panel, not a trading-calendar or corporate-action
    audit. Dates must already be timezone-naive, unique, sorted, and at midnight.
    """
    _check_tickers(tickers)
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("price panel must be a nonempty DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("price panel requires a DatetimeIndex")
    if (prices.index.tz is not None or prices.index.hasnans
            or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not prices.index.equals(prices.index.normalize())):
        raise ValueError("price dates must be unique, sorted, timezone-naive daily dates")
    if not prices.columns.is_unique:
        raise ValueError("price columns must be unique")
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"price panel is missing tickers: {missing}")
    selected = prices.loc[:, list(tickers)]
    try:
        values = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be numeric") from exc
    bad = ~np.isfinite(values) | (values <= 0.0)
    if bad.any():
        row, column = np.argwhere(bad)[0]
        raise ValueError(
            f"missing, nonfinite, or nonpositive price for {tickers[column]} "
            f"on {selected.index[row].date()}; no filling is performed"
        )
    result = pd.DataFrame(values, index=selected.index.rename("Date"), columns=list(tickers))
    result.attrs = deepcopy(prices.attrs)
    return result


def external_directory(path: Path | str) -> Path:
    """Resolve a generated-state directory, rejecting checkout and OneDrive locations."""
    resolved = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    forbidden = [repository, Path.home() / "OneDrive"]
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(name):
            forbidden.append(Path(os.environ[name]).expanduser().resolve())
    if (any(resolved.is_relative_to(root) for root in forbidden)
            or any(part.casefold() == "onedrive" or part.casefold().startswith("onedrive - ")
                   for part in resolved.parts)):
        raise ValueError("generated state must be outside the checkout and OneDrive")
    if os.name == "nt" and resolved.drive.casefold() != "c:":
        raise ValueError("Windows generated state must be on the C: drive")
    return resolved


def default_cache_directory() -> Path:
    """Choose an external cache, honoring the repository's configured agent workspace."""
    if os.environ.get("AGENT_LOCAL_ROOT"):
        root = Path(os.environ["AGENT_LOCAL_ROOT"]) / "cache"
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        root = root / "OptimalPortfolios"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        root = root / "optimalportfolios"
    return external_directory(root / "oss_portfolio_pipeline")


def _date(value: str, label: str) -> str:
    """Validate and canonicalize a date-only request boundary."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a date") from exc
    if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize():
        raise ValueError(f"{label} must be a timezone-naive date")
    return stamp.strftime("%Y-%m-%d")


def _check_window(prices: pd.DataFrame, start: str, end: str) -> None:
    """Reject rows outside the inclusive-start, exclusive-end Yahoo request."""
    if prices.index[0] < pd.Timestamp(start) or prices.index[-1] >= pd.Timestamp(end):
        raise ValueError("price dates fall outside the requested [start, end) interval")


def _adjusted_close(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Extract adjusted Close from either Yahoo MultiIndex column orientation."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Yahoo returned no prices")
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [i for i in range(raw.columns.nlevels)
                  if "Close" in raw.columns.get_level_values(i)]
        if len(levels) != 1:
            raise ValueError("Yahoo response does not identify a unique Close field")
        prices = raw.xs("Close", axis=1, level=levels[0]).copy()
    elif len(tickers) == 1 and "Close" in raw.columns:
        prices = raw["Close"].to_frame(tickers[0])
    else:
        raise ValueError("Yahoo response has an unsupported price-column structure")
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return validate_prices(prices, tickers)


def _atomic_write(path: Path, content: bytes) -> None:
    """Publish one cache file atomically without leaving partial files after failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_cache(manifest: Path, request: dict, key: str) -> pd.DataFrame:
    """Verify cached bytes and request metadata before parsing the price panel."""
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1 or metadata["request"] != request:
            raise ValueError("cache request or schema does not match")
        digest = metadata["csv_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid cache checksum")
        data_path = manifest.parent / f"{key}-{digest}.csv"
        content = data_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("cache checksum mismatch")
        # Python's float conversion preserves the stored 17-digit round trip without a
        # deprecated pandas parser precision option.
        prices = pd.read_csv(data_path, index_col="Date", parse_dates=["Date"], dtype=str)
        prices = validate_prices(prices, tuple(request["tickers"]))
        _check_window(prices, request["start"], request["end"])
        if (metadata["rows"] != len(prices)
                or metadata["first_date"] != prices.index[0].strftime("%Y-%m-%d")
                or metadata["last_date"] != prices.index[-1].strftime("%Y-%m-%d")):
            raise ValueError("cache dimensions or dates do not match")
        prices.attrs["yahoo_download"] = metadata
        return prices
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Yahoo cache {manifest}; use refresh=True to replace it") from exc


def download_prices(tickers: tuple[str, ...], start: str, end: str,
                    cache_dir: Path | str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download complete adjusted daily prices, or read their verified external cache.

    ``start`` is inclusive and ``end`` exclusive. Returned prices include Yahoo's split and
    dividend adjustments via ``auto_adjust=True``. Missing quotes fail explicitly. The cache
    records request options, yfinance version, UTC retrieval time and a SHA-256 checksum;
    existing cache data are never silently repaired or refreshed.
    """
    if isinstance(tickers, str):
        raise ValueError("tickers must be a sequence, not a single string")
    tickers = tuple(tickers)
    _check_tickers(tickers)
    start, end = _date(start, "start"), _date(end, "end")
    if start >= end:
        raise ValueError("start must precede the exclusive end date")
    directory = default_cache_directory() if cache_dir is None else external_directory(cache_dir)
    request = {"tickers": list(tickers), "start": start, "end": end,
               "options": dict(DOWNLOAD_OPTIONS)}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = directory / f"{key}.json"
    if manifest.exists() and not refresh:
        return _read_cache(manifest, request, key)

    import yfinance as yf

    directory.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(directory / "yfinance"))
    raw = yf.download(tickers=list(tickers), start=start, end=end, **DOWNLOAD_OPTIONS)
    prices = _adjusted_close(raw, tickers)
    _check_window(prices, start, end)
    content = prices.to_csv(index_label="Date", float_format="%.17g").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    metadata = {
        "schema_version": 1,
        "request": request,
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": digest,
        "rows": len(prices),
        "first_date": prices.index[0].strftime("%Y-%m-%d"),
        "last_date": prices.index[-1].strftime("%Y-%m-%d"),
    }
    # Content-addressed CSVs and an atomic manifest avoid mixed generations during refresh.
    _atomic_write(directory / f"{key}-{digest}.csv", content)
    _atomic_write(manifest, json.dumps(metadata, indent=2).encode("utf-8"))
    prices.attrs["yahoo_download"] = metadata
    return prices
