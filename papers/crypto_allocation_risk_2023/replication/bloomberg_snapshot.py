"""Create immutable Bloomberg-only snapshots for the crypto-allocation paper.

The published scripts mixed Bloomberg, Yahoo Finance, CoinMarketCap, and local
workbooks.  This module is the acquisition boundary for the 2026 update: every
source observation comes from Bloomberg Desktop API, while the transformations
(60/40 portfolio and proxy splices) retain the paper's stated conventions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import qis


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bloomberg"
DEFAULT_START_DATE = pd.Timestamp("1997-12-31")
DEFAULT_AS_OF = pd.Timestamp("2026-09-04")

EQUITY_SOURCES: dict[str, str] = {
    "SPY US Equity": "SPY",
    "IEF US Equity": "IEF",
    "PSP US Equity": "PE",
    "IYR US Equity": "IYR",
    "REET US Equity": "REET",
    "GSG US Equity": "GSG",
    "COMT US Equity": "COMT",
    "GLD US Equity": "Gold",
}

INDEX_SOURCES: dict[str, str] = {
    "XBTUSD Curncy": "BTC",
    "XETUSD Curncy": "ETH",
    "HFRXGL Index": "HFs",
    "HFRIMDT Index": "Macro",
    "NEIXCTA Index": "SG CTA",
}

RATE_SOURCES: dict[str, str] = {"GB3 Govt": "RiskFree"}

RAW_COLUMNS = (
    "BTC",
    "ETH",
    "SPY",
    "IEF",
    "HFs",
    "PE",
    "IYR",
    "REET",
    "Macro",
    "SG CTA",
    "GSG",
    "COMT",
    "Gold",
)

# The persisted source panel also retains the Bloomberg percentage quote behind
# the derived decimal risk-free series, so that every transformation can be
# independently recomputed from the immutable raw file.
SNAPSHOT_RAW_COLUMNS = RAW_COLUMNS + ("RiskFreePct",)

ANALYSIS_COLUMNS = (
    "60/40",
    "BTC",
    "ETH",
    "HFs",
    "PE",
    "RealEstate",
    "Macro",
    "SG CTA",
    "Commodities",
    "Gold",
)

# HFRIMDT is published monthly and had a 2026-07-31 observation at the 2026-09-04 cut.
# All other sources are expected to be daily, including seven-day crypto closes.
MAX_STALENESS_DAYS: dict[str, int] = {
    "BTC": 3,
    "ETH": 3,
    "SPY": 7,
    "IEF": 7,
    "HFs": 7,
    "PE": 7,
    "IYR": 7,
    "REET": 7,
    "Macro": 45,
    "SG CTA": 7,
    "GSG": 7,
    "COMT": 7,
    "Gold": 7,
    "RiskFree": 7,
}

MAX_INTERNAL_GAP_DAYS: dict[str, int] = {
    "BTC": 14,
    "ETH": 7,
    "SPY": 10,
    "IEF": 10,
    # HFRXGL is monthly in its earliest history (33-day gaps) and daily later.
    "HFs": 35,
    "PE": 10,
    "IYR": 10,
    "REET": 10,
    "Macro": 45,
    "SG CTA": 10,
    "GSG": 10,
    "COMT": 10,
    "Gold": 10,
    "RiskFree": 10,
}

MAX_PUBLISHED_WINDOW_GAP_DAYS = {
    **MAX_INTERNAL_GAP_DAYS,
    "HFs": 10,
}

PUBLISHED_HISTORY_START = pd.Timestamp("2010-07-19")

Fetcher = Callable[..., pd.DataFrame | None]


@dataclass(frozen=True)
class SnapshotPaths:
    """Paths belonging to one immutable Bloomberg snapshot."""

    root: Path
    raw: Path
    prices_legacy: Path
    prices_observed_eth: Path
    risk_free: Path
    manifest: Path


def snapshot_tag(as_of: pd.Timestamp | str) -> str:
    """Return the stable tag for an inclusive Bloomberg cutoff."""
    return f"bbg_{_date(as_of):%Y%m%d}"


def get_snapshot_paths(tag: str, data_path: Path = DATA_PATH) -> SnapshotPaths:
    """Return all paths for ``tag`` below the private paper data directory."""
    root = Path(data_path) / tag
    return SnapshotPaths(
        root=root,
        raw=root / "raw_bloomberg_px_last.csv",
        prices_legacy=root / "analysis_prices_legacy_eth_proxy.csv",
        prices_observed_eth=root / "analysis_prices_observed_eth.csv",
        risk_free=root / "risk_free_gb3.csv",
        manifest=root / "MANIFEST.json",
    )


def _date(value: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize an input to a timezone-naive calendar date."""
    value = pd.Timestamp(value)
    if value.tz is not None:
        value = value.tz_localize(None)
    return value.normalize()


def _normalize_panel(panel: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Normalize Bloomberg output without filling missing observations."""
    if panel is None or panel.empty:
        raise ValueError("Bloomberg returned no data")
    panel = panel.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).tz_localize(None).normalize()
    if panel.index.has_duplicates:
        duplicates = panel.index[panel.index.duplicated()].unique().strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Bloomberg returned duplicate dates: {duplicates[:5]}")
    panel = panel.sort_index().reindex(columns=list(columns))
    converted = panel.apply(pd.to_numeric, errors="coerce")
    malformed = converted.isna() & panel.notna()
    if malformed.any().any():
        bad_columns = malformed.any()[malformed.any()].index.tolist()
        raise ValueError(f"Bloomberg returned non-numeric observations for: {bad_columns}")
    panel = converted
    missing = [column for column in columns if column not in panel.columns or panel[column].isna().all()]
    if missing:
        raise ValueError(f"Bloomberg returned no observations for: {missing}")
    finite = pd.DataFrame(np.isfinite(panel), index=panel.index, columns=panel.columns)
    invalid = panel.notna() & ~finite
    if invalid.any().any():
        bad_columns = invalid.any()[invalid.any()].index.tolist()
        raise ValueError(f"Bloomberg returned non-finite observations for: {bad_columns}")
    return panel


def _locate_request_range(
    panel: pd.DataFrame,
    start_date: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Reject non-null observations outside the explicit Bloomberg request."""
    outside = (panel.index < start_date) | (panel.index > as_of)
    if outside.any() and panel.loc[outside].notna().any().any():
        raise ValueError("Bloomberg returned observations outside the requested date range")
    return panel.loc[start_date:as_of]


def _validate_source_coverage(panel: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, dict[str, object]]:
    """Validate levels and trailing coverage, returning manifest-ready statistics."""
    coverage: dict[str, dict[str, object]] = {}
    for column in panel.columns:
        series = panel[column].dropna()
        if column != "RiskFree" and (series <= 0.0).any():
            raise ValueError(f"Bloomberg source {column} contains non-positive levels")
        first = pd.Timestamp(series.index[0])
        last = pd.Timestamp(series.index[-1])
        gaps = series.index.to_series().diff().dt.days.dropna()
        maximum_gap = int(gaps.max()) if not gaps.empty else 0
        allowed_gap = MAX_INTERNAL_GAP_DAYS[column]
        if maximum_gap > allowed_gap:
            raise ValueError(
                f"Bloomberg source {column} has an internal gap of {maximum_gap} days "
                f"(maximum {allowed_gap})"
            )
        published_series = series.loc[PUBLISHED_HISTORY_START:]
        published_gaps = published_series.index.to_series().diff().dt.days.dropna()
        published_maximum_gap = int(published_gaps.max()) if not published_gaps.empty else 0
        published_allowed_gap = MAX_PUBLISHED_WINDOW_GAP_DAYS[column]
        if published_maximum_gap > published_allowed_gap:
            raise ValueError(
                f"Bloomberg source {column} has a {published_maximum_gap}-day gap in the "
                f"published analysis window (maximum {published_allowed_gap})"
            )
        staleness_days = int((as_of - last).days)
        max_staleness = MAX_STALENESS_DAYS[column]
        if staleness_days < 0:
            raise ValueError(f"Bloomberg source {column} extends beyond cutoff {as_of.date()}")
        if staleness_days > max_staleness:
            raise ValueError(
                f"Bloomberg source {column} is stale by {staleness_days} days "
                f"(maximum {max_staleness})"
            )
        coverage[column] = {
            "first_observation": first.strftime("%Y-%m-%d"),
            "last_observation": last.strftime("%Y-%m-%d"),
            "observations": int(series.size),
            "maximum_internal_gap_days": maximum_gap,
            "max_internal_gap_days": allowed_gap,
            "maximum_gap_since_published_start_days": published_maximum_gap,
            "max_gap_since_published_start_days": published_allowed_gap,
            "staleness_days": staleness_days,
            "max_staleness_days": max_staleness,
        }
    return coverage


def fetch_bloomberg_sources(
    start_date: pd.Timestamp | str = DEFAULT_START_DATE,
    as_of: pd.Timestamp | str = DEFAULT_AS_OF,
    fetcher: Fetcher | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict[str, object]]]:
    """Fetch and validate every raw input needed by the updated paper analysis."""
    start_date = _date(start_date)
    as_of = _date(as_of)
    if start_date > as_of:
        raise ValueError("start_date must not be after as_of")
    if fetcher is None:
        from bbg_fetch import fetch_field_timeseries_per_tickers

        fetcher = fetch_field_timeseries_per_tickers

    equity = fetcher(
        tickers=EQUITY_SOURCES,
        field="PX_LAST",
        CshAdjNormal=True,
        CshAdjAbnormal=True,
        CapChg=True,
        start_date=start_date,
        end_date=as_of,
        freq=None,
    )
    equity = _normalize_panel(equity, tuple(EQUITY_SOURCES.values()))
    equity = _locate_request_range(equity, start_date=start_date, as_of=as_of)

    indices = fetcher(
        tickers=INDEX_SOURCES,
        field="PX_LAST",
        CshAdjNormal=False,
        CshAdjAbnormal=False,
        CapChg=False,
        start_date=start_date,
        end_date=as_of,
        freq=None,
    )
    indices = _normalize_panel(indices, tuple(INDEX_SOURCES.values()))
    indices = _locate_request_range(indices, start_date=start_date, as_of=as_of)

    rate_start = max(start_date, pd.Timestamp("2003-12-31"))
    rates = fetcher(
        tickers=RATE_SOURCES,
        field="PX_LAST",
        CshAdjNormal=False,
        CshAdjAbnormal=False,
        CapChg=False,
        start_date=rate_start,
        end_date=as_of,
        freq=None,
    )
    rates = _normalize_panel(rates, tuple(RATE_SOURCES.values()))
    rates = _locate_request_range(rates, start_date=rate_start, as_of=as_of)

    raw = pd.concat(
        [indices, equity, rates["RiskFree"].rename("RiskFreePct")], axis=1
    ).sort_index().reindex(columns=list(SNAPSHOT_RAW_COLUMNS))
    coverage_panel = raw.rename(columns={"RiskFreePct": "RiskFree"})
    coverage = _validate_source_coverage(coverage_panel, as_of=as_of)

    risk_free = _to_business_series(raw["RiskFreePct"]).div(100.0)
    risk_free = risk_free.loc[:as_of].rename("RiskFree")
    if not risk_free.between(-0.01, 0.25).all():
        raise ValueError("GB3 risk-free yield is not a plausible decimal annual rate")
    return raw, risk_free, coverage


def _to_business_series(series: pd.Series) -> pd.Series:
    """Move non-business observations forward, then forward-fill a business-day grid."""
    series = series.dropna().sort_index().copy()
    series.index = pd.DatetimeIndex(
        [pd.offsets.BDay().rollforward(pd.Timestamp(date)) for date in series.index]
    )
    series = series.groupby(level=0).last()
    return series.asfreq("B").ffill().rename(series.name)


def build_analysis_prices(raw: pd.DataFrame, use_legacy_eth_proxy: bool) -> pd.DataFrame:
    """Build the paper panel from raw Bloomberg observations.

    ``use_legacy_eth_proxy=True`` reproduces the existing updated-paper convention:
    Bloomberg ETH is extended backwards with scaled Bloomberg BTC.  ``False`` retains
    missing values before the first observed Bloomberg ETH price.
    """
    raw = _normalize_panel(raw, RAW_COLUMNS)
    business_raw = pd.concat(
        [_to_business_series(raw[column]) for column in RAW_COLUMNS], axis=1
    ).reindex(columns=list(RAW_COLUMNS))
    balanced_legs = business_raw[["SPY", "IEF"]].dropna()
    balanced = qis.backtest_model_portfolio(
        prices=balanced_legs,
        weights=[0.6, 0.4],
        rebalancing_freq="QE",
        is_rebalanced_at_first_date=True,
        rebalancing_costs=0.005,
        ticker="60/40",
    ).nav.rename("60/40")

    real_estate = qis.bfill_timeseries(
        df_newer=business_raw["REET"], df_older=business_raw["IYR"], is_prices=True
    ).rename("RealEstate")
    commodities = qis.bfill_timeseries(
        df_newer=business_raw["COMT"], df_older=business_raw["GSG"], is_prices=True
    ).rename("Commodities")
    if use_legacy_eth_proxy:
        eth = qis.bfill_timeseries(
            df_newer=business_raw["ETH"],
            df_older=business_raw["BTC"],
            freq="B",
            is_prices=True,
        ).rename("ETH")
    else:
        eth = business_raw["ETH"].rename("ETH")

    prices = pd.concat(
        [
            balanced,
            business_raw["BTC"],
            eth,
            business_raw["HFs"],
            business_raw["PE"],
            real_estate,
            business_raw["Macro"],
            business_raw["SG CTA"],
            commodities,
            business_raw["Gold"],
        ],
        axis=1,
    ).sort_index()
    prices = prices.asfreq("B").ffill().reindex(columns=list(ANALYSIS_COLUMNS))
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Derived analysis panel must have a unique increasing index")
    observed = prices.notna()
    invalid = ((prices <= 0.0) & observed).any()
    if invalid.any():
        raise ValueError(f"Derived analysis series contain non-positive levels: {invalid[invalid].index.tolist()}")
    return prices


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_csv(frame: pd.DataFrame | pd.Series, path: Path) -> None:
    """Write one snapshot CSV with a stable date index representation."""
    frame.to_csv(path, index_label="date", date_format="%Y-%m-%d", float_format="%.12g")


def create_bloomberg_snapshot(
    start_date: pd.Timestamp | str = DEFAULT_START_DATE,
    as_of: pd.Timestamp | str = DEFAULT_AS_OF,
    data_path: Path = DATA_PATH,
    fetcher: Fetcher | None = None,
) -> SnapshotPaths:
    """Fetch, transform, and atomically publish one immutable Bloomberg snapshot."""
    start_date = _date(start_date)
    as_of = _date(as_of)
    tag = snapshot_tag(as_of)
    final_paths = get_snapshot_paths(tag=tag, data_path=data_path)
    if final_paths.root.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {final_paths.root}")

    raw, risk_free, coverage = fetch_bloomberg_sources(
        start_date=start_date,
        as_of=as_of,
        fetcher=fetcher,
    )
    prices_legacy = build_analysis_prices(raw=raw, use_legacy_eth_proxy=True).loc[:as_of]
    prices_observed = build_analysis_prices(raw=raw, use_legacy_eth_proxy=False).loc[:as_of]
    if start_date <= PUBLISHED_HISTORY_START <= as_of:
        if raw["BTC"].first_valid_index() != PUBLISHED_HISTORY_START:
            raise ValueError("Bloomberg BTC history does not match the published 2010-07-19 start")
        if prices_legacy.dropna().index[0] != PUBLISHED_HISTORY_START:
            raise ValueError("The complete legacy analysis panel does not cover the published start")

    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{tag}_", dir=data_path))
    temporary_paths = SnapshotPaths(
        root=temporary_root,
        raw=temporary_root / final_paths.raw.name,
        prices_legacy=temporary_root / final_paths.prices_legacy.name,
        prices_observed_eth=temporary_root / final_paths.prices_observed_eth.name,
        risk_free=temporary_root / final_paths.risk_free.name,
        manifest=temporary_root / final_paths.manifest.name,
    )
    try:
        _write_csv(raw, temporary_paths.raw)
        _write_csv(prices_legacy, temporary_paths.prices_legacy)
        _write_csv(prices_observed, temporary_paths.prices_observed_eth)
        _write_csv(risk_free, temporary_paths.risk_free)

        file_paths: Mapping[str, Path] = {
            temporary_paths.raw.name: temporary_paths.raw,
            temporary_paths.prices_legacy.name: temporary_paths.prices_legacy,
            temporary_paths.prices_observed_eth.name: temporary_paths.prices_observed_eth,
            temporary_paths.risk_free.name: temporary_paths.risk_free,
        }
        manifest = {
            "schema_version": 1,
            "tag": tag,
            "provider": "Bloomberg Desktop API",
            "requested_start_date": start_date.strftime("%Y-%m-%d"),
            "inclusive_as_of": as_of.strftime("%Y-%m-%d"),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "field": "PX_LAST",
            "sources": {
                "equities_adjusted": EQUITY_SOURCES,
                "indices_unadjusted": INDEX_SOURCES,
                "risk_free_unadjusted": RATE_SOURCES,
            },
            "coverage": coverage,
            "transformations": {
                "60/40": "SPY/IEF 60/40, quarterly rebalanced, 50 bp rebalance cost",
                "RealEstate": "REET extended backward with scaled IYR using qis.bfill_timeseries",
                "Commodities": "COMT extended backward with scaled GSG using qis.bfill_timeseries",
                "legacy_ETH": "XETUSD backfilled by scaled XBTUSD",
                "observed_ETH": "XETUSD observations only; pre-history remains missing",
                "risk_free": "GB3 Govt PX_LAST divided by 100",
                "macro_proxy": "HFRIMDT Index, the explicit 2024 update substitution",
                "macro_timestamp_convention": (
                    "Month-end observations become usable on the following business day; "
                    "no separate publication lag is applied for published-paper parity"
                ),
            },
            "packages": {
                "python": platform.python_version(),
                "pandas": _version("pandas"),
                "qis": _version("qis"),
                "optimalportfolios": _version("optimalportfolios"),
                "bbg-fetch": _version("bbg-fetch"),
                "blpapi": _version("blpapi"),
            },
            "files": {
                name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for name, path in file_paths.items()
            },
        }
        temporary_paths.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_bloomberg_snapshot(
            tag=tag,
            data_path=data_path,
            _paths=temporary_paths,
        )
        temporary_root.rename(final_paths.root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return final_paths


def verify_bloomberg_snapshot(
    tag: str,
    data_path: Path = DATA_PATH,
    _paths: SnapshotPaths | None = None,
) -> dict[str, object]:
    """Verify hashes, schemas, dates, and transformations for a saved snapshot."""
    paths = _paths or get_snapshot_paths(tag=tag, data_path=data_path)
    if not paths.manifest.is_file():
        raise FileNotFoundError(f"Missing Bloomberg manifest: {paths.manifest}")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Bloomberg snapshot manifest schema")
    if manifest.get("tag") != tag or manifest.get("provider") != "Bloomberg Desktop API":
        raise ValueError("Bloomberg snapshot manifest identity mismatch")
    expected_files = {
        paths.raw.name,
        paths.prices_legacy.name,
        paths.prices_observed_eth.name,
        paths.risk_free.name,
    }
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("Bloomberg snapshot file inventory mismatch")
    for filename, metadata in manifest["files"].items():
        path = paths.root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Bloomberg snapshot file: {path}")
        if _sha256(path) != metadata["sha256"]:
            raise ValueError(f"Bloomberg snapshot hash mismatch: {filename}")

    raw = pd.read_csv(paths.raw, index_col=0, parse_dates=True)
    legacy = pd.read_csv(paths.prices_legacy, index_col=0, parse_dates=True)
    observed = pd.read_csv(paths.prices_observed_eth, index_col=0, parse_dates=True)
    risk_free = pd.read_csv(paths.risk_free, index_col=0, parse_dates=True).iloc[:, 0]
    if tuple(raw.columns) != SNAPSHOT_RAW_COLUMNS:
        raise ValueError("Raw Bloomberg columns do not match the paper schema")
    raw_prices = _normalize_panel(raw, RAW_COLUMNS)
    for label, prices in (("legacy", legacy), ("observed", observed)):
        if tuple(prices.columns) != ANALYSIS_COLUMNS:
            raise ValueError(f"{label} analysis columns do not match the paper schema")
        if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
            raise ValueError(f"{label} analysis dates are not unique and increasing")
    if observed["ETH"].first_valid_index() != raw["ETH"].first_valid_index():
        raise ValueError("Observed-ETH panel contains synthetic pre-history")
    if legacy["ETH"].first_valid_index() != raw["BTC"].first_valid_index():
        raise ValueError("Legacy-ETH panel does not reproduce the BTC proxy start")
    if not risk_free.dropna().between(-0.01, 0.25).all():
        raise ValueError("Saved risk-free data are not decimal annual rates")
    if risk_free.index.has_duplicates or not risk_free.index.is_monotonic_increasing:
        raise ValueError("Saved risk-free dates are not unique and increasing")
    as_of = _date(manifest["inclusive_as_of"])
    if any(frame.index.max() > as_of for frame in (raw, legacy, observed, risk_free.to_frame())):
        raise ValueError("Snapshot contains observations beyond its inclusive cutoff")
    coverage_panel = raw.rename(columns={"RiskFreePct": "RiskFree"})
    recomputed_coverage = _validate_source_coverage(coverage_panel, as_of=as_of)
    if recomputed_coverage != manifest.get("coverage"):
        raise ValueError("Bloomberg snapshot coverage metadata mismatch")

    expected_legacy = build_analysis_prices(
        raw=raw_prices, use_legacy_eth_proxy=True
    ).loc[:as_of]
    expected_observed = build_analysis_prices(
        raw=raw_prices, use_legacy_eth_proxy=False
    ).loc[:as_of]
    pd.testing.assert_frame_equal(
        legacy,
        expected_legacy,
        check_freq=False,
        check_names=False,
        rtol=1e-9,
        atol=1e-10,
        obj="saved legacy-ETH analysis panel",
    )
    pd.testing.assert_frame_equal(
        observed,
        expected_observed,
        check_freq=False,
        check_names=False,
        rtol=1e-9,
        atol=1e-10,
        obj="saved observed-ETH analysis panel",
    )
    expected_risk_free = _to_business_series(raw["RiskFreePct"]).div(100.0).loc[:as_of]
    expected_risk_free = expected_risk_free.rename("RiskFree")
    pd.testing.assert_series_equal(
        risk_free,
        expected_risk_free,
        check_freq=False,
        check_names=False,
        rtol=1e-9,
        atol=1e-10,
        obj="saved decimal risk-free series",
    )
    requested_start = _date(manifest["requested_start_date"])
    if requested_start <= PUBLISHED_HISTORY_START <= as_of:
        if raw["BTC"].first_valid_index() != PUBLISHED_HISTORY_START:
            raise ValueError("Saved BTC history does not match the published start")
        if legacy.dropna().index[0] != PUBLISHED_HISTORY_START:
            raise ValueError("Saved legacy panel does not cover the published start")
    return manifest


def load_bloomberg_prices(
    tag: str,
    use_legacy_eth_proxy: bool = True,
    data_path: Path = DATA_PATH,
) -> pd.DataFrame:
    """Load a verified Bloomberg analysis panel."""
    verify_bloomberg_snapshot(tag=tag, data_path=data_path)
    paths = get_snapshot_paths(tag=tag, data_path=data_path)
    path = paths.prices_legacy if use_legacy_eth_proxy else paths.prices_observed_eth
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_bloomberg_risk_free(tag: str, data_path: Path = DATA_PATH) -> pd.Series:
    """Load the verified Bloomberg GB3 decimal annual yield."""
    verify_bloomberg_snapshot(tag=tag, data_path=data_path)
    path = get_snapshot_paths(tag=tag, data_path=data_path).risk_free
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename("RiskFree")
