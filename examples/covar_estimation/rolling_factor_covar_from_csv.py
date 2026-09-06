"""Build a rolling factor risk model from a portable CSV input bundle.

This example has two deliberately separate stages:

1. ``fetch`` downloads free Yahoo data and writes six CSV files.
2. ``load`` reads those files, constructs ``qis.FactorsData`` and
   ``qis.FxRatesData``, and estimates ``RollingFactorCovarData``.

The load stage never imports or calls yfinance. A user who receives the CSV
directory therefore needs only qis, factorlasso and optimalportfolios::

    python -m examples.covar_estimation.rolling_factor_covar_from_csv fetch `
        --data-dir C:\\data\\risk_model_inputs
    python -m examples.covar_estimation.rolling_factor_covar_from_csv load `
        --data-dir C:\\data\\risk_model_inputs

Running without a mode performs both stages using the ignored ``tmp/`` tree::

    python -m examples.covar_estimation.rolling_factor_covar_from_csv

The Yahoo series are pedagogical proxies, not the proprietary MATF factor
definitions. The four asset-class proxies are fully FX-hedged into CHF and a
separate factor carries USD/CHF risk. Replace ``futures_risk_factors.csv``
with the delivered MATF factor NAV panel and update the ``factor_names`` row
in ``risk_model_settings.csv`` to validate its ordered column and currency
basis contract. The CHF short rate in the Yahoo bundle is also illustrative:
Yahoo provides the USD 13-week Treasury-bill yield but not a reliable CHF
three-month history. Production inputs must replace that curve. Review the
applicable data-provider terms before redistributing downloaded Yahoo data;
generated CSVs stay in ``tmp/`` by default and are not repository artifacts.

CSV bundle
----------
``futures_risk_factors.csv``
    Factor price/NAV levels consumed by ``qis.FactorsData.load``.
``fx_hedging_data_fx_spots.csv``
    FX spots in the qis convention: USD per one unit of each currency.
``fx_hedging_data_domestic_rates.csv``
    Annualised domestic short rates expressed as decimal fractions.
``asset_prices.csv``
    Native-currency adjusted asset prices.
``asset_metadata.csv``
    Per-asset currency, hedge ratio and return frequency.
``risk_model_settings.csv``
    Reference currency, return convention and LASSO/HCGL calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import qis

import optimalportfolios as opt


FACTOR_FILE = "futures_risk_factors"
FX_FILE = "fx_hedging_data"
ASSET_PRICES_FILE = "asset_prices"
ASSET_METADATA_FILE = "asset_metadata"
SETTINGS_FILE = "risk_model_settings"

FETCH_START = "2007-12-31"
FETCH_END = "2026-01-01"  # yfinance treats end as exclusive
REFERENCE_CCY = "CHF"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "tmp" / "yahoo_factor_risk_model"

FACTOR_TICKERS = {
    "SPY": "Equity",
    "TLT": "Rates",
    "LQD": "Credit",
    "GLD": "Commodities",
}

ASSET_SPECS = {
    "QQQ": {"currency": "USD", "hedge_ratio": 0.0, "return_frequency": "ME"},
    "EFA": {"currency": "USD", "hedge_ratio": 0.0, "return_frequency": "ME"},
    "EEM": {"currency": "USD", "hedge_ratio": 0.0, "return_frequency": "ME"},
    "IEF": {"currency": "USD", "hedge_ratio": 1.0, "return_frequency": "ME"},
    "HYG": {"currency": "USD", "hedge_ratio": 1.0, "return_frequency": "ME"},
    "VNQ": {"currency": "USD", "hedge_ratio": 0.0, "return_frequency": "ME"},
    "GSG": {"currency": "USD", "hedge_ratio": 0.0, "return_frequency": "ME"},
}

# Yahoo's CHF quote is CHF per USD. QIS stores USD per CHF, so the fetch stage inverts it.
FX_TICKER = "CHF=X"
USD_RATE_TICKER = "^IRX"
FX_FACTOR = "Fx"


@dataclass(frozen=True)
class RiskModelSettings:
    """Risk-model calibration loaded from ``risk_model_settings.csv``."""

    reference_ccy: str
    is_log_returns: bool
    is_excess_returns: bool
    factor_returns_freq: str
    factor_names: tuple[str, ...]
    factor_covar_span: int
    rebalancing_freq: str
    lasso_model_type: str
    reg_lambda: float
    beta_span: int
    warmup_period: int
    demean: bool
    solver: str
    estimation_start: pd.Timestamp
    estimation_end: pd.Timestamp

    @classmethod
    def yahoo_demo(cls) -> "RiskModelSettings":
        """Return the compact HCGL calibration used by the Yahoo example."""
        return cls(
            reference_ccy=REFERENCE_CCY,
            is_log_returns=True,
            is_excess_returns=False,
            factor_returns_freq="ME",
            factor_names=tuple(FACTOR_TICKERS.values()) + (FX_FACTOR,),
            factor_covar_span=36,
            rebalancing_freq="YE",
            lasso_model_type="HIERARCHICAL_CLUSTER_GROUP_LASSO",
            reg_lambda=1.0e-5,
            beta_span=36,
            warmup_period=36,
            demean=True,
            solver="CLARABEL",
            estimation_start=pd.Timestamp("2019-12-31"),
            estimation_end=pd.Timestamp("2025-12-31"),
        )

    def to_frame(self) -> pd.DataFrame:
        """Serialise the settings to a human-editable one-column frame."""
        values = {
            "reference_ccy": self.reference_ccy,
            "is_log_returns": self.is_log_returns,
            "is_excess_returns": self.is_excess_returns,
            "factor_returns_freq": self.factor_returns_freq,
            "factor_names": "|".join(self.factor_names),
            "factor_covar_span": self.factor_covar_span,
            "rebalancing_freq": self.rebalancing_freq,
            "lasso_model_type": self.lasso_model_type,
            "reg_lambda": self.reg_lambda,
            "beta_span": self.beta_span,
            "warmup_period": self.warmup_period,
            "demean": self.demean,
            "solver": self.solver,
            "estimation_start": self.estimation_start.date().isoformat(),
            "estimation_end": self.estimation_end.date().isoformat(),
        }
        return pd.Series(values, name="value").to_frame().rename_axis("setting")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "RiskModelSettings":
        """Parse settings previously written by :meth:`to_frame`."""
        if "value" not in frame.columns:
            raise ValueError("risk_model_settings.csv must contain a 'value' column")
        values = frame["value"].to_dict()
        required = set(cls.__dataclass_fields__)
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(f"risk_model_settings.csv is missing settings: {missing}")
        return cls(
            reference_ccy=str(values["reference_ccy"]),
            is_log_returns=_parse_bool(values["is_log_returns"]),
            is_excess_returns=_parse_bool(values["is_excess_returns"]),
            factor_returns_freq=str(values["factor_returns_freq"]),
            factor_names=tuple(str(values["factor_names"]).split("|")),
            factor_covar_span=int(float(values["factor_covar_span"])),
            rebalancing_freq=str(values["rebalancing_freq"]),
            lasso_model_type=str(values["lasso_model_type"]),
            reg_lambda=float(values["reg_lambda"]),
            beta_span=int(float(values["beta_span"])),
            warmup_period=int(float(values["warmup_period"])),
            demean=_parse_bool(values["demean"]),
            solver=str(values["solver"]),
            estimation_start=pd.Timestamp(values["estimation_start"]),
            estimation_end=pd.Timestamp(values["estimation_end"]),
        )


@dataclass(frozen=True)
class CsvRiskModelInputs:
    """All model inputs reconstructed from one CSV directory."""

    factors_data: qis.FactorsData
    fx_rates_data: qis.FxRatesData
    asset_prices: pd.DataFrame
    asset_metadata: pd.DataFrame
    settings: RiskModelSettings


def _parse_bool(value: object) -> bool:
    """Parse a CSV scalar as a strict boolean."""
    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _download_close(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close levels while keeping yfinance fetch-only."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Yahoo fetching needs the optional data dependencies. Install "
            "optimalportfolios[data], then rerun the fetch stage."
        ) from exc

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
        # This example already runs beside other examples in CI. Avoid nested request fan-out.
        threads=False,
        multi_level_index=True,
    )
    if raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise ValueError("Yahoo returned no adjusted close data")

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    missing = [ticker for ticker in tickers if ticker not in close.columns]
    if missing:
        raise ValueError(f"Yahoo returned no Close column for: {missing}")
    empty = [ticker for ticker in tickers if close[ticker].dropna().empty]
    if empty:
        raise ValueError(f"Yahoo returned only missing Close values for: {empty}")

    close = close.reindex(columns=list(tickers)).sort_index()
    close.index = pd.DatetimeIndex(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.loc[~close.index.duplicated(keep="last")]
    return close.rename_axis(index="date", columns=None)


def fetch_and_save_yahoo_csvs(
    data_dir: Path,
    start: str = FETCH_START,
    end: str = FETCH_END,
) -> None:
    """Fetch Yahoo proxies and write the complete portable CSV bundle.

    Args:
        data_dir: Destination directory. Existing bundle files are replaced.
        start: Inclusive Yahoo download start date.
        end: Exclusive Yahoo download end date.
    """
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(ASSET_SPECS) + list(FACTOR_TICKERS) + [FX_TICKER, USD_RATE_TICKER]
    tickers = list(dict.fromkeys(tickers))
    close = _download_close(tickers=tickers, start=start, end=end)

    # Use one common business-day grid so every delivered CSV has identical date support.
    close = close.asfreq("B").ffill().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(close.index) < 5 * 252:
        raise ValueError(
            f"only {len(close.index)} common daily observations were downloaded; "
            "at least five years are required"
        )

    asset_prices = close[list(ASSET_SPECS)].copy()
    asset_prices.index.name = "date"

    fx_spots = pd.DataFrame(index=close.index)
    fx_spots["USD"] = 1.0
    fx_spots["CHF"] = 1.0 / close[FX_TICKER]
    fx_spots.index.name = "date"

    usd_rate = close[USD_RATE_TICKER] / 100.0
    domestic_rates = pd.DataFrame(
        {
            "USD": usd_rate,
            # Illustrative only. Replace with a real CHF three-month rate in production.
            "CHF": usd_rate - 0.01,
        },
        index=close.index,
    )
    domestic_rates.index.name = "date"

    settings = RiskModelSettings.yahoo_demo()
    # Fully hedge the asset-class proxies into CHF and keep FX as its own factor. An unhedged
    # USD asset can then load on both its asset class and USD/CHF, while a hedged asset need not
    # acquire an offsetting FX beta. The delivered FactorsData CSV is already a self-contained
    # CHF-basis NAV panel, so the intern does not need Yahoo or ROSAA.
    fx_rates_data = qis.FxRatesData(
        fx_spots=fx_spots.copy(),
        domestic_rates=domestic_rates.copy(),
    )
    native_factor_prices = close[list(FACTOR_TICKERS)].rename(columns=FACTOR_TICKERS)
    factor_names = native_factor_prices.columns
    factor_prices, _ = fx_rates_data.compute_returns_in_reference_ccy(
        asset_prices=native_factor_prices,
        hedge_ratios=pd.Series(1.0, index=factor_names),
        local_ccys=pd.Series("USD", index=factor_names),
        reference_ccy=REFERENCE_CCY,
        freq=settings.factor_returns_freq,
        is_log_returns=True,
        is_excess_returns=False,
    )
    factor_prices[FX_FACTOR] = fx_rates_data.get_fx_total_return_nav(
        local_ccy="USD",
        reference_ccy=REFERENCE_CCY,
        freq=settings.factor_returns_freq,
    ).reindex(factor_prices.index).ffill()
    factor_prices = factor_prices.dropna(how="any")
    factor_prices.index.name = "date"
    qis.FactorsData(factors_prices=factor_prices)

    asset_metadata = pd.DataFrame.from_dict(ASSET_SPECS, orient="index")
    asset_metadata.index.name = "asset"

    qis.save_df_to_csv(
        df=factor_prices,
        file_name=FACTOR_FILE,
        local_path=str(data_dir),
    )
    qis.save_df_dict_to_csv(
        datasets={"fx_spots": fx_spots, "domestic_rates": domestic_rates},
        file_name=FX_FILE,
        local_path=str(data_dir),
    )
    qis.save_df_to_csv(
        df=asset_prices,
        file_name=ASSET_PRICES_FILE,
        local_path=str(data_dir),
    )
    qis.save_df_to_csv(
        df=asset_metadata,
        file_name=ASSET_METADATA_FILE,
        local_path=str(data_dir),
    )
    qis.save_df_to_csv(
        df=settings.to_frame(),
        file_name=SETTINGS_FILE,
        local_path=str(data_dir),
    )

    print(f"Wrote Yahoo CSV bundle to {data_dir}")
    for path in _expected_paths(data_dir):
        print(f"  {path.name}")


def _expected_paths(data_dir: Path) -> tuple[Path, ...]:
    """Return every file required by the CSV-only load stage."""
    return (
        data_dir / f"{FACTOR_FILE}.csv",
        data_dir / f"{FX_FILE}_fx_spots.csv",
        data_dir / f"{FX_FILE}_domestic_rates.csv",
        data_dir / f"{ASSET_PRICES_FILE}.csv",
        data_dir / f"{ASSET_METADATA_FILE}.csv",
        data_dir / f"{SETTINGS_FILE}.csv",
    )


def _require_csv_bundle(data_dir: Path) -> None:
    """Fail with one actionable error when a bundle is incomplete."""
    missing = [path.name for path in _expected_paths(data_dir) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"CSV bundle in {data_dir} is incomplete; missing {missing}. "
            "Run this module in fetch mode or supply the delivered files."
        )


def load_inputs_from_csv(data_dir: Path) -> CsvRiskModelInputs:
    """Load every market-data and configuration object from CSV files."""
    data_dir = Path(data_dir).resolve()
    _require_csv_bundle(data_dir)

    factors_data = qis.FactorsData.load(
        local_path=str(data_dir),
        file_name=FACTOR_FILE,
    )
    fx_rates_data = qis.FxRatesData.load(local_path=str(data_dir))
    asset_prices = qis.load_df_from_csv(
        file_name=ASSET_PRICES_FILE,
        local_path=str(data_dir),
    ).sort_index()
    asset_metadata = qis.load_df_from_csv(
        file_name=ASSET_METADATA_FILE,
        local_path=str(data_dir),
        parse_dates=False,
    )
    settings_frame = qis.load_df_from_csv(
        file_name=SETTINGS_FILE,
        local_path=str(data_dir),
        parse_dates=False,
    )
    settings = RiskModelSettings.from_frame(settings_frame)

    factor_names = tuple(str(name) for name in factors_data.get_prices().columns)
    if not settings.factor_names or len(set(settings.factor_names)) != len(
        settings.factor_names
    ):
        raise ValueError("risk_model_settings.csv factor_names must be non-empty and unique")
    if factor_names != settings.factor_names:
        raise ValueError(
            "futures_risk_factors.csv columns must match the ordered factor_names setting; "
            f"actual={factor_names}, expected={settings.factor_names}"
        )

    for name, frame in {
        "futures_risk_factors.csv": factors_data.get_prices(),
        "asset_prices.csv": asset_prices,
        "fx_hedging_data_fx_spots.csv": fx_rates_data.fx_spots,
        "fx_hedging_data_domestic_rates.csv": fx_rates_data.domestic_rates,
    }.items():
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex")
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError(f"{name} dates must be sorted and unique")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain only finite numeric values")

    if (asset_prices <= 0.0).any().any():
        raise ValueError("asset_prices.csv must contain strictly positive prices")
    if (factors_data.get_prices() <= 0.0).any().any():
        raise ValueError("futures_risk_factors.csv must contain strictly positive NAVs")
    if (fx_rates_data.fx_spots <= 0.0).any().any():
        raise ValueError("fx_hedging_data_fx_spots.csv must contain strictly positive spots")
    if not settings.is_log_returns:
        raise ValueError(
            "FactorCovarEstimator computes log factor returns; set is_log_returns=true"
        )

    required_metadata = {"currency", "hedge_ratio", "return_frequency"}
    missing_metadata = sorted(required_metadata.difference(asset_metadata.columns))
    if missing_metadata:
        raise ValueError(f"asset_metadata.csv is missing columns: {missing_metadata}")

    missing_assets = [asset for asset in asset_prices.columns if asset not in asset_metadata.index]
    extra_assets = [asset for asset in asset_metadata.index if asset not in asset_prices.columns]
    if missing_assets or extra_assets:
        raise ValueError(
            "asset_prices.csv and asset_metadata.csv must have the same assets; "
            f"missing metadata={missing_assets}, extra metadata={extra_assets}"
        )
    asset_metadata = asset_metadata.reindex(asset_prices.columns)
    asset_metadata["hedge_ratio"] = pd.to_numeric(asset_metadata["hedge_ratio"])
    if not asset_metadata["hedge_ratio"].between(0.0, 1.0).all():
        raise ValueError("asset_metadata.csv hedge_ratio values must be between zero and one")
    for frequency in asset_metadata["return_frequency"].astype(str).unique():
        try:
            pd.tseries.frequencies.to_offset(frequency)
        except ValueError as exc:
            raise ValueError(
                f"asset_metadata.csv contains invalid return_frequency={frequency!r}"
            ) from exc

    required_ccys = set(asset_metadata["currency"].astype(str)) | {settings.reference_ccy}
    missing_spots = sorted(required_ccys.difference(fx_rates_data.fx_spots.columns))
    missing_rates = sorted(required_ccys.difference(fx_rates_data.domestic_rates.columns))
    if missing_spots or missing_rates:
        raise ValueError(
            "FX CSVs do not cover all asset/reference currencies; "
            f"missing spots={missing_spots}, missing rates={missing_rates}"
        )

    common_last_date = min(
        asset_prices.index[-1],
        factors_data.get_prices().index[-1],
        fx_rates_data.fx_spots.index[-1],
        fx_rates_data.domestic_rates.index[-1],
    )
    if settings.estimation_start > settings.estimation_end:
        raise ValueError("estimation_start must not be after estimation_end")
    if settings.estimation_end > common_last_date:
        raise ValueError(
            f"estimation_end={settings.estimation_end.date()} exceeds the common CSV end "
            f"date {common_last_date.date()}"
        )

    return CsvRiskModelInputs(
        factors_data=factors_data,
        fx_rates_data=fx_rates_data,
        asset_prices=asset_prices,
        asset_metadata=asset_metadata,
        settings=settings,
    )


def _make_estimator(settings: RiskModelSettings) -> opt.FactorCovarEstimator:
    """Construct the configured public OP estimator from CSV settings."""
    try:
        model_type = opt.LassoModelType[settings.lasso_model_type]
    except KeyError as exc:
        allowed = [model.name for model in opt.LassoModelType]
        raise ValueError(
            f"unknown lasso_model_type={settings.lasso_model_type!r}; choose from {allowed}"
        ) from exc

    lasso_model = opt.LassoModel(
        model_type=model_type,
        reg_lambda=settings.reg_lambda,
        span=settings.beta_span,
        warmup_period=settings.warmup_period,
        demean=settings.demean,
        solver=settings.solver,
    )
    return opt.FactorCovarEstimator(
        rebalancing_freq=settings.rebalancing_freq,
        lasso_model=lasso_model,
        factor_returns_freq=settings.factor_returns_freq,
        factor_covar_span=settings.factor_covar_span,
        demean=settings.demean,
    )


def _verify_rolling_decomposition(rolling: opt.RollingFactorCovarData) -> float:
    """Verify every dated covariance against beta-factor-residual arithmetic."""
    if len(rolling) == 0:
        raise ValueError("rolling covariance fit returned no snapshots")

    max_error = 0.0
    residual_column = opt.VarianceColumns.RESIDUAL_VARS.value
    for date in rolling.dates:
        snapshot = rolling.data[date]
        if snapshot.estimation_date is None or pd.Timestamp(snapshot.estimation_date) != date:
            raise ValueError(f"snapshot at {date} has estimation_date={snapshot.estimation_date}")

        betas = snapshot.y_betas
        factor_covar = snapshot.x_covar.reindex(index=betas.columns, columns=betas.columns)
        residual_vars = snapshot.y_variances[residual_column].reindex(betas.index)
        if factor_covar.isna().any().any() or residual_vars.isna().any():
            raise ValueError(f"component labels do not align at {date}")

        expected = (
            betas.to_numpy()
            @ factor_covar.to_numpy()
            @ betas.to_numpy().T
            + np.diag(residual_vars.to_numpy())
        )
        actual = snapshot.get_y_covar().reindex(index=betas.index, columns=betas.index)
        if not np.isfinite(actual.to_numpy()).all():
            raise ValueError(f"asset covariance contains non-finite values at {date}")
        np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1.0e-12, atol=1.0e-14)
        max_error = max(max_error, float(np.max(np.abs(actual.to_numpy() - expected))))

        min_eigenvalue = float(np.linalg.eigvalsh(actual.to_numpy()).min())
        if min_eigenvalue < -1.0e-10:
            raise ValueError(
                f"asset covariance is not positive semidefinite at {date}: "
                f"minimum eigenvalue={min_eigenvalue}"
            )
    return max_error


def fit_rolling_risk_model_from_csv(
    data_dir: Path,
) -> tuple[opt.RollingFactorCovarData, qis.RiskModel]:
    """Load the CSV bundle and return the rolling factor data and qis model."""
    inputs = load_inputs_from_csv(data_dir=data_dir)
    metadata = inputs.asset_metadata
    settings = inputs.settings

    asset_returns_dict = inputs.fx_rates_data.compute_fx_adjusted_returns(
        prices=inputs.asset_prices,
        hedge_ratios=metadata["hedge_ratio"],
        local_ccys=metadata["currency"].astype(str),
        reference_ccy=settings.reference_ccy,
        freq=metadata["return_frequency"].astype(str),
        is_log_returns=settings.is_log_returns,
        is_excess_returns=settings.is_excess_returns,
    )

    rolling = _make_estimator(settings).fit_rolling_factor_covars(
        risk_factor_prices=inputs.factors_data.get_prices(),
        asset_returns_dict=asset_returns_dict,
        assets=inputs.asset_prices.columns,
        time_period=qis.TimePeriod(
            start=settings.estimation_start,
            end=settings.estimation_end,
        ),
    )
    max_error = _verify_rolling_decomposition(rolling)
    risk_model = opt.build_risk_model(rolling)

    latest_date = rolling.dates[-1]
    latest = rolling.get_latest()
    equal_weights = pd.Series(
        1.0 / len(latest.y_betas.index),
        index=latest.y_betas.index,
    )
    exposures = risk_model.compute_exposures_at_date(equal_weights, latest_date)
    residual_vols = np.sqrt(
        latest.y_variances[opt.VarianceColumns.RESIDUAL_VARS.value]
    ).rename("residual_vol")

    print(f"Loaded CSV bundle from {Path(data_dir).resolve()}")
    print(f"Rolling snapshots: {len(rolling)}")
    print(f"Latest snapshot: {latest_date.date()}")
    print(f"Maximum covariance reconstruction error: {max_error:.3e}")
    print("\nLatest factor loadings (assets x factors):")
    print(latest.y_betas.round(3).to_string())
    print("\nLatest annualised factor covariance:")
    print(latest.x_covar.round(6).to_string())
    print("\nLatest annualised residual volatilities:")
    print(residual_vols.round(4).to_string())
    print("\nEqual-weight portfolio factor exposures:")
    print(exposures.round(3).to_string())

    return rolling, risk_model


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run Yahoo fetch, CSV-only loading, or both stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("fetch", "load", "all"),
        nargs="?",
        default="all",
        help="fetch CSVs, load and fit from existing CSVs, or do both (default)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"CSV bundle directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args(argv)

    if args.mode in {"fetch", "all"}:
        fetch_and_save_yahoo_csvs(data_dir=args.data_dir)
    if args.mode in {"load", "all"}:
        fit_rolling_risk_model_from_csv(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
