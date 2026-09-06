"""Run the Bloomberg-only update of the published crypto-allocation analysis.

This runner deliberately uses the same four method labels and parameter values as
the 2024 update while making the data snapshot, ETH proxy choice, implementation
lag, output cutoff, and current package versions explicit in machine-readable
manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis
from qis import PerfParams, PerfStat, TimePeriod

from optimalportfolios import Constraints, EwmaCovarEstimator, rolling_maximise_diversification
from optimalportfolios.reports.marginal_backtest import (
    OptimisationParams,
    OptimisationType,
    backtest_marginal_optimal_portfolios,
)
from papers.crypto_allocation_risk_2023.replication.bloomberg_snapshot import (
    DEFAULT_AS_OF,
    create_bloomberg_snapshot,
    get_snapshot_paths,
    load_bloomberg_prices,
    load_bloomberg_risk_free,
    snapshot_tag,
    verify_bloomberg_snapshot,
)
from papers.crypto_allocation_risk_2023.replication.load_prices import OUTPUT_PATH
from papers.crypto_allocation_risk_2023.replication.parity_2024 import (
    Parity2024Config,
    backtest_marginal_2024,
    monthly_log_returns_2024,
    require_parity_dependencies,
)


METHODS = (
    OptimisationType.ERC,
    OptimisationType.MAX_DIV,
    OptimisationType.MAX_SHARPE,
    OptimisationType.MIXTURE,
)

EngineId = Literal["published_2024", "current_v7_1"]
PUBLISHED_ENGINE: EngineId = "published_2024"
CURRENT_ENGINE: EngineId = "current_v7_1"

PARAMS = OptimisationParams(
    first_asset_target_weight=0.75,
    rebalancing_freq="QE",
    roll_window=60,
    returns_freq="ME",
    span=30,
    carra=0.5,
    n_mixures=3,
    rebalancing_costs=0.005,
    weight_implementation_lag=1,
    marginal_asset_ew_weight=0.02,
)

PERFORMANCE_COLUMNS = (
    "Total",
    "P.a. return",
    "P.a. excess return",
    "Vol",
    "Log Ex Sharpe",
    "Max DD",
    "Skewness",
    "Alpha",
    "Beta",
    "R2",
)


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


def _input_weights(portfolio: qis.PortfolioData) -> pd.DataFrame:
    """Return the optimiser's target weights rather than drifted daily weights."""
    weights = portfolio.get_input_weights()
    if not isinstance(weights, pd.DataFrame):
        weights = portfolio.get_weights(freq=None)
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("Expected a DataFrame of rolling target weights")
    return weights.dropna(how="all")


def summarize_crypto_weights(weights: pd.DataFrame, crypto_asset: str) -> dict[str, object]:
    """Return the allocation statistics used in the original paper exhibits."""
    if crypto_asset not in weights:
        raise KeyError(f"Missing {crypto_asset} in target weights")
    if weights.empty or weights.index.has_duplicates or not weights.index.is_monotonic_increasing:
        raise ValueError("Target weights must have nonempty, unique, increasing dates")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Target weights contain non-finite values")
    series = weights[crypto_asset].dropna()
    if series.empty:
        raise ValueError(f"No target weights for {crypto_asset}")
    return {
        "first_weight_date": series.index[0].strftime("%Y-%m-%d"),
        "last_weight_date": series.index[-1].strftime("%Y-%m-%d"),
        "observations": int(series.size),
        "min_crypto_weight": float(series.min()),
        "median_crypto_weight": float(series.median()),
        "mean_crypto_weight": float(series.mean()),
        "max_crypto_weight": float(series.max()),
        "last_crypto_weight": float(series.iloc[-1]),
        "max_weight_sum_error": float((weights.sum(axis=1) - 1.0).abs().max()),
        "minimum_portfolio_weight": float(weights.min().min()),
        "maximum_portfolio_weight": float(weights.max().max()),
    }


def _validate_target_weights(
    weights: pd.DataFrame,
    expected_columns: pd.Index,
    label: str,
    fixed_first_weight: float | None = None,
) -> dict[str, float]:
    """Validate one target-weight panel and return aggregate audit metrics."""
    if weights.empty:
        raise AssertionError(f"{label}: target weights are empty")
    if not weights.columns.equals(expected_columns):
        raise AssertionError(f"{label}: target-weight columns do not match the universe")
    if weights.index.has_duplicates or not weights.index.is_monotonic_increasing:
        raise AssertionError(f"{label}: target-weight dates are not unique and increasing")
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"{label}: target weights contain non-finite values")
    sum_error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
    minimum = float(values.min())
    maximum = float(values.max())
    if sum_error > 2e-5:
        raise AssertionError(f"{label}: maximum weight-sum error is {sum_error}")
    if minimum < -2e-5 or maximum > 1.0 + 2e-5:
        raise AssertionError(f"{label}: long-only bounds failed ({minimum}, {maximum})")
    if fixed_first_weight is not None:
        first_error = float(np.max(np.abs(values[:, 0] - fixed_first_weight)))
        if first_error > 2e-5:
            raise AssertionError(
                f"{label}: first sleeve differs from {fixed_first_weight:.2%} by {first_error}"
            )
    else:
        first_error = 0.0
    return {
        "max_weight_sum_error": sum_error,
        "minimum_weight": minimum,
        "maximum_weight": maximum,
        "max_fixed_first_weight_error": first_error,
    }


def _periods(as_of: pd.Timestamp) -> dict[str, TimePeriod]:
    """Return full, published, and genuinely out-of-sample reporting windows."""
    end = as_of.strftime("%d%b%Y")
    return {
        "full_since_2016Q1": TimePeriod("31Mar2016", end),
        "published_to_2023Q2": TimePeriod("31Mar2016", "30Jun2023"),
        "post_published_sample": TimePeriod("30Jun2023", end),
        "post_2024_update": TimePeriod("16Aug2024", end),
        "since_2021": TimePeriod("31Dec2020", end),
    }


def _common_reporting_start(prices: pd.DataFrame) -> pd.Timestamp:
    """Return the first quarter after every method has 60 monthly returns."""
    config = Parity2024Config(end_date=prices.index[-1])
    returns = monthly_log_returns_2024(prices=prices, config=config)
    if len(returns) < config.roll_window:
        raise ValueError(
            f"The analysis requires {config.roll_window} monthly returns; found {len(returns)}"
        )
    full_window_end = pd.Timestamp(returns.index[config.roll_window - 1])
    first_common_quarter = full_window_end.to_period("Q").end_time.normalize()
    published_start = pd.Timestamp(config.reporting_start)
    return max(published_start, first_common_quarter)


def _run_pair(
    prices: pd.DataFrame,
    crypto_asset: str,
    method: OptimisationType,
    universe: str,
    as_of: pd.Timestamp,
    engine_id: EngineId = PUBLISHED_ENGINE,
    reporting_start: pd.Timestamp | None = None,
) -> tuple[qis.PortfolioData, qis.PortfolioData]:
    """Run one with/without-crypto pair using the paper's parameterization."""
    if engine_id == PUBLISHED_ENGINE:
        config = Parity2024Config(end_date=as_of)
        if reporting_start is not None:
            config = Parity2024Config(end_date=as_of, reporting_start=reporting_start)
        result = backtest_marginal_2024(
            prices=prices,
            marginal_asset=crypto_asset,
            method=method,
            is_alternatives=universe == "alternatives",
            config=config,
        )
        return result.without_asset, result.with_asset
    if engine_id != CURRENT_ENGINE:
        raise ValueError(f"Unsupported engine_id: {engine_id}")
    time_period = TimePeriod("19Jul2010", as_of.strftime("%d%b%Y"))
    perf_start = pd.Timestamp("2016-03-31") if reporting_start is None else reporting_start
    perf_time_period = TimePeriod(perf_start.strftime("%d%b%Y"), as_of.strftime("%d%b%Y"))
    is_alternatives = universe == "alternatives"
    # backtest_marginal_optimal_portfolios is rejected for MaxDiv here because v7.1.0
    # passes the with-crypto covariance set to both universes.  The without-crypto result
    # then drops the crypto weight and is under-invested.  The paper runner keeps the
    # public estimator/optimiser and estimates each investable universe separately.
    if method == OptimisationType.MAX_DIV:
        return _run_max_div_pair(
            prices=prices,
            crypto_asset=crypto_asset,
            time_period=time_period,
            perf_time_period=perf_time_period,
            is_alternatives=is_alternatives,
        )
    return backtest_marginal_optimal_portfolios(
        prices=prices,
        marginal_asset=crypto_asset,
        time_period=time_period,
        perf_time_period=perf_time_period,
        is_alternatives=is_alternatives,
        optimisation_type=method,
        **PARAMS.to_dict(),
    )


def _run_max_div_pair(
    prices: pd.DataFrame,
    crypto_asset: str,
    time_period: TimePeriod,
    perf_time_period: TimePeriod,
    is_alternatives: bool,
) -> tuple[qis.PortfolioData, qis.PortfolioData]:
    """Run MaxDiv with covariance matrices matching each investable universe."""
    prices_with = prices
    prices_without = prices.drop(columns=crypto_asset)
    if is_alternatives:
        constraints_with = Constraints()
        constraints_without = Constraints()
    else:
        min_with = pd.Series(0.0, index=prices_with.columns)
        max_with = pd.Series(1.0, index=prices_with.columns)
        min_without = pd.Series(0.0, index=prices_without.columns)
        max_without = pd.Series(1.0, index=prices_without.columns)
        min_with.iloc[0] = max_with.iloc[0] = PARAMS.first_asset_target_weight
        min_without.iloc[0] = max_without.iloc[0] = PARAMS.first_asset_target_weight
        constraints_with = Constraints(min_weights=min_with, max_weights=max_with)
        constraints_without = Constraints(min_weights=min_without, max_weights=max_without)

    estimator = EwmaCovarEstimator(
        returns_freq=PARAMS.returns_freq,
        rebalancing_freq=PARAMS.rebalancing_freq,
        span=PARAMS.span,
    )
    covars_without = estimator.fit_rolling_covars(
        prices=prices_without, time_period=time_period
    )
    covars_with = estimator.fit_rolling_covars(prices=prices_with, time_period=time_period)
    weights_without = rolling_maximise_diversification(
        prices=prices_without,
        constraints=constraints_without,
        covar_dict=covars_without,
    )
    weights_with = rolling_maximise_diversification(
        prices=prices_with,
        constraints=constraints_with,
        covar_dict=covars_with,
    )
    weights_without = perf_time_period.locate(weights_without)
    weights_with = perf_time_period.locate(weights_with)

    portfolio_without = qis.backtest_model_portfolio(
        prices=qis.truncate_prior_to_start(prices_without, start=weights_without.index[0]),
        weights=weights_without,
        rebalancing_freq=PARAMS.rebalancing_freq,
        is_rebalanced_at_first_date=True,
        rebalancing_costs=PARAMS.rebalancing_costs,
        weight_implementation_lag=PARAMS.weight_implementation_lag,
        ticker=f"{OptimisationType.MAX_DIV.value} w/o {crypto_asset}",
    )
    portfolio_with = qis.backtest_model_portfolio(
        prices=qis.truncate_prior_to_start(prices_with, start=weights_with.index[0]),
        weights=weights_with,
        rebalancing_freq=PARAMS.rebalancing_freq,
        is_rebalanced_at_first_date=True,
        rebalancing_costs=PARAMS.rebalancing_costs,
        weight_implementation_lag=PARAMS.weight_implementation_lag,
        ticker=f"{OptimisationType.MAX_DIV.value} with {crypto_asset}",
    )
    return portfolio_without, portfolio_with


def _performance_table(
    benchmark: pd.Series,
    portfolio_without: qis.PortfolioData,
    portfolio_with: qis.PortfolioData,
    perf_params: PerfParams,
    period: TimePeriod,
) -> pd.DataFrame:
    """Compute a numeric benchmark-relative performance table through public qis APIs."""
    navs = pd.concat(
        [benchmark.rename("100% Balanced"), portfolio_without.nav, portfolio_with.nav], axis=1
    ).dropna()
    navs = period.locate(navs)
    table = qis.compute_ra_perf_table_with_benchmark(
        prices=navs,
        benchmark="100% Balanced",
        perf_params=perf_params,
    )
    return table.reindex(columns=list(PERFORMANCE_COLUMNS))


def _build_report(
    allocation_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    weight_series: dict[tuple[str, str], pd.DataFrame],
    nav_series: dict[tuple[str, str, str], pd.DataFrame],
    report_path: Path,
    perf_params: PerfParams,
    engine_label: str,
) -> Path:
    """Build the deterministic PDF report using qis plotting utilities."""
    figures: list[plt.Figure] = []

    coverage_view = coverage[
        [
            "first_observation",
            "last_observation",
            "observations",
            "staleness_days",
            "max_staleness_days",
            "maximum_gap_since_published_start_days",
        ]
    ].rename(
        columns={
            "first_observation": "First",
            "last_observation": "Last",
            "observations": "Obs",
            "staleness_days": "Stale d",
            "max_staleness_days": "Stale limit",
            "maximum_gap_since_published_start_days": "Max gap d",
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    qis.plot_df_table(
        df=coverage_view,
        title="Bloomberg source coverage for the immutable update snapshot",
        add_index_as_column=True,
        index_column_name="Series",
        fontsize=8,
        ax=ax,
    )
    figures.append(fig)

    weight_view = allocation_summary[
        [
            "min_crypto_weight",
            "median_crypto_weight",
            "mean_crypto_weight",
            "max_crypto_weight",
            "last_crypto_weight",
        ]
    ]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
    qis.plot_df_table(
        df=qis.df_to_str(weight_view, var_format="{:.2%}"),
        title=f"Crypto allocation summary — {engine_label}",
        add_index_as_column=True,
        index_column_name="Universe / asset / method",
        fontsize=7,
        ax=ax,
    )
    figures.append(fig)

    for (universe, crypto_asset), weights in weight_series.items():
        fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
        qis.plot_time_series(
            df=weights,
            title=f"{crypto_asset} target weights — {universe}",
            var_format="{:.1%}",
            legend_stats=qis.LegendStats.FIRST_AVG_LAST,
            y_limits=(0.0, None),
            ax=ax,
        )
        figures.append(fig)

    perf_view = performance_summary[
        ["P.a. return", "Vol", "Log Ex Sharpe", "Max DD", "Alpha", "Beta", "R2"]
    ]
    percent_columns = {"P.a. return", "Vol", "Max DD", "Alpha", "R2"}
    for (universe, crypto_asset, period_name), group in perf_view.groupby(level=[0, 1, 2]):
        table = group.droplevel([0, 1, 2]).copy()
        for column in table:
            value_format = "{:.2%}" if column in percent_columns else "{:.2f}"
            table[column] = table[column].map(
                lambda value, fmt=value_format: fmt.format(value)
            )
        fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
        qis.plot_df_table(
            df=table,
            title=f"{period_name} performance — {universe}, {crypto_asset}",
            add_index_as_column=True,
            index_column_name="Method / portfolio",
            fontsize=7,
            ax=ax,
        )
        figures.append(fig)

    for (universe, crypto_asset, method), navs in nav_series.items():
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
        qis.plot_prices_with_dd(
            prices=navs,
            title=f"{method}: {universe} portfolio with and without {crypto_asset}",
            perf_stats_labels=[PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_LOG_EXCESS],
            perf_params=perf_params,
            axs=axs,
        )
        figures.append(fig)

    written = qis.save_figs_to_pdf(
        figures,
        file_name=report_path.stem,
        local_path=str(report_path.parent),
        add_current_date=False,
    )
    plt.close("all")
    return Path(written)


def run_published_update(
    tag: str,
    use_legacy_eth_proxy: bool = True,
    output_root: Path = OUTPUT_PATH,
    methods: Iterable[OptimisationType] = METHODS,
    build_report: bool = True,
    engine_id: EngineId = PUBLISHED_ENGINE,
) -> Path:
    """Run and verify the complete four-method BTC/ETH paper update."""
    methods = tuple(methods)
    if not methods:
        raise ValueError("At least one optimisation method is required")
    if len(set(methods)) != len(methods):
        raise ValueError("Optimisation methods must be unique")
    if engine_id == PUBLISHED_ENGINE:
        engine_runtime = require_parity_dependencies(methods).to_dict()
        engine_label = "August 2024 published engine (commit 6038fba)"
    elif engine_id == CURRENT_ENGINE:
        engine_runtime = None
        engine_label = "current optimalportfolios 7.1 engine"
    else:
        raise ValueError(f"Unsupported engine_id: {engine_id}")

    manifest = verify_bloomberg_snapshot(tag=tag)
    as_of = pd.Timestamp(manifest["inclusive_as_of"])
    prices = load_bloomberg_prices(tag=tag, use_legacy_eth_proxy=use_legacy_eth_proxy)
    risk_free = load_bloomberg_risk_free(tag=tag)
    perf_params = PerfParams(
        freq_vol="ME",
        freq_reg="ME",
        freq_drawdown="ME",
        rates_data=risk_free,
    )
    mode = "legacy_eth_proxy" if use_legacy_eth_proxy else "observed_eth"
    output_path = Path(output_root) / tag / engine_id / mode
    output_path.mkdir(parents=True, exist_ok=True)

    allocation_rows: list[dict[str, object]] = []
    performance_frames: list[pd.DataFrame] = []
    attribution_rows: list[dict[str, object]] = []
    period_coverage_rows: list[dict[str, object]] = []
    weight_audits: list[dict[str, object]] = []
    weight_series_raw: dict[tuple[str, str], dict[str, pd.Series]] = {}
    nav_series: dict[tuple[str, str, str], pd.DataFrame] = {}
    reporting_starts: dict[str, str] = {}
    periods = _periods(as_of)

    for crypto_asset in ("BTC", "ETH"):
        selected = prices.drop(columns=["ETH" if crypto_asset == "BTC" else "BTC"]).dropna()
        reporting_start = _common_reporting_start(selected)
        reporting_starts[crypto_asset] = reporting_start.strftime("%Y-%m-%d")
        benchmark = selected["60/40"].rename("100% Balanced")
        for universe in ("alternatives", "balanced_risk_budget"):
            model_prices = selected.drop(columns=["60/40"]) if universe == "alternatives" else selected
            for method in methods:
                portfolio_without, portfolio_with = _run_pair(
                    prices=model_prices,
                    crypto_asset=crypto_asset,
                    method=method,
                    universe=universe,
                    as_of=as_of,
                    engine_id=engine_id,
                    reporting_start=reporting_start,
                )
                weights_without = _input_weights(portfolio_without)
                weights_with = _input_weights(portfolio_with)
                fixed_first_weight = (
                    PARAMS.first_asset_target_weight
                    if universe == "balanced_risk_budget" and method != OptimisationType.ERC
                    else None
                )
                for portfolio_label, weights, expected_columns in (
                    (
                        "without_crypto",
                        weights_without,
                        model_prices.columns.drop(crypto_asset),
                    ),
                    ("with_crypto", weights_with, model_prices.columns),
                ):
                    audit = _validate_target_weights(
                        weights=weights,
                        expected_columns=expected_columns,
                        label=f"{universe}/{crypto_asset}/{method.value}/{portfolio_label}",
                        fixed_first_weight=fixed_first_weight,
                    )
                    audit.update(
                        {
                            "universe": universe,
                            "crypto_asset": crypto_asset,
                            "method": method.value,
                            "portfolio": portfolio_label,
                        }
                    )
                    weight_audits.append(audit)

                summary = summarize_crypto_weights(
                    weights=weights_with, crypto_asset=crypto_asset
                )
                summary.update(
                    {
                        "universe": universe,
                        "crypto_asset": crypto_asset,
                        "method": method.value,
                    }
                )
                allocation_rows.append(summary)
                weight_series_raw.setdefault((universe, crypto_asset), {})[
                    method.value
                ] = weights_with[crypto_asset]

                navs = pd.concat(
                    [benchmark, portfolio_without.nav, portfolio_with.nav], axis=1
                ).dropna()
                for period_name, period in periods.items():
                    period_navs = period.locate(navs).dropna()
                    period_coverage_rows.append(
                        {
                            "universe": universe,
                            "crypto_asset": crypto_asset,
                            "method": method.value,
                            "period": period_name,
                            "available": len(period_navs) >= 2,
                            "effective_start": (
                                period_navs.index[0].strftime("%Y-%m-%d")
                                if len(period_navs) else None
                            ),
                            "effective_end": (
                                period_navs.index[-1].strftime("%Y-%m-%d")
                                if len(period_navs) else None
                            ),
                            "observations": len(period_navs),
                        }
                    )
                    if len(period_navs) < 2:
                        continue
                    table = _performance_table(
                        benchmark=benchmark,
                        portfolio_without=portfolio_without,
                        portfolio_with=portfolio_with,
                        perf_params=perf_params,
                        period=period,
                    )
                    for portfolio_label, portfolio in (
                        ("without_crypto", portfolio_without),
                        ("with_crypto", portfolio_with),
                    ):
                        row_name = portfolio.nav.name
                        if row_name not in table.index:
                            raise AssertionError(
                                f"Missing performance row {row_name!r} for {method.value}"
                            )
                        row = table.loc[[row_name]].copy()
                        row["universe"] = universe
                        row["crypto_asset"] = crypto_asset
                        row["period"] = period_name
                        row["method"] = method.value
                        row["portfolio"] = portfolio_label
                        performance_frames.append(row.reset_index(drop=True))

                    attribution = portfolio_with.get_performance_attribution(time_period=period)
                    crypto_attribution = float(attribution[crypto_asset])
                    if not np.isfinite(crypto_attribution):
                        raise AssertionError(
                            f"Non-finite attribution for {universe}/{crypto_asset}/"
                            f"{method.value}/{period_name}"
                        )
                    attribution_rows.append(
                        {
                            "universe": universe,
                            "crypto_asset": crypto_asset,
                            "method": method.value,
                            "period": period_name,
                            "crypto_performance_attribution": crypto_attribution,
                        }
                    )
                full_navs = periods["full_since_2016Q1"].locate(navs).dropna()
                if len(full_navs) >= 2:
                    nav_series[(universe, crypto_asset, method.value)] = full_navs

    allocation_summary = pd.DataFrame(allocation_rows).set_index(
        ["universe", "crypto_asset", "method"]
    ).sort_index()
    scenario_summary = allocation_summary.groupby(level=[0, 1])[
        "median_crypto_weight"
    ].median().to_frame("median_across_method_medians")
    performance_summary = pd.concat(performance_frames, ignore_index=True).set_index(
        ["universe", "crypto_asset", "period", "method", "portfolio"]
    ).sort_index()
    attribution_summary = pd.DataFrame(attribution_rows).set_index(
        ["universe", "crypto_asset", "method", "period"]
    ).sort_index()
    weight_series = {
        key: pd.DataFrame(value).sort_index() for key, value in weight_series_raw.items()
    }
    weight_audit = pd.DataFrame(weight_audits).set_index(
        ["universe", "crypto_asset", "method", "portfolio"]
    ).sort_index()
    period_coverage = pd.DataFrame(period_coverage_rows).set_index(
        ["universe", "crypto_asset", "method", "period"]
    ).sort_index()

    max_sum_error = float(weight_audit["max_weight_sum_error"].max())
    minimum_weight = float(weight_audit["minimum_weight"].min())
    maximum_weight = float(weight_audit["maximum_weight"].max())
    max_fixed_first_weight_error = float(
        weight_audit["max_fixed_first_weight_error"].max()
    )
    expected_runs = 2 * 2 * len(methods)
    if len(allocation_summary) != expected_runs:
        raise AssertionError(f"Expected {expected_runs} completed runs, got {len(allocation_summary)}")
    if max_sum_error > 2e-5:
        raise AssertionError(f"Target weights are not fully invested: max error={max_sum_error}")
    if minimum_weight < -2e-5 or maximum_weight > 1.0 + 2e-5:
        raise AssertionError(
            f"Long-only bounds failed: min={minimum_weight}, max={maximum_weight}"
        )
    performance_values = performance_summary.loc[:, list(PERFORMANCE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_performance = ~np.isfinite(performance_values)
    if invalid_performance.any().any():
        coordinates = invalid_performance.stack()
        coordinates = coordinates[coordinates].index.tolist()[:5]
        raise AssertionError(f"Performance summary contains non-finite values: {coordinates}")

    allocation_path = output_path / "allocation_summary.csv"
    scenario_path = output_path / "scenario_summary.csv"
    performance_path = output_path / "performance_summary.csv"
    attribution_path = output_path / "performance_attribution.csv"
    period_coverage_path = output_path / "period_coverage.csv"
    weight_audit_path = output_path / "weight_audit.csv"
    allocation_summary.to_csv(allocation_path)
    scenario_summary.to_csv(scenario_path)
    performance_summary.to_csv(performance_path)
    attribution_summary.to_csv(attribution_path)
    period_coverage.to_csv(period_coverage_path)
    weight_audit.to_csv(weight_audit_path)
    weight_paths: list[Path] = []
    for (universe, crypto_asset), weights in weight_series.items():
        weight_path = output_path / f"weights_{universe}_{crypto_asset}.csv"
        weights.to_csv(weight_path, index_label="date")
        weight_paths.append(weight_path)

    coverage = pd.DataFrame(manifest["coverage"]).T
    report_path = output_path / f"published_update_{tag}_{engine_id}_{mode}.pdf"
    if build_report:
        report_path = _build_report(
            allocation_summary=allocation_summary,
            performance_summary=performance_summary,
            coverage=coverage,
            weight_series=weight_series,
            nav_series=nav_series,
            report_path=report_path,
            perf_params=perf_params,
            engine_label=engine_label,
        )

    output_files = [
        allocation_path,
        scenario_path,
        performance_path,
        attribution_path,
        period_coverage_path,
        weight_audit_path,
        *weight_paths,
    ]
    if build_report:
        output_files.append(report_path)
    analysis_manifest = {
        "schema_version": 1,
        "data_snapshot": tag,
        "data_manifest_sha256": _sha256(get_snapshot_paths(tag).manifest),
        "analysis_cutoff": as_of.strftime("%Y-%m-%d"),
        "analysis_mode": mode,
        "engine_id": engine_id,
        "engine": engine_label,
        "engine_runtime": engine_runtime,
        "known_methodology_notes": [
            "Balanced ERC applies a 75% risk budget to the 60/40 sleeve; it is not a 75% capital weight.",
            (
                "Published-engine MaxSharpe uses a rolling 60-month arithmetic mean and "
                "the historical ECOS_BB Charnes-Cooper solve."
                if engine_id == PUBLISHED_ENGINE
                else "Current-engine MaxSharpe uses EWMA expected returns."
            ),
            (
                "Legacy ETH mode backfills XETUSD with scaled XBTUSD before Bloomberg ETH history begins."
                if use_legacy_eth_proxy
                else "Observed ETH mode uses XETUSD only and begins all methods after a common 60-month warm-up."
            ),
            "HFRIMDT is the explicit 2024 HFRI Macro substitution, not the original SG Macro index.",
            (
                "For published-paper parity, HFRIMDT month-end observations become usable on "
                "the following business day without a separate publication lag; revised-history "
                "and release-timing bias remain a stated limitation."
            ),
        ],
        "parameters": PARAMS.to_dict(),
        "reporting_start_by_crypto_asset": reporting_starts,
        "methods": [method.value for method in methods],
        "completed_runs": len(allocation_summary),
        "verification": {
            "data_manifest_verified": True,
            "weights_fully_invested": True,
            "both_with_and_without_portfolios_checked": True,
            "long_only_bounds_verified": True,
            "fixed_first_sleeve_checked_where_applicable": True,
            "performance_values_finite": True,
            "max_weight_sum_error": max_sum_error,
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "max_fixed_first_weight_error": max_fixed_first_weight_error,
            "configured_weight_implementation_lag": PARAMS.weight_implementation_lag,
        },
        "headline": {
            "median_of_scenario_method_medians_all": float(
                scenario_summary["median_across_method_medians"].median()
            ),
            "median_of_scenario_method_medians_alternatives": float(
                scenario_summary.xs("alternatives")[
                    "median_across_method_medians"
                ].median()
            ),
            "median_of_scenario_method_medians_balanced_risk_budget": float(
                scenario_summary.xs("balanced_risk_budget")[
                    "median_across_method_medians"
                ].median()
            ),
            "scenario_method_medians": {
                f"{universe}/{crypto_asset}": float(value)
                for (universe, crypto_asset), value in scenario_summary[
                    "median_across_method_medians"
                ].items()
            },
        },
        "packages": {
            "python": platform.python_version(),
            "pandas": _version("pandas"),
            "qis": _version("qis"),
            "optimalportfolios": _version("optimalportfolios"),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        },
    }
    manifest_path = output_path / "ANALYSIS_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis_manifest["headline"], indent=2))
    print(f"Verified analysis outputs: {output_path}")
    return output_path


def main() -> None:
    """Run the command-line Bloomberg acquisition and/or paper analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "analyse", "all"))
    parser.add_argument("--as-of", default=DEFAULT_AS_OF.strftime("%Y-%m-%d"))
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--observed-eth", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--engine",
        choices=(PUBLISHED_ENGINE, CURRENT_ENGINE),
        default=PUBLISHED_ENGINE,
    )
    args = parser.parse_args()

    tag = args.snapshot or snapshot_tag(args.as_of)
    if args.action in ("fetch", "all"):
        paths = create_bloomberg_snapshot(as_of=args.as_of)
        tag = paths.root.name
        print(f"Created Bloomberg snapshot: {paths.root}")
    if args.action in ("analyse", "all"):
        run_published_update(
            tag=tag,
            use_legacy_eth_proxy=not args.observed_eth,
            build_report=not args.no_report,
            engine_id=args.engine,
        )


if __name__ == "__main__":
    main()
