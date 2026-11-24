"""Assemble Yahoo inputs, FM excess factors and the fixed monthly QIS benchmark."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qis

from examples.oss_portfolio_pipeline.config import (
    PipelineConfig, all_tickers, benchmark_weights, universe_table,
)
from examples.oss_portfolio_pipeline.data import (
    download_prices, external_directory, validate_prices,
)
from examples.oss_portfolio_pipeline.factors import build_factors, factor_model, month_end_dates


@dataclass
class PipelineInputs:
    """Prepared public inputs, with total-return asset prices and excess-return factors."""

    source_prices: pd.DataFrame
    prices: pd.DataFrame
    factor_navs: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_weights: dict[str, pd.DataFrame]
    benchmark_weights: pd.Series
    benchmark_targets: pd.DataFrame
    benchmark: qis.PortfolioData
    universe: pd.DataFrame


def build_inputs(prices: pd.DataFrame, config: PipelineConfig) -> PipelineInputs:
    """Build from a supplied complete adjusted-price panel, without accessing the network."""
    prices = validate_prices(prices, all_tickers())
    prices = prices.loc[(prices.index >= config.start) & (prices.index < config.end)]
    if prices.empty or prices.index[0] > pd.Timestamp(config.factor_start):
        raise ValueError('Prices must include the factor start and construction prehistory')
    if prices.index[-1] < pd.Timestamp(config.backtest_start):
        raise ValueError('Prices must reach the backtest start')
    if (pd.Timestamp(config.end) - prices.index[-1]).days > 7:
        raise ValueError('Price history ends more than seven days before the requested end')
    built = build_factors(prices, config)
    anchor = prices.index.asof(pd.Timestamp(config.factor_start))
    for name, targets in built.target_weights.items():
        if not targets.loc[:anchor].abs().sum(axis=1).gt(0).any():
            raise ValueError(f'{name}: factor start precedes the volatility warm-up')
    factor_navs = built.navs.loc[anchor:].div(built.navs.loc[anchor]).mul(100.0)
    factor_returns = qis.to_returns(factor_navs, is_log_returns=True, ffill_nans=False)
    universe = universe_table()
    asset_prices = prices.loc[:, universe.index]
    initial = pd.Timestamp(config.backtest_start)
    if initial not in asset_prices.index:
        raise ValueError('Backtest start must be an observed trading session')
    backtest_prices = asset_prices.loc[initial:]
    dates = month_end_dates(backtest_prices.index, config.end).union(pd.DatetimeIndex([initial]))
    strategic = benchmark_weights()
    targets = pd.DataFrame(np.tile(strategic.to_numpy(), (len(dates), 1)),
                           index=dates, columns=strategic.index)
    benchmark = qis.backtest_model_portfolio(
        prices=backtest_prices, weights=targets, ticker='Strategic 50/35/10/5')
    return PipelineInputs(
        source_prices=prices, prices=asset_prices,
        factor_navs=factor_navs, factor_returns=factor_returns,
        factor_weights=built.target_weights, benchmark_weights=strategic,
        benchmark_targets=targets, benchmark=benchmark, universe=universe)


def run_pipeline(config: PipelineConfig, cache_dir: Path | str | None = None,
                 refresh: bool = False) -> PipelineInputs:
    """Download or reuse the 40-source Yahoo panel and build the example inputs."""
    prices = download_prices(all_tickers(), config.start, config.end, cache_dir, refresh)
    return build_inputs(prices, config)


def save_inputs(inputs: PipelineInputs, config: PipelineConfig,
                output_dir: Path | str) -> Path:
    """Save a local reproducibility bundle outside the repository and OneDrive."""
    output = external_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        'source_adjusted_prices.csv': inputs.source_prices,
        'asset_prices.csv': inputs.prices,
        'universe.csv': inputs.universe,
        'factor_excess_navs.csv': inputs.factor_navs,
        'factor_excess_log_returns.csv': inputs.factor_returns,
        'factor_target_weights.csv': pd.concat(inputs.factor_weights, names=['factor', 'date']),
        'benchmark_targets.csv': inputs.benchmark_targets,
        'benchmark_nav.csv': inputs.benchmark.get_portfolio_nav(),
        'benchmark_realised_weights.csv': inputs.benchmark.weights,
    }
    hashes = {}
    for name, table in tables.items():
        content = table.to_csv(float_format='%.17g').encode('utf-8')
        (output / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    spec = factor_model(config)
    manifest = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'file_sha256': hashes,
        'source_download': inputs.source_prices.attrs.get('yahoo_download'),
        'config': asdict(config), 'currency': 'USD',
        'asset_price_type': 'Yahoo dividend/split adjusted close',
        'factor_return_type': 'log returns of BIL-financed excess-P&L NAVs; no second cash debit',
        'construction_returns_frequency': 'W-WED',
        'rebalance_rule': 'last observed US session at completed calendar month-end',
        'factor_anchor_observation': str(inputs.factor_navs.index[0].date()),
        'factor_definitions': {f.name: f.to_text() for f in spec.baskets.values()},
        'benchmark_weights': inputs.benchmark_weights.to_dict(),
        'costs': 'gross of trading/borrow costs; fund fees embedded in Yahoo adjusted prices',
        'versions': {name: version(name) for name in ('factormodels', 'factorlasso', 'qis',
                                                    'yfinance', 'pandas', 'numpy')},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output
