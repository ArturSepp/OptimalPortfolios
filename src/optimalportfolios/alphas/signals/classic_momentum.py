"""Classic fixed-window momentum with standard and cluster scoring.

Classic momentum sums a fixed number of completed log returns after a hard
skip.  With the defaults, the signal at a monthly formation date contains
exactly 12 monthly returns and excludes the latest one.  It is deliberately
separate from :mod:`optimalportfolios.alphas.signals.momentum`, whose raw
signal is benchmark-relative, volatility-normalised, and EWMA-filtered.

Three public entry points share the same raw signal:

* :func:`compute_classic_momentum_from_returns` computes the fixed-window sum.
* :func:`compute_classic_momentum_alpha` scores globally or within fixed groups.
* :func:`compute_classic_momentum_cluster_alpha` scores within rolling clusters.

Both support a uniform return cadence or a per-asset cadence Series.  Lookback
and skip periods can be scalars or mappings keyed by cadence.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import qis

from optimalportfolios.alphas.signals.utils import resolve_span, score_within_clusters


PeriodSpec = Union[int, Mapping[str, int]]


def _resolve_skip_periods(skip_periods: PeriodSpec, freq: str) -> int:
    """Resolve a non-negative skipped-period count for one reporting cadence."""
    if isinstance(skip_periods, Mapping) and str(freq) not in skip_periods:
        raise ValueError(
            f"skip_periods covers {sorted(skip_periods)} but an asset reports at "
            f"{str(freq)!r}; add the cadence rather than inheriting another horizon"
        )
    if isinstance(skip_periods, Mapping):
        skip_periods = skip_periods[str(freq)]
    if isinstance(skip_periods, bool) or not isinstance(skip_periods, (int, np.integer)):
        raise ValueError(
            "skip_periods must be a non-negative int number of periods, or a "
            f"per-cadence mapping of them, got {skip_periods!r}"
        )
    if skip_periods < 0:
        raise ValueError(f"skip_periods must be >= 0, got {skip_periods!r}")
    return int(skip_periods)


# ``qis.compute_ewm_long_short_filtered_ra_returns`` cannot be used here: it
# applies an exponential filter and its short span is not a hard return skip.
def compute_classic_momentum_from_returns(
        returns: pd.DataFrame,
        lookback_periods: int = 12,
        skip_periods: int = 1,
) -> pd.DataFrame:
    """Sum a fixed number of aligned returns after a hard period skip.

    Args:
        returns: Already sampled return panel. Log returns produce the usual
            continuously compounded classic momentum signal.
        lookback_periods: Exact number of included observations.
        skip_periods: Latest observations excluded before forming the window.

    Returns:
        Fixed-window momentum in the same shape as ``returns``.
    """
    lookback = resolve_span(
        lookback_periods, freq='returns', name='lookback_periods'
    )
    skip = _resolve_skip_periods(skip_periods, freq='returns')
    return returns.shift(skip).rolling(
        window=lookback,
        min_periods=lookback,
    ).sum()


def _compute_raw_classic_momentum_single_freq(
        prices: pd.DataFrame,
        returns_freq: str = 'ME',
        lookback_periods: int = 12,
        skip_periods: int = 1,
) -> pd.DataFrame:
    """Compute a fixed-window sum of log returns after a hard period skip."""
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    return compute_classic_momentum_from_returns(
        returns=returns,
        lookback_periods=lookback_periods,
        skip_periods=skip_periods,
    )


def _compute_raw_classic_momentum_mixed_freq(
        prices: pd.DataFrame,
        returns_freqs: pd.Series,
        lookback_periods: PeriodSpec = 12,
        skip_periods: PeriodSpec = 1,
) -> pd.DataFrame:
    """Compute raw classic momentum independently within each cadence bucket."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)
    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        bucket_lookback = resolve_span(
            lookback_periods, freq=freq, name='lookback_periods'
        )
        bucket_skip = _resolve_skip_periods(skip_periods, freq=freq)
        all_raw.append(
            _compute_raw_classic_momentum_single_freq(
                prices=prices[asset_tickers],
                returns_freq=freq,
                lookback_periods=bucket_lookback,
                skip_periods=bucket_skip,
            )
        )
    return pd.concat(all_raw, axis=1, sort=True)[prices.columns].ffill()


def _compute_classic_momentum_alpha_single_freq(
        prices: pd.DataFrame,
        returns_freq: str = 'ME',
        group_data: Optional[pd.Series] = None,
        lookback_periods: PeriodSpec = 12,
        skip_periods: PeriodSpec = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute one-cadence classic momentum and cross-sectional scores."""
    lookback = resolve_span(
        lookback_periods, freq=returns_freq, name='lookback_periods'
    )
    skip = _resolve_skip_periods(skip_periods, freq=returns_freq)
    raw_momentum = _compute_raw_classic_momentum_single_freq(
        prices=prices,
        returns_freq=returns_freq,
        lookback_periods=lookback,
        skip_periods=skip,
    )
    if group_data is None:
        momentum_score = qis.df_to_cross_sectional_score(df=raw_momentum)
    else:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data)
        group_scores = [
            qis.df_to_cross_sectional_score(df=raw_momentum[group_prices.columns])
            for group_prices in grouped_prices.values()
        ]
        momentum_score = pd.concat(group_scores, axis=1, sort=True)[prices.columns]
    return momentum_score, raw_momentum


def _compute_classic_momentum_alpha_mixed_freq(
        prices: pd.DataFrame,
        returns_freqs: pd.Series,
        group_data: Optional[pd.Series] = None,
        lookback_periods: PeriodSpec = 12,
        skip_periods: PeriodSpec = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and merge scores for every cadence and optional fixed group."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)
    all_scores = []
    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        if group_data is None:
            grouped_prices = {'_': freq_prices}
        else:
            freq_group_data = group_data.loc[group_data.index.intersection(asset_tickers)]
            grouped_prices = qis.split_df_by_groups(
                df=freq_prices, group_data=freq_group_data
            )
        for group_prices in grouped_prices.values():
            score, raw = _compute_classic_momentum_alpha_single_freq(
                prices=group_prices,
                returns_freq=freq,
                group_data=None,
                lookback_periods=lookback_periods,
                skip_periods=skip_periods,
            )
            all_scores.append(score)
            all_raw.append(raw)
    momentum_score = pd.concat(all_scores, axis=1, sort=True)[prices.columns].ffill()
    raw_momentum = pd.concat(all_raw, axis=1, sort=True)[prices.columns].ffill()
    return momentum_score, raw_momentum


def compute_classic_momentum_alpha(
        prices: pd.DataFrame,
        returns_freq: Union[str, pd.Series] = 'ME',
        group_data: Optional[pd.Series] = None,
        lookback_periods: PeriodSpec = 12,
        skip_periods: PeriodSpec = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute classic fixed-window momentum and cross-sectional scores.

    The raw signal is the rolling sum of ``lookback_periods`` log returns
    after shifting the return panel by ``skip_periods``.  The default therefore
    includes exactly 12 monthly observations and excludes the latest month.
    No benchmark subtraction, volatility scaling, mean adjustment, or EWMA
    filtering is applied.

    Args:
        prices: Asset price panel with dates on the index and tickers in columns.
        returns_freq: Uniform return cadence, or a Series mapping assets to cadences.
        group_data: Optional fixed group label per asset for within-group scoring.
        lookback_periods: Included return observations, as a positive integer or
            a mapping by cadence such as ``{'ME': 12, 'QE': 4}``.
        skip_periods: Latest return observations excluded, as a non-negative
            integer or mapping by cadence.

    Returns:
        Tuple of cross-sectional momentum scores and raw fixed-window returns.
    """
    if isinstance(returns_freq, pd.Series):
        return _compute_classic_momentum_alpha_mixed_freq(
            prices=prices,
            returns_freqs=returns_freq,
            group_data=group_data,
            lookback_periods=lookback_periods,
            skip_periods=skip_periods,
        )
    return _compute_classic_momentum_alpha_single_freq(
        prices=prices,
        returns_freq=returns_freq,
        group_data=group_data,
        lookback_periods=lookback_periods,
        skip_periods=skip_periods,
    )


def compute_classic_momentum_cluster_alpha(
        prices: pd.DataFrame,
        rolling_clusters: Optional[Dict[pd.Timestamp, pd.Series]] = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        lookback_periods: PeriodSpec = 12,
        skip_periods: PeriodSpec = 1,
        min_cluster_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute classic momentum scored within time-varying clusters.

    Args:
        prices: Asset price panel with dates on the index and tickers in columns.
        rolling_clusters: Mapping from estimation dates to ticker-cluster labels.
        returns_freq: Uniform return cadence, or a Series mapping assets to cadences.
        lookback_periods: Included return observations, scalar or cadence mapping.
        skip_periods: Latest excluded return observations, scalar or cadence mapping.
        min_cluster_size: Minimum size for within-cluster statistics; smaller
            clusters use the global cross-sectional score parameters.

    Returns:
        Tuple of cluster-scored momentum and the unaltered raw momentum panel.
    """
    if isinstance(returns_freq, pd.Series):
        raw_momentum = _compute_raw_classic_momentum_mixed_freq(
            prices=prices,
            returns_freqs=returns_freq,
            lookback_periods=lookback_periods,
            skip_periods=skip_periods,
        )
    else:
        lookback = resolve_span(
            lookback_periods, freq=returns_freq, name='lookback_periods'
        )
        skip = _resolve_skip_periods(skip_periods, freq=returns_freq)
        raw_momentum = _compute_raw_classic_momentum_single_freq(
            prices=prices,
            returns_freq=returns_freq,
            lookback_periods=lookback,
            skip_periods=skip,
        )
    momentum_score = score_within_clusters(
        raw_signal=raw_momentum,
        rolling_clusters=rolling_clusters or {},
        min_cluster_size=min_cluster_size,
    )
    return momentum_score, raw_momentum
