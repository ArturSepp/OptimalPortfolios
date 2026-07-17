"""
Carry alpha computation (standard + cluster scoring).

Computes cross-sectional risk-adjusted carry scores from an instrument yield
(carry) panel divided by trailing realised volatility. Ranking by yield PER UNIT
OF RISK removes the duration / credit-risk tilt of raw yield: the highest-yielding
instrument is usually the riskiest, so a raw-carry ranking overweights risk, while
risk-adjusted carry does not.

Two public entry points share the same raw signal:
    * ``compute_ra_carry_alpha`` — scores cross-sectionally, optionally within
      fixed user-defined groups (``group_data``).
    * ``compute_ra_carry_cluster_alpha`` — scores within time-varying statistical
      clusters (``rolling_clusters``) rather than fixed groups.

Unlike momentum and low-beta, carry cannot be derived from prices alone: the
``carry`` (yield) panel is a required input. Prices are used only for the
volatility normalisation.

Pipeline:
    yield / trailing EWMA vol → risk-adjusted carry
            → cross-sectional score (global / within groups / within clusters)
"""
from __future__ import annotations

import pandas as pd
import qis as qis
from typing import Optional, Tuple, Union, Dict

from optimalportfolios.alphas.signals.utils import score_within_clusters


# ---------------------------------------------------------------------------
# raw signal — shared by the standard and cluster entry points
# ---------------------------------------------------------------------------
def _compute_raw_ra_carry_single_freq(prices: pd.DataFrame,
                                      carry: pd.DataFrame,
                                      returns_freq: str = 'W-WED',
                                      vol_span: Optional[int] = 13,
                                      mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                                      ) -> pd.DataFrame:
    """Compute raw risk-adjusted carry (yield / trailing vol), before scoring."""
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    ewm_vol = qis.compute_ewm_vol(data=returns,
                                  span=vol_span,
                                  mean_adj_type=mean_adj_type,
                                  annualize=True)
    ra_carry = carry.reindex(index=ewm_vol.index, method='ffill').divide(ewm_vol)
    return ra_carry


def _compute_raw_ra_carry_mixed_freq(prices: pd.DataFrame,
                                     carry: pd.DataFrame,
                                     returns_freqs: pd.Series = None,
                                     vol_span: Optional[int] = 13,
                                     mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                                     ) -> pd.DataFrame:
    """Mixed-frequency: compute raw risk-adjusted carry per frequency group, merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        raw = _compute_raw_ra_carry_single_freq(
            prices=freq_prices, carry=carry[asset_tickers],
            returns_freq=freq, vol_span=vol_span,
            mean_adj_type=mean_adj_type)
        all_raw.append(raw)

    return pd.concat(all_raw, axis=1)[prices.columns].ffill()


# ---------------------------------------------------------------------------
# standard — cross-sectional scoring (optionally within fixed groups)
# ---------------------------------------------------------------------------
def compute_ra_carry_alpha(prices: pd.DataFrame,
                           carry: pd.DataFrame,
                           returns_freq: Union[str, pd.Series] = 'W-WED',
                           group_data: Optional[pd.Series] = None,
                           vol_span: Optional[int] = 13,
                           mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional risk-adjusted carry alpha scores.

    Divides the instrument yield by its trailing EWMA volatility, then converts
    the risk-adjusted carry to a cross-sectional score (high yield-per-risk →
    high score).

    Handles both single-frequency and mixed-frequency universes:
    - ``returns_freq`` is a string: all assets at the same frequency.
    - ``returns_freq`` is a pd.Series: per-asset frequency.

    When ``group_data`` is provided, cross-sectional scoring is computed within
    each group independently.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used for the
            volatility normalisation only.
        carry: Instrument yield (carry) panel. Index=dates, columns=tickers.
        returns_freq: Return frequency for the vol estimate. String or pd.Series
            mapping tickers to frequencies for mixed-frequency computation.
        group_data: Optional group labels per asset for within-group scoring.
        vol_span: EWMA span for volatility normalisation.
        mean_adj_type: Mean adjustment type for EWMA vol computation.

    Returns:
        Tuple of (carry_score, raw_ra_carry).
    """
    if isinstance(returns_freq, pd.Series):
        return _compute_ra_carry_alpha_mixed_freq(
            prices=prices, carry=carry,
            returns_freqs=returns_freq, group_data=group_data,
            vol_span=vol_span, mean_adj_type=mean_adj_type)
    else:
        return _compute_ra_carry_alpha_single_freq(
            prices=prices, carry=carry,
            returns_freq=returns_freq, group_data=group_data,
            vol_span=vol_span, mean_adj_type=mean_adj_type)


def _compute_ra_carry_alpha_single_freq(prices: pd.DataFrame,
                                        carry: pd.DataFrame,
                                        returns_freq: str = 'W-WED',
                                        group_data: Optional[pd.Series] = None,
                                        vol_span: Optional[int] = 13,
                                        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-frequency carry: raw risk-adjusted carry, then cross-sectional scoring."""
    raw_ra_carry = _compute_raw_ra_carry_single_freq(
        prices=prices, carry=carry,
        returns_freq=returns_freq, vol_span=vol_span, mean_adj_type=mean_adj_type)

    if group_data is not None:
        grouped_carry = qis.split_df_by_groups(df=raw_ra_carry, group_data=group_data)
        group_scores = []
        for group, gcarry in grouped_carry.items():
            group_scores.append(qis.df_to_cross_sectional_score(df=gcarry))
        carry_score = pd.concat(group_scores, axis=1)[raw_ra_carry.columns]
    else:
        carry_score = qis.df_to_cross_sectional_score(df=raw_ra_carry)

    return carry_score, raw_ra_carry


def _compute_ra_carry_alpha_mixed_freq(prices: pd.DataFrame,
                                       carry: pd.DataFrame,
                                       returns_freqs: pd.Series = None,
                                       group_data: Optional[pd.Series] = None,
                                       vol_span: Optional[int] = 13,
                                       mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-frequency carry: compute per (frequency × group), merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_scores = []
    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        freq_carry = carry[asset_tickers]
        if group_data is not None:
            freq_group_data = group_data.loc[group_data.index.intersection(asset_tickers)]
            grouped_prices = qis.split_df_by_groups(df=freq_prices, group_data=freq_group_data)
        else:
            grouped_prices = {'_': freq_prices}

        for group, gprice in grouped_prices.items():
            score, raw = _compute_ra_carry_alpha_single_freq(
                prices=gprice, carry=freq_carry[gprice.columns],
                returns_freq=freq, group_data=None,
                vol_span=vol_span, mean_adj_type=mean_adj_type)
            all_scores.append(score)
            all_raw.append(raw)

    carry_score = pd.concat(all_scores, axis=1)[prices.columns].ffill()
    raw_ra_carry = pd.concat(all_raw, axis=1)[prices.columns].ffill()
    return carry_score, raw_ra_carry


# ---------------------------------------------------------------------------
# cluster — scoring within time-varying statistical clusters
# ---------------------------------------------------------------------------
def compute_ra_carry_cluster_alpha(
        prices: pd.DataFrame,
        carry: pd.DataFrame,
        rolling_clusters: Dict[pd.Timestamp, pd.Series] = None,
        returns_freq: Union[str, pd.Series] = 'W-WED',
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE,
        min_cluster_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute risk-adjusted carry scores with time-varying cluster-based scoring.

    Identical to compute_ra_carry_alpha for the raw signal computation
    (yield / trailing vol), but applies cross-sectional scoring within
    statistical clusters that evolve over time rather than fixed user-defined
    groups.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used for the
            volatility normalisation only.
        carry: Instrument yield (carry) panel. Index=dates, columns=tickers.
        rolling_clusters: Dict mapping dates to pd.Series (ticker → cluster_id).
            Extracted from RollingFactorCovarData via extract_rolling_clusters().
        returns_freq: Return frequency for the vol estimate. String or pd.Series
            for mixed-freq.
        vol_span: EWMA span for volatility normalisation.
        mean_adj_type: Mean adjustment type for EWMA vol computation.
        min_cluster_size: Minimum cluster size for within-cluster scoring.
            Clusters with size <= min_cluster_size use global statistics.

    Returns:
        Tuple of (carry_cluster_score, raw_ra_carry).
    """
    if rolling_clusters is None:
        rolling_clusters = {}

    if isinstance(returns_freq, pd.Series):
        raw_ra_carry = _compute_raw_ra_carry_mixed_freq(
            prices=prices, carry=carry,
            returns_freqs=returns_freq, vol_span=vol_span,
            mean_adj_type=mean_adj_type)
    else:
        raw_ra_carry = _compute_raw_ra_carry_single_freq(
            prices=prices, carry=carry,
            returns_freq=returns_freq, vol_span=vol_span,
            mean_adj_type=mean_adj_type)

    # score within time-varying clusters
    carry_cluster_score = score_within_clusters(
        raw_signal=raw_ra_carry,
        rolling_clusters=rolling_clusters,
        min_cluster_size=min_cluster_size)

    return carry_cluster_score, raw_ra_carry


# ---------------------------------------------------------------------------
# backwards-compatible alias for the prior global cross-section entry point
# ---------------------------------------------------------------------------
def compute_ra_carry_alphas(prices: pd.DataFrame,
                            carry: pd.DataFrame,
                            returns_freq: str = 'W-WED',
                            vol_span: Optional[int] = 13,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                            ) -> pd.DataFrame:
    """global cross-sectional risk-adjusted carry score (legacy single-return entry point).

    Retained for callers of the original signature. Returns only the score;
    use ``compute_ra_carry_alpha`` for the (score, raw) tuple and group support,
    or ``compute_ra_carry_cluster_alpha`` for cluster scoring.
    """
    carry_score, _ = _compute_ra_carry_alpha_single_freq(
        prices=prices, carry=carry, returns_freq=returns_freq,
        group_data=None, vol_span=vol_span, mean_adj_type=mean_adj_type)
    return carry_score