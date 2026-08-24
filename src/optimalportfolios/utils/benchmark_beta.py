"""Compute benchmark-beta loadings and ex-ante portfolio beta.

The covariance-based helpers in this module keep benchmark-beta analytics
consistent with the covariance matrix used for portfolio optimisation.  In
particular, the joint-covariance variant is the same labelled matrix slice
used by :class:`qis.RiskModel`.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _validate_benchmark_variance(benchmark_variance: float) -> None:
    """Require a finite, strictly positive benchmark variance."""
    if not np.isfinite(benchmark_variance):
        raise ValueError(
            f"benchmark variance must be finite and positive, got {benchmark_variance}")
    if benchmark_variance <= 0.0:
        raise ValueError(f"benchmark variance must be positive, got {benchmark_variance}")


def _validate_beta_loadings(beta_loadings: pd.Series) -> None:
    """Require every computed benchmark-beta loading to be finite."""
    if not np.isfinite(beta_loadings.to_numpy()).all():
        raise ValueError("benchmark beta loadings must be finite")


def compute_benchmark_beta_loadings(asset_betas: pd.DataFrame,
                                    benchmark_betas: pd.Series,
                                    factor_covar: pd.DataFrame,
                                    benchmark_idio_var: float = 0.0,
                                    ) -> pd.Series:
    """Per-asset loadings of portfolio beta to a benchmark under a factor model.

    With joint factor covariance F, asset loadings B_a (assets x factors),
    benchmark loadings b (factors,) and benchmark idiosyncratic variance
    d_idio, the ex-ante beta of portfolio w to the benchmark is linear:

        beta(w) = w' @ beta_loadings,
        beta_loadings = (B_a @ F @ b) / (b' @ F @ b + d_idio)

    The cross-covariance carries no idiosyncratic term (factor-model
    residuals are independent across instruments), so only the benchmark
    variance in the denominator picks up its idio component.

    Args:
        asset_betas: Factor loadings of assets (assets x factors).
        benchmark_betas: Factor loadings of the benchmark (indexed by factor).
        factor_covar: Factor covariance F (factors x factors).
        benchmark_idio_var: Benchmark idiosyncratic variance (same
            periodicity as factor_covar).

    Returns:
        pd.Series of loadings indexed by asset; beta(w) = loadings @ w.

    Raises:
        ValueError: If benchmark variance is not finite and positive or the
            resulting loadings are non-finite.
    """
    factors = factor_covar.index
    b = benchmark_betas.reindex(factors).fillna(0.0).to_numpy()
    ba = asset_betas.reindex(columns=factors).fillna(0.0).to_numpy()
    f = factor_covar.to_numpy()
    fb = f @ b
    denom = float(b @ fb) + float(benchmark_idio_var)
    _validate_benchmark_variance(denom)
    beta_loadings = pd.Series(ba @ fb / denom, index=asset_betas.index)
    _validate_beta_loadings(beta_loadings)
    return beta_loadings


def compute_benchmark_beta_loadings_from_covar(covar: pd.DataFrame,
                                               benchmark_weights: pd.Series,
                                               asset_tickers: List[str],
                                               ) -> pd.Series:
    """Per-asset beta loadings sliced from ONE joint covariance matrix.

    The fully consistent variant of ``compute_benchmark_beta_loadings``:
    when the benchmark constituents are members of the same estimated
    covariance as the assets (one joint fit), the loadings are a pure
    slice — the beta the optimiser enforces then derives from the exact
    matrix its TRE terms use:

        c = Sigma[assets, cons] @ b / (b' Sigma[cons, cons] b),
        beta(w) = c' w

    Args:
        covar: Joint covariance (labelled DataFrame) covering assets AND
            benchmark constituents — e.g. one date of the extended-universe
            ``get_y_covars`` dict.
        benchmark_weights: Static benchmark composition indexed by
            constituent ticker (need not sum to 1; the ratio normalises).
        asset_tickers: Portfolio asset order of w.

    Returns:
        pd.Series of loadings indexed by ``asset_tickers``.

    Raises:
        KeyError: If a benchmark constituent is absent from ``covar``.
        ValueError: If benchmark variance is not finite and positive or the
            resulting loadings are non-finite.
    """
    cons = list(benchmark_weights.index)
    missing = [t for t in cons if t not in covar.index]
    if missing:
        raise KeyError(
            f"benchmark constituents missing from joint covariance: {missing} — "
            f"estimate covariance over the joint portfolio and benchmark universe")
    b = benchmark_weights.to_numpy()
    sig_ab = covar.loc[asset_tickers, cons].to_numpy()
    sig_bb = covar.loc[cons, cons].to_numpy()
    denom = float(b @ sig_bb @ b)
    _validate_benchmark_variance(denom)
    beta_loadings = pd.Series(sig_ab @ b / denom, index=asset_tickers)
    _validate_beta_loadings(beta_loadings)
    return beta_loadings


def compute_benchmark_beta_loadings_ts(
        covar_dict: Dict[pd.Timestamp, pd.DataFrame],
        benchmark_weights: pd.Series,
        asset_tickers: List[str],
) -> pd.DataFrame:
    """Compute joint-covariance beta loadings for each rebalancing date.

    Args:
        covar_dict: Covariance matrices over the joint portfolio-asset and
            benchmark-constituent universe, keyed by rebalancing date.
        benchmark_weights: Static benchmark composition indexed by constituent.
        asset_tickers: Portfolio assets in the optimiser's weight order.

    Returns:
        DataFrame indexed by sorted rebalancing date with assets in columns.

    Raises:
        KeyError: If any covariance omits a benchmark constituent.
        ValueError: If any date produces invalid benchmark variance or loadings.
    """
    rows = {
        date: compute_benchmark_beta_loadings_from_covar(
            covar=covar,
            benchmark_weights=benchmark_weights,
            asset_tickers=asset_tickers,
        )
        for date, covar in covar_dict.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def compute_ex_ante_beta_ts(
        weights: pd.DataFrame,
        beta_loadings: pd.DataFrame,
) -> pd.Series:
    """Compute the ex-ante portfolio-beta time series ``c_t @ w_t``.

    Loading observations are aligned to weight dates using as-of lookback.
    Every weight column must be represented in the loadings panel; otherwise
    a partial intersection could silently bias the reported beta toward zero.

    Args:
        weights: Portfolio weights indexed by observation date.
        beta_loadings: Per-asset benchmark-beta loadings indexed by estimation date.

    Returns:
        Ex-ante beta indexed like ``weights`` and named ``ex_ante_beta``.

    Raises:
        ValueError: If loadings are non-finite or do not cover every portfolio asset.
    """
    if not np.isfinite(beta_loadings.to_numpy()).all():
        raise ValueError("beta_loadings must contain only finite values")
    loadings = beta_loadings.reindex(index=weights.index, method="ffill")
    missing = [column for column in weights.columns if column not in loadings.columns]
    if missing:
        raise ValueError(
            f"beta loadings do not cover {len(missing)} weight columns "
            f"(first: {missing[:3]}) — pass asset_tickers matching the "
            f"portfolio universe to compute_benchmark_beta_loadings_ts")
    beta = (weights.fillna(0.0) * loadings[weights.columns]).sum(axis=1)
    beta.name = "ex_ante_beta"
    return beta
