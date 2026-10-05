"""Subset-size, concentration and removal diagnostics for frozen alpha estimates.

These descriptive calculations use the approved diagonal residual model. They do
not identify independent bets, estimate sampling uncertainty in alpha, or solve a
long-only frontier. CMA alpha, betas and residual variances are held fixed; the
derived GLS projection is recomputed on each subset to evaluate that subset's K.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optimalportfolios.matf.opportunity import (
    OpportunityModel, _alpha_dispersion_inputs, alpha_dispersion,
)


def subset_dispersion(model: OpportunityModel, members):
    """Return canonical dispersion for an exact nonempty subset of model identities."""
    ids = pd.Index(members)
    if ids.empty or ids.has_duplicates or not ids.isin(model.betas.index).all():
        raise ValueError('members must be unique known asset identities in a nonempty subset')
    return _alpha_dispersion_inputs(model.betas.loc[ids], model.residual_variances.loc[ids],
                                    model.alpha.loc[ids])


def dispersion_concentration(model: OpportunityModel):
    """Return A/J/K concentration and ranked instrument contributions.

    Effective contributors equals one over the sum of squared contribution shares.
    It is not independent breadth. If a capacity is at most 1e-24 times A, its
    residual norm is at most 1e-12 of the alpha norm: shares are marked unavailable
    rather than interpreting floating-point projection error as opportunity.
    The underlying reported capacities are never floored or altered.
    """
    metrics, detail = alpha_dispersion(model)
    rows, instruments = [], []
    threshold = metrics['A'] * 1e-24
    for name in ('A', 'J', 'K'):
        contributions = detail[name + '_contribution'].sort_values(ascending=False, kind='stable')
        defined = metrics[name] > threshold
        shares = contributions / metrics[name] if defined else contributions * np.nan
        cumulative = shares.cumsum()
        rows.append(dict(metric=name, capacity=metrics[name], N=len(contributions),
                         status='defined' if defined else 'numerically_zero',
                         effective_contributors=float(1. / (shares @ shares))
                         if defined else np.nan,
                         top1_share=float(shares.iloc[0]) if defined else np.nan,
                         top5_share=float(shares.iloc[:5].sum()) if defined else np.nan,
                         top10_share=float(shares.iloc[:10].sum()) if defined else np.nan,
                         instruments_for_50pct=int(np.searchsorted(cumulative, .5) + 1)
                         if defined else np.nan,
                         instruments_for_80pct=int(np.searchsorted(cumulative, .8) + 1)
                         if defined else np.nan))
        instruments.append(pd.DataFrame(dict(asset=contributions.index, metric=name,
                                              contribution=contributions.to_numpy(),
                                              share=shares.to_numpy(),
                                              cumulative_share=cumulative.to_numpy(),
                                              rank=np.arange(1, len(contributions) + 1))))
    return pd.DataFrame(rows).set_index('metric'), pd.concat(instruments, ignore_index=True)


def removal_diagnostics(model: OpportunityModel):
    """Recompute A/J/K after each single removal under unchanged fitted moments.

    True removal loss differs from the asset's current contribution because the
    GLS projection changes. It also equals the gain from adding that asset back
    to the remaining universe; it is not a unique additive attribution. Direct
    recomputation remains valid when removal lowers factor/funding rank.
    """
    if len(model.betas) < 2:
        raise ValueError('removal diagnostics require at least two assets')
    full, detail = alpha_dispersion(model)
    rows = []
    for asset in model.betas.index:
        remaining = model.betas.index[model.betas.index != asset]
        reduced, _ = subset_dispersion(model, remaining)
        row = dict(asset=asset, rank_before=full['rank_funding_B'],
                   rank_after=reduced['rank_funding_B'], alpha=float(model.alpha[asset]),
                   K_direct_contribution=float(detail.loc[asset, 'K_contribution']))
        for name in ('A', 'J', 'K'):
            row[name + '_after'] = reduced[name]
            row[name + '_loss'] = full[name] - reduced[name]
            row[name + '_loss_fraction'] = row[name + '_loss'] / full[name] \
                if full[name] > full['A'] * 1e-24 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('asset')


def bundle_removal_diagnostics(model: OpportunityModel, bundles):
    """Measure named joint removals, including departure from summed single losses."""
    full, _ = alpha_dispersion(model)
    singles = removal_diagnostics(model)
    rows = []
    for name, members in bundles.items():
        ids = pd.Index(members)
        if ids.empty or ids.has_duplicates or not ids.isin(model.betas.index).all():
            raise ValueError('bundle members must be unique known asset identities')
        remaining = model.betas.index[~model.betas.index.isin(ids)]
        reduced, _ = subset_dispersion(model, remaining)
        row = dict(bundle=name, removed_N=len(ids), remaining_N=len(remaining),
                   rank_after=reduced['rank_funding_B'])
        for metric in ('A', 'J', 'K'):
            loss = full[metric] - reduced[metric]
            single_sum = float(singles.loc[ids, metric + '_loss'].sum())
            row.update({metric + '_loss': loss, metric + '_sum_single_losses': single_sum,
                        metric + '_nonadditivity': loss - single_sum})
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_alpha_covariance(model, covariance):
    """Delegate PSD validation to FactorLasso once for a batch of subsets."""
    from factorlasso import gaussian_quadratic_summary
    if covariance is not None:
        if (not isinstance(covariance, pd.DataFrame)
                or not covariance.index.equals(model.betas.index)
                or not covariance.columns.equals(model.betas.index)):
            raise ValueError('alpha covariance must match the exact asset order')
        gaussian_quadratic_summary(model.alpha, covariance, np.eye(len(model.betas)))


def _subset_summary(model, ids, covariance):
    """Recompute subset geometry, concentration and optional conditional noise."""
    from optimalportfolios.matf.sharpe_accounting import gls_projector
    metrics, detail = subset_dispersion(model, ids)
    contributions = detail.K_contribution.to_numpy()
    metrics['K_effective_contributors'] = (metrics['K']**2/(contributions @ contributions)
        if metrics['K'] > metrics['A']*1e-24 else np.nan)
    metrics['neutral_dimension'] = len(ids)-metrics['rank_funding_B']
    if covariance is not None:
        joint = pd.concat([pd.Series(1., index=ids, name='funding'),
                           model.betas.loc[ids]], axis=1, sort=False)
        p = gls_projector(joint, model.residual_variances.loc[ids])
        metric = p.T @ (p/model.residual_variances.loc[ids].to_numpy()[:, None])
        noise = float(np.sum(metric*covariance.loc[ids, ids].to_numpy().T))
        adjusted = metrics['K']-noise
        metrics.update(K_noise=noise, K_noise_adjusted=adjusted,
                       K_adjusted_per_asset=adjusted/len(ids),
                       K_adjusted_rms=np.sqrt(adjusted/len(ids)) if adjusted >= 0 else np.nan)
    return metrics


def compare_universe_dispersion(model: OpportunityModel, baseline, scenarios, *,
                                 alpha_covariance=None):
    """Compare exact named asset sets on one unchanged fitted model and covariance.

    ``scenarios`` maps unique names to nonempty sets of model identities; baseline
    is reserved. Return levels, signed scenario-minus-baseline changes and exact
    memberships. Both the GLS projection and noise trace are recomputed for each
    set, including rank changes. Negative noise-adjusted dispersion is retained.
    These are conditional descriptive changes, not independent contributions,
    calibrated confidence intervals, or portfolio opportunity gains.
    """
    if not scenarios or 'baseline' in scenarios:
        raise ValueError('provide named scenarios; baseline is reserved')
    _validate_alpha_covariance(model, alpha_covariance)
    rows, members = [], []
    for name, selection in [('baseline', baseline), *scenarios.items()]:
        if not isinstance(name, str) or not name:
            raise ValueError('scenario names must be nonempty strings')
        ids = pd.Index(selection)
        metrics = _subset_summary(model, ids, alpha_covariance)
        rows.append(dict(scenario=name, **metrics))
        members.append(pd.DataFrame(dict(scenario=name, asset=ids)))
    table = pd.DataFrame(rows).set_index('scenario')
    deltas = table.subtract(table.loc['baseline'], axis=1).add_prefix('delta_')
    return pd.concat([table, deltas], axis=1, sort=False), pd.concat(members, ignore_index=True)


def sample_universe_dispersion(model: OpportunityModel, sizes, *, repetitions=1000, seed=20261004,
                                alpha_covariance=None):
    """Sample exact-size subsets without replacement using a local NumPy generator.

    Return one row per draw plus its exact membership. Full-universe size is
    evaluated once. The same seed/order reproduces membership across alpha
    policies; no state is taken from NumPy's global generator. Bands across draws
    measure subset composition variation, not estimation confidence intervals.
    Optional aligned alpha-estimation covariance adds conditional plug-in noise
    adjustments for K using a new projection on each subset. Negative adjusted
    values are retained; their RMS is undefined, not floored to zero.
    """
    n = len(model.betas)
    _validate_alpha_covariance(model, alpha_covariance)
    sizes = list(sizes)
    if not sizes or any(isinstance(s, bool) or not isinstance(s, (int, np.integer))
                        or s < 1 or s > n for s in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError('sizes must be distinct integers between one and universe size')
    if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)) \
            or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    rng = np.random.default_rng(seed)
    rows, memberships = [], []
    for size in sizes:
        for draw in range(1 if size == n else repetitions):
            positions = np.arange(n) if size == n else np.sort(
                rng.choice(n, size=size, replace=False))
            ids = model.betas.index[positions]
            metrics = _subset_summary(model, ids, alpha_covariance)
            sample_id = f'n{size}_draw{draw}'
            rows.append(dict(sample_id=sample_id, sample_size=size, draw=draw,
                             neutral_dimension=metrics.pop('neutral_dimension'), **metrics))
            memberships.append(pd.DataFrame(dict(sample_id=sample_id, asset=ids,
                                                  asset_position=positions)))
    return pd.DataFrame(rows), pd.concat(memberships, ignore_index=True)
