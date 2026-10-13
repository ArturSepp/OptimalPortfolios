"""Construction-aware CMA audit on a frozen annual excess-return factor model.

The GLS audit leaves forecasts unchanged. Its diagonal residual geometry and
unconstrained Sharpe accounting are distinct from portfolio risk attribution in
qis.RiskModel and from the funding-neutral opportunity metric in this package.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from optimalportfolios.matf import sharpe_accounting as sa


@dataclass(frozen=True)
class AdmissionPolicy:
    """Explicit illustrative admission limits; these never alter a forecast."""

    name: str = 'paper_illustrative'
    max_positive_ir: float = 0.5
    max_admission_share: float = 0.5
    kappa: float = 0.5

    def __post_init__(self):
        """Require a named policy with finite nonnegative limits."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('policy name must be nonempty')
        for name in ('max_positive_ir', 'max_admission_share', 'kappa'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class CmaDiagnosticSpec:
    """Relative SVD cutoff and scale-aware numerical reconciliation tolerances."""

    rcond: float | None = None
    atol: float = 1e-10
    rtol: float = 1e-8

    def __post_init__(self):
        """Reject invalid reconciliation tolerances or singular-value cutoffs."""
        for name in ('atol', 'rtol'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.rcond is not None and (not np.isfinite(self.rcond) or not 0 <= self.rcond < 1):
            raise ValueError('rcond must be finite in [0, 1)')


@dataclass(frozen=True)
class CmaDiagnosticInputs:
    """Owned, labelled annual decimal moments for one exact universe and currency.

    Channels exclude B times calibrated premia. Scenarios contain absolute excess
    CMAs. Metadata and provenance are optional descriptive inputs, not estimators.
    The caller resolves currency, observation conventions and cash exclusions.
    """

    betas: pd.DataFrame
    residual_variances: pd.Series
    excess_cma: pd.Series
    factor_covariance: pd.DataFrame | None = None
    factor_premia: pd.Series | None = None
    channels: pd.DataFrame | None = None
    admitted_alpha: pd.Series | None = None
    scenarios: pd.DataFrame | None = None
    asset_metadata: pd.DataFrame | None = None
    production_covariance: pd.DataFrame | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        """Own copies so the audit cannot mutate the caller's CMA snapshot."""
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, deepcopy(getattr(self, name)))


@dataclass(frozen=True)
class CmaDiagnosticResult:
    """Full-precision audit tables consumed directly by reports and persistence."""

    summary: dict
    assets: pd.DataFrame
    factors: pd.DataFrame
    channels: pd.DataFrame
    policy_checks: pd.DataFrame
    scenarios: pd.DataFrame
    validation: pd.DataFrame
    provenance: dict
    factor_contributions: pd.DataFrame = field(default_factory=pd.DataFrame)


def _labelled(value, kind, labels, name, *, finite=True):
    """Validate an exactly aligned labelled vector or table without coercing axes."""
    if not isinstance(value, kind):
        raise ValueError(f'{name} must be a labelled {kind.__name__}')
    if value.index.has_duplicates or value.index.hasnans or not value.index.equals(labels):
        raise ValueError(f'{name} must match the exact unique index')
    if isinstance(value, pd.DataFrame) and (
            value.columns.has_duplicates or value.columns.hasnans):
        raise ValueError(f'{name} columns must be unique and nonmissing')
    dtypes = value.dtypes if isinstance(value, pd.DataFrame) else [value.dtype]
    if finite and (not all(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes)
                   or np.iscomplexobj(value)
                   or not np.isfinite(value.to_numpy(dtype=float)).all()):
        raise ValueError(f'{name} must contain finite real values')
    return value


def _check(rows, name, error, scale, spec):
    """Append a scale-aware identity check without concealing approximation errors."""
    tolerance = spec.atol + spec.rtol * scale
    rows.append(dict(check=name, error=float(error), tolerance=float(tolerance),
                     status='pass' if error <= tolerance else 'fail'))


def _max_abs(value):
    """Return the maximum absolute magnitude, including empty optional panels."""
    return float(np.max(np.abs(value), initial=0.))


def _ratio(numerator, denominator):
    """Leave a ratio unavailable when its denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else None


def _moments(inputs, ids, factors):
    """Validate optional covariance/premia; preserve a singular PSD factor matrix."""
    premia = inputs.factor_premia
    if premia is not None:
        _labelled(premia, pd.Series, factors, 'factor_premia')
    covariance = inputs.factor_covariance
    positive = False
    if covariance is not None:
        _labelled(covariance, pd.DataFrame, factors, 'factor_covariance')
        if not covariance.columns.equals(factors):
            raise ValueError('factor_covariance columns must match the factor axis')
        matrix = covariance.to_numpy(dtype=float)
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError('factor_covariance must be symmetric')
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() < -1e-12 * max(1., _max_abs(matrix)):
            raise ValueError('factor_covariance must be positive semidefinite')
        positive = eigenvalues.min() > 0.
    production = inputs.production_covariance
    if production is not None:
        _labelled(production, pd.DataFrame, ids, 'production_covariance')
        if not production.columns.equals(ids):
            raise ValueError('production_covariance columns must match assets')
        sa._covariance(production, len(ids), 'production_covariance')
    return covariance, premia, positive


def _admission(inputs, ids, spec):
    """Resolve admitted alpha and independently validate its declared channel."""
    admission = inputs.admitted_alpha
    if admission is not None:
        _labelled(admission, pd.Series, ids, 'admitted_alpha')
    if inputs.channels is not None and 'admitted_alpha' in inputs.channels:
        channel = inputs.channels.admitted_alpha
        if admission is not None and not np.allclose(
                admission, channel, atol=spec.atol, rtol=spec.rtol):
            raise ValueError('admitted_alpha differs from its construction channel')
        admission = channel
    return admission


def without_admitted_alpha(inputs: CmaDiagnosticInputs, *, spec=None) -> CmaDiagnosticInputs:
    """Create a diagnostic counterfactual removing only the supplied admitted alpha.

    Keep risk, calibrated premia, every other construction channel and scenario
    shocks fixed. Unknown admission is an error, not an assumption of zero.
    The returned input owns its data; the production forecast is never modified.
    """
    admission = _admission(inputs, inputs.betas.index, spec or CmaDiagnosticSpec())
    if admission is None:
        raise ValueError('admitted_alpha is not supplied; cannot construct a zero-alpha view')
    channels = inputs.channels.copy() if inputs.channels is not None else None
    if channels is not None and 'admitted_alpha' in channels:
        channels['admitted_alpha'] = 0.
    scenarios = (inputs.scenarios.sub(admission, axis=0)
                 if inputs.scenarios is not None else None)
    return replace(inputs, excess_cma=inputs.excess_cma-admission, channels=channels,
                   admitted_alpha=admission * 0., scenarios=scenarios,
                   provenance={**inputs.provenance, 'alpha_view': 'zero_admitted_alpha',
                               'alpha_view_convention': 'Subtract specified admission from base '
                               'and scenario CMAs; all other inputs and shocks held fixed.'})


def _policy_table(assets, summary, policy):
    """Measure configured policy limits with explicit applicability/availability."""
    columns = ['asset', 'check', 'value', 'limit', 'status']
    rows = []
    if policy is None:
        return pd.DataFrame(columns=columns)
    for asset, row in assets.iterrows():
        for check, field_name, limit, applicable in (
                ('cap1_positive_ir', 'admission_ir', policy.max_positive_ir,
                 row.admitted_alpha > 0),
                ('cap2_admission_share', 'admission_share', policy.max_admission_share,
                 row.excess_cma > 0)):
            value = row[field_name]
            status = ('unavailable' if pd.isna(row.admitted_alpha) else
                      'not_applicable' if not applicable else
                      'breach' if value > limit else 'pass')
            rows.append(dict(asset=asset, check=check, value=value, limit=limit, status=status))
    budget = summary['admission_budget']
    score = summary['admission_score']
    status = ('unavailable' if budget is None or score is None else
              'breach' if score > budget else 'pass')
    rows.append(dict(asset='__universe__', check='cap3_admission_budget',
                     value=score, limit=budget, status=status))
    return pd.DataFrame(rows, columns=columns)


def _risk_accounting(inputs, geometry, fitted, mean, summary, validation, spec, positive):
    """Reconcile forecast Sharpe in asset space and calibrated factor access."""
    if inputs.factor_covariance is None:
        summary['availability']['risk_accounting'] = 'factor_covariance_not_supplied'
        return
    b, d = geometry.beta, geometry.residual
    f = inputs.factor_covariance.to_numpy(dtype=float)
    covariance = b @ f @ b.T + d
    sa._covariance(covariance, len(b), 'audit_covariance')
    total = float(mean @ np.linalg.solve(covariance, mean))
    systematic = float(fitted @ np.linalg.solve(covariance, fitted))
    summary.update(total_sharpe2=total, implied_systematic_sharpe2=systematic,
                   total_sharpe=float(np.sqrt(max(0., total))))
    _check(validation, 'sharpe_accounting', abs(total-systematic-summary['residual_sharpe2']),
           max(total, systematic+summary['residual_sharpe2']), spec)
    if inputs.production_covariance is not None:
        _check(validation, 'production_covariance_reconciliation',
               _max_abs(covariance-inputs.production_covariance.to_numpy()),
               _max_abs(covariance), spec)
    if inputs.factor_premia is None:
        summary['availability']['factor_access'] = 'factor_premia_not_supplied'
        return
    calibrated = b @ inputs.factor_premia.to_numpy(dtype=float)
    summary['calibrated_systematic_sharpe2'] = float(
        calibrated @ np.linalg.solve(covariance, calibrated))
    if positive:
        ceiling = sa.factor_ceiling(inputs.factor_premia, inputs.factor_covariance)
        summary.update(factor_ceiling2=ceiling,
                       fpir=_ratio(summary['calibrated_systematic_sharpe2'], ceiling))
        if ceiling == 0:
            summary['availability']['fpir'] = 'zero_factor_ceiling'
    else:
        summary['availability']['factor_access'] = 'factor_covariance_not_positive_definite'


def compute_cma_diagnostics(inputs: CmaDiagnosticInputs, *, spec=None, policy=None):
    """Audit one frozen universe without fitting, repairing or changing its forecasts.

    Returns full-precision labelled tables, numerical integrity checks and explicit
    unavailable statuses. A truncated nonzero loading direction can invalidate the
    original covariance identity: this is reported as a failed reconciliation.
    Invalid inputs and an unexplained construction bridge raise ValueError.
    """
    if not isinstance(inputs, CmaDiagnosticInputs):
        raise TypeError('inputs must be CmaDiagnosticInputs')
    spec = CmaDiagnosticSpec() if spec is None else spec
    if not isinstance(spec, CmaDiagnosticSpec):
        raise TypeError('spec must be CmaDiagnosticSpec')
    if policy is not None and not isinstance(policy, AdmissionPolicy):
        raise TypeError('policy must be AdmissionPolicy')
    if not isinstance(inputs.betas, pd.DataFrame):
        raise ValueError('betas must be a labelled DataFrame')
    ids, factor_ids = inputs.betas.index, inputs.betas.columns
    _labelled(inputs.betas, pd.DataFrame, ids, 'betas')
    _labelled(inputs.residual_variances, pd.Series, ids, 'residual_variances')
    _labelled(inputs.excess_cma, pd.Series, ids, 'excess_cma')
    geometry = sa._gls_geometry(inputs.betas, inputs.residual_variances, rcond=spec.rcond)
    _, premia, positive = _moments(inputs, ids, factor_ids)
    for name in ('channels', 'scenarios', 'asset_metadata'):
        value = getattr(inputs, name)
        if value is not None:
            _labelled(value, pd.DataFrame, ids, name, finite=name != 'asset_metadata')
    if inputs.channels is not None and premia is None:
        raise ValueError('construction channels require calibrated factor_premia')
    admission = _admission(inputs, ids, spec)
    b, d, p = geometry.beta, inputs.residual_variances.to_numpy(), geometry.residual_projector
    m = inputs.excess_cma.to_numpy(dtype=float)
    implied = geometry.white_pinv @ np.linalg.solve(geometry.chol, m)
    fitted = b @ implied
    h = p @ m
    score = float(h @ (h/d))
    rank, n, k = geometry.rank, len(ids), len(factor_ids)
    identification = 'saturated' if rank == n else 'full_rank' if rank == k else 'rank_deficient'
    summary = dict(schema_version='1.0', n_assets=n, n_factors=k, rank=rank,
                   residual_dimension=n-rank, identification_status=identification,
                   singular_values=geometry.singular_values.tolist(),
                   rank_cutoff=geometry.cutoff, relative_cutoff=geometry.relative_cutoff,
                   condition_number=(float(geometry.singular_values[0] /
                                           geometry.singular_values[-1])
                                     if rank == min(n, k) else None),
                   residual_sharpe2=score, total_sharpe2=None, total_sharpe=None,
                   implied_systematic_sharpe2=None, calibrated_systematic_sharpe2=None,
                   factor_ceiling2=None, fpir=None, admission_score=None,
                   admission_residual_score=None, admission_in_span_share=None,
                   admission_budget=None, admission_budget_ratio=None,
                   policy_name=policy.name if policy else None, availability={})
    # Projection roundoff is not economically meaningful contributor concentration.
    concentration_defined = score > max(float(m @ (m/d)), 1e-300) * 1e-24
    contributions = h*h/d
    shares = contributions/score if concentration_defined else np.full(n, np.nan)
    summary.update(top1_residual_share=float(np.max(shares)) if concentration_defined else None,
                   top5_residual_share=float(np.sort(shares)[-5:].sum())
                   if concentration_defined else None)
    assets = pd.DataFrame(dict(excess_cma=m, fitted_excess_cma=fitted, residual=h,
                               residual_vol=np.sqrt(d), standardized_residual=h/np.sqrt(d),
                               residual_score=contributions, residual_share=shares), index=ids)
    for name in ('admitted_alpha', 'admission_ir', 'admission_share', 'projected_admission'):
        assets[name] = np.nan
    if admission is not None:
        a = admission.to_numpy(dtype=float)
        ha = p @ a
        raw, projected = float(a @ (a/d)), float(ha @ (ha/d))
        assets['admitted_alpha'], assets['projected_admission'] = a, ha
        assets['admission_ir'] = a/np.sqrt(d)
        assets['admission_share'] = np.divide(a, m, out=np.full(n, np.nan), where=m > 0)
        summary.update(admission_score=raw, admission_residual_score=projected,
                       admission_in_span_share=1.-projected/raw if raw > 0 else None)
    else:
        summary['availability']['admission'] = 'admitted_alpha_not_supplied'
    if inputs.asset_metadata is not None:
        overlap = assets.columns.intersection(inputs.asset_metadata.columns)
        if len(overlap):
            raise ValueError(f'asset metadata shadows calculated columns: {list(overlap)}')
        assets = assets.join(inputs.asset_metadata)
    validation = []
    _check(validation, 'forecast_reconstruction', _max_abs(m-fitted-h), _max_abs(m), spec)
    _check(validation, 'projector_idempotence', _max_abs(p@p-p), _max_abs(p), spec)
    _check(validation, 'projector_factor_annihilation', _max_abs(p@b), _max_abs(b), spec)
    _check(validation, 'weighted_self_adjointness', _max_abs(p.T/d-p/d[:, None]),
           _max_abs(p/d[:, None]), spec)
    _check(validation, 'gls_orthogonality', _max_abs(b.T@(h/d)),
           _max_abs(b.T@(m/d)), spec)
    _risk_accounting(inputs, geometry, fitted, m, summary, validation, spec, positive)
    calibrated = premia.to_numpy() if premia is not None else np.full(k, np.nan)
    identified = rank == k
    factors = pd.DataFrame(dict(calibrated_premium=calibrated, implied_premium=implied,
                                revision=implied-calibrated,
                                identification_scale=(np.sqrt(np.sum(geometry.white_pinv**2,
                                                                     axis=1))
                                                      if identified else np.full(k, np.nan)),
                                status='identified' if identified else 'minimum_norm'),
                           index=factor_ids)
    # These are GLS premium-revision contributions, not portfolio risk contributions.
    # qis.RiskModel attributes portfolio risk and does not decompose fitted CMA premia.
    factor_contributions = pd.DataFrame()
    factors['revision_attribution_offset'] = np.nan
    if premia is not None:
        weights = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T).T
        adjustment = m-b@calibrated
        factor_contributions = pd.DataFrame((weights*adjustment[None, :]).T,
                                            index=ids, columns=factor_ids)
        # With deficient rank, A B is a coordinate projection, not the identity.
        # Keep this non-asset offset explicit instead of allocating it to assets.
        offset = weights@(b@calibrated)-calibrated
        factors['revision_attribution_offset'] = offset
        _check(validation, 'factor_revision_attribution',
               _max_abs(factor_contributions.sum().to_numpy()+offset-(implied-calibrated)),
               max(_max_abs(implied), _max_abs(calibrated)), spec)
    channel_rows = []
    if inputs.channels is not None:
        reconstructed = b@calibrated+inputs.channels.sum(axis=1).to_numpy()
        if not np.allclose(reconstructed, m, atol=spec.atol, rtol=spec.rtol):
            raise ValueError('CMA construction channels do not reconcile to excess_cma')
        _check(validation, 'construction_reconciliation', _max_abs(reconstructed-m),
               _max_abs(m), spec)
        for name in inputs.channels:
            values = inputs.channels[name].to_numpy(dtype=float)
            projected = p@values
            channel_rows.append(dict(channel=name, raw_score=float(values@(values/d)),
                                     projected_score=float(projected@(projected/d)),
                                     residual_attribution=float(projected@(h/d))))
    channels = pd.DataFrame(channel_rows, columns=['channel', 'raw_score', 'projected_score',
                                                  'residual_attribution']).set_index('channel')
    if policy is not None and summary['calibrated_systematic_sharpe2'] is not None:
        budget = policy.kappa*summary['calibrated_systematic_sharpe2']
        summary['admission_budget'] = budget
        if summary['admission_score'] is not None:
            summary['admission_budget_ratio'] = _ratio(summary['admission_score'], budget)
    scenarios = inputs.scenarios.copy() if inputs.scenarios is not None else pd.DataFrame(index=ids)
    checks = pd.DataFrame(validation)
    summary['accounting_status'] = 'pass' if checks.status.eq('pass').all() else 'fail'
    provenance = deepcopy(inputs.provenance)
    provenance.update(audit_covariance='B F B.T + diagonal D',
                      return_units='annual_arithmetic_excess_decimal',
                      asset_order=ids.tolist(), factor_order=factor_ids.tolist())
    return CmaDiagnosticResult(summary, assets, factors, channels,
                              _policy_table(assets, summary, policy), scenarios, checks, provenance,
                              factor_contributions=factor_contributions)
