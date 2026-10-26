"""Constrained opportunity optimisation on supplied labelled risk and alpha arrays.

The caller owns estimation, units, currency, statistical projections and inference.
Full covariance is authoritative. Factor and residual components are optional
prepared reporting/constraint inputs, never inferred or fitted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cvx
import numpy as np
import pandas as pd
import qis

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import validate_solution


def _align(value, labels):
    """Require exact labelled order without sorting or repairing caller inputs."""
    if not isinstance(value, (pd.Series, pd.DataFrame)) or not value.index.equals(labels):
        raise ValueError('label/order mismatch or duplicate labels')
    if np.iscomplexobj(value):
        raise ValueError('supplied inputs must be real')
    if value.index.has_duplicates or not np.isfinite(value.to_numpy()).all():
        raise ValueError('label/order mismatch or nonfinite values')


def _vector(value, labels, name):
    """Validate a finite labelled vector on the supplied asset axis."""
    if not isinstance(value, pd.Series):
        raise ValueError(f'{name} must be a labelled Series')
    _align(value, labels)
    return value.to_numpy(dtype=float)


def _covariance(value, labels, name):
    """Validate a supplied PSD covariance, preserving exact order and null risk."""
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f'{name} must be a labelled DataFrame')
    _align(value, labels)
    if not value.columns.equals(labels) or not np.allclose(value, value.T, rtol=0, atol=1e-12):
        raise ValueError(f'{name} must have symmetric values and exact label/order')
    factorize_covariance(value.to_numpy(), eigenvalue_floor=0.)
    return value.copy(deep=True)


@dataclass(frozen=True)
class OpportunityModel:
    """Complete supplied inputs for constrained opportunity optimisation.

    Covariance and alpha use the caller's common horizon/currency; no scaling is
    performed. Optional factor loadings/covariance and full residual covariance
    support risk limits and component reporting. Their sum must reconcile with
    the authoritative covariance when all components are supplied. Orthogonal
    alpha is prepared upstream, including its funding and GLS convention.
    """

    covariance: pd.DataFrame
    alpha: pd.Series
    factor_loadings: pd.DataFrame | None = None
    factor_covariance: pd.DataFrame | None = None
    residual_covariance: pd.DataFrame | None = None
    orthogonal_alpha: pd.Series | None = None
    date: pd.Timestamp = pd.Timestamp('2000-01-01')
    risk_model: object = field(init=False, repr=False, compare=False)
    factor_risk_model: object = field(init=False, repr=False, compare=False)
    residual_risk_model: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate owned inputs and create QIS views without estimating a model."""
        if not isinstance(self.alpha, pd.Series):
            raise ValueError('alpha must be a labelled Series')
        ids = self.alpha.index
        _vector(self.alpha, ids, 'alpha')
        if ids.empty:
            raise ValueError('alpha must contain at least one asset')
        object.__setattr__(self, 'alpha', self.alpha.copy(deep=True))
        covariance = _covariance(self.covariance, ids, 'covariance')
        object.__setattr__(self, 'covariance', covariance)
        date = pd.Timestamp(self.date)
        if pd.isna(date):
            raise ValueError('date must be finite')
        object.__setattr__(self, 'date', date)
        object.__setattr__(self, 'risk_model', qis.RiskModel(covar={date: covariance.copy()}))
        if (self.factor_loadings is None) != (self.factor_covariance is None):
            raise ValueError('factor loadings and covariance must be supplied together')
        factor_model = None
        systematic = None
        if self.factor_loadings is not None:
            if not isinstance(self.factor_loadings, pd.DataFrame):
                raise ValueError('factor_loadings must be a labelled DataFrame')
            _align(self.factor_loadings, ids)
            b = self.factor_loadings.copy(deep=True)
            f = _covariance(self.factor_covariance, b.columns, 'factor_covariance')
            try:
                np.linalg.cholesky(f)
            except np.linalg.LinAlgError as error:
                raise ValueError('factor_covariance must be positive definite') from error
            object.__setattr__(self, 'factor_loadings', b)
            object.__setattr__(self, 'factor_covariance', f)
            systematic = b @ f @ b.T
            factor_model = qis.RiskModel(covar={date: systematic})
        object.__setattr__(self, 'factor_risk_model', factor_model)
        residual_model = None
        if self.residual_covariance is not None:
            residual = _covariance(self.residual_covariance, ids, 'residual_covariance')
            object.__setattr__(self, 'residual_covariance', residual)
            residual_model = qis.RiskModel(covar={date: residual.copy()})
            if systematic is not None and not np.allclose(
                    covariance, systematic + residual, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    'supplied factor/residual components do not reconcile to covariance')
        object.__setattr__(self, 'residual_risk_model', residual_model)
        if self.orthogonal_alpha is not None:
            _vector(self.orthogonal_alpha, ids, 'orthogonal_alpha')
            object.__setattr__(self, 'orthogonal_alpha', self.orthogonal_alpha.copy(deep=True))


def _tracking_error(risk, reference, weights, date):
    """Delegate risk to QIS, preserving cancellation at an exact PSD null."""
    with np.errstate(invalid='ignore'):
        value = risk.compute_tre_at_date(reference, weights, date)
    if np.isfinite(value):
        return value
    covariance = risk.covar[date]
    root = factorize_covariance(covariance.to_numpy(), eigenvalue_floor=0.).factor
    positions = pd.Series(root.T @ (weights-reference).to_numpy(), index=covariance.index)
    coordinates = qis.RiskModel(covar={date: pd.DataFrame(
        np.eye(len(positions)), index=positions.index, columns=positions.index)})
    return coordinates.compute_tre_at_date(positions*0., positions, date)


def portfolio_opportunity_metrics(model, weights, reference=None):
    """Report supplied-book funding, alpha and risk without statistical inference.

    Returns generic metrics, optional factor exposures/contributions, and actual
    holdings. GLS decomposition and statistical attribution are upstream reports.
    """
    _vector(weights, model.alpha.index, 'weights')
    if reference is None:
        reference = weights * 0.
    _vector(reference, model.alpha.index, 'reference')
    delta = weights - reference
    date = model.date
    rows = dict(net_exposure=float(weights.sum()),
                reference_net_exposure=float(reference.sum()),
                min_weight=float(weights.min()), max_weight=float(weights.max()),
                concentration=float(weights @ weights), alpha_level=float(model.alpha @ weights),
                active_l1=float(delta.abs().sum()),
                held_assets=int((weights.abs() > 1e-7).sum()),
                alpha_increment=float(model.alpha @ delta),
                total_te=float(_tracking_error(model.risk_model, reference, weights, date)),
                portfolio_vol=float(_tracking_error(model.risk_model, weights*0., weights, date)))
    exposures = pd.DataFrame()
    if model.factor_risk_model is not None:
        factor_te = _tracking_error(model.factor_risk_model, reference, weights, date)
        factor_vol = _tracking_error(model.factor_risk_model, weights*0., weights, date)
        rows.update(factor_te=float(factor_te), factor_variance=float(factor_vol)**2)
        f = model.factor_loadings.T @ weights
        df = model.factor_loadings.T @ delta
        fm = qis.RiskModel(covar={date: model.factor_covariance})
        marginal = fm.compute_marginal_tre_at_date(f*0., f, date)
        active = fm.compute_marginal_tre_at_date(df*0., df, date)
        exposures = pd.DataFrame(dict(exposure=f, active_exposure=df,
            variance_contribution=marginal.mcte*factor_vol,
            active_variance_contribution=active.mcte*factor_te))
    if model.residual_risk_model is not None:
        rows.update(
            residual_te=float(_tracking_error(model.residual_risk_model, reference, weights, date)),
            residual_variance=float(_tracking_error(
                model.residual_risk_model, weights*0., weights, date))**2)
    holdings = pd.DataFrame(dict(weight=weights, reference=reference, active_weight=delta))
    return rows, exposures, holdings


@dataclass(frozen=True)
class OpportunitySolution:
    """A validated optimum or explicit failure, never fallback portfolio holdings."""

    status: str
    accepted: bool
    weights: pd.Series | None
    metrics: dict
    objective_value: float | None = None
    max_constraint_violation: float | None = None


def _constraint_order(constraints, index):
    """Check all asset-indexed fields before compiling positional solver rows."""
    fields = ('min_weights', 'max_weights', 'benchmark_weights', 'weights_0',
              'turnover_costs', 'asset_returns')
    for name in fields:
        value = getattr(constraints, name)
        if value is not None:
            _align(value, index)
            _vector(value, index, name)
    nested = dict(group_lower_upper_constraints='group_loadings',
                  group_tracking_error_constraint='group_loadings',
                  group_turnover_constraint='group_loadings',
                  sector_deviation_constraints='factor_loading_mat',
                  style_deviation_constraints='factor_loading_mat',
                  benchmark_beta_constraint='beta_loadings',
                  linear_constraints='loadings')
    for name, attribute in nested.items():
        value = getattr(constraints, name)
        if value is not None:
            _align(getattr(value, attribute), index)


def solve_opportunity(model, reference, *, investable=None, constraints=None,
                      total_te=None, factor_te=None, objective='upper', scores=None,
                      solver='CLARABEL', tolerance=1e-7, alpha_covariance=None,
                      uncertainty_radius=None):
    """Solve a long-only, unit-funded frontier on an unchanged joint risk model.

    Objectives: upper/lower statistical alpha, orthogonal_upper/orthogonal_lower,
    min_total_te, min_factor_te, or score (dimensionless, evaluated separately).
    OP Constraints compile mandate restrictions as hard rows. Additional correlated
    factor-TE norms have no counterpart in the existing Constraints API. Zero risk
    budgets use exact affine equalities. All returned holdings pass post-solve
    risk and compiled-row checks. Infeasibility returns no weights.

    Optional alpha_covariance and uncertainty_radius specify an ellipsoidal
    uncertainty penalty for alpha objectives. The caller owns joint calibration;
    a pointwise normal critical value does not imply simultaneous confidence.
    Covariance must describe the selected objective: callers supply any projected
    covariance for an orthogonal objective; OP performs no statistical projection.
    """
    _vector(reference, model.alpha.index, 'reference')
    if reference.min() < -tolerance or abs(reference.sum() - 1.) > 1e-6:
        raise ValueError('reference must be long-only with unit exposure; '
                         'do not normalize saved books')
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('tolerance must be finite and positive')
    uncertainty_root = None
    if (alpha_covariance is None) != (uncertainty_radius is None):
        raise ValueError('alpha covariance and uncertainty radius must be supplied together')
    if alpha_covariance is not None:
        if (objective not in ('upper', 'lower', 'orthogonal_upper', 'orthogonal_lower')
                or not np.isfinite(uncertainty_radius) or uncertainty_radius < 0):
            raise ValueError('nonnegative finite uncertainty radius requires an alpha objective')
        if (not isinstance(alpha_covariance, pd.DataFrame)
                or not alpha_covariance.index.equals(model.alpha.index)
                or not alpha_covariance.columns.equals(model.alpha.index)):
            raise ValueError('alpha covariance must match the exact asset order')
        v = alpha_covariance.to_numpy()
        # Calibration and any GLS projection are supplied by the caller.
        # OP validates numerical convexity without estimating uncertainty.
        _covariance(alpha_covariance, model.alpha.index, 'alpha_covariance')
        eig, vectors = np.linalg.eigh((v+v.T)/2)
        uncertainty_root = vectors*np.sqrt(np.maximum(eig, 0.))
    for name, value in [('total_te', total_te), ('factor_te', factor_te)]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    spec = constraints if constraints is not None else Constraints()
    if not spec.is_long_only or spec.min_exposure != 1 or spec.max_exposure != 1:
        raise ValueError('constraints must specify long-only unit exposure')
    if spec.constraint_enforcement_type.name != 'FORCED_CONSTRAINTS':
        raise ValueError('frontier mandate restrictions must be hard constraints')
    _constraint_order(spec, model.alpha.index)
    n = len(model.alpha)
    w = cvx.Variable(n)
    delta = w - reference.to_numpy()
    covariance = model.risk_model.covar[model.date].to_numpy()
    rows = spec.set_cvx_all_constraints(w, covar=cvx.psd_wrap(covariance))
    if investable is not None:
        _align(investable, model.alpha.index)
        if not isinstance(investable, pd.Series) or investable.dtype != bool:
            raise ValueError('investable must be a labelled boolean Series')
        if (~investable).any():
            rows += [w[np.flatnonzero(~investable)] == 0.]
    factor_coordinates = None
    if model.factor_loadings is not None:
        factor_coordinates = (np.linalg.cholesky(model.factor_covariance).T
                              @ model.factor_loadings.to_numpy().T @ delta)
    if factor_coordinates is None and (factor_te is not None or objective == 'min_factor_te'):
        raise ValueError(
            'factor constraints/objectives require supplied factor loadings/covariance')
    try:
        covariance_root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        # Fully retained empirical dependence may be rank deficient. Preserve
        # null risk directions using the canonical factorizer with a zero floor;
        # materially indefinite inputs still fail its validation.
        covariance_root = factorize_covariance(covariance, eigenvalue_floor=0.).factor
    total_coordinates = covariance_root.T @ delta
    for coordinates, cap in [(factor_coordinates, factor_te), (total_coordinates, total_te)]:
        if cap is not None:
            rows += [coordinates == 0.] if cap == 0 else [cvx.norm(coordinates, 2) <= cap]
    if objective in ('min_factor_te', 'min_total_te'):
        coordinates = factor_coordinates if objective == 'min_factor_te' else total_coordinates
        target = cvx.Minimize(cvx.norm(coordinates, 2))
    else:
        if objective in ('upper', 'lower'):
            values = model.alpha.to_numpy()
        elif objective in ('orthogonal_upper', 'orthogonal_lower'):
            if model.orthogonal_alpha is None:
                raise ValueError('orthogonal objectives require supplied orthogonal_alpha')
            values = model.orthogonal_alpha.to_numpy()
        elif objective == 'score':
            values = _vector(scores, model.alpha.index, 'scores')
        else:
            raise ValueError(f'unknown objective: {objective}')
        sign = -1. if objective in ('lower', 'orthogonal_lower') else 1.
        penalty = (uncertainty_radius*cvx.norm(uncertainty_root.T @ delta, 2)
                   if uncertainty_root is not None and uncertainty_radius > 0 else 0.)
        target = cvx.Maximize(sign * values @ delta - penalty)
    problem = cvx.Problem(target, rows)
    try:
        problem.solve(solver=solver)
    except cvx.error.SolverError as error:
        return OpportunitySolution('solver_error: ' + str(error), False, None, {})
    if problem.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE) or w.value is None:
        return OpportunitySolution(str(problem.status), False, None, {})
    accepted = validate_solution(w.value, problem.status, spec, n, solver=solver,
                                 context='opportunity', covar=covariance,
                                 budget_atol=tolerance, bound_atol=tolerance,
                                 constraint_atol=tolerance).accepted
    violation = max(float(np.max(row.violation())) for row in rows)
    weights = pd.Series(w.value, index=model.alpha.index)
    metrics = portfolio_opportunity_metrics(model, weights, reference)[0]
    if uncertainty_root is not None:
        metrics['alpha_estimation_standard_error'] = float(np.linalg.norm(
            uncertainty_root.T @ (weights-reference).to_numpy()))
        metrics['alpha_uncertainty_penalty'] = (
            uncertainty_radius*metrics['alpha_estimation_standard_error'])
    if objective == 'score':
        metrics['score_level'] = float(values @ weights)
        metrics['score_increment'] = float(values @ (weights - reference))
    accepted = accepted and violation <= tolerance
    for name, cap in [('total_te', total_te), ('factor_te', factor_te)]:
        accepted = accepted and (cap is None or metrics[name] <= cap + tolerance)
    return OpportunitySolution(str(problem.status) if accepted else 'constraint_violation',
                               bool(accepted), weights if accepted else None, metrics,
                               float(problem.value), violation)
