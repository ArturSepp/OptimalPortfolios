"""Long-only opportunity diagnostics on a frozen, annualized factor model.

Statistical alpha describes longer-horizon residual means, not tactical forecasts.
The caller owns estimation, currency, membership and source provenance. Invalid
assets must be resolved before construction; nothing is floored or normalized.
Risk reporting delegates to qis through OP's canonical risk-model adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cvx
import numpy as np
import pandas as pd
from factorlasso import CurrentFactorCovarData, ResidualCorrelationData, ResidualType

from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
from optimalportfolios.matf import sharpe_accounting as sa
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import validate_solution


@dataclass(frozen=True)
class OpportunityModel:
    """Owned labelled inputs and a canonical risk model; treat frames as immutable.

    Residual variances must be strictly positive. A joint asset axis may include
    reference-only instruments; an investability mask controls candidate holdings.
    All moments and alpha use annualized decimal units in one reference currency.
    Optional prepared residual correlation changes portfolio risk only; dispersion
    and GLS attribution retain the explicitly diagonal-D reference geometry.
    """

    betas: pd.DataFrame
    residual_variances: pd.Series
    factor_covariance: pd.DataFrame
    alpha: pd.Series
    date: pd.Timestamp = pd.Timestamp('2000-01-01')
    residual_correlation: ResidualCorrelationData | None = None
    residual_corr_weight: float = 1.
    risk_model: object = field(init=False, repr=False, compare=False)
    factor_risk_model: object = field(init=False, repr=False, compare=False)
    residual_risk_model: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Reject invalid axes/moments, copy inputs, and build canonical risk."""
        for name, kind in [('betas', pd.DataFrame), ('residual_variances', pd.Series),
                           ('factor_covariance', pd.DataFrame), ('alpha', pd.Series)]:
            value = getattr(self, name)
            if not isinstance(value, kind):
                raise ValueError(f'{name} must be a labelled {kind.__name__}')
            object.__setattr__(self, name, value.copy(deep=True))
        b, d, f = self.betas, self.residual_variances, self.factor_covariance
        sa._model(pd.Series(0., index=b.columns), f, b, d)
        sa._vector(self.alpha, b, 'alpha')
        date = pd.Timestamp(self.date)
        if pd.isna(date):
            raise ValueError('date must be finite')
        object.__setattr__(self, 'date', date)
        current = CurrentFactorCovarData(x_covar=f, y_betas=b,
            y_variances=pd.DataFrame({'residual_var': d}),
            residual_correlation=self.residual_correlation)
        diagonal = build_risk_model({date: current})
        object.__setattr__(self, 'factor_risk_model', diagonal)
        kind = (ResidualType.EMPIRICAL if self.residual_correlation is not None
                else ResidualType.ORTHOGONAL)
        options = dict(residual_type=kind, residual_corr_weight=self.residual_corr_weight)
        residual = current.get_residual_covar(**options)
        if self.residual_correlation is not None:
            self.residual_correlation.get_corr(date=date, assets=b.index)
        correlated = self.residual_correlation is not None and self.residual_corr_weight > 0
        object.__setattr__(self, 'risk_model',
            build_risk_model({date: current.get_y_covar(**options)}) if correlated else diagonal)
        object.__setattr__(self, 'residual_risk_model',
            build_risk_model({date: residual}) if correlated else None)


def _alpha_dispersion_inputs(b, d, a):
    """Evaluate canonical dispersion without constructing unused portfolio risk models."""
    _, h = sa.gls_split(a, b, d)
    c = pd.DataFrame(np.column_stack([np.ones(len(b)), b]), index=b.index)
    _, hc = sa.gls_split(a, c, d)
    centre = float((a / d).sum() / (1. / d).sum())
    positive = a.clip(lower=0.)
    detail = pd.DataFrame({'alpha': a, 'h_factor': h, 'h_joint': hc,
                           'residual_vol': np.sqrt(d), 'A_contribution': a * a / d,
                           'J_contribution': (a - centre)**2 / d,
                           'K_contribution': hc * hc / d})
    rows = dict(N=len(b), rank_B=int(np.linalg.matrix_rank(b)),
                rank_funding_B=int(np.linalg.matrix_rank(c)),
                A=float((a * a / d).sum()), J=float(((a - centre)**2 / d).sum()),
                S_h2=float((h * h / d).sum()), K=float((hc * hc / d).sum()),
                A_plus=float((positive**2 / d).sum()), alpha_common_level=centre,
                alpha_mean=float(a.mean()), alpha_median=float(a.median()),
                alpha_q25=float(a.quantile(.25)), alpha_q75=float(a.quantile(.75)),
                positive_count=int((a > 0).sum()), negative_count=int((a < 0).sum()))
    for key in ('A', 'J', 'K'):
        rows[key + '_per_asset'] = rows[key] / len(b)
        rows[key + '_rms'] = np.sqrt(rows[key] / len(b))
    rows['J_weighted_alpha_std'] = np.sqrt(rows['J'] / (1. / d).sum())
    if positive.max() > 0:
        weights = positive / d
        detail['residual_reference_weight'] = weights / weights.sum()
        rows['residual_reference_signed_sharpe'] = np.sqrt(rows['A_plus'])
    else:
        weights = pd.Series(0., index=b.index)
        weights.loc[(a / np.sqrt(d)).idxmax()] = 1.
        detail['residual_reference_weight'] = weights
        rows['residual_reference_signed_sharpe'] = float((a / np.sqrt(d)).max())
    return rows, detail


def alpha_dispersion(model: OpportunityModel):
    """Return D1-D6 capacities and asset-level GLS residuals on the supplied axis.

    K removes the joint span of funding and loadings, using canonical GLS.
    A+ is a positive-part residual-risk reference, not a full-model frontier.
    """
    return _alpha_dispersion_inputs(model.betas, model.residual_variances, model.alpha)


def factor_access_metrics(model: OpportunityModel, premia: pd.Series | None = None):
    """Return premium-free access modes and optional canonical F1-F4/F3 results."""
    b, d, f = model.betas, model.residual_variances, model.factor_covariance
    gram = b.to_numpy().T @ (b.to_numpy() / d.to_numpy()[:, None])
    root = np.linalg.cholesky(f)
    modes = np.linalg.eigvalsh(root.T @ gram @ root)
    spectrum = pd.DataFrame({'mode': np.arange(1, len(modes) + 1),
                             'g': modes, 'access_fraction': modes / (1. + modes)})
    summary, factors = {}, pd.DataFrame(index=b.columns)
    if premia is not None:
        ceiling = sa.factor_ceiling(premia, f)
        systematic = sa.achievable_sharpe2(premia, f, b, d)
        hurdle, conditional, gain = sa.hurdle_premia(premia, f)
        z = np.linalg.solve(f, premia)
        summary = dict(factor_ceiling2=ceiling, systematic_sharpe2=systematic,
                       access_loss=ceiling - systematic,
                       FPIR=systematic / ceiling if ceiling > 0 else np.nan)
        factors = pd.DataFrame({'premium': premia, 'tangency_direction': z,
                                'premium_sensitivity': 2 * z, 'hurdle': hurdle,
                                'conditional_variance': conditional, 'drop_one_gain': gain})
    return summary, factors, spectrum


def candidate_access_gain(model, premia, candidate_beta, candidate_variance):
    """Fixed-model systematic capacity gain for one independent-residual addition.

    Existing OP achievable_sharpe2 computes total capacity, but does not expose
    the Sherman-Morrison marginal-addition identity used here. No refit occurs.
    """
    sa._model(premia, model.factor_covariance, model.betas, model.residual_variances)
    sa._align(candidate_beta, model.betas.columns)
    beta = sa._array(candidate_beta, 'candidate_beta', 1)
    if len(beta) != len(premia) or not np.isfinite(candidate_variance) or candidate_variance <= 0:
        raise ValueError('candidate requires aligned loadings and positive residual variance')
    precision = np.linalg.solve(model.factor_covariance, np.eye(len(premia)))
    b, d = model.betas.to_numpy(), model.residual_variances.to_numpy()
    h = precision + b.T @ (b / d[:, None])
    hb = np.linalg.solve(h, beta)
    return float((hb @ precision @ premia)**2 / (candidate_variance + beta @ hb))


def _tracking_error(risk, reference, weights, date):
    """Delegate risk to QIS, using canonical coordinates for cancellation at a PSD null.

    QIS's covariance quadratic may become slightly negative through cancellation
    for a signed, exactly zero-risk book. The existing OP factorizer rejects
    material indefiniteness and preserves null directions with a zero floor.
    QIS then evaluates the same book in uncorrelated risk coordinates.
    """
    with np.errstate(invalid='ignore'):
        value = risk.compute_tre_at_date(reference, weights, date)
    if np.isfinite(value):
        return value
    covariance = risk.covar[date]
    root = factorize_covariance(covariance.to_numpy(), eigenvalue_floor=0.).factor
    positions = pd.Series(root.T @ (weights-reference).to_numpy(), index=covariance.index)
    coordinates = build_risk_model({date: pd.DataFrame(
        np.eye(len(positions)), index=positions.index, columns=positions.index)})
    return coordinates.compute_tre_at_date(positions*0., positions, date)


def portfolio_opportunity_metrics(model, weights, reference=None):
    """Report actual funding, exposures, risk and statistical-alpha attribution.

    Actual saved books need not be unit funded. Their funding is reported exactly;
    only the optimizer requires a long-only unit reference. No holdings are scaled.
    """
    sa._vector(weights, model.betas, 'weights')
    if reference is None:
        reference = weights * 0.
    sa._vector(reference, model.betas, 'reference')
    risk, date = model.risk_model, model.date
    delta = weights - reference
    factor_model = model.factor_risk_model
    f = factor_model.compute_exposures_at_date(weights, date)
    df = factor_model.compute_exposures_at_date(delta, date)
    active = factor_model.compute_tre_decomposition_at_date(reference, weights, date)
    absolute = factor_model.compute_tre_decomposition_at_date(weights * 0., weights, date)
    residual_te, residual_vol = active.residual_te, absolute.residual_te
    if model.residual_risk_model is not None:
        residual_te = _tracking_error(model.residual_risk_model, reference, weights, date)
        residual_vol = _tracking_error(model.residual_risk_model, weights * 0., weights, date)
    wf, wh, _ = sa.holdings_decomposition(weights, model.betas, model.residual_variances)
    _, h = sa.gls_split(model.alpha, model.betas, model.residual_variances)
    joint = pd.DataFrame(np.column_stack([np.ones(len(weights)), model.betas]),
                         index=model.betas.index)
    joint_coefficients, hc = sa.gls_split(model.alpha, joint, model.residual_variances)
    fm = build_risk_model({date: model.factor_covariance})
    factor_risk = fm.compute_marginal_tre_at_date(f * 0., f, date)
    active_risk = fm.compute_marginal_tre_at_date(df * 0., df, date)
    rows = dict(net_exposure=float(weights.sum()), reference_net_exposure=float(reference.sum()),
                min_weight=float(weights.min()), max_weight=float(weights.max()),
                concentration=float(weights @ weights), alpha_level=float(model.alpha @ weights),
                active_l1=float(delta.abs().sum()),
                held_assets=int((weights.abs() > 1e-7).sum()),
                alpha_increment=float(model.alpha @ delta),
                orthogonal_alpha_increment=float(h @ delta),
                joint_orthogonal_alpha_increment=float(hc @ delta),
                joint_common_level_increment=float(joint_coefficients[0] * delta.sum()),
                joint_factor_aligned_alpha_increment=float(joint_coefficients[1:] @ df),
                factor_aligned_alpha_increment=float((model.alpha.to_numpy() - h) @ delta),
                total_te=float(_tracking_error(risk, reference, weights, date)),
                factor_te=float(active.factor_te), residual_te=float(residual_te),
                portfolio_vol=float(_tracking_error(risk, weights * 0., weights, date)),
                factor_variance=float(absolute.factor_te)**2,
                residual_variance=float(residual_vol)**2,
                gls_factor_book_vol=float(_tracking_error(
                    risk, weights * 0., pd.Series(wf, index=weights.index), date)),
                gls_neutral_book_vol=float(_tracking_error(
                    risk, weights * 0., pd.Series(wh, index=weights.index), date)))
    exposures = pd.DataFrame({'exposure': f, 'active_exposure': df,
                               'variance_contribution': factor_risk.mcte * absolute.factor_te,
                               'active_variance_contribution': active_risk.mcte * active.factor_te})
    holdings = pd.DataFrame({'weight': weights, 'reference': reference, 'active_weight': delta,
                             'w_factor': wf, 'w_neutral': wh})
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
            sa._align(value, index)
            sa._array(value, name, 1)
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
            sa._align(getattr(value, attribute), index)


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
    """
    sa._vector(reference, model.betas, 'reference')
    if reference.min() < -tolerance or abs(reference.sum() - 1.) > 1e-6:
        raise ValueError('reference must be long-only with unit exposure; '
                         'do not normalize saved books')
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('tolerance must be finite and positive')
    uncertainty_root = None
    if (alpha_covariance is None) != (uncertainty_radius is None):
        raise ValueError('alpha covariance and uncertainty radius must be supplied together')
    if alpha_covariance is not None:
        from factorlasso import gaussian_quadratic_summary
        if (objective not in ('upper', 'lower', 'orthogonal_upper', 'orthogonal_lower')
                or not np.isfinite(uncertainty_radius) or uncertainty_radius < 0):
            raise ValueError('nonnegative finite uncertainty radius requires an alpha objective')
        if (not isinstance(alpha_covariance, pd.DataFrame)
                or not alpha_covariance.index.equals(model.betas.index)
                or not alpha_covariance.columns.equals(model.betas.index)):
            raise ValueError('alpha covariance must match the exact asset order')
        v = alpha_covariance.to_numpy()
        # FactorLasso owns validation of the estimation covariance, including
        # materially indefinite matrices; only eigensolver roundoff is removed.
        gaussian_quadratic_summary(model.alpha, v, np.eye(len(v)))
        if objective.startswith('orthogonal'):
            joint = pd.DataFrame(np.column_stack([np.ones(len(v)), model.betas]),
                                 index=model.betas.index)
            projector = sa.gls_projector(joint, model.residual_variances)
            v = projector @ v @ projector.T
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
    _constraint_order(spec, model.betas.index)
    n = len(model.betas)
    w = cvx.Variable(n)
    delta = w - reference.to_numpy()
    covariance = model.risk_model.covar[model.date].to_numpy()
    rows = spec.set_cvx_all_constraints(w, covar=cvx.psd_wrap(covariance))
    if investable is not None:
        sa._align(investable, model.betas.index)
        if not isinstance(investable, pd.Series) or investable.dtype != bool:
            raise ValueError('investable must be a labelled boolean Series')
        if (~investable).any():
            rows += [w[np.flatnonzero(~investable)] == 0.]
    factor_coordinates = (np.linalg.cholesky(model.factor_covariance).T
                          @ model.betas.to_numpy().T @ delta)
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
            joint = pd.DataFrame(np.column_stack([np.ones(n), model.betas]),
                                 index=model.betas.index)
            _, values = sa.gls_split(model.alpha, joint, model.residual_variances)
        elif objective == 'score':
            values = sa._vector(scores, model.betas, 'scores')
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
    weights = pd.Series(w.value, index=model.betas.index)
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
