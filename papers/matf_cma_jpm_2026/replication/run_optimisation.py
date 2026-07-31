"""
Optimizer port for the MATF-CMA mandate exhibits (roadmap Stage J4a).

The single place where a portfolio is solved. Every exhibit script downstream
calls into here, so the mandate design, the moment construction, and the
reporting convention are stated once.

Moments. The asset covariance is the factor model of Equation (2),

    Sigma = beta Sigma_F beta' + diag(sigma_eps^2)

and the expected-return vector is the EXCESS CMA of the adopted convention
(roadmap B9),

    mu_x = factor_excess_cma + w_paper * alpha

with equity_regional_addon already inside factor_excess_cma. The reference
cash rate enters ONLY at the reporting layer, total = excess + rf_rate. The
mandate optimum is invariant to that anchor because the budget constraint
annihilates it (see excess_vs_total_optimisation.py for the numerical proof),
so the same solve serves every reference currency.

Two solve designs, both from the production process (roadmap B7):

  solve_mandate        max mu_x'(w - w_b) subject to the tracking-error cap
                       ||w - w_b||_Sigma <= TE, a +-50% box around the
                       benchmark, and full investment. Run through
                       optimalportfolios.wrapper_maximise_alpha_over_tre with
                       FORCED_CONSTRAINTS. This is the mandate the committee
                       actually holds.
  solve_max_return_at_vol   max mu_x'w subject to w >= 0, 1'w = 1 and a
                       volatility cap, through cvxpy/CLARABEL. This is the
                       unconstrained comparator that shows what the admission
                       policy would do without the mandate guardrails.

Benchmarks come from cma_data.benchmarks, which is D8-correct: the R2 exhibit
build's benchmark input transposed Asia ex-Japan against EM ex-Asia, so every
mandate optimum here moves against R2 for two reasons at once, the July premia
config and the benchmark fix.

Units: decimals per annum throughout; percent only in printed reports.
Main entry point: run_local_test(local_test).

Does not belong here: figures (the run_*_exhibits scripts), the admission
sweep and scenario repricing (run_admission_exhibits.py), and the bootstrap
(run_bootstrap_q2.py).
"""
# packages
import numpy as np
import pandas as pd
import cvxpy as cvx
from enum import Enum
from typing import Dict, List, Optional, Tuple
# qis / project
from optimalportfolios import Constraints, ConstraintEnforcementType, wrapper_maximise_alpha_over_tre
from local_path import load_cma_data
from governed_cma_projection import SNAPSHOT, load_paper_inputs

_cma_data = load_cma_data()

MANDATE = 'Balanced with Alts'          # the headline mandate of the main text
BAND = 0.50                             # +-50% box around the benchmark weights
TE_CONSTRAINT = 0.015                   # tracking-error cap, annualized
VOL_GRID_POINTS = 61                    # long-only frontier grid resolution
SOLVER = cvx.CLARABEL
PSD_RIDGE = 1e-10                       # regularizer under psd_wrap

# R2 anchor points, printed beside the regenerated values (never asserted: the
# premia config K2 and the D8 benchmark fix both move them by design).
R2_ANCHORS: Dict[str, float] = {
    'balanced_total_return': 0.0792,
    'balanced_sharpe': 0.35,
    'balanced_alternatives': 0.37,
    'balanced_benchmark_vol': 0.093,
    'unconstrained_alternatives': 0.78,
    'unconstrained_private_equity': 0.51,
    'unconstrained_sharpe': 0.51,
}


def build_moments(inputs,
                  admission_scale: float = 1.0,            # s on the admitted-alpha channel
                  admission_weights: Optional[pd.Series] = None,   # default: w_paper column
                  ) -> Tuple[pd.DataFrame, pd.Series, float]:
    """Sigma, the excess CMA vector, and the reference cash rate on the paper universe.

    admission_scale is the dial s of the main text: the admitted-alpha channel
    enters as s * w_i * alpha_i, so s = 0 is the pure market-return book and
    s = 1 is the production policy. The factor-implied component never scales.
    """
    if admission_scale < 0.0:
        raise ValueError(f"admission scale must be non-negative, got {admission_scale!r}")
    assets = inputs.assets
    betas = inputs.betas.values
    sigma = betas @ inputs.factor_covar.values @ betas.T + np.diag(assets['resid_vol'].values ** 2)
    covar = pd.DataFrame(sigma, index=assets.index, columns=assets.index)
    w = assets['w_paper'] if admission_weights is None else admission_weights.reindex(assets.index)
    if w.isna().any():
        raise ValueError(f"admission weights carry NaN after reindex, got {list(w[w.isna()].index)!r}")
    mu_x = (assets['factor_excess_cma'] + admission_scale * w * assets['alpha']).rename('mu_excess')
    return covar, mu_x, float(assets['rf_rate'].iloc[0])


def get_benchmark(inputs, mandate: str = MANDATE) -> pd.Series:
    """one mandate benchmark reindexed onto the snapshot asset order, NaN-checked."""
    weights = _cma_data.get_benchmark_weights(mandate=mandate).reindex(inputs.assets.index)
    if weights.isna().any():
        raise ValueError(f"benchmark misaligned with the snapshot universe, "
                         f"got NaN for {list(weights[weights.isna()].index)!r}")
    return weights


def solve_mandate(covar: pd.DataFrame,
                  cmas: pd.Series,
                  benchmark_weights: pd.Series,
                  band: float = BAND,                       # box half-width around the benchmark
                  tracking_err_vol_constraint: float = TE_CONSTRAINT,
                  ) -> pd.Series:
    """the production mandate solve: max excess alpha over tracking error, boxed and fully invested."""
    if not 0.0 < band <= 1.0:
        raise ValueError(f"band must lie in (0, 1], got {band!r}")
    constraints = Constraints(min_weights=(1.0 - band) * benchmark_weights,
                              max_weights=(1.0 + band) * benchmark_weights,
                              benchmark_weights=benchmark_weights,
                              weights_0=benchmark_weights.rename('Current'),
                              constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
                              tracking_err_vol_constraint=tracking_err_vol_constraint)
    weights = wrapper_maximise_alpha_over_tre(pd_covar=covar,
                                              alphas=cmas,
                                              benchmark_weights=benchmark_weights,
                                              constraints=constraints,
                                              weights_0=None)
    if weights.isna().any():
        raise ValueError(f"mandate solve returned NaN weights for "
                         f"{list(weights[weights.isna()].index)!r}")
    return weights


def solve_max_return_at_vol(covar: pd.DataFrame,
                            cmas: pd.Series,
                            vol_target: float,              # annualized volatility cap
                            ) -> Optional[pd.Series]:
    """long-only full-investment maximum excess return at a volatility cap; None if infeasible."""
    if vol_target <= 0.0:
        raise ValueError(f"vol target must be positive, got {vol_target!r}")
    n = len(cmas)
    w = cvx.Variable(n, nonneg=True)
    matrix = covar.values + PSD_RIDGE * np.eye(n)
    problem = cvx.Problem(cvx.Maximize(cmas.values @ w),
                          [cvx.sum(w) == 1.0,
                           cvx.quad_form(w, cvx.psd_wrap(matrix)) <= vol_target ** 2])
    problem.solve(solver=SOLVER)
    if w.value is None:
        return None
    return pd.Series(np.asarray(w.value), index=cmas.index)


def solve_long_only_frontier(covar: pd.DataFrame,
                             cmas: pd.Series,
                             vol_grid: np.ndarray,
                             ) -> pd.DataFrame:
    """the long-only frontier over a volatility grid; infeasible points drop out with a count."""
    rows, dropped = {}, 0
    for vol_target in vol_grid:
        weights = solve_max_return_at_vol(covar=covar, cmas=cmas, vol_target=float(vol_target))
        if weights is None:
            dropped += 1
            continue
        rows[float(vol_target)] = weights
    if not rows:
        raise ValueError(f"every frontier solve failed on the grid {vol_grid[[0, -1]]!r}")
    if dropped:
        print(f"frontier: {dropped} of {len(vol_grid)} vol targets infeasible and dropped")
    frontier = pd.DataFrame(rows).T
    frontier.index.name = 'vol_target'
    return frontier


def build_vol_grid(covar: pd.DataFrame,
                   cmas: pd.Series,
                   n_points: int = VOL_GRID_POINTS,
                   ) -> np.ndarray:
    """a feasible volatility grid from just above minimum variance to just below the best asset."""
    min_variance = solve_max_return_at_vol(covar=covar, cmas=0.0 * cmas, vol_target=1.0)
    if min_variance is None:
        raise ValueError("minimum-variance solve failed; check the covariance")
    lower = 1.02 * float(np.sqrt(min_variance @ covar.values @ min_variance))
    upper = 0.98 * float(np.sqrt(covar.loc[cmas.idxmax(), cmas.idxmax()]))
    if upper <= lower:
        raise ValueError(f"empty vol grid, got lower {lower!r} and upper {upper!r}")
    return np.linspace(lower, upper, n_points)


def report_portfolio(weights: pd.Series,
                     covar: pd.DataFrame,
                     cmas: pd.Series,
                     rf_rate: float,
                     inputs,
                     benchmark_weights: Optional[pd.Series] = None,
                     ) -> pd.Series:
    """the reporting layer: excess and total expected return, volatility, Sharpe, class split."""
    excess = float(cmas @ weights)
    vol = float(np.sqrt(weights @ covar.values @ weights))
    split = weights.groupby(inputs.assets['asset_class']).sum()
    stats = {'excess_return': excess,
             'total_return': excess + rf_rate,
             'vol': vol,
             'excess_sharpe': excess / vol,
             'bonds': float(split.get('Bonds', 0.0)),
             'equities': float(split.get('Equities', 0.0)),
             'alternatives': float(split.get('Alternatives', 0.0))}
    if benchmark_weights is not None:
        active = weights - benchmark_weights
        stats['tracking_error'] = float(np.sqrt(active @ covar.values @ active))
    return pd.Series(stats)


def compute_factor_exposures(weights: pd.Series, inputs) -> pd.Series:
    """book factor exposures beta' w in the canonical factor order."""
    return pd.Series(inputs.betas.values.T @ weights.values, index=inputs.betas.columns)


def compute_factor_risk_contributions(weights: pd.Series, inputs) -> pd.Series:
    """percentage risk contribution of each factor plus the residual, summing to one.

    Splits w' Sigma w = x' Sigma_F x + w' D w with x = beta' w, and attributes
    the systematic part by factor through x_j (Sigma_F x)_j.
    """
    exposures = compute_factor_exposures(weights=weights, inputs=inputs)
    systematic = exposures.values * (inputs.factor_covar.values @ exposures.values)
    residual = float((weights.values ** 2) @ (inputs.assets['resid_vol'].values ** 2))
    contributions = pd.concat([pd.Series(systematic, index=inputs.betas.columns),
                               pd.Series({'Residual': residual})])
    total = float(contributions.sum())
    if total <= 0.0:
        raise ValueError(f"non-positive book variance, got {total!r}")
    return contributions / total


def solve_all_mandates(inputs,
                       admission_scale: float = 1.0,
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """solve all eight mandates; return the weight matrix and the per-mandate statistics."""
    covar, mu_x, rf_rate = build_moments(inputs=inputs, admission_scale=admission_scale)
    weights, stats = {}, {}
    for mandate in _cma_data.MANDATES:
        benchmark = get_benchmark(inputs=inputs, mandate=mandate)
        book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
        weights[mandate] = book
        stats[mandate] = report_portfolio(weights=book, covar=covar, cmas=mu_x,
                                          rf_rate=rf_rate, inputs=inputs,
                                          benchmark_weights=benchmark)
        stats[mandate]['benchmark_excess_return'] = float(mu_x @ benchmark)
        stats[mandate]['benchmark_total_return'] = float(mu_x @ benchmark) + rf_rate
        stats[mandate]['benchmark_vol'] = float(np.sqrt(benchmark @ covar.values @ benchmark))
    return pd.DataFrame(weights), pd.DataFrame(stats).T


def run_optimisation_report(snapshot: str = SNAPSHOT) -> Dict[str, pd.DataFrame]:
    """the Stage J4a acceptance report: mandate anchors against R2, and the unconstrained book."""
    inputs = load_paper_inputs(snapshot=snapshot)
    covar, mu_x, rf_rate = build_moments(inputs=inputs)
    print('=' * 78)
    print(f"Stage J4a — optimizer port, cut {snapshot}, band +-{BAND:.0%}, "
          f"TE cap {TE_CONSTRAINT:.1%}, r_f {rf_rate:.2%}")
    print('=' * 78)

    weights, stats = solve_all_mandates(inputs=inputs)
    print('\n--- mandate optima, all eight mandates ---')
    print(stats.round(4).to_string())
    print('\n--- mandate weights ---')
    print(weights.round(4).to_string())

    balanced = stats.loc[MANDATE]
    benchmark_vol = float(balanced['benchmark_vol'])
    print(f"\n--- R2 anchors vs regenerated ({MANDATE}) ---")
    for label, key, value, reference in (
            ('total expected return', 'balanced_total_return', balanced['total_return'],
             R2_ANCHORS['balanced_total_return']),
            ('excess Sharpe', 'balanced_sharpe', balanced['excess_sharpe'],
             R2_ANCHORS['balanced_sharpe']),
            ('alternatives weight', 'balanced_alternatives', balanced['alternatives'],
             R2_ANCHORS['balanced_alternatives']),
            ('benchmark volatility', 'balanced_benchmark_vol', benchmark_vol,
             R2_ANCHORS['balanced_benchmark_vol'])):
        print(f"  {label:<24s} R2 {reference:>8.4f}   new {value:>8.4f}   "
              f"delta {value - reference:>+8.4f}")

    # unconstrained comparator at the Balanced benchmark volatility
    unconstrained = solve_max_return_at_vol(covar=covar, cmas=mu_x, vol_target=benchmark_vol)
    if unconstrained is None:
        raise ValueError(f"unconstrained solve infeasible at vol target {benchmark_vol!r}")
    unconstrained_stats = report_portfolio(weights=unconstrained, covar=covar, cmas=mu_x,
                                           rf_rate=rf_rate, inputs=inputs)
    print(f"\n--- unconstrained long-only book at the Balanced benchmark vol "
          f"{benchmark_vol:.2%} ---")
    print(unconstrained_stats.round(4).to_string())
    print(f"  alternatives   R2 {R2_ANCHORS['unconstrained_alternatives']:.2f}   "
          f"new {unconstrained_stats['alternatives']:.2f}")
    print(f"  private equity R2 {R2_ANCHORS['unconstrained_private_equity']:.2f}   "
          f"new {float(unconstrained['MP503001 Index']):.2f}")
    print(f"  excess Sharpe  R2 {R2_ANCHORS['unconstrained_sharpe']:.2f}   "
          f"new {unconstrained_stats['excess_sharpe']:.2f}")

    exposures = compute_factor_exposures(weights=weights[MANDATE], inputs=inputs)
    risk = compute_factor_risk_contributions(weights=weights[MANDATE], inputs=inputs)
    print(f"\n--- {MANDATE}: factor exposures and percentage risk contributions ---")
    order = list(inputs.betas.columns) + ['Residual']       # canonical order, residual last
    print(pd.DataFrame({'exposure': exposures.reindex(order),
                        'risk_share': risk.reindex(order)}).round(4).to_string())
    return {'weights': weights, 'stats': stats,
            'unconstrained': unconstrained.to_frame('weight'),
            'unconstrained_stats': unconstrained_stats.to_frame('value')}


class LocalTests(str, Enum):
    OPTIMISATION_REPORT = 'optimisation_report'
    BALANCED_ONLY = 'balanced_only'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

    if local_test == LocalTests.OPTIMISATION_REPORT:
        run_optimisation_report()

    elif local_test == LocalTests.BALANCED_ONLY:
        inputs = load_paper_inputs()
        covar, mu_x, rf_rate = build_moments(inputs=inputs)
        benchmark = get_benchmark(inputs=inputs)
        book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
        print(report_portfolio(weights=book, covar=covar, cmas=mu_x, rf_rate=rf_rate,
                               inputs=inputs, benchmark_weights=benchmark).round(4).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.OPTIMISATION_REPORT)
