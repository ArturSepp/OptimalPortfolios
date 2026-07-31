"""
Total vs excess CMAs in the SAA optimization: exact equivalence and where it breaks.

Verifies numerically what the algebra states. The production objective is
max_w mu'(w - w_b) s.t. (w - w_b)' Sigma (w - w_b) <= TE^2, 1'w = 1, bounds
(optimalportfolios.maximise_alpha_over_tre). Writing mu_total = mu_x + r_f 1
gives mu_total'(w - w_b) = mu_x'(w - w_b) + r_f (1'w - 1'w_b) = mu_x'(w - w_b),
so the cash anchor annihilates EXACTLY: total and excess CMAs produce the
identical mandate book for every reference currency. The equivalence needs
the linear objective and the budget constraint on both sides. It breaks for
ratio objectives: maximizing the TOTAL-return Sharpe mu_total'w / vol(w)
adds the constant r_f to the numerator, which tilts the maximizer toward
low-volatility books, by an amount that grows with r_f - so USD (r_f 4.18%)
and CHF (r_f ~0%) mandates diverge purely through the cash anchor.

Panel 1 runs the production mandate solve under total, excess, and
CHF-anchored total CMAs (identical weights expected). Panel 2 runs long-only
maximum-Sharpe with total-return numerators under USD and CHF anchors
against the excess-Sharpe book (divergence expected). Units: decimals p.a.
Main entry point: run_local_test(local_test). 2026q2 frozen cut.

The published excess CMA vector is mu_x = factor_excess_cma + w_paper * alpha.
equity_regional_addon is NOT added on top: the identity
factor_excess_cma = beta @ lambda + equity_regional_addon holds on the
snapshot to 1e-13 bp, so adding it again double-counts the regional blend
(2-311 bp on the seven equity sleeves). Corrected 2026-07-30, roadmap Stage
J0b; the identity is asserted in tests/test_snapshot_parity.py.

Does not belong here: the mandate exhibit builds (Decision One scripts).
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple
import cvxpy as cvx
from optimalportfolios import Constraints, ConstraintEnforcementType, wrapper_maximise_alpha_over_tre
# shared paper-data layer, imported by file location (no sys.path mutation)
from local_path import load_cma_data
from governed_cma_projection import load_paper_inputs, SNAPSHOT

_cma_data = load_cma_data()

MANDATE = 'Balanced with Alts'
BAND = 0.50                      # +-50% box around benchmark weights
TE_CONSTRAINT = 0.015            # tracking error cap, annualized
RF_CHF = 0.0009                  # CHF 3Y cash anchor (paper Appendix C worked example)
VOL_GRID = np.linspace(0.04, 0.16, 61)   # long-only frontier grid for the ratio objective


def build_moments(inputs) -> Tuple[pd.DataFrame, pd.Series, float]:
    """asset covariance Sigma = B Sigma_F B' + D, the excess CMA vector, and the reference cash rate.

    mu_x = factor_excess_cma + w_paper * alpha. The regional add-on is already
    inside factor_excess_cma (see the module docstring) and is not added again.
    """
    b = inputs.betas.values
    sigma = b @ inputs.factor_covar.values @ b.T + np.diag(inputs.assets['resid_vol'].values ** 2)
    covar = pd.DataFrame(sigma, index=inputs.assets.index, columns=inputs.assets.index)
    assets = inputs.assets
    mu_x = (assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']).rename('mu_excess')
    return covar, mu_x, float(assets['rf_rate'].iloc[0])


def solve_mandate(covar: pd.DataFrame,
                  cmas: pd.Series,
                  benchmark_weights: pd.Series,
                  ) -> pd.Series:
    """the production mandate solve of run_optimisation.py on the frozen inputs."""
    constraints = Constraints(min_weights=(1.0 - BAND) * benchmark_weights,
                              max_weights=(1.0 + BAND) * benchmark_weights,
                              benchmark_weights=benchmark_weights,
                              weights_0=benchmark_weights.rename('Current'),
                              constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
                              tracking_err_vol_constraint=TE_CONSTRAINT)
    return wrapper_maximise_alpha_over_tre(pd_covar=covar,
                                           alphas=cmas,
                                           benchmark_weights=benchmark_weights,
                                           constraints=constraints,
                                           weights_0=None)


def solve_max_return_at_vol(covar: np.ndarray, mu: np.ndarray, vol: float) -> np.ndarray:
    """long-only full-investment maximum return at a volatility cap."""
    n = len(mu)
    w = cvx.Variable(n, nonneg=True)
    problem = cvx.Problem(cvx.Maximize(mu @ w),
                          [cvx.sum(w) == 1.0, cvx.quad_form(w, cvx.psd_wrap(covar)) <= vol ** 2])
    problem.solve(solver=cvx.CLARABEL)
    if w.value is None:
        raise ValueError(f"solver failed at vol target {vol!r}")
    return np.asarray(w.value)


def solve_max_ratio(covar: pd.DataFrame, numerator: pd.Series) -> pd.Series:
    """long-only book maximizing numerator'w / vol(w) over the frontier grid."""
    best, best_ratio = None, -np.inf
    for vol in VOL_GRID:
        w = solve_max_return_at_vol(covar=covar.values, mu=numerator.values, vol=vol)
        ratio = float(numerator.values @ w) / float(np.sqrt(w @ covar.values @ w))
        if ratio > best_ratio:
            best, best_ratio = w, ratio
    return pd.Series(best, index=covar.index)


def class_split(weights: pd.Series, inputs) -> pd.Series:
    return weights.groupby(inputs.assets['asset_class']).sum()


def run_report(snapshot: str = SNAPSHOT) -> None:
    inputs = load_paper_inputs(snapshot=snapshot)
    covar, mu_x, rf_usd = build_moments(inputs=inputs)
    bench = _cma_data.get_benchmark_weights(mandate=MANDATE)
    bench.index = [f"{t} Index" if not t.endswith('Index') else t for t in bench.index]
    bench = bench.reindex(inputs.assets.index)

    print(f"snapshot {snapshot}, mandate {MANDATE!r}, band +-{BAND:.0%}, TE cap {TE_CONSTRAINT:.1%}")
    print(f"r_f USD = {rf_usd:.4%}, r_f CHF = {RF_CHF:.4%}\n")

    # Panel 1: production mandate solve - identical books expected
    w_total = solve_mandate(covar=covar, cmas=mu_x + rf_usd, benchmark_weights=bench)
    w_excess = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=bench)
    w_chf = solve_mandate(covar=covar, cmas=mu_x + RF_CHF, benchmark_weights=bench)
    panel1 = pd.DataFrame({'sleeve': inputs.assets['sleeve'],
                           'total_usd': w_total, 'excess': w_excess, 'total_chf_anchor': w_chf})
    print("Panel 1 - mandate solve (budget + TE + box): total vs excess vs CHF-anchored total")
    print(panel1.round(4).to_string())
    print(f"max |w_total - w_excess|      = {float((w_total - w_excess).abs().max()):.2e}")
    print(f"max |w_total - w_chf_anchor|  = {float((w_total - w_chf).abs().max()):.2e}\n")

    # Panel 2: ratio objective - the anchor is live, books diverge
    w_sharpe_x = solve_max_ratio(covar=covar, numerator=mu_x)
    w_sharpe_usd = solve_max_ratio(covar=covar, numerator=mu_x + rf_usd)
    w_sharpe_chf = solve_max_ratio(covar=covar, numerator=mu_x + RF_CHF)
    panel2 = pd.DataFrame({'excess (any ccy)': class_split(w_sharpe_x, inputs),
                           'total, USD anchor': class_split(w_sharpe_usd, inputs),
                           'total, CHF anchor': class_split(w_sharpe_chf, inputs)})
    print("Panel 2 - long-only max Sharpe with the numerator as labeled, asset-class split")
    print(panel2.round(3).to_string())
    dist = 0.5 * float((w_sharpe_usd - w_sharpe_chf).abs().sum())
    vol_usd = float(np.sqrt(w_sharpe_usd @ covar.values @ w_sharpe_usd))
    vol_chf = float(np.sqrt(w_sharpe_chf @ covar.values @ w_sharpe_chf))
    print(f"\nUSD-anchor vs CHF-anchor book distance (half L1) = {dist:.1%}")
    print(f"book volatility: USD anchor {vol_usd:.1%} vs CHF anchor {vol_chf:.1%}")
    print(f"excess-Sharpe book volatility (anchor-invariant) = "
          f"{float(np.sqrt(w_sharpe_x @ covar.values @ w_sharpe_x)):.1%}")


class LocalTests(str, Enum):
    REPORT = 'report'


def run_local_test(local_test: LocalTests) -> None:
    if local_test == LocalTests.REPORT:
        run_report()
    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.REPORT)
