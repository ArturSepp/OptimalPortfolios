"""Complete one dated allocation from supplied inputs, then inspect its outputs.

Runs offline with the base installed optimalportfolios package; TF and FL are not
required. The independent equality-constrained KKT solution checks the optimizer,
and a Cholesky norm checks the QIS risk handoff. No files are written by default.
Call save_tables with a new directory to explicitly save inspectable CSV/JSON tables.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import optimalportfolios as op


def main():
    """Return the completed result and its separate accepted solver outcome."""
    assets = pd.Index(['Bonds', 'Credit', 'Equity'])
    date = pd.Timestamp('2025-12-31')
    covariance = pd.DataFrame(np.diag([.01, .0225, .04]), index=assets, columns=assets)
    alpha = pd.Series([.04, .06, .08], index=assets, name='annual_alpha')
    benchmark = pd.Series(1 / 3, index=assets, name='benchmark')
    constraints = op.Constraints(is_long_only=True)
    risk_aversion = 3.0
    weights, outcome = op.wrapper_quadratic_optimisation(
        pd_covar=covariance, means=alpha, constraints=constraints,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
        carra=risk_aversion, context=str(date.date()),
        optimiser_config=op.OptimiserConfig(solver='CLARABEL'),
    )
    assert outcome.accepted and outcome.compliant
    assert outcome.status in ('optimal', 'optimal_inaccurate')
    assert outcome.fallback_source is None
    audit = outcome.residuals_frame()
    assert not audit.empty and audit.loc[audit['hard'], 'passed'].all()

    supplied_weights = weights.rename('allocation').copy()
    result = op.PortfolioOptimisationResult(
        weights=supplied_weights, benchmark_weights=benchmark,
        covariance=covariance, expected_return=alpha,
        optimisation_date=date, risk_model_date=date,
        reference_ccy='USD', constraints=constraints,
    )

    # Independent constrained stationary point, rather than a second CVXPY solve.
    inverse = np.diag(1 / np.diag(covariance))
    ones = np.ones(len(assets))
    multiplier = (ones @ inverse @ alpha - risk_aversion) / (ones @ inverse @ ones)
    reference = inverse @ (alpha - multiplier * ones) / risk_aversion
    assert (reference > 0).all()  # Long-only bounds are inactive in this fixture.
    np.testing.assert_allclose(result.get_weights('allocation'), reference, atol=2e-6)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-8)
    assert list(result.tickers) == list(assets)
    assert result.risk_model_date == date
    assert outcome.context == '2025-12-31'

    # These are test references only; production risk uses QIS through the result.
    root = np.linalg.cholesky(covariance.to_numpy())
    active = weights.to_numpy() - benchmark.to_numpy()
    np.testing.assert_allclose(result.compute_portfolio_vol(),
                               np.linalg.norm(root.T @ weights), atol=1e-12)
    np.testing.assert_allclose(result.compute_tracking_error(),
                               np.linalg.norm(root.T @ active), atol=1e-12)
    owned_weights = result.get_weights('allocation').copy()
    owned_covariance = result.covariance.copy()
    supplied_weights.iloc[0] = 0  # Constructor-owned numerical context stays unchanged.
    covariance.iloc[0, 0] = 999
    pd.testing.assert_series_equal(result.get_weights('allocation'), owned_weights)
    pd.testing.assert_frame_equal(result.covariance, owned_covariance)
    # The raw public ``weights`` attribute remains mutable; restore the illustrative
    # input mutation before returning a consistent inspection object.
    supplied_weights.iloc[:] = owned_weights
    return result, outcome


def save_tables(result, outcome, output_dir):
    """Explicitly export review tables; this is not a result deserialization format."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    result.to_weights_df().to_csv(destination / 'weights.csv')
    result.covariance.to_csv(destination / 'covariance.csv')
    result.summary().to_csv(destination / 'summary.csv')
    outcome.residuals_frame().to_csv(destination / 'constraints.csv', index=False)
    receipt = {
        'optimisation_date': result.optimisation_date.isoformat(),
        'risk_model_date': result.risk_model_date.isoformat(),
        'currency': result.reference_ccy,
        'return_units': 'annual arithmetic decimal alpha',
        'covariance_units': 'annual decimal return squared',
        'solver': outcome.solver, 'status': outcome.status,
        'accepted': outcome.accepted, 'compliant': outcome.compliant,
        'fallback_source': outcome.fallback_source,
        'format': 'inspection tables; not an OP persistence schema',
    }
    (destination / 'status.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    completed, solver_outcome = main()
    print(completed.to_weights_df().round(6).to_string())
    print(f'status={solver_outcome.status}; accepted={solver_outcome.accepted}; '
          f'compliant={solver_outcome.compliant}; date={completed.optimisation_date.date()}')
    print(completed.summary().round(6).to_string())
