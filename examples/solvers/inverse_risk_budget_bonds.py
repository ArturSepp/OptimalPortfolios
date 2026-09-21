"""Explain the XXX inverse risk-budget boundary with three offline examples.

Inverse inference chooses non-negative risk budgets so that rolling forward
risk-budgeted weights average to the given central weights. This fails for two
different reasons in XXX-shaped data. At the central mix, LUATTRUU has a
non-positive *average* marginal risk contribution: a positive holding that
hedges the portfolio cannot be supported by a positive risk budget, since
``w_i * (Sigma @ w)_i = budget_i * portfolio_variance``. LD19TRUU has a
positive average contribution, but even a nearly zero *positive* budget leaves
its average forward weight above target. Lowering the budget floor cannot
make that inverse fit converge.

The remedy is to hold these assets at their central weights in every forward
solve and fit budgets for the remaining assets using the full covariance.
Their zero *reported* fitted budgets mean exclusion from the inverse fit, not
zero portfolio weight or zero economic risk contribution.

The two index weights and 65% equity sleeve mirror XXX's central mix. The
remaining 19.8925% is compressed into one synthetic sleeve. The three
covariance matrices are illustrative, not licensed production data; changing
stock/bond correlations reproduce both boundary cases.

Run from the repository root::

    python examples/solvers/inverse_risk_budget_bonds.py

No Bloomberg terminal, network, or local files are required.
"""

import warnings

import numpy as np
import pandas as pd

from optimalportfolios import (
    Constraints,
    rolling_risk_budgeting,
    solve_for_risk_budgets_from_given_weights,
)

PRODUCTION_INFERENCE_FLOOR = 1e-6
DEEP_BOUNDARY_PROBE = 1e-12


def make_xxx_shaped_inputs() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return central weights and synthetic covariance regimes for the examples."""
    assets = ['XXX equity', 'XXX other', 'LUATTRUU Index', 'LD19TRUU Index']
    target = pd.Series([0.65, 0.198925, 0.1253, 0.025775], index=assets)
    vol = np.array([0.20, 0.12, 0.06, 0.07])
    covars = {}
    for date, equity_ld_corr in zip(
            pd.DatetimeIndex(['2020-03-31', '2023-03-31', '2026-03-31']),
            (-0.3, 0.2, 0.2)):
        corr = np.array([
            [1.0, 0.4, -0.7, equity_ld_corr],
            [0.4, 1.0, -0.2, 0.0],
            [-0.7, -0.2, 1.0, 0.0],
            [equity_ld_corr, 0.0, 0.0, 1.0],
        ])
        covars[date] = pd.DataFrame(np.outer(vol, vol) * corr,
                                    index=assets, columns=assets)
    # Only the column grid matters here: the covariance path is supplied above.
    prices = pd.DataFrame(1.0, index=list(covars), columns=assets)
    return prices, target, covars


def fixed_weight_constraints(target: pd.Series, assets: list[str]) -> Constraints:
    """Keep named zero-budget assets inside the full covariance solve."""
    minimum = pd.Series(0.0, index=target.index)
    maximum = pd.Series(1.0, index=target.index)
    minimum.loc[assets] = target.loc[assets]
    maximum.loc[assets] = target.loc[assets]
    return Constraints(is_long_only=True, min_weights=minimum, max_weights=maximum)


def run_examples() -> None:
    """Show the floor discontinuity, automatic pinning, and manual parity."""
    prices, target, covars = make_xxx_shaped_inputs()

    # Case 1: LUATTRUU is pinned, but LD19TRUU is left at a tiny *positive*
    # budget. Its average forward weight is still far above its 2.58% target.
    floor_budget = pd.Series(
        [0.91 - DEEP_BOUNDARY_PROBE, 0.09, 0.0, DEEP_BOUNDARY_PROBE],
        index=target.index)
    floor_weights = rolling_risk_budgeting(
        prices=prices, covar_dict=covars, risk_budget=floor_budget,
        constraints=fixed_weight_constraints(target, ['LUATTRUU Index']))
    floor_average = floor_weights.mean()
    assert floor_average['LD19TRUU Index'] > target['LD19TRUU Index'] + 0.10
    print('Case 1: LD19TRUU with a 1e-12 positive boundary probe')
    print(f"  target={target['LD19TRUU Index']:.4%}; "
          f"average forward weight={floor_average['LD19TRUU Index']:.4%}")

    # Case 2: the inverse solver detects both fixed sleeves without a ticker
    # override. Zero reported budget does not mean zero portfolio weight.
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always', UserWarning)
        inferred = solve_for_risk_budgets_from_given_weights(
            prices=prices, given_weights=target, covar_dict=covars,
            min_risk_budget=PRODUCTION_INFERENCE_FLOOR)
    fixed = target.index[(target > 0.0) & inferred.eq(0.0)].tolist()
    assert fixed == ['LUATTRUU Index', 'LD19TRUU Index']
    reproduced = rolling_risk_budgeting(
        prices=prices, covar_dict=covars, risk_budget=inferred,
        constraints=fixed_weight_constraints(target, fixed))
    average = reproduced.mean()
    np.testing.assert_allclose(average, target, atol=1e-3)
    print('Case 2: automatic fixed-weight detection')
    print(pd.concat([target.rename('central_weight'), inferred.rename('risk_budget'),
                     average.rename('average_forward_weight')], axis=1).to_string())
    print('  warnings:', [str(item.message).split(':')[0] for item in emitted])

    # Case 3: the existing explicit override remains available for governance.
    # It must agree with the automatically discovered solution in this example.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        explicit = solve_for_risk_budgets_from_given_weights(
            prices=prices, given_weights=target, covar_dict=covars,
            min_risk_budget=PRODUCTION_INFERENCE_FLOOR,
            fixed_weight_assets=('LD19TRUU Index',))
    np.testing.assert_allclose(inferred, explicit, atol=1e-6)
    print('Case 3: explicit LD19TRUU override agrees with automatic inference')


if __name__ == '__main__':
    run_examples()
