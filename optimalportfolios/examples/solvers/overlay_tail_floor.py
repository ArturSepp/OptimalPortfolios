"""
overlay optimisation with a fixed core and a linear tail floor.

synthetic example for the pattern documented in docs/overlay_tail_floor.md:
maximise the sharpe ratio of core + overlays with the core fixed at 100%,
long-only overlays summing to 100%, and a floor on the portfolio's bear-regime
return contribution passed as a homogeneous linear constraint through
Constraints(asset_returns=..., target_return=0.0).

all inputs are synthetic and illustrative.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional
# qis / project
from optimalportfolios import Constraints, cvx_maximize_portfolio_sharpe

CORE = 'Core'


def solve_overlay_tail_floor(means: pd.Series,
                             covar: pd.DataFrame,
                             bear_contributions: pd.Series,
                             overlay_budget: float = 1.0,
                             floor_b0: Optional[float] = 0.0,
                             ) -> pd.Series:
    """maximise sharpe of core + overlays with a linear bear-contribution floor.

    Parameters
    ----------
    means : pd.Series
        annualised expected excess returns, core first.
    covar : pd.DataFrame
        annualised covariance on the same index.
    bear_contributions : pd.Series
        per-asset a_i such that a' w is the portfolio bear-regime return
        contribution, for example ann_vol * bear_regime_sharpe.
    overlay_budget : float
        fixed sum of overlay weights.
    floor_b0 : float, optional
        floor on a' w, folded into the coefficient vector to keep the
        constraint homogeneous under the charnes-cooper transform.
        None disables the floor.

    Returns
    -------
    pd.Series
        optimal weights with weights[CORE] = 1.
    """
    if CORE not in means.index:
        raise ValueError(f"means must contain the fixed asset {CORE!r}")
    total_exposure = 1.0 + overlay_budget
    min_w = pd.Series(0.0, index=means.index)
    min_w[CORE] = 1.0
    max_w = pd.Series(overlay_budget, index=means.index)
    max_w[CORE] = 1.0
    kwargs = dict(is_long_only=True,
                  min_weights=min_w,
                  max_weights=max_w,
                  min_exposure=total_exposure,  # equality routes to the convex path
                  max_exposure=total_exposure)
    if floor_b0 is not None:
        # homogeneous encoding: a' w >= b0  <=>  (a - (b0 / E) e)' w >= 0 given e' w = E
        kwargs.update(asset_returns=bear_contributions - floor_b0 / total_exposure,
                      target_return=0.0)
    constraints = Constraints(**kwargs)
    weights = cvx_maximize_portfolio_sharpe(covar=covar.to_numpy(),
                                            means=means.to_numpy(),
                                            constraints=constraints)
    return pd.Series(weights, index=means.index)


def create_synthetic_inputs() -> tuple:
    """one core and four overlays with a one-factor covariance and stylised bear stats."""
    names = [CORE, 'Defensive A', 'Defensive B', 'Carry C', 'Carry D']
    vols = pd.Series([0.10, 0.15, 0.12, 0.10, 0.08], index=names)
    sharpes = pd.Series([0.60, 0.35, 0.30, 0.90, 1.00], index=names)
    betas = pd.Series([1.00, -0.30, -0.20, 0.60, 0.40], index=names)
    bear_sharpes = pd.Series([-0.80, 0.60, 0.35, -0.45, -0.30], index=names)
    means = sharpes * vols
    factor_vol = vols[CORE]
    idio = np.sqrt(np.maximum(vols ** 2 - (betas * factor_vol) ** 2, 1e-6))
    covar = pd.DataFrame(np.outer(betas, betas) * factor_vol ** 2 + np.diag(idio ** 2),
                         index=names, columns=names)
    bear_contributions = vols * bear_sharpes
    return means, covar, bear_contributions


class UnitTests(Enum):
    NO_FLOOR = 1
    ZERO_FLOOR = 2


def run_unit_test(unit_test: UnitTests) -> None:
    means, covar, bear_contributions = create_synthetic_inputs()
    if unit_test == UnitTests.NO_FLOOR:
        w = solve_overlay_tail_floor(means=means, covar=covar,
                                     bear_contributions=bear_contributions, floor_b0=None)
    elif unit_test == UnitTests.ZERO_FLOOR:
        w = solve_overlay_tail_floor(means=means, covar=covar,
                                     bear_contributions=bear_contributions, floor_b0=0.0)
    print(f"{unit_test.name}: weights\n{w.round(3).to_string()}")
    print(f"bear contribution a'w = {bear_contributions @ w:+.4f}, "
          f"model sharpe = {means @ w / np.sqrt(w @ covar.to_numpy() @ w):.2f}")


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.NO_FLOOR)
    run_unit_test(unit_test=UnitTests.ZERO_FLOOR)
