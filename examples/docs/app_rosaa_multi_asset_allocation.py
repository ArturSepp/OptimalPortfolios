"""Canonical script of docs/app_rosaa_multi_asset_allocation.md.

The case study reports the study of Sepp, Ossa and Kastenholz (2026) from the article; this script
does not reproduce its numbers. It builds the same three-layer configuration on a synthetic panel
shaped like the study, with known factor loadings, and asserts the mechanism the article relies
on: the strategic allocation meets its risk budgets, and the tactical allocation spends its
tracking-error budget in the direction of the alphas. It runs offline:

    python -m examples.docs.app_rosaa_multi_asset_allocation

``exhibit`` draws the page's figure for ``tools/docs_analytics/teaching.py``.
"""
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType

import optimalportfolios as op

SEED = 11
FACTORS = ['Equity', 'Rates', 'Credit']
ASSETS = ['Govt', 'IG credit', 'HY credit', 'DM equity', 'EM equity', 'Real estate',
          'Hedge funds', 'Commodities']
# Monthly factor loadings, residual and factor volatilities of the synthetic panel.
LOADINGS = np.array([
    [-0.1, 0.9, 0.0],
    [0.1, 0.6, 0.5],
    [0.4, 0.2, 0.8],
    [1.0, 0.0, 0.1],
    [1.2, 0.0, 0.3],
    [0.7, 0.3, 0.2],
    [0.3, 0.0, 0.2],
    [0.4, -0.1, 0.1],
])
RESIDUAL_VOLS = np.array([0.01, 0.01, 0.02, 0.02, 0.04, 0.04, 0.02, 0.06])
FACTOR_VOLS = np.array([0.045, 0.02, 0.02])
FACTOR_CORR = np.array([[1.0, -0.2, 0.5], [-0.2, 1.0, 0.1], [0.5, 0.1, 1.0]])
RISK_BUDGETS = [0.10, 0.10, 0.10, 0.25, 0.15, 0.10, 0.10, 0.10]
TRACKING_ERROR = 0.03


def simulated_panel(seed: int) -> tuple:
    """Simulate monthly factor and asset prices, December 2004 to June 2025, with known loadings."""
    dates = pd.date_range('2004-12-31', '2025-06-30', freq='ME')
    rng = np.random.default_rng(seed)
    factor_covar = np.outer(FACTOR_VOLS, FACTOR_VOLS) * FACTOR_CORR
    factors = rng.multivariate_normal(np.full(3, 0.004), factor_covar, size=len(dates) - 1)
    assets = factors @ LOADINGS.T + rng.normal(0.0, RESIDUAL_VOLS,
                                               size=(len(dates) - 1, len(ASSETS)))

    def prices(returns, columns):
        """Turn monthly log returns into prices starting at 100."""
        levels = np.exp(np.vstack([np.zeros((1, returns.shape[1])), np.cumsum(returns, 0)]))
        return pd.DataFrame(100.0 * levels, index=dates, columns=columns)

    return prices(factors, FACTORS), prices(assets, ASSETS), rng


def alpha_scores(dates, rng) -> pd.DataFrame:
    """Draw standardised alpha scores for each rebalancing date."""
    return pd.DataFrame(rng.normal(size=(len(dates), len(ASSETS))), index=dates, columns=ASSETS)


def run() -> dict:
    """Build the three layers and return the covariance, SAA, TAA and alphas."""
    factor_prices, asset_prices, rng = simulated_panel(SEED)
    asset_returns = np.log(asset_prices).diff().iloc[1:]

    # Layer 1: a hierarchical-clustering group LASSO factor model, refitted every quarter end.
    estimator = op.FactorCovarEstimator(
        lasso_model=LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                               reg_lambda=1e-5, span=36, warmup_period=24),
        factor_returns_freq='ME', factor_covar_span=36, rebalancing_freq='QE')
    covar_dict = estimator.fit_rolling_covars(
        risk_factor_prices=factor_prices, asset_returns_dict={'ME': asset_returns},
        time_period=qis.TimePeriod('2014-12-31', '2025-06-30'))

    # Layer 2: the strategic allocation by risk budgets.
    budgets = pd.Series(RISK_BUDGETS, index=ASSETS)
    saa = op.rolling_risk_budgeting(prices=asset_prices, constraints=op.Constraints(),
                                    risk_budget=budgets, covar_dict=covar_dict)

    # Layer 3: the tactical allocation, alpha over tracking error against the strategic one.
    alphas = alpha_scores(list(covar_dict), rng)
    taa = op.rolling_maximise_alpha_over_tre(
        prices=asset_prices, alphas=alphas, benchmark_weights=saa, covar_dict=covar_dict,
        constraints=op.Constraints(tracking_err_vol_constraint=TRACKING_ERROR))
    return {'covar_dict': covar_dict, 'budgets': budgets, 'saa': saa, 'taa': taa,
            'alphas': alphas}


def main() -> None:
    """Run the three layers and assert the mechanism the case study describes."""
    result = run()
    covar_dict, budgets = result['covar_dict'], result['budgets']
    saa, taa, alphas = result['saa'], result['taa'], result['alphas']
    assert len(covar_dict) == 43

    # Strategic layer: every quarter-end allocation meets its risk budgets.
    for date, covar in covar_dict.items():
        w = saa.loc[date].to_numpy()
        contributions = w * (covar.to_numpy() @ w)
        assert np.allclose(contributions / contributions.sum(), budgets, atol=1e-6)
    assert saa.iloc[-1]['Govt'] > 3 * budgets['Govt']

    # Tactical layer: the tracking error against the strategic allocation is at its 3% budget,
    # and the active weights follow the alphas.
    for date, covar in covar_dict.items():
        active = (taa.loc[date] - saa.loc[date]).to_numpy()
        assert abs(np.sqrt(active @ covar.to_numpy() @ active) - TRACKING_ERROR) < 1e-4
    alignment = [np.corrcoef(alphas.loc[d], taa.loc[d] - saa.loc[d])[0, 1] for d in covar_dict]
    assert np.median(alignment) > 0.5


def exhibit(path) -> dict:
    """Draw the page's figure: strategic weights against budgets, tactical tilts against alphas.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result = run()
    covar_dict, budgets = result['covar_dict'], result['budgets']
    saa, taa, alphas = result['saa'], result['taa'], result['alphas']
    last = list(covar_dict)[-1]
    active = (taa - saa).loc[list(covar_dict)]
    tracking_errors = [np.sqrt(a @ covar_dict[d].to_numpy() @ a)
                       for d, a in zip(active.index, active.to_numpy())]
    table = pd.DataFrame({'risk_budget': budgets, 'saa_weight_last': saa.loc[last],
                          'taa_active_weight_last': active.loc[last],
                          'alpha_last': alphas.loc[last]})

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.35, 1.0]})
    x = np.arange(len(ASSETS))
    width = 0.38
    left.bar(x - width / 2 - 0.01, budgets, width, color=orange, label='Risk budget')
    left.bar(x + width / 2 + 0.01, saa.loc[last], width, color=blue,
             label='Strategic weight that meets it')
    left.set_xticks(x, ASSETS, rotation=35, ha='right', fontsize=10)
    left.set_ylabel('Share')
    left.set_ylim(0.0, 0.5)
    left.set_title('Strategic layer: budgets are not weights', loc='left', color=ink)
    left.legend(frameon=False, loc='upper right', fontsize=10, labelcolor=ink)
    right.scatter(alphas.loc[list(covar_dict)].to_numpy().ravel(), active.to_numpy().ravel(),
                  s=14, color=blue, edgecolors=surface, linewidths=0.6)
    right.axhline(0.0, color=muted, linewidth=0.8)
    right.set_xlabel('Alpha score')
    right.set_ylabel('Active weight')
    right.set_title('Tactical layer: tilts follow the alphas', loc='left', color=ink)
    right.text(0.02, 0.97, f'Ex-ante tracking error {TRACKING_ERROR:.0%}\n'
               f'at all {len(covar_dict)} quarter ends', transform=right.transAxes,
               ha='left', va='top', color=ink, fontsize=10)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    checks = {
        'tracking_error_at_budget': bool(np.allclose(tracking_errors, TRACKING_ERROR, atol=1e-4)),
        'govt_weight_exceeds_three_budgets': bool(saa.loc[last, 'Govt'] > 3 * budgets['Govt']),
        'tilts_follow_alphas': bool(np.corrcoef(alphas.loc[list(covar_dict)].to_numpy().ravel(),
                                                active.to_numpy().ravel())[0, 1] > 0.5),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
