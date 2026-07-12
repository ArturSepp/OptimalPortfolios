"""
lazy loading of symbols that require the optional factorlasso dependency.

factorlasso is licensed GPL-3.0-or-later and is not installed by default:
install with ``pip install optimalportfolios[factorlasso]``.

The package ``__init__`` modules route attribute access for the symbols below
through ``load_factorlasso_symbol`` (PEP 562), so ``import optimalportfolios``
succeeds without factorlasso and the actionable error is raised at the point
of first use.
"""
import importlib
from typing import Any, Dict

# symbol -> defining module
FACTORLASSO_SYMBOLS: Dict[str, str] = {
    # re-exports from factorlasso
    'LassoModel': 'factorlasso',
    'LassoModelType': 'factorlasso',
    'CurrentFactorCovarData': 'factorlasso',
    'RollingFactorCovarData': 'factorlasso',
    'VarianceColumns': 'factorlasso',
    # optimalportfolios modules that import factorlasso at module level
    'FactorCovarEstimator': 'optimalportfolios.covar_estimation.factor_covar_estimator',
    'estimate_lasso_factor_covar_data': 'optimalportfolios.covar_estimation.factor_covar_estimator',
    'plot_current_covar_data': 'optimalportfolios.covar_estimation.covar_reporting',
    'plot_hcgl_covar_data': 'optimalportfolios.covar_estimation.covar_reporting',
    'run_rolling_covar_report': 'optimalportfolios.covar_estimation.covar_reporting',
    'PortfolioOptimisationResult': 'optimalportfolios.optimization.portfolio_result',
    'plot_efficient_frontier': 'optimalportfolios.reports.portfolio_result_plots',
}


def load_factorlasso_symbol(name: str) -> Any:
    """
    import the module defining name and return the symbol.

    Parameters
    ----------
    name : str
        key of FACTORLASSO_SYMBOLS.

    Returns
    -------
    Any
        the requested symbol.

    Raises
    ------
    ImportError
        when factorlasso is not installed, with the install command.
    """
    module_name = FACTORLASSO_SYMBOLS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split('.')[0] == 'factorlasso':
            raise ImportError(
                f"{name} requires the optional dependency factorlasso (GPL-3.0-or-later), "
                f"which is not installed by default: "
                f"run `pip install optimalportfolios[factorlasso]`, got missing module {exc.name!r}"
            ) from exc
        raise
    return getattr(module, name)
