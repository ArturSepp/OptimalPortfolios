"""Public API of the reporting layer."""

from optimalportfolios.reports.portfolio_result_plots import plot_efficient_frontier


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'plot_efficient_frontier',
]
