"""Public API of ``optimalportfolios``: covariance estimation, optimisers, universes
and the alpha and reporting layers, re-exported from their subpackages.
"""

import optimalportfolios.local_path  # noqa: F401  re-exported as optimalportfolios.local_path

from optimalportfolios.config import PortfolioObjective as PortfolioObjective

# Each subpackage below declares an ``__all__``, so these stars re-export exactly the list
# written down in that subpackage's ``__init__`` and nothing else — the surface is auditable
# there rather than being whatever happened to be left in a namespace. F403 is silenced
# because the rule's premise ("unable to detect undefined names") no longer holds.
from optimalportfolios.utils.__init__ import *  # noqa: F403

from optimalportfolios.covar_estimation.__init__ import *  # noqa: F403

from optimalportfolios.optimization.__init__ import *  # noqa: F403

from optimalportfolios.universe.__init__ import *  # noqa: F403

from optimalportfolios.reports.__init__ import *  # noqa: F403

from optimalportfolios.alphas.__init__ import *  # noqa: F403

"""Backward-compatible re-exports from factorlasso."""
from factorlasso import (  # noqa: F401
    DependenceMeasure,
    DistanceTransform,
    LassoModel,
    LassoModelType,
    CurrentFactorCovarData,
    RollingFactorCovarData,
    VarianceColumns,
    compute_dependence_matrix,
    compute_gerber_matrix,
)