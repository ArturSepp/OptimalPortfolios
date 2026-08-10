"""Public API of ``optimalportfolios``: covariance estimation, optimisers, universes
and the alpha and reporting layers, re-exported from their subpackages.
"""

import optimalportfolios.local_path

from optimalportfolios.config import PortfolioObjective

# Each star below re-exports its subpackage's namespace, so a subpackage's public surface is
# the imports written in its own ``__init__`` and nothing beside them. F401 and F403 are off
# for ``__init__.py`` in pyproject.toml, which is what keeps that arrangement lintable:
# adding a public import here is one edit, not an import plus a list entry.
from optimalportfolios.utils.__init__ import *

from optimalportfolios.covar_estimation.__init__ import *

from optimalportfolios.optimization.__init__ import *

from optimalportfolios.universe.__init__ import *

from optimalportfolios.reports.__init__ import *

from optimalportfolios.alphas.__init__ import *

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