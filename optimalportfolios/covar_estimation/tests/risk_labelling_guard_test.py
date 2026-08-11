"""Compatibility tests for the deprecated OptimalPortfolios lineage import path."""

import importlib
import sys

import pytest

from factorlasso import analyze_cluster_lineage, run_cluster_lineage_report


def test_risk_labelling_shim_warns_once_and_reexports_factorlasso() -> None:
    """A fresh shim import emits one warning and preserves both legacy callable identities."""
    module_name = "optimalportfolios.covar_estimation.risk_labelling"
    sys.modules.pop(module_name, None)

    with pytest.warns(DeprecationWarning, match="factorlasso") as warnings_seen:
        module = importlib.import_module(module_name)

    assert len(warnings_seen) == 1
    assert module.analyze_risk_clusters is analyze_cluster_lineage
    assert module.run_risk_label_report is run_cluster_lineage_report
