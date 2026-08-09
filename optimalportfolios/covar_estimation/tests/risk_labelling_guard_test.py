"""Regression tests for optional dependencies in risk-cluster matching."""

import sys

import pytest

from optimalportfolios.covar_estimation.risk_labelling import _match_panel_mcf


def test_mcf_import_error_names_extra_and_dependency_free_alternative(monkeypatch) -> None:
    """The networkx guard names both the install extra and Hungarian alternative."""
    monkeypatch.setitem(sys.modules, "networkx", None)

    with pytest.raises(ImportError) as exc_info:
        _match_panel_mcf(
            snapshots={},
            x_covars={},
            overlap_metric="overlap",
            combine="gated",
            overlap_band=(0.20, 0.60),
            spread_vol_cut=0.025,
            w_overlap=0.6,
            bridge_window=1,
        )

    message = str(exc_info.value)
    assert "optimalportfolios[clustering]" in message
    assert "hungarian" in message
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
