"""Verify the finite supplied-input example, its oracle and explicit persistence."""
import json
import runpy

import pandas as pd
import pytest


def test_completed_output_inspection_and_explicit_export(root, tmp_path):
    """No default files; explicit tables retain values, status, units and decision date."""
    namespace = runpy.run_path(str(root / 'examples/docs/completed_output.py'))
    result, outcome = namespace['main']()
    assert not list(tmp_path.iterdir())
    destination = tmp_path / 'inspection'
    namespace['save_tables'](result, outcome, destination)
    assert {path.name for path in destination.iterdir()} == {
        'weights.csv', 'covariance.csv', 'summary.csv', 'constraints.csv', 'status.json'}
    weights = pd.read_csv(destination / 'weights.csv', index_col=0)
    pd.testing.assert_frame_equal(weights, result.to_weights_df(), check_exact=False, atol=1e-14)
    receipt = json.loads((destination / 'status.json').read_text(encoding='utf-8'))
    assert receipt['accepted'] and receipt['compliant']
    assert receipt['fallback_source'] is None
    assert receipt['risk_model_date'] == '2025-12-31T00:00:00'
    assert receipt['covariance_units'] == 'annual decimal return squared'
    with pytest.raises(FileExistsError):
        namespace['save_tables'](result, outcome, destination)


def test_independent_reference_rejects_a_plausible_wrong_allocation(root, monkeypatch):
    """A budget-neutral weight defect must fail even with unchanged accepted status."""
    import optimalportfolios as op

    namespace = runpy.run_path(str(root / 'examples/docs/completed_output.py'))
    solve = op.wrapper_quadratic_optimisation

    def defective_solution(**kwargs):
        """Inject a finite, budget-neutral perturbation after the real solver accepts."""
        weights, outcome = solve(**kwargs)
        weights = weights.copy()
        weights.iloc[0] += .01
        weights.iloc[1] -= .01
        return weights, outcome

    monkeypatch.setattr(op, 'wrapper_quadratic_optimisation', defective_solution)
    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        namespace['main']()
