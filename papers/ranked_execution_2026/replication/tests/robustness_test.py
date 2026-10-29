"""Independent numerical contracts for the public robustness extension."""
from dataclasses import replace
import numpy as np
import pytest
from papers.ranked_execution_2026.replication.robustness import (
    synthetic_case, perturb, rebuild_corridor, changed_metrics, solve,
)
from papers.ranked_execution_2026.replication.benchmark import independent_audit


@pytest.mark.parametrize('n,b,width,band', [(12, 3, .005, 'narrow'), (36, 6, .08, 'wide')])
def test_synthetic_domain_is_funded_audited_and_repeatable(n, b, width, band):
    """The generated base is feasible before any solver and no private inputs are needed."""
    problem = synthetic_case(n, b, width, band, 41001)
    repeat = synthetic_case(n, b, width, band, 41001)
    assert independent_audit(problem, problem.base)['passed']
    assert problem.rules.ticket_cap == b+2
    assert problem.rules.audit_tolerance == 1e-7
    assert problem.rules.zero_tolerance == 1e-8
    np.testing.assert_array_equal(problem.covariance, repeat.covariance)
    assert np.linalg.eigvalsh(problem.covariance).min() > 0.


def test_perturbations_isolate_inputs_and_preserve_domain():
    """Target mass and covariance PSD are preserved without changing hard constraints."""
    problem = synthetic_case(12, 3, .005, 'narrow', 41001)
    target = perturb(problem, 71, .001, .05, 'target')
    risk = perturb(problem, 71, .001, .05, 'covariance')
    assert .5*np.abs(target.target-problem.target).sum() == pytest.approx(.001)
    assert target.target.sum() == pytest.approx(problem.target.sum())
    np.testing.assert_array_equal(target.covariance, problem.covariance)
    np.testing.assert_array_equal(risk.target, problem.target)
    np.testing.assert_array_equal(risk.groups, problem.groups)
    np.testing.assert_array_equal(target.lower, problem.lower)
    assert np.linalg.eigvalsh(risk.covariance).min() > 0.


def test_rebuilt_corridor_does_not_change_product_limits_or_lifecycle():
    """Domain rebuilding changes only optional target corridors, never the hard product domain."""
    problem = synthetic_case(12, 3, .005, 'narrow', 41001)
    changed = perturb(problem, 71, .005, .05, 'joint')
    rebuilt = rebuild_corridor(problem, changed)
    for field in ('full_lower', 'full_upper', 'groups', 'group_lower', 'group_upper',
                  'mandatory', 'current'):
        np.testing.assert_array_equal(getattr(problem, field), getattr(rebuilt, field))
    assert rebuilt.rules == problem.rules


def test_independent_audit_keeps_true_group_breach():
    """A deliberately violated cap is rejected at the unchanged audit threshold."""
    problem = synthetic_case(12, 3, .005, 'narrow', 41001)
    exposure = problem.groups@problem.base
    bad = replace(problem, group_upper=exposure-2e-7)
    assert 'groups' in independent_audit(bad, bad.base)['violations']


def test_ranked_smoke_and_common_reference_zero_change():
    """Actual V2 sizing delivers an audited portfolio and self-comparison is zero."""
    problem = synthetic_case(12, 3, .005, 'narrow', 41001)
    result = solve(problem)
    assert result['audit']['passed']
    metrics = changed_metrics(problem, result, problem, result)
    assert metrics['half_l1'] == 0.
    assert metrics['support_symmetric_difference'] == 0


def test_summary_rejects_smoke_run(tmp_path):
    """A partial run cannot become a headline robustness table."""
    import json
    from papers.ranked_execution_2026.replication.summarize_robustness import summarize
    (tmp_path/'complete.json').write_text(json.dumps({'full_sample': False, 'study': 'all'}))
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='complete frozen study'):
        summarize(tmp_path, tmp_path/'summary')


def test_complete_failed_run_retains_denominators(tmp_path):
    """Failed preparation remains a failed case rather than disappearing from the study."""
    import json
    from papers.ranked_execution_2026.replication.robustness import PROTOCOL
    from papers.ranked_execution_2026.replication.summarize_robustness import summarize
    spec = json.loads(PROTOCOL.read_text())
    spec['sensitivity']['case_count'] = 1
    for key in ('optional_assets', 'discretionary_allowance', 'group_half_width', 'cash_band', 'seeds'):
        spec['scarcity'][key] = spec['scarcity'][key][:1]
    (tmp_path/'protocol.json').write_text(json.dumps({'specification': spec, 'versions': {}}))
    (tmp_path/'complete.json').write_text(json.dumps({'full_sample': True, 'study': 'all'}))
    (tmp_path/'sensitivity_case_0001.json').write_text(json.dumps({'id': 'case_0001', 'variants': []}))
    (tmp_path/'scarcity_000.json').write_text(json.dumps({'id': 'scarcity_000', 'methods': {},
        'budget': 3, 'optional_assets': 12, 'width': .005, 'cash_band': 'narrow'}))
    summary = summarize(tmp_path, tmp_path/'summary')
    assert summary['sensitivity']['expected_variants'] == 12
    assert summary['sensitivity']['audited_variants'] == 0
    assert summary['scarcity']['ranked_comparisons']['ExactRanked']['expected'] == 1
    assert summary['scarcity']['ranked_comparisons']['ExactRanked']['pairs'] == 0
    assert 'failed' in (tmp_path/'summary'/'robustness.tex').read_text()


def test_archived_grids_pair_by_inputs_and_fail_on_unpaired_cases(tmp_path):
    """Combining an old holdout cannot silently pair different ordinal case IDs."""
    import json
    from papers.ranked_execution_2026.replication.summarize_robustness import compare_previous
    current, paths = [], []
    for index, seed in enumerate((41001, 51001)):
        path = tmp_path/str(seed)
        path.mkdir()
        paths.append(path)
        (path/'protocol.json').write_text('{}')
        old = dict(id='scarcity_000', optional_assets=36, budget=6, width=.005,
                   cash_band='wide', seed=seed, methods={
                       'Ranked': dict(te_bp=10.+index, audit=dict(passed=True)),
                       'ExactRanked': dict(te_bp=9.+index)})
        (path/'scarcity_000.json').write_text(json.dumps(old))
        current.append(dict(old, id=f'scarcity_{index+100}', methods={
            'Ranked': dict(te_bp=9.+index, audit=dict(passed=True))}))
    result = compare_previous(list(reversed(current)), paths, 1e-4)
    assert result['pairs'] == result['improved'] == result['old_misses_recovered'] == 2
    with pytest.raises(ValueError, match='Duplicate'):
        compare_previous(current, paths+paths, 1e-4)
    with pytest.raises(ValueError, match='Missing'):
        compare_previous(current, paths[:1], 1e-4)
    with pytest.raises(ValueError, match='Unpaired'):
        compare_previous(current[:1], paths, 1e-4)
    current[0]['methods']['Ranked']['audit']['passed'] = False
    with pytest.raises(ValueError, match='audited paired'):
        compare_previous(current, paths, 1e-4)
