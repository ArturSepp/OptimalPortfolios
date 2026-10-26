"""Offline checks for the public instance, audit and reporting contracts."""
import json

import numpy as np
import pytest

from optimalportfolios.execution.v2 import ExecutionProblem, ExecutionRules
from papers.ranked_execution_2026.replication.benchmark import (
    ARRAYS, independent_audit, load_case, sha256, tracking_error_bp, verify_bundle,
)
from papers.ranked_execution_2026.replication.summarize import (
    METHODS, aggregate, manuscript_results, summarize,
)


@pytest.fixture
def problem():
    """Provide a four-instrument problem with analytically checkable constraints."""
    current = np.array([.1, .3, .3, .3])
    return ExecutionProblem(names=('Cash', 'Asset001', 'Asset002', 'Asset003'), cash=0,
        current=current, base=current, target=np.array([.1, .4, .2, .3]),
        lower=np.zeros(4), upper=np.ones(4), full_lower=np.zeros(4), full_upper=np.ones(4),
        groups=np.array([[0., 1., 1., 0.]]), group_lower=np.array([.5]),
        group_upper=np.array([.7]), mandatory=np.zeros(4, bool),
        categories=('',)*4, eligible=np.ones(4, bool), frozen=np.zeros(4, bool),
        rules=ExecutionRules(minimum_ticket=.05, holding_floor=.1, ticket_cap=2),
        covariance=np.eye(4)*.01)


@pytest.fixture
def bundle(tmp_path, problem):
    """Write the exact public format without any private research data."""
    arrays = {name: getattr(problem, name) for name in ARRAYS}
    arrays.update(holding_floor=problem.rules.holding_floor,
                  exemption_mask=problem.rules.exemption_mask)
    np.savez_compressed(tmp_path/'case_0001.npz', **arrays)
    entry = dict(id='case_0001', array_file='case_0001.npz', names=problem.names,
                 cash=0, categories=problem.categories,
                 rules=dict(minimum_ticket=.05, ticket_cap=2))
    manifest = dict(schema='ranked-execution-benchmark-1', cases=[entry],
                    files={'case_0001.npz': sha256(tmp_path/'case_0001.npz')})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, entry


def test_round_trip_and_changed_bytes_rejected(bundle, problem):
    """Checksums cover the arrays actually consumed, including harmless byte changes."""
    root, entry = bundle
    verify_bundle(root)
    actual = load_case(root, entry)
    np.testing.assert_array_equal(actual.target, problem.target)
    with (root/'case_0001.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_bundle(root)


def test_escaping_manifest_is_rejected(bundle):
    """A manifest cannot verify a file outside its dataset directory."""
    root, _ = bundle
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['files'] = {'../outside.npz': 'bad'}
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_bundle(root)


@pytest.mark.parametrize(('weights', 'violation'), [
    ([.09, .31, .3, .3], 'minimum_ticket'),
    ([.31, .09, .3, .3], 'holding_floor'),
    ([.1, .5, .3, .1], 'groups'),
    ([.1, .4, .4, .1], 'ticket_cap'),
    ([.2, .4, .2, .3], 'funding'),
    ([-.1, .5, .3, .3], 'lower'),
    ([.1, float('nan'), .3, .3], 'invalid_weights'),
])
def test_independent_audit_rejects_real_breaches(problem, weights, violation):
    """Each semantic defect is detected without consulting solver status."""
    audit = independent_audit(problem, weights)
    assert not audit['passed'] and violation in audit['violations']


def test_audit_boundary_does_not_hide_genuine_group_breach(problem):
    """The same declared tolerance admits rounding but not a larger exposure breach."""
    inside = [.1, .40000005, .3, .19999995]
    outside = [.1, .4000002, .3, .1999998]
    assert independent_audit(problem, inside)['passed']
    assert 'groups' in independent_audit(problem, outside)['violations']


def test_actual_tickets_use_original_holdings_and_zero_threshold(problem):
    """Tiny zero residuals stay unticketed; a real subminimum trade is rejected."""
    tiny = [.1-5e-9, .3+5e-9, .3, .3]
    small = [.1-2e-8, .3+2e-8, .3, .3]
    assert independent_audit(problem, tiny)['tickets'] == 0
    assert independent_audit(problem, tiny)['passed']
    assert 'minimum_ticket' in independent_audit(problem, small)['violations']
    audit = independent_audit(problem, problem.target)
    assert audit['tickets'] == 2 and audit['turnover'] == pytest.approx(.2)


def test_audit_is_separate_from_package_gate(problem, monkeypatch):
    """A broken package return gate cannot silently bless an invalid benchmark result."""
    import optimalportfolios.execution.v2.domain as domain
    monkeypatch.setattr(domain, 'audit_execution', lambda *args, **kwargs: {'passed': True})
    assert not independent_audit(problem, [.1, .5, .3, .1])['passed']


def test_qis_te_units_match_closed_form(problem):
    """Two opposite ten-percent shifts with ten-percent independent vols give this TE."""
    assert tracking_error_bp(problem, problem.current) == pytest.approx(100.*np.sqrt(2.))
    assert tracking_error_bp(problem, problem.target) == pytest.approx(0.)


def test_reporting_retains_failures_and_pair_denominators():
    """A failed preparation cannot disappear from the headline denominator."""
    valid = dict(id='case_0001', preparation={'mode': 'CORRIDOR'},
        preparation_wall_seconds=.2, base_te_bp=4., optional_candidates=3, mandatory=1, cap=11,
        methods={m: dict(status='audited', seconds=.1, end_to_end_seconds=.3,
            te_bp=1. if m == 'Ranked' else 2.,
            audit=dict(passed=True, tickets=2, turnover=.2), diagnostics={}) for m in METHODS})
    failed = dict(id='case_0002', preparation={}, preparation_wall_seconds=.3, methods={})
    summary = aggregate([valid, failed])
    assert (summary['cases'], summary['prepared'], summary['common']) == (2, 1, 1)
    assert summary['preparation_failures'] == ['case_0002']
    assert summary['comparisons']['Greedy']['ranked_lower'] == 1
    assert 'All 2 decisions were attempted' in manuscript_results(summary)


def test_smoke_results_cannot_be_manuscript_results(tmp_path):
    """The quick-start sample cannot masquerade as a complete empirical experiment."""
    (tmp_path/'complete.json').write_text(json.dumps(dict(cases=3, full_sample=False)))
    with pytest.raises(ValueError, match='smoke'):
        summarize(tmp_path, tmp_path/'summary')
