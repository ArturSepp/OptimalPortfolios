"""Protect matched domains, independent budget calls and complete cost accounting."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from papers.ranked_execution_2026.replication import joint
from papers.ranked_execution_2026.replication.breadth import case_grid, generated_case


def specification():
    """Read the public, frozen design without private fixtures."""
    return json.loads(joint.PROTOCOL.read_text())


def problem():
    """Supply one tiny public-generator instance for protocol tests."""
    spec = specification()
    return generated_case(case_grid(spec)[-1], spec)


def fake_controls(monkeypatch, ranked_fails=False):
    """Record calls and deliberately mutate the supplied start to check isolation."""
    calls = []
    ticks = iter(range(50))
    monkeypatch.setattr(joint, 'perf_counter', lambda: next(ticks))

    def ranked(p, config):
        """Return only the common audited base for deterministic control checks."""
        if ranked_fails:
            raise RuntimeError('deliberate heuristic failure')
        return SimpleNamespace(weights=p.base.copy(), status='test', diagnostics={})

    def exact(p, domain, **kwargs):
        """Capture the original supplied start before emulating a mutating solver."""
        start = kwargs['mip_start']
        calls.append(dict(domain=domain, **dict(kwargs, mip_start=None if start is None else start.copy())))
        if start is not None:
            start[:] = 0.
        return SimpleNamespace(weights=p.base.copy(), status='test', diagnostics={})

    monkeypatch.setattr(joint, 'wrapper_ranked_execution', ranked)
    monkeypatch.setattr(joint, 'wrapper_exact_execution', exact)
    return calls


@pytest.mark.parametrize('index', [0, 1])
def test_all_budgets_have_separate_starts_and_full_costs(monkeypatch, index):
    """Each native arm retains the domain and charges its own requested start."""
    p, spec = problem(), specification()
    calls = fake_controls(monkeypatch)
    records, order = joint.serial_arms(p, spec, index=index, preparation_seconds=3.)
    assert set(records) == set(joint.methods(spec)) and len(records) == 7
    budgets = spec['exact_seconds'][::(-1 if index else 1)]
    assert [c['time_limit_s'] for c in calls] == [s for s in budgets for _ in range(2)]
    for name, call in zip(order[1:], calls):
        assert call['domain'] == 'CORRIDOR' and call['threads'] == 1
        assert call['gap_rel'] == spec['exact_gap_rel']
        item = records[name]
        if name.startswith('JointRanked'):
            np.testing.assert_array_equal(call['mip_start'], p.base)
            assert item['end_to_end_seconds'] == 5.
        else:
            assert call['mip_start'] is None and item['end_to_end_seconds'] == 4.
    np.testing.assert_array_equal(records['Ranked']['weights'], p.base)


def test_failed_heuristic_remains_visible_and_cannot_supply_a_start(monkeypatch):
    """A failed V2 arm remains in the denominator and its time is still charged."""
    calls = fake_controls(monkeypatch, ranked_fails=True)
    records, _ = joint.serial_arms(problem(), specification(), preparation_seconds=3.)
    assert len(records) == 7 and records['Ranked']['status'] == 'exception'
    assert not records['Ranked']['audit']['passed']
    assert all(c['mip_start'] is None for c in calls)
    for name, item in records.items():
        if name.startswith('JointRanked'):
            assert item['requested_start'] == 'Ranked'
            assert item['supplied_start'] == 'common_base'
            assert item['start_generation_seconds'] == 1.


def test_protocol_preserves_thresholds_and_declares_every_case():
    """Larger universes change scale while retaining the original rule magnitudes."""
    spec = specification()
    cases = case_grid(spec)
    assert len(cases) == 26 and sum(c['sample'] == 'broad' for c in cases) == 24
    assert {c['optional_assets'] for c in cases if c['sample'] == 'broad'} == {24, 72, 144}
    assert spec['minimum_ticket'] == .005 and spec['audit_tolerance'] == 1e-7
    assert spec['zero_tolerance'] == 1e-8 and spec['objective_tolerance'] == 1e-12
    for case in cases:
        p = generated_case(case, spec)
        assert np.isfinite(p.current).all() and (p.current >= 0).all()
        assert p.rules.ticket_cap == case['budget'] + 2


def test_mutated_prepared_problem_aborts_protocol(monkeypatch):
    """Shared-domain mutation must fail rather than contaminate subsequent arms."""
    fake_controls(monkeypatch)

    def invalid(p, config):
        """Emulate an illegal replacement of the raw objective target."""
        object.__setattr__(p, 'target', p.target + .01)
        return SimpleNamespace(weights=p.base.copy(), status='test', diagnostics={})

    monkeypatch.setattr(joint, 'wrapper_ranked_execution', invalid)
    with pytest.raises(ValueError, match='mutated'):
        joint.serial_arms(problem(), specification())


def reporting_case():
    """Include a failed arm and a diagnostic bound contradicted by a feasible vector."""
    entries = {m:dict(audit={'passed': True, 'optional_tickets': 2}, te_bp=10.,
        seconds=1., end_to_end_seconds=2., status='audited_incumbent', diagnostics={})
        for m in joint.methods(specification())}
    entries['JointBase10']['audit']['passed'] = False
    entries['JointBase60']['te_bp'] = 20.
    entries['JointBase60']['diagnostics'] = dict(native_bound=2.25e-6, native_status='optimal')
    return dict(id='joint_000', sample='broad', optional_assets=24, optional_candidates=20,
                budget=2, methods=entries, relaxations={'plain':{'lower_bound':.00000025}})


def test_summary_preserves_failures_and_separates_native_from_valid_bounds():
    """A native optimal label or higher native bound must not close a valid gap."""
    from papers.ranked_execution_2026.replication.summarize_joint import sample_summary
    case = reporting_case()
    failed = dict(case, id='joint_001', methods={})
    summary = sample_summary([case, failed], specification())
    assert summary['expected'] == 2 and summary['prepared'] == 1
    assert summary['methods']['JointBase10']['audited'] == 0
    assert summary['methods']['JointBase10']['seconds']['n'] == 1
    assert summary['methods']['JointBase60']['expected'] == 2
    assert len(summary['methods']['JointBase60']['native_bound_above_best']) == 1
    assert summary['conservative_gap_bp']['maximum'] == pytest.approx(5.)
    assert summary['bound_closed_at_band'] == 0
    case['relaxations']['plain']['lower_bound'] = .01
    invalid = sample_summary([case], specification())
    assert invalid['bound_inconsistencies'] == 1
    assert invalid['conservative_gap_bp']['n'] == 0


@pytest.mark.parametrize('fault', ['missing_arm', 'changed_input', 'smoke', 'identity', 'duplicate'])
def test_manuscript_loader_rejects_incomplete_or_changed_runs(tmp_path, fault):
    """The publication denominator and frozen input identity cannot silently shrink."""
    from papers.ranked_execution_2026.replication.benchmark import sha256
    from papers.ranked_execution_2026.replication.summarize_joint import load_complete
    (tmp_path/'inputs').mkdir()
    data = tmp_path/'inputs/joint_000.npz'
    data.write_bytes(b'frozen public fixture')
    row = reporting_case()
    declared = dict(id=row['id'], sample='broad', input_sha256=sha256(data))
    row.update(declared)
    protocol = dict(cases=[declared], methods=joint.methods(specification()), specification=specification())
    completion = dict(cases=1, full_sample=True)
    if fault == 'missing_arm':
        row['methods'].pop('JointBase60')
    elif fault == 'changed_input':
        data.write_bytes(b'different input')
    elif fault == 'smoke':
        completion['full_sample'] = False
    elif fault == 'identity':
        row['sample'] = 'tiny'
    else:
        protocol['cases'].append(declared)
    for name, value in [('protocol',protocol), ('complete',completion), ('joint_000',row)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_complete(tmp_path)


def test_compact_manuscript_keeps_sensitivity_and_archives_development():
    """Compact presentation must preserve numerical sensitivity values and its anchor."""
    from papers.ranked_execution_2026.replication.summarize_robustness import render_tex
    path = joint.PROTOCOL.with_name('robustness_expected_summary.json')
    summary = json.loads(path.read_text())
    text = render_tex(summary, compact=True)
    assert 'Controlled input sensitivity' in text and r'\label{sec:robustness}' in text
    assert 'Earlier scarce-ticket cohorts' not in text
    assert 'sensitivity.pdf' in text
    assert 'Earlier scarce-ticket cohorts' in render_tex(summary)
