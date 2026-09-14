"""Execute the constraints article and verify its analytical and navigation contracts."""

import ast
from pathlib import Path
import re
import runpy

import numpy as np
import pytest


# Public section fragments before the M2a reorganization; changing heading levels must not
# invalidate existing links. The renamed notation section has an explicit portable anchor.
LEGACY_ANCHORS = frozenset({
    'portfolio-constraints', 'notation-and-units', 'the-constraint-map',
    'exposure-long-only-and-instrument-boxes', 'constructor-validation',
    'target-return-and-portfolio-volatility', 'minimum-target-return',
    'maximum-portfolio-volatility', 'benchmark-relative-risk', 'total-tracking-error',
    'group-tracking-error', 'group-allocation-and-benchmark-deviations',
    'absolute-group-allocation', 'sector-and-style-deviations',
    'turnover-and-trading-constraints', 'total-turnover', 'group-turnover', 'benchmark-beta',
    'hard-and-utility-enforcement', 'forced-constraints', 'utility-constraints',
    'group-precedence', 'solver-specific-utility-paths', 'backend-capability-matrix',
    'a-complete-forced-constraint-example', 'converting-the-example-to-utility-mode',
    'universe-alignment-and-rebalancing-policy', 'use-the-production-alignment-method',
    'current-to-model-eligibility-corridor', 'frozen-positions', 'frozen-group-bound-waivers',
    'feasibility-and-diagnostics', 'before-solving', 'after-solving', 'configuration-checklist',
})


@pytest.fixture(scope='module')
def article(root: Path) -> str:
    """Read the authoritative article; the shared root fixture skips installed-wheel runs."""
    return (root / 'docs/constraints.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def example_state(article: str) -> dict:
    """Execute the actual documented Python sequence, excluding its one contextual fragment."""
    blocks = list(re.finditer(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S))
    executable = [block for block in blocks if not block[1].strip()]
    skipped = [block for block in blocks if block[1].strip()]
    assert len(executable) == 15, 'Do not silently retire or rename the executable examples.'
    assert len(skipped) == 1
    assert skipped[0][1].strip() == '+SKIP'
    assert skipped[0][2].startswith('outcome.compliant')
    state = {'__name__': '__constraints_article__'}
    for block in executable:
        line = article[:block.start(2)].count('\n') + 1
        exec(compile(block[2], f'constraints.md (line {line})', 'exec'), state)
    return state


def test_constraints_article_meets_source_standard(article, root):
    """Require source compliance while the separate external-viewer gates remain explicit."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](article, root / 'docs/constraints.md', root)


def test_legacy_section_fragments_survive(article, root):
    """Preserve the existing 35 public fragments across the structural reorganization."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        heading = re.match(r'^#{1,6} (.+)', line)
        if heading:
            anchors.add(re.sub(r'[^\w -]', '', heading[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors


def test_forced_result_has_independent_optimality_certificate(example_state):
    """Use three linear rows and a dual upper bound independently of the CVXPY compiler."""
    # Full investment, upper active Risk-assets exposure, and a supporting affine row of
    # weighted L1 turnover. The latter is valid for all weights, not only this trade pattern.
    rows = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, -0.5, -2.0]])
    rhs = np.array([1.0, 0.68, 0.125])
    reference = np.linalg.solve(rows, rhs)
    solution = example_state['solution'].to_numpy()
    np.testing.assert_allclose(solution, reference, atol=1e-6, rtol=0.0)
    returns = example_state['constraints'].asset_returns.to_numpy()
    multipliers = np.linalg.solve(rows.T, returns)
    assert np.all(multipliers[1:] >= 0.0)
    np.testing.assert_allclose(multipliers, [0.04, 0.02, 0.01], atol=1e-12, rtol=0.0)
    upper_bound = float(multipliers @ rhs)
    assert upper_bound == pytest.approx(0.05485, abs=1e-12)
    assert float(returns @ solution) == pytest.approx(upper_bound, abs=1e-7)
    residuals = example_state['evaluate_constraint_residuals'](
        solution, example_state['constraints'], covar=example_state['covar'].to_numpy())
    assert residuals
    assert all(record.passed for record in residuals if record.hard)


def test_displayed_allocations_match_executed_examples(article, example_state):
    """Catch stale displayed numbers even when the example itself still executes."""
    table = re.search(r'```text\nasset\n(.*?)dtype: float64\n```', article, re.S)
    assert table, 'Keep an explicit displayed forced allocation.'
    weights = [float(value) for value in re.findall(
        r'^(?:Equity|Bond|Gold)\s+([0-9.]+)$', table[1], re.M)]
    assert len(weights) == 3
    np.testing.assert_allclose(weights, example_state['solution'], atol=1e-6, rtol=0.0)
    utility = re.search(r'Equity=([0-9.]+)`, `Bond=([0-9.]+)`, and `Gold=([0-9.]+)', article)
    assert utility, 'Keep the displayed utility allocation.'
    np.testing.assert_allclose(
        [float(value) for value in utility.groups()],
        example_state['utility_solution'], atol=1e-6, rtol=0.0)


def test_soft_breaches_remain_visible_without_determining_compliance(article, example_state):
    """Soft limits report positive violations while retaining passed=True."""
    displayed = re.search(r'```text\n(\[\("turnover".*?\])\n```', article, re.S)
    assert displayed
    expected = ast.literal_eval(displayed[1])
    actual = [(row.constraint_type, round(row.violation, 6), row.passed)
              for row in example_state['soft_violations']]
    assert actual == expected
    assert all(not row.hard and row.violation > 0 for row in example_state['soft_violations'])


def test_alignment_examples_preserve_corridor_and_frozen_waiver(article, example_state):
    """Protect the documented eligibility and one-period frozen-position examples."""
    np.testing.assert_allclose(example_state['lower'], [0.2, 0.3, 0.0, 0.0])
    np.testing.assert_allclose(example_state['upper'], [0.5, 0.3, 0.5, 0.0])
    np.testing.assert_array_equal(example_state['indicators'], [1, 1, 1, 0])
    group = example_state['aligned'].group_lower_upper_constraints
    expected = 0.25 + 1e-8
    assert group.group_max_allocation['Illiquid'] == pytest.approx(expected, abs=1e-12)
    displayed = re.search(r'group_max_allocation\["Illiquid"\]\n# ([0-9.]+)', article)
    assert displayed, 'Keep the displayed frozen-bound waiver.'
    assert float(displayed[1]) == pytest.approx(expected, abs=1e-12)
    assert not example_state['breaches'].empty
