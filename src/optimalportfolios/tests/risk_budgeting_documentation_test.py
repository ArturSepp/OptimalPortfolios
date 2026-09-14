"""Verify the risk-budgeting article's examples, numerical references and navigation."""

from pathlib import Path
import re
import runpy

import numpy as np
import pytest

from optimalportfolios.optimization.tests.risk_budgeting_full_investment_test import (
    conic_reference,
)


LEGACY_ANCHORS = frozenset({
    'risk-budgeting', 'inputs-and-conventions', 'minimal-offline-example',
    'single-date-versus-rolling-use', 'group-risk-budgets', 'hierarchical-risk-parity',
    'missing-data-frozen-assets-and-feasibility', 'see-also',
})


@pytest.fixture(scope='module')
def article(root: Path) -> str:
    """Read the migrated article; root skips repository checks for installed wheels."""
    assert not (root / 'docs/risk_budgeting.rst').exists()
    return (root / 'docs/risk_budgeting.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article: str) -> dict:
    """Execute the three canonical Python blocks in their documented order."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 3, 'Do not silently retire an executable example.'
    assert all(not options.strip() for options, _ in blocks)
    state = {'__name__': '__risk_budgeting_article__'}
    for index, (_, code) in enumerate(blocks, start=1):
        exec(compile(code, f'risk_budgeting.md (block {index})', 'exec'), state)
    return state


def test_article_meets_source_and_link_standard(article, root):
    """Require portable methodology format and valid local references."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](article, root / 'docs/risk_budgeting.md', root)


def test_legacy_risk_budgeting_fragments_survive(article, root):
    """Keep the eight public section links through the RST-to-Markdown conversion."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        heading = re.match(r'^#{1,6} (.+)', line)
        if heading:
            anchors.add(re.sub(r'[^\w -]', '', heading[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors


def test_correlated_example_matches_independent_conic_reference(examples):
    """Compare CCD/ADMM weights with the existing independent volatility/log conic solve."""
    bounds = np.column_stack([np.zeros(3), np.full(3, 0.8)])
    reference = conic_reference(
        examples['covar'].to_numpy(), examples['budgets'].to_numpy(), bounds)
    np.testing.assert_allclose(examples['weights'], reference, atol=2e-6, rtol=0.0)
    assert examples['weights'].sum() == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(
        examples['realised_budgets'], examples['budgets'], atol=1e-6, rtol=0.0)


def test_displayed_correlated_result_matches_execution(article, examples):
    """Detect stale table values even when the code still runs."""
    rows = re.findall(
        r'^\| (Equity|Bonds|Diversifier) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$',
        article, re.M)
    assert len(rows) == 3
    assert [row[0] for row in rows] == examples['assets']
    displayed = np.array([[float(value) for value in row[1:]] for row in rows])
    np.testing.assert_allclose(displayed[:, 0], examples['weights'], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(displayed[:, 1], examples['budgets'], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        displayed[:, 2], examples['realised_budgets'], atol=1e-6, rtol=0.0)


def test_diagonal_example_has_closed_form_weights_and_invariant_scores(article, examples):
    """Use diagonal risk-budget equations as an independent reference, not analytics code."""
    # QIS reports achieved contributions above; this closed form verifies the optimizer.
    volatilities = np.sqrt(np.diag(examples['diagonal_covar'].to_numpy()))
    reference = np.sqrt(examples['diagonal_budgets'].to_numpy()) / volatilities
    reference /= reference.sum()
    np.testing.assert_allclose(reference, [0.5, 0.5], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(examples['diagonal_weights'], reference, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(examples['diagonal_shares'], [0.8, 0.2], atol=1e-6, rtol=0.0)
    assert '# [0.5, 0.5]' in article and '# [0.8, 0.2]' in article
    opt = examples['opt']
    scaled = opt.wrapper_risk_budgeting(
        pd_covar=examples['diagonal_covar'] * 12.0,
        constraints=opt.Constraints(is_long_only=True),
        risk_budget=examples['diagonal_budgets'] * 100.0)
    np.testing.assert_allclose(scaled, reference, atol=1e-6, rtol=0.0)


def test_partial_group_example_preserves_budget_and_exclusion(article, examples):
    """Check the equal-group result, including its displayed missing-classification output."""
    actual = examples['group_budgets']
    np.testing.assert_allclose(actual, [0.5, 0.25, 0.25, 0.0], atol=1e-12, rtol=0.0)
    assert actual.sum() == pytest.approx(1.0)
    assert actual['Unclassified'] == 0.0
    assert '# [0.5, 0.25, 0.25, 0.0]' in article
