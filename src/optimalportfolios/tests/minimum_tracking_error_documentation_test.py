"""Verify the tracking-error article's executable results, units, timing and navigation."""

from pathlib import Path
import re
import runpy

import numpy as np
import pandas as pd
import pytest


LEGACY_ANCHORS = frozenset({
    'minimum-tracking-error', 'inputs-units-and-alignment', 'minimal-offline-example',
    'single-date-versus-rolling-use', 'constraints-and-failure-modes', 'see-also',
})


@pytest.fixture(scope='module')
def article(root: Path) -> str:
    """Read the migrated article; root skips checkout-only checks in installed wheels."""
    assert not (root / 'docs/minimum_tracking_error.rst').exists()
    return (root / 'docs/minimum_tracking_error.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def blocks(article: str) -> list:
    """Extract the two canonical, sequential Python examples from the article."""
    matches = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(matches) == 2, 'Do not silently retire an executable example.'
    assert all(not options.strip() for options, _ in matches)
    return [code for _, code in matches]


@pytest.fixture(scope='module')
def examples(blocks: list) -> dict:
    """Execute the actual article blocks rather than a separately maintained copy."""
    state = {'__name__': '__minimum_tracking_error_article__'}
    for index, code in enumerate(blocks, start=1):
        exec(compile(code, f'minimum_tracking_error.md (block {index})', 'exec'), state)
    return state


def test_article_meets_source_and_link_standard(article, root):
    """Require portable methodology format and valid local file links."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](
        article, root / 'docs/minimum_tracking_error.md', root)


def test_legacy_tracking_error_fragments_survive(article, root):
    """Keep the six existing section links through the RST-to-Markdown conversion."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        heading = re.match(r'^#{1,6} (.+)', line)
        if heading:
            anchors.add(re.sub(r'[^\w -]', '', heading[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors


def test_capped_allocation_has_independent_optimality_certificate(examples):
    """Check a closed-form active-set solution and its first-order conditions."""
    covariance = examples['covar'].to_numpy()
    benchmark = examples['benchmark'].to_numpy()
    # Solve the B-to-C transfer derivative, independently of CVXPY.
    transfer = 0.00195 / 0.028
    reference = np.array([0.35, 0.30 + transfer, 0.35 - transfer])
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    assert reference.sum() == pytest.approx(1.0)
    assert np.all(reference >= 0.0)
    assert np.all(reference <= examples['constraints'].max_weights.to_numpy())
    gradient = covariance @ (reference - benchmark)
    assert gradient[1] == pytest.approx(gradient[2], abs=1e-14)
    assert gradient[0] < gradient[2]  # Inward movement from A's cap increases the objective.
    np.testing.assert_allclose(examples['weights'], reference, atol=2e-6, rtol=0.0)
    outcome = examples['outcome']
    assert outcome.accepted and outcome.compliant
    assert outcome.fallback_source is None
    assert outcome.covar_factorization.n_eigenvalues_floored == 0
    np.testing.assert_allclose(
        outcome.covar_factorization.covar, covariance, atol=1e-15, rtol=0.0)


def test_displayed_allocation_and_tracking_error_match_execution(article, examples):
    """Catch stale weight cells or percent scaling despite runnable example code."""
    rows = re.findall(
        r'^\| ([ABC]) \| ([0-9.]+) \| ([0-9.]+) \| (-?[0-9.]+) \|$', article, re.M)
    assert len(rows) == 3
    assert [row[0] for row in rows] == examples['assets']
    displayed = np.array([[float(value) for value in row[1:]] for row in rows])
    np.testing.assert_allclose(displayed, examples['result'], atol=1e-6, rtol=0.0)
    match = re.search(r'Annualized tracking error: \*\*([0-9.]+)%\*\*', article)
    assert match
    assert float(match[1]) == pytest.approx(100.0 * examples['tracking_error'], abs=1e-6)


def test_covariance_scale_changes_risk_units_but_not_this_allocation(examples):
    """Verify a scale identity through qis without adding another TE implementation."""
    opt = examples['opt']
    scaled_covar = examples['covar'] / 12.0
    scaled_model = opt.build_risk_model({examples['date']: scaled_covar})
    scaled_te = scaled_model.compute_tre_at_date(
        benchmark_weights=examples['benchmark'],
        portfolio_weights=examples['weights'],
        date=examples['date'])
    assert scaled_te * np.sqrt(12.0) == pytest.approx(
        examples['tracking_error'], rel=1e-12)
    weights, outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=scaled_covar, benchmark_weights=examples['benchmark'],
        constraints=examples['constraints'], weights_0=examples['benchmark'])
    assert outcome.accepted and outcome.compliant
    assert outcome.covar_factorization.n_eigenvalues_floored == 0
    np.testing.assert_allclose(weights, examples['weights'], atol=2e-6, rtol=0.0)


def test_rolling_table_matches_execution_without_future_benchmark_leakage(article, examples):
    """Perturb the future observation and check that only its decision date changes."""
    dates = examples['dates']
    actual = examples['rolling_weights']
    expected = examples['benchmarks'].reindex(dates, method='ffill')
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.0)
    pd.testing.assert_index_equal(actual.index, dates)
    pd.testing.assert_index_equal(actual.columns, examples['prices'].columns)
    rows = re.findall(
        r'^\| (2024-\d{2}-\d{2}) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$',
        article, re.M)
    assert len(rows) == 3
    assert [row[0] for row in rows] == list(dates.strftime('%Y-%m-%d'))
    displayed = np.array([[float(value) for value in row[1:]] for row in rows])
    np.testing.assert_allclose(displayed, actual, atol=1e-6, rtol=0.0)
    perturbed = examples['benchmarks'].copy()
    perturbed.loc[dates[-1]] = [0.10, 0.20, 0.70]
    opt = examples['opt']
    changed = opt.rolling_minimise_tracking_error(
        prices=examples['prices'], constraints=opt.Constraints(is_long_only=True),
        benchmark_weights=perturbed,
        covar_dict={date: examples['covar'] for date in dates})
    np.testing.assert_allclose(changed.iloc[:2], actual.iloc[:2], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(changed.iloc[-1], perturbed.iloc[-1], atol=2e-6, rtol=0.0)


@pytest.mark.parametrize(
    'defect', ['benchmark.drop("C")', 'benchmark.mask(benchmark.index == "C")'])
def test_example_guard_rejects_missing_benchmark_coverage(blocks, defect):
    """The documented pre-solve guard rejects both absent labels and explicit NaNs."""
    original = 'benchmark = benchmark.reindex(covar.columns)'
    assert blocks[0].count(original) == 1
    code = blocks[0].replace(
        original, f'benchmark = {defect}\n{original}')
    with pytest.raises(ValueError, match='benchmark must cover every covariance asset'):
        exec(compile(code, 'minimum_tracking_error.md (coverage defect)', 'exec'), {})
