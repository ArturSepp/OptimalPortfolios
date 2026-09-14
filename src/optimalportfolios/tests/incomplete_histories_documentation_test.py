"""Execute incomplete-history examples and check data policies against independent references."""

from dataclasses import replace
from fractions import Fraction
import re
import runpy

import numpy as np
import pandas as pd
import pytest

LEGACY_ANCHORS = {
    'incomplete-histories-and-frozen-positions', 'missing-observations',
    'eligibility-versus-freezing', 'price-gaps-at-implementation', 'operational-checks', 'see-also',
}


@pytest.fixture(scope='module')
def article(root):
    """Load the canonical article only when the repository documentation exists."""
    return (root / 'docs/incomplete_histories.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article):
    """Execute the six sequential examples without allowing skipped or empty blocks."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 6 and all(not options.strip() for options, _ in blocks)
    state = {'__name__': '__incomplete_histories_article__'}
    for index, (_, code) in enumerate(blocks, start=1):
        exec(compile(code, f'incomplete_histories.md (block {index})', 'exec'), state)
    return state


@pytest.fixture(scope='module')
def ledger():
    """Value fixed units and one explicit trade using exact fractions, without a backtest loop."""
    first_liquid, first_gapped = Fraction(3, 5), Fraction(2, 5)
    priced_nav = first_liquid * 110
    new_liquid = priced_nav * Fraction(3, 5) / 110
    cash = (first_liquid - new_liquid) * 110
    return {
        'held_gap': {
            'units': [[first_liquid, first_gapped]] * 4,
            'nav': [100, priced_nav, first_liquid * 120 + first_gapped * 110,
                    first_liquid * 130 + first_gapped * 120],
            'cash': [0] * 4,
        },
        'traded_gap': {
            'units': [[first_liquid, first_gapped]] + [[new_liquid, 0]] * 3,
            'nav': [100, priced_nav, new_liquid * 120 + cash, new_liquid * 130 + cash],
            'cash': [0, cash, cash, cash],
        },
        'late_entry': {
            'units': [[first_liquid, 0]] * 4,
            'nav': [100, first_liquid * 110 + 40, first_liquid * 120 + 40,
                    first_liquid * 130 + 40],
            'cash': [40] * 4,
        },
    }


def test_article_structure_links_and_legacy_fragments(article, root):
    """Keep the standard structure, source links and all six original section anchors."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](
        article, root / 'docs/incomplete_histories.md', root)
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        match = re.match(r'^#{1,6} (.+)', line)
        if match:
            anchors.add(re.sub(r'[^\w -]', '', match[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors
    assert not (root / 'docs/incomplete_histories.rst').exists()


def test_policy_panels_are_distinct_and_complete(examples):
    """The preserved example encodes two different universes with explicit binary rows."""
    eligibility, tradability = examples['eligibility'], examples['can_rebalance']
    assert eligibility.columns.tolist() == ['Liquid', 'Late Starter']
    assert tradability.columns.tolist() == ['Liquid', 'Locked Fund']
    pd.testing.assert_index_equal(eligibility.index, tradability.index)
    np.testing.assert_array_equal(eligibility, [[1, 0], [1, 1]])
    np.testing.assert_array_equal(tradability, [[1, 0], [1, 0]])


def test_first_alignment_uses_stored_baseline(examples):
    """A stored 40% holding pins both fund bounds without a supplied current-weight argument."""
    aligned = examples['aligned']
    np.testing.assert_allclose(aligned.min_weights, [0, .4], atol=0)
    np.testing.assert_allclose(aligned.max_weights, [1, .4], atol=0)
    np.testing.assert_allclose(examples['spec'].min_weights, 0, atol=0)
    np.testing.assert_allclose(examples['spec'].max_weights, 1, atol=0)


@pytest.mark.parametrize('missing_side', ['min_weights', 'max_weights'])
def test_freeze_does_not_create_missing_box_side(examples, missing_side):
    """An absent box side stays absent, so a lone configured side is not an equality pin."""
    spec = replace(examples['spec'], **{missing_side: None})
    aligned = spec.update_with_valid_tickers(
        valid_tickers=examples['assets'],
        rebalancing_indicators=examples['can_rebalance'].iloc[0],
    )
    assert getattr(aligned, missing_side) is None
    other = 'max_weights' if missing_side == 'min_weights' else 'min_weights'
    assert getattr(aligned, other)['Locked Fund'] == .4


def test_supplied_baseline_overrides_stored_and_absent_baseline_does_not_freeze(examples):
    """Current holdings take precedence; no baseline leaves the original box intact."""
    kwargs = dict(valid_tickers=examples['assets'],
                  rebalancing_indicators=examples['can_rebalance'].iloc[0])
    aligned = examples['spec'].update_with_valid_tickers(
        weights_0=pd.Series([.7, .3], index=examples['assets']), **kwargs)
    assert aligned.min_weights['Locked Fund'] == aligned.max_weights['Locked Fund'] == .3
    cold = replace(examples['spec'], weights_0=None).update_with_valid_tickers(**kwargs)
    assert cold.min_weights['Locked Fund'] == 0
    assert cold.max_weights['Locked Fund'] == 1


def test_flat_drift_leg_shares_nav_denominator(examples):
    """Currency values 72 and 40 imply 112 NAV, so the flat leg cannot retain a 40% weight."""
    expected = np.array([Fraction(72, 112), Fraction(40, 112)], dtype=float)
    np.testing.assert_allclose(examples['drifted'], expected, atol=1e-14)
    assert examples['drifted']['Locked Fund'] < .4


@pytest.mark.parametrize('name', ['held_gap', 'traded_gap', 'late_entry'])
def test_missing_price_paths_against_units_and_cash(examples, ledger, name):
    """Validate every displayed NAV, held unit and residual cash value from a currency ledger."""
    portfolio, expected = examples[name], ledger[name]
    np.testing.assert_allclose(portfolio.nav, np.array(expected['nav'], float), atol=1e-12)
    np.testing.assert_allclose(portfolio.units, np.array(expected['units'], float), atol=1e-12)
    marked_value = (portfolio.units * portfolio.prices).sum(axis=1)
    np.testing.assert_allclose(
        portfolio.nav - marked_value, np.array(expected['cash'], float), atol=1e-12)
    np.testing.assert_allclose(portfolio.realized_costs, 0, atol=0)


def test_displayed_nav_table_matches_independent_ledger(article, ledger):
    """The four quoted rows must agree with the reference at the displayed currency precision."""
    rows = re.findall(
        r'^\| (2024-01-0[2-5]) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$', article, re.M)
    assert [row[0] for row in rows] == [f'2024-01-0{day}' for day in range(2, 6)]
    actual = np.array([[float(value) for value in row[1:]] for row in rows])
    expected = np.array([ledger[name]['nav'] for name in
                         ('held_gap', 'traded_gap', 'late_entry')], float).T
    np.testing.assert_allclose(actual, expected, atol=.005, rtol=0)


def test_missing_price_warnings_are_retained(examples):
    """Both interior gaps and missing execution quotes must remain visible to the reader."""
    messages = examples['warning_messages']
    assert sum('inside the reported history' in message for message in messages) == 2
    assert sum('have no price on their traded date' in message for message in messages) == 2


def test_filtering_floor_and_explicit_eligibility(examples):
    """Only positive diagonals survive; a floor changes the tiny survivor, not eligibility."""
    for name in ('filtered', 'floored'):
        assert examples[name].columns.tolist() == ['Liquid', 'Warmup']
    np.testing.assert_array_equal(np.diag(examples['filtered']), [.04, 1e-12])
    np.testing.assert_array_equal(np.diag(examples['floored']), [.04, 1e-6])
    assert examples['eligible'].columns.tolist() == ['Liquid']
    assert examples['covariance'].loc['Warmup', 'Warmup'] == 1e-12


@pytest.mark.parametrize('position', [(0, 0), (0, 1)])
def test_filter_helper_alone_does_not_validate_finite_covariance(examples, position):
    """Non-finite survivors require validation in the selected solver path."""
    covariance = pd.DataFrame([[.04, 0.], [0., .09]], index=['A', 'B'], columns=['A', 'B'])
    covariance.iloc[position] = np.inf
    filtered, _ = examples['op'].filter_covar_and_vectors_for_nans(covariance)
    assert filtered.shape == (2, 2)
    assert np.isinf(filtered.iloc[position])


def test_ewma_state_reset_against_three_explicit_updates(examples):
    """Missing updates reset covariance entries; zero-return input would instead decay them."""
    scale = Fraction(1, 10000)
    expected = np.array([
        [[Fraction(1, 2), 1], [1, 2]],
        [[0, 0], [0, Fraction(11, 2)]],
        [[Fraction(1, 2), 2], [2, Fraction(43, 4)]],
    ], dtype=float) * float(scale)
    np.testing.assert_allclose(examples['covariance_states'], expected, atol=1e-18)
    assert float(Fraction(1, 2) * Fraction(1, 2) * scale) == .000025
