"""Execute covariance-estimator documentation and verify units, decomposition and timing."""

from dataclasses import replace
from fractions import Fraction
import re
import runpy

import numpy as np
import pandas as pd
import pytest

LEGACY_ANCHORS = {
    'covariance-estimators', 'estimator-choice', 'ewma-covariance',
    'factor-and-hcgl-covariance', 'units-and-validation', 'see-also',
}


@pytest.fixture(scope='module')
def article(root):
    """Load the canonical article through the checkout-only root fixture."""
    return (root / 'docs/covariance_estimators.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article):
    """Execute all six sequential blocks, including the unchanged original estimator example."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 6 and all(not options.strip() for options, _ in blocks)
    state = {'__name__': '__covariance_article__'}
    for index, (_, code) in enumerate(blocks, start=1):
        exec(compile(code, f'covariance_estimators.md (block {index})', 'exec'), state)
    return state


# The QIS estimator is under test; this reference uses pandas mean adjustment and a finite sum.
def _weighted_reference(prices, span, annualization, demean):
    """Compute the final covariance independently of QIS return and covariance helpers."""
    returns = pd.DataFrame(np.diff(np.log(prices.to_numpy()), axis=0))
    if demean:
        returns = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1 - 2 / (span + 1)
    weights = (1 - decay) * decay ** np.arange(len(returns) - 1, -1, -1)
    return annualization * np.einsum('t,ti,tj->ij', weights, returns, returns)


def _fresh_factor(examples, **overrides):
    """Construct a factor estimator with the article configuration and no prior fitted state."""
    configured = examples['factor_estimator']
    model = examples['LassoModel'](**configured.lasso_model.get_params())
    return replace(configured, lasso_model=model, **overrides)


def test_article_structure_links_and_legacy_fragments(article, root):
    """Preserve the methodology structure, required attribution and six old section anchors."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](article, root / 'docs/covariance_estimators.md', root)
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        match = re.match(r'^#{1,6} (.+)', line)
        if match:
            anchors.add(re.sub(r'[^\w -]', '', match[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors
    assert 'https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff' in article
    assert not (root / 'docs/covariance_estimators.rst').exists()


def test_original_ewma_matches_independent_demeaned_weighted_sum(examples):
    """The original weekly example uses contemporaneous means, zero covariance seed and 52/year."""
    expected = _weighted_reference(examples['prices'], span=52, annualization=52, demean=True)
    actual = examples['current_covar']
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
    assert actual.index.equals(actual.columns)
    assert actual.columns.tolist() == ['Equity', 'Bonds']


def test_three_monthly_observations_match_exact_fraction_reference(examples):
    """Three zero-seeded updates carry weights 1/8, 1/4, 1/2 and annualize by twelve."""
    observations = [
        [Fraction(1, 100), Fraction(2, 100)],
        [Fraction(-2, 100), Fraction(1, 100)],
        [Fraction(3, 100), Fraction(-1, 100)],
    ]
    weights = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)]
    expected = [[float(12 * sum(weight * row[i] * row[j]
                               for weight, row in zip(weights, observations)))
                 for j in range(2)] for i in range(2)]
    np.testing.assert_allclose(examples['small_covar'], expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        examples['small_returns'], np.array(observations, dtype=float), rtol=0, atol=1e-15)


def test_displayed_annual_covariance_table_matches_independent_numbers(article):
    """Quoted covariance entries must stay in annual fractional-return-squared units."""
    rows = re.findall(
        r'^\| ([AB]) \| (-?[0-9.]+) \| (-?[0-9.]+) \|$', article, re.M)
    assert [row[0] for row in rows] == ['A', 'B']
    actual = np.array([[float(x) for x in row[1:]] for row in rows])
    np.testing.assert_allclose(actual, [[.00675, -.0021], [-.0021, .0015]], atol=.5e-6, rtol=0)


@pytest.mark.parametrize('name,weight', [
    ('factor_covar', 1.0), ('factor_only', 0.0), ('scaled_residual_covar', 0.35),
])
def test_factor_assembly_against_scalar_component_reference(examples, name, weight):
    """Sum each asset/factor pair explicitly, then add the specified residual diagonal."""
    from factorlasso import VarianceColumns

    data = examples['factor_data']
    betas, factors = data.y_betas, data.x_covar
    residuals = data.y_variances[VarianceColumns.RESIDUAL_VARS.value]
    expected = np.empty((len(betas), len(betas)))
    for i, left in enumerate(betas.index):
        for j, right in enumerate(betas.index):
            expected[i, j] = sum(betas.at[left, f] * factors.at[f, g] * betas.at[right, g]
                                 for f in betas.columns for g in betas.columns)
        expected[i, i] += weight * residuals.loc[left]
    assert (residuals > 0).all()
    np.testing.assert_allclose(examples[name], expected, rtol=1e-10, atol=1e-14)
    assert examples[name].index.tolist() == ['Equity', 'Bonds', 'Balanced']


def test_factor_covariance_annualization_uses_its_monthly_cadence(examples):
    """The factor covariance equals a separate monthly demeaned EWMA sum multiplied by twelve."""
    data = examples['factor_data']
    prices = examples['factor_prices'].loc[:examples['as_of']]
    expected = _weighted_reference(prices, span=24, annualization=12, demean=True)
    np.testing.assert_allclose(data.x_covar, expected, rtol=1e-10, atol=1e-14)
    assert data.y_betas.shape == (3, 2)
    assert data.estimation_date == pd.Timestamp('2022-12-31')
    assert data.clusters.notna().all()
    assert all(value.startswith('ME:') for value in data.clusters)


def test_current_factor_inputs_are_explicitly_truncated(examples):
    """The canonical fit must agree with an independently invoked prefix-only current fit."""
    expected = _fresh_factor(examples).fit_current_covar(
        risk_factor_prices=examples['factor_prices'].loc[:examples['as_of']],
        asset_returns_dict={'ME': examples['asset_returns'].loc[:examples['as_of']]},
        estimation_date=examples['as_of'],
    )
    np.testing.assert_allclose(examples['factor_covar'], expected, rtol=1e-10, atol=1e-14)


def test_estimation_date_alone_does_not_truncate_ordinary_current_factor_fit(examples):
    """A metadata date on the complete history differs from the correctly truncated example."""
    whole = _fresh_factor(examples).fit_current_covar(
        risk_factor_prices=examples['factor_prices'],
        asset_returns_dict={'ME': examples['asset_returns']},
        estimation_date=examples['as_of'],
    )
    assert np.max(np.abs(whole - examples['factor_covar']).to_numpy()) > 1e-5


def test_top_level_factor_demean_does_not_toggle_internal_factor_demeaning(examples):
    """The stored field is currently not wired into the internal factor-covariance call."""
    changed = _fresh_factor(examples, demean=False).fit_current_factor_covars(
        risk_factor_prices=examples['factor_prices'].loc[:examples['as_of']],
        asset_returns_dict={'ME': examples['asset_returns'].loc[:examples['as_of']]},
        estimation_date=examples['as_of'],
    )
    np.testing.assert_allclose(changed.x_covar, examples['factor_data'].x_covar, atol=1e-14)
    np.testing.assert_allclose(changed.y_betas, examples['factor_data'].y_betas, atol=1e-12)


def test_supplied_factor_covariance_is_used_without_additional_annualization(examples):
    """A supplied annual matrix is not multiplied by twelve again."""
    supplied = examples['factor_data'].x_covar * 1.7
    data = _fresh_factor(examples).fit_current_factor_covars(
        risk_factor_prices=examples['factor_prices'].loc[:examples['as_of']],
        asset_returns_dict={'ME': examples['asset_returns'].loc[:examples['as_of']]},
        estimation_date=examples['as_of'], x_covar=supplied,
    )
    pd.testing.assert_frame_equal(data.x_covar, supplied)


def test_rolling_output_dates_match_each_estimator_grid(examples):
    """The direct estimator reports weekly observations while factor fits use calendar anchors."""
    assert list(examples['rolling_covars']) == list(pd.to_datetime(
        ['2022-04-06', '2022-07-06', '2022-10-05']))
    assert list(examples['rolling_factor_covars']) == list(pd.to_datetime(
        ['2022-12-31', '2023-03-31', '2023-06-30']))


def test_direct_rolling_ewma_is_unchanged_by_future_prices(examples):
    """Perturbing prices after the final reporting date cannot change ordinary earlier estimates."""
    cutoff = max(examples['rolling_covars'])
    future = examples['prices'].copy()
    mask = future.index > cutoff
    future.loc[mask, 'Equity'] *= np.linspace(1.1, 3.0, mask.sum())
    changed = examples['estimator'].fit_rolling_covars(future, examples['ewma_period'])
    for date, covariance in examples['rolling_covars'].items():
        pd.testing.assert_frame_equal(covariance, changed[date])
        prefix = examples['estimator'].fit_current_covar(examples['prices'].loc[:date])
        np.testing.assert_allclose(covariance, prefix, rtol=1e-10, atol=1e-14)


def test_normalized_rolling_ewma_is_unchanged_by_future_prices(examples):
    """QIS 5.31 seeds the normalized-return volatility with the first square, not the full array."""
    cutoff = max(examples['rolling_covars'])
    future = examples['prices'].copy()
    mask = future.index > cutoff
    future.loc[mask, 'Equity'] *= np.linspace(1.1, 3.0, mask.sum())
    estimator = replace(examples['estimator'], is_apply_vol_normalised_returns=True)
    before = estimator.fit_rolling_covars(examples['prices'], examples['ewma_period'])
    after = estimator.fit_rolling_covars(future, examples['ewma_period'])
    assert list(before) == list(after)
    for date, covariance in before.items():
        np.testing.assert_allclose(covariance, after[date], rtol=1e-10, atol=1e-14)


def test_rolling_factor_fit_is_unchanged_by_future_inputs(examples):
    """Future factor prices and asset observations cannot alter earlier rolling factor estimates."""
    cutoff = max(examples['rolling_factor_covars'])
    factors, returns = examples['factor_prices'].copy(), examples['asset_returns'].copy()
    factors.loc[factors.index > cutoff, 'Growth'] *= 2.0
    returns.loc[returns.index > cutoff, 'Equity'] += .10
    changed = _fresh_factor(examples).fit_rolling_covars(
        risk_factor_prices=factors, asset_returns_dict={'ME': returns},
        time_period=examples['factor_period'],
    )
    for date, covariance in examples['rolling_factor_covars'].items():
        np.testing.assert_allclose(covariance, changed[date], rtol=1e-10, atol=1e-14)


def test_factor_residual_panel_uses_annual_scaling_without_intercept_subtraction(examples):
    """Reported residuals are scaled deviations, not the raw observations used to estimate D."""
    data = examples['factor_data']
    factors = examples['factor_prices'].loc[:examples['as_of']]
    asset_returns = examples['asset_returns'].loc[:examples['as_of']]
    factor_returns = pd.DataFrame(np.diff(np.log(factors), axis=0),
                                 index=factors.index[1:], columns=factors.columns)
    expected = 12 * (asset_returns - factor_returns @ data.y_betas.T)
    # The first factor observation in the adapter has no preceding aligned factor price.
    pd.testing.assert_index_equal(data.residuals.index, expected.index)
    np.testing.assert_allclose(data.residuals.iloc[1:], expected.iloc[1:], atol=1e-14)
    assert data.residuals.iloc[0].isna().all()
