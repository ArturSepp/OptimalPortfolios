"""Calibration provenance, paired removal and fixed-book interval contracts."""
import numpy as np
import pandas as pd
import pytest
from factorlasso import linear_confidence_intervals

from optimalportfolios.matf import uncertainty as u
from optimalportfolios.matf.tests.uncertainty_test import model


@pytest.fixture
def inputs():
    """Return a frozen model, covariance and externally calibrated alpha interval table."""
    m = model()
    covariance = pd.DataFrame(np.eye(4) * .0001, index=m.alpha.index, columns=m.alpha.index)
    table = pd.DataFrame(linear_confidence_intervals(m.alpha, covariance), index=m.alpha.index)
    return m, covariance, table


@pytest.mark.parametrize('field,value', [
    ('estimate', .4), ('standard_error', .2), ('standard_error', -.01), ('confidence', .8),
])
def test_supplied_calibration_must_match_the_reported_estimates(inputs, field, value):
    """Intervals for another point estimate, covariance or confidence cannot be reused."""
    m, covariance, table = inputs
    table.loc['a', field] = value
    with pytest.raises(ValueError, match='current estimates/covariance'):
        u.alpha_uncertainty_metrics(m, covariance, draws=2, intervals={'alpha': table})


@pytest.mark.parametrize('lower,upper', [(np.nan, .1), (-.1, np.nan), (-np.inf, .1), (.2, .1)])
def test_supplied_calibration_requires_available_ordered_finite_pairs(inputs, lower, upper):
    """Half-missing, infinite or inverted endpoints cannot be presented as valid bounds."""
    m, covariance, table = inputs
    table.loc['a', ['lower', 'upper']] = [lower, upper]
    with pytest.raises(ValueError, match='ordered finite pairs'):
        u.alpha_uncertainty_metrics(m, covariance, draws=2, intervals={'alpha': table})


def test_pointwise_intervals_cannot_claim_simultaneous_scope(inputs):
    """A family-wise report requires family-calibrated provenance for every projected asset."""
    m, covariance, _ = inputs
    result = u.alpha_uncertainty_metrics(m, covariance, draws=2)
    table = pd.DataFrame(linear_confidence_intervals(result['contributors'].h_joint,
                         result['projected_covariance']), index=m.alpha.index)
    with pytest.raises(ValueError, match='family-calibrated'):
        u.alpha_uncertainty_metrics(m, covariance, draws=2,
                                   intervals={'projected_simultaneous': table})


@pytest.mark.parametrize('intervals', [['alpha'], {'unknown_family': pd.DataFrame()}])
def test_interval_mapping_requires_registered_target_families(inputs, intervals):
    """Unregistered families cannot be ignored while the report claims calibration."""
    m, covariance, _ = inputs
    with pytest.raises(ValueError, match='named alpha/projected target families'):
        u.alpha_uncertainty_metrics(m, covariance, draws=2, intervals=intervals)


@pytest.mark.parametrize('scenarios', [np.zeros((1, 4)), np.full((2, 4), np.nan)])
def test_scenario_inputs_preserve_the_declared_draw_count_and_asset_axis(inputs, scenarios):
    """Wrong-size or unavailable paired scenarios must fail rather than broadcast."""
    m, covariance, _ = inputs
    with pytest.raises(ValueError, match='finite with shape'):
        u.alpha_uncertainty_metrics(m, covariance, draws=2, scenarios=scenarios)


def test_dispersion_reconciliation_detects_an_inconsistent_calibration(inputs, monkeypatch):
    """A mismatched quadratic point estimate cannot silently change canonical capacity."""
    m, covariance, _ = inputs
    original = u.quadratic_confidence_summary
    def inconsistent(*args, **kwargs):
        """Retain a real calibrated summary but corrupt its independently checked point."""
        result = original(*args, **kwargs)
        result['observed'] += 1.
        return result
    monkeypatch.setattr(u, 'quadratic_confidence_summary', inconsistent)
    with pytest.raises(ValueError, match='dispersion reconciliation'):
        u.alpha_uncertainty_metrics(m, covariance, draws=2)


@pytest.mark.parametrize('scenarios', [np.zeros(4), np.zeros((2, 3)), np.full((2, 4), np.nan)])
def test_removal_scenarios_require_the_complete_asset_axis(inputs, scenarios):
    """Removal uncertainty retains paired shocks to every original instrument."""
    m, covariance, _ = inputs
    with pytest.raises(ValueError, match='complete asset axis'):
        u.removal_uncertainty(m, covariance, scenarios)


@pytest.mark.parametrize('members', [[], ['a', 'a'], ['unknown'], ['a', 'b', 'c', 'd']])
def test_removal_uncertainty_rejects_invalid_or_complete_bundles(inputs, members):
    """A removal must identify unique fitted assets and leave a nonempty model."""
    m, covariance, _ = inputs
    with pytest.raises(ValueError):
        u.removal_uncertainty(m, covariance, np.tile(m.alpha, (2, 1)),
                              bundles={'invalid': members})


@pytest.mark.parametrize('bad_book', [[-.1, .3, .4, .4], [.1, .2, .3, .3]])
@pytest.mark.parametrize('field', ['weights', 'reference'])
def test_fixed_book_intervals_require_long_only_unit_funded_books(inputs, bad_book, field):
    """Invalid actual funding cannot be silently normalized inside an alpha interval."""
    m, covariance, _ = inputs
    books = dict(weights=pd.Series(.25, index=m.alpha.index),
                 reference=pd.Series(.25, index=m.alpha.index))
    books[field] = pd.Series(bad_book, index=m.alpha.index)
    with pytest.raises(ValueError, match='long-only and unit funded'):
        u.fixed_portfolio_alpha_interval(m, covariance, **books)


def test_projected_fixed_book_interval_matches_independent_linear_propagation(inputs):
    """An ordinary joint projector gives the active estimate and correlated error variance."""
    m, covariance, _ = inputs
    reference = pd.Series(.25, index=m.alpha.index)
    weights = pd.Series([.1, .2, .4, .3], index=m.alpha.index)
    design = np.column_stack([np.ones(4), m.betas])
    projector = np.eye(4) - design @ np.linalg.pinv(design)
    loading = projector.T @ (weights - reference).to_numpy()
    result = u.fixed_portfolio_alpha_interval(m, covariance, weights, reference, projected=True)
    assert result['estimate'] == pytest.approx(loading @ m.alpha, abs=1e-14)
    assert result['standard_error'] == pytest.approx(
        np.sqrt(loading @ covariance.to_numpy() @ loading), abs=1e-14)
    z = 1.959963984540054
    assert result['lower'] == pytest.approx(result['estimate'] - z * result['standard_error'])
    assert result['upper'] == pytest.approx(result['estimate'] + z * result['standard_error'])
