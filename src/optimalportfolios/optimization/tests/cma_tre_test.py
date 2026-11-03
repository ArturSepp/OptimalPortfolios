"""Independent closed-form and rejection checks for the experimental CMA layer."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization._cma_tre import solve_cma_tre


def inputs():
    """Two independent assets with a sparse policy benchmark."""
    index = pd.Index(['AGG', 'OTHER'])
    return (pd.Series([.01, .08], index=index),
            pd.DataFrame(np.eye(2)*.04, index=index, columns=index),
            pd.Series([1., 0.], index=index))


@pytest.mark.parametrize('tre', [.01, .02, .04, .05])
def test_off_benchmark_asset_matches_two_asset_closed_form(tre):
    """Budget and independence give risk sqrt(.08)*w_OTHER, not a zero weight."""
    means, covariance, benchmark = inputs()
    weights, audit = solve_cma_tre(means, covariance, benchmark, tre, pd.Timestamp('2020-12-31'))
    np.testing.assert_allclose(weights, [1-tre/np.sqrt(.08), tre/np.sqrt(.08)], atol=1e-6)
    assert audit['tracking_error'] == pytest.approx(tre, abs=1e-6)
    assert audit['expected_active_return_pa'] > 0


@pytest.mark.parametrize('defect', ['sum', 'nan', 'labels', 'covariance', 'tre'])
def test_invalid_inputs_never_normalize_fill_or_realign(defect):
    """Invalid user declarations must fail before optimisation."""
    means, covariance, benchmark = inputs()
    tre = .02
    if defect == 'sum':
        benchmark.iloc[0] = 1.2
    elif defect == 'nan':
        means.iloc[1] = np.nan
    elif defect == 'labels':
        benchmark = benchmark.iloc[::-1]
    elif defect == 'covariance':
        covariance.iloc[0, 1] = .1
    else:
        tre = 0.
    with pytest.raises(ValueError):
        solve_cma_tre(means, covariance, benchmark, tre, pd.Timestamp('2020-12-31'))


def test_fallback_and_inaccurate_weights_are_rejected():
    """Finite fallback weights cannot enter the research performance sample."""
    means, covariance, benchmark = inputs()
    outcome = SimpleNamespace(accepted=True, status='optimal_inaccurate',
                              fallback_source=None, reason='inaccurate')
    with patch('optimalportfolios.optimization._cma_tre.wrapper_maximise_alpha_over_tre',
               return_value=(benchmark, outcome)):
        with pytest.raises(RuntimeError, match='Uncertified'):
            solve_cma_tre(means, covariance, benchmark, .02, pd.Timestamp('2020-12-31'))


def bounded_inputs():
    """Twenty explicit assets with a policy reference outside the name cap."""
    index = pd.Index([f'E{i}' for i in range(8)]+[f'B{i}' for i in range(12)])
    means = pd.Series(np.r_[np.linspace(.30, .23, 8), np.linspace(.12, .01, 12)],
                      index=index)
    covariance = pd.DataFrame(np.eye(20)*.0001, index=index, columns=index)
    benchmark = pd.Series(0., index=index)
    benchmark.loc[['E0', 'B0']] = [.2, .8]
    groups = pd.DataFrame({'Equity': [1.]*8+[0.]*12,
                           'Fixed Income': [0.]*8+[1.]*12}, index=index)
    limits = dict(max_weights=pd.Series(.1, index=index), group_loadings=groups,
                  group_min=pd.Series([.05, .65], index=groups.columns),
                  group_max=pd.Series([.35, .95], index=groups.columns))
    return means, covariance, benchmark, limits


@pytest.mark.parametrize('prefer_bonds', [False, True])
def test_name_caps_and_class_bands_match_independent_linear_solution(prefer_bonds):
    """Wide TRE leaves a unique linear optimum with binding name/class limits."""
    means, covariance, benchmark, limits = bounded_inputs()
    expected = pd.Series(0., index=means.index)
    if prefer_bonds:
        means.loc['B0':] += 1.
        expected.loc['E0'] = .05
        expected.loc[[f'B{i}' for i in range(9)]] = .1
        expected.loc['B9'] = .05
    else:
        expected.loc[['E0', 'E1', 'E2']] = .1
        expected.loc['E3'] = .05
        expected.loc[[f'B{i}' for i in range(6)]] = .1
        expected.loc['B6'] = .05
    actual, audit = solve_cma_tre(means, covariance, benchmark, .10,
                                  pd.Timestamp('2020-12-31'), **limits)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert actual.max() <= .1+2e-6
    assert audit['max_group_violation'] <= 2e-6
    assert audit['group_allocations']['Equity'] == pytest.approx(
        .05 if prefer_bonds else .35, abs=2e-6)


@pytest.mark.parametrize('defect', ['cap_labels', 'cap_nan', 'cap_negative',
                                   'missing_block', 'group_labels', 'empty_group',
                                   'bound_labels', 'bound_order', 'group_nan'])
def test_explicit_allocation_inputs_reject_incomplete_or_misaligned_data(defect):
    """No alignment, zero-column dropping or skipped group bounds is allowed."""
    means, covariance, benchmark, limits = bounded_inputs()
    if defect == 'cap_labels':
        limits['max_weights'] = limits['max_weights'].iloc[::-1]
    elif defect == 'cap_nan':
        limits['max_weights'].iloc[0] = np.nan
    elif defect == 'cap_negative':
        limits['max_weights'].iloc[0] = -.1
    elif defect == 'missing_block':
        del limits['group_max']
    elif defect == 'group_labels':
        limits['group_loadings'] = limits['group_loadings'].iloc[::-1]
    elif defect == 'empty_group':
        limits['group_loadings']['Equity'] = 0.
    elif defect == 'bound_labels':
        limits['group_min'] = limits['group_min'].iloc[::-1]
    elif defect == 'bound_order':
        limits['group_min'].iloc[0] = .5
    else:
        limits['group_loadings'].iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        solve_cma_tre(means, covariance, benchmark, .1, pd.Timestamp('2020-12-31'),
                      **limits)


def test_postsolve_audit_rejects_silent_name_or_group_violations():
    """An optimal status cannot conceal violated allocation limits."""
    means, covariance, benchmark, limits = bounded_inputs()
    for bad_weights in (benchmark, pd.Series(.05, index=means.index)):
        outcome = SimpleNamespace(accepted=True, status='optimal', fallback_source=None,
                                  reason='mock', weights=bad_weights,
                                  covar_factorization=None, solver='CLARABEL')
        with patch('optimalportfolios.optimization._cma_tre.wrapper_maximise_alpha_over_tre',
                   return_value=(bad_weights, outcome)):
            with pytest.raises(RuntimeError, match='violates'):
                solve_cma_tre(means, covariance, benchmark, .1,
                              pd.Timestamp('2020-12-31'), **limits)
