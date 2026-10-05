"""Numerical contracts for equal-size and instrument-dispersion diagnostics."""
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.matf.opportunity import OpportunityModel, alpha_dispersion
from optimalportfolios.matf.universe_diagnostics import (
    dispersion_concentration, removal_diagnostics, sample_universe_dispersion,
)


def example(alpha=None):
    """A four-asset universe with constant factor exposure and equal residual risk."""
    index = pd.Index(['a', 'b', 'c', 'd'])
    return OpportunityModel(
        pd.DataFrame({'factor': [1., 1., 1., 1.]}, index=index),
        pd.Series(1., index=index),
        pd.DataFrame([[.04]], index=['factor'], columns=['factor']),
        pd.Series([-1., -1., 1., 1.] if alpha is None else alpha, index=index))


def test_sample_noise_adjustment_uses_subset_projection_and_signed_values():
    """The equal-risk demeaned three-asset metric has trace two, not three."""
    m = example()
    covariance = pd.DataFrame(4*np.eye(4), index=m.alpha.index, columns=m.alpha.index)
    draws, members = sample_universe_dispersion(m, [3, 4], repetitions=5,
                                                alpha_covariance=covariance)
    np.testing.assert_allclose(draws.K_noise, 4*(draws.sample_size-1))
    np.testing.assert_allclose(draws.K_noise_adjusted, draws.K-draws.K_noise)
    assert (draws.K_noise_adjusted < 0).all()
    assert draws.K_adjusted_rms.isna().all()
    assert len(members) == 19
    with pytest.raises(ValueError, match='order'):
        sample_universe_dispersion(m, [3], alpha_covariance=covariance.iloc[::-1])


def test_equal_contributors_and_zero_dispersion():
    """Equal K contributions imply exactly four contributors; zero K has no shares."""
    summary, detail = dispersion_concentration(example())
    k = summary.loc['K']
    assert k.effective_contributors == pytest.approx(4.)
    assert k.top1_share == pytest.approx(.25)
    assert k.top5_share == pytest.approx(1.)
    assert detail.loc[detail.metric.eq('K'), 'share'].sum() == pytest.approx(1.)
    summary, detail = dispersion_concentration(example([.02] * 4))
    assert summary.loc['K', 'status'] == 'numerically_zero'
    assert pd.isna(summary.loc['K', 'effective_contributors'])
    assert detail.loc[detail.metric.eq('K'), 'share'].isna().all()


def test_subset_noise_matches_whitened_svd_with_correlated_errors():
    """A second covariance/projection path detects ignored cross-asset alpha errors."""
    rng = np.random.default_rng(771)
    ids = pd.Index(list('abcde'))
    b = pd.DataFrame(rng.normal(size=(5, 1)), index=ids, columns=['factor'])
    d = pd.Series([.2, .7, .9, 1.1, 1.3], index=ids)
    m = OpportunityModel(b, d, pd.DataFrame([[.04]], index=b.columns, columns=b.columns),
                         pd.Series(rng.normal(size=5), index=ids))
    root = rng.normal(size=(5, 3))
    covariance = pd.DataFrame(root @ root.T, index=ids, columns=ids)
    draws, members = sample_universe_dispersion(m, [4], repetitions=7, alpha_covariance=covariance)
    for row in draws.itertuples():
        chosen = pd.Index(members.loc[members.sample_id.eq(row.sample_id), 'asset'])
        whitening = np.diag(1/np.sqrt(d.loc[chosen]))
        design = whitening @ np.column_stack([np.ones(4), b.loc[chosen]])
        left, _, _ = np.linalg.svd(design, full_matrices=True)
        metric = whitening @ left[:, 2:] @ left[:, 2:].T @ whitening
        assert row.K_noise == pytest.approx(np.trace(metric @ covariance.loc[chosen, chosen]))


def test_removal_reprojects_instead_of_subtracting_contribution():
    """Removing an observation changes centering: true loss is 4/3, not one."""
    table = removal_diagnostics(example())
    np.testing.assert_allclose(table.K_loss, 4 / 3, atol=1e-12)
    np.testing.assert_allclose(table.K_direct_contribution, 1., atol=1e-12)
    assert (table.rank_after == table.rank_before).all()
    assert (table.K_loss > table.K_direct_contribution).all()


def test_removal_handles_loss_of_factor_rank():
    """A unique loading can vanish without destroying existing neutral dispersion."""
    m = example()
    b = m.betas.copy()
    b['factor'] = [0., 0., 0., 1.]
    m = OpportunityModel(b, m.residual_variances, m.factor_covariance,
                         pd.Series([-1., 0., 1., 5.], index=b.index))
    table = removal_diagnostics(m)
    assert table.loc['d', 'rank_after'] < table.loc['d', 'rank_before']
    assert table.loc['d', 'K_loss'] == pytest.approx(0., abs=1e-12)


def test_subset_schedule_is_repeatable_and_has_no_duplicate_members():
    """Exact-size samples repeat with seed and preserve the full-set endpoint."""
    m = example()
    first, members = sample_universe_dispersion(m, [2, 3, 4], repetitions=12, seed=19)
    again, other_members = sample_universe_dispersion(m, [2, 3, 4], repetitions=12, seed=19)
    pd.testing.assert_frame_equal(first, again, check_exact=True)
    pd.testing.assert_frame_equal(members, other_members, check_exact=True)
    assert len(first.loc[first.sample_size.eq(4)]) == 1
    assert first.loc[first.sample_size.eq(4), 'K'].iloc[0] == pytest.approx(
        alpha_dispersion(m)[0]['K'])
    for sample_id, group in members.groupby('sample_id'):
        size = first.set_index('sample_id').loc[sample_id, 'sample_size']
        assert len(group) == size
        assert group.asset.nunique() == size
    with pytest.raises(ValueError, match='size'):
        sample_universe_dispersion(m, [5], repetitions=2, seed=19)


def test_sampled_K_agrees_with_independent_whitened_svd():
    """Audit a nonconstant multi-factor example without using canonical GLS."""
    rng = np.random.default_rng(48)
    index = pd.Index([f'asset{i}' for i in range(9)])
    factors = pd.Index(['f1', 'f2'])
    b = pd.DataFrame(rng.normal(size=(9, 2)), index=index, columns=factors)
    d = pd.Series(rng.uniform(.01, .09, 9), index=index)
    m = OpportunityModel(b, d, pd.DataFrame(np.eye(2), index=factors, columns=factors),
                         pd.Series(rng.normal(size=9), index=index))
    draws, members = sample_universe_dispersion(m, [6], repetitions=3, seed=9)
    for row in draws.itertuples():
        ids = members.loc[members.sample_id.eq(row.sample_id), 'asset'].tolist()
        root = np.sqrt(d.loc[ids].to_numpy())
        c = np.column_stack([np.ones(len(ids)), b.loc[ids]]) / root[:, None]
        u, s, _ = np.linalg.svd(c, full_matrices=False)
        q = u[:, s > s[0] * 1e-12]
        y = m.alpha.loc[ids].to_numpy() / root
        residual = y - q @ (q.T @ y)
        assert row.K == pytest.approx(residual @ residual, abs=1e-10)


def test_named_changes_reproject_and_retain_signed_noise_adjustment():
    """A four-asset centered sum of squares has K=4 versus 8/3 on three assets."""
    from optimalportfolios.matf.universe_diagnostics import compare_universe_dispersion
    model = example()
    ids = model.alpha.index
    covariance = pd.DataFrame(4*np.eye(4), index=ids, columns=ids)
    table, membership = compare_universe_dispersion(model, ids[:3], {'add_d': ids},
                                                    alpha_covariance=covariance)
    assert table.loc['baseline', 'K'] == pytest.approx(8/3)
    assert table.loc['add_d', 'delta_K'] == pytest.approx(4/3)
    assert table.loc['add_d', 'delta_K_noise'] == pytest.approx(4.)
    assert table.loc['add_d', 'delta_K_noise_adjusted'] == pytest.approx(-8/3)
    assert table.loc['add_d', 'delta_K_per_asset'] == pytest.approx(1/9)
    assert table.K_adjusted_rms.isna().all()
    assert membership.groupby('scenario').size().to_dict() == {'add_d': 4, 'baseline': 3}
    with pytest.raises(ValueError, match='baseline'):
        compare_universe_dispersion(model, ids[:3], {'baseline': ids})
    with pytest.raises(ValueError, match='unique'):
        compare_universe_dispersion(model, ids[:3], {'duplicate': ['a', 'a']})


def test_named_changes_allow_factor_rank_loss_without_spurious_gain():
    """A unique factor loading disappears; the remaining neutral variation is unchanged."""
    from optimalportfolios.matf.universe_diagnostics import compare_universe_dispersion
    model = example()
    b = model.betas.copy()
    b['factor'] = [0., 0., 0., 1.]
    model = OpportunityModel(b, model.residual_variances, model.factor_covariance,
                             pd.Series([-1., 0., 1., 5.], index=b.index))
    table, _ = compare_universe_dispersion(model, b.index, {'remove_d': b.index[:3]})
    assert table.loc['remove_d', 'delta_K'] == pytest.approx(0., abs=1e-12)
    assert table.loc['remove_d', 'delta_rank_funding_B'] == -1
    assert table.loc['remove_d', 'delta_N'] == -1
