"""Contracts for reusable benchmark-beta loadings and ex-ante beta analytics."""

import numpy as np
import pandas as pd
import pytest
import qis

import optimalportfolios as op
import optimalportfolios.utils as utils_api
from optimalportfolios.optimization import constraints
from optimalportfolios.utils import benchmark_beta


CONSTITUENTS = ['Equity', 'Bonds']
BENCHMARK_WEIGHTS = pd.Series([0.6, 0.4], index=CONSTITUENTS)
DATES = pd.date_range('2020-03-31', periods=4, freq='QE')


def _make_covar_dict(covar: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Repeat one covariance at every test rebalancing date."""
    return {date: covar for date in DATES}


def _make_benchmark_covar() -> pd.DataFrame:
    """Return a positive-definite covariance for the benchmark constituents."""
    vols = np.array([0.18, 0.06])
    corr = np.array([[1.0, 0.15], [0.15, 1.0]])
    return pd.DataFrame(
        np.outer(vols, vols) * corr,
        index=CONSTITUENTS,
        columns=CONSTITUENTS,
    )


def test_benchmark_has_beta_one_against_itself() -> None:
    """The benchmark composition has beta one against itself at every date."""
    loadings = benchmark_beta.compute_benchmark_beta_loadings_ts(
        covar_dict=_make_covar_dict(_make_benchmark_covar()),
        benchmark_weights=BENCHMARK_WEIGHTS,
        asset_tickers=CONSTITUENTS,
    )
    weights = pd.DataFrame(
        [BENCHMARK_WEIGHTS.to_numpy()] * len(DATES),
        index=DATES,
        columns=CONSTITUENTS,
    )

    beta = benchmark_beta.compute_ex_ante_beta_ts(
        weights=weights, beta_loadings=loadings)

    np.testing.assert_allclose(beta.to_numpy(), 1.0, rtol=1e-12, atol=1e-15)
    assert beta.name == 'ex_ante_beta'


def test_ex_ante_beta_is_linear_in_weights() -> None:
    """Scaling every risky position scales ex-ante beta by the same amount."""
    loadings = benchmark_beta.compute_benchmark_beta_loadings_ts(
        covar_dict=_make_covar_dict(_make_benchmark_covar()),
        benchmark_weights=BENCHMARK_WEIGHTS,
        asset_tickers=CONSTITUENTS,
    )
    weights = pd.DataFrame(
        [BENCHMARK_WEIGHTS.to_numpy()] * len(DATES),
        index=DATES,
        columns=CONSTITUENTS,
    )

    base = benchmark_beta.compute_ex_ante_beta_ts(weights, loadings)
    levered = benchmark_beta.compute_ex_ante_beta_ts(2.5 * weights, loadings)
    cash = benchmark_beta.compute_ex_ante_beta_ts(0.0 * weights, loadings)

    np.testing.assert_allclose(levered, 2.5 * base, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(cash, 0.0, atol=1e-15)


def test_loadings_dates_are_sorted_and_forward_filled_to_weight_dates() -> None:
    """Loadings sort covariance dates and use the latest prior row for each weight date."""
    first = pd.Timestamp('2024-01-31')
    second = pd.Timestamp('2024-03-31')
    covar = _make_benchmark_covar()
    loadings = benchmark_beta.compute_benchmark_beta_loadings_ts(
        covar_dict={second: 2.0 * covar, first: covar},
        benchmark_weights=BENCHMARK_WEIGHTS,
        asset_tickers=CONSTITUENTS,
    )
    assert loadings.index.tolist() == [first, second]

    weight_dates = pd.DatetimeIndex(
        [first, pd.Timestamp('2024-02-29'), second, pd.Timestamp('2024-04-30')])
    weights = pd.DataFrame(
        [[1.0, 0.0]] * len(weight_dates),
        index=weight_dates,
        columns=CONSTITUENTS,
    )
    beta = benchmark_beta.compute_ex_ante_beta_ts(weights, loadings)

    expected = pd.Series(
        [loadings.loc[first, 'Equity'], loadings.loc[first, 'Equity'],
         loadings.loc[second, 'Equity'], loadings.loc[second, 'Equity']],
        index=weight_dates,
        name='ex_ante_beta',
    )
    pd.testing.assert_series_equal(beta, expected)


def test_weight_dates_before_first_loading_retain_zero_beta() -> None:
    """A leading weight date retains the legacy zero result before loadings exist."""
    loading_date = pd.Timestamp('2024-01-31')
    weight_dates = pd.DatetimeIndex([pd.Timestamp('2023-12-31'), loading_date])
    loadings = pd.DataFrame(
        [[0.8, 1.2]], index=[loading_date], columns=CONSTITUENTS)
    weights = pd.DataFrame(
        [[0.6, 0.4], [0.6, 0.4]], index=weight_dates, columns=CONSTITUENTS)

    beta = benchmark_beta.compute_ex_ante_beta_ts(weights, loadings)

    expected = pd.Series([0.0, 0.96], index=weight_dates, name='ex_ante_beta')
    pd.testing.assert_series_equal(beta, expected)


def test_loadings_missing_a_weight_column_raise() -> None:
    """Partial loading coverage must not silently bias reported beta toward zero."""
    loadings = pd.DataFrame(
        [[0.8, 1.2]], index=[DATES[0]], columns=CONSTITUENTS)
    weights = pd.DataFrame(
        [[0.6, 0.4]], index=[DATES[0]], columns=['Equity', 'Credit'])

    with pytest.raises(ValueError, match='do not cover 1 weight columns'):
        benchmark_beta.compute_ex_ante_beta_ts(weights, loadings)


def test_missing_benchmark_constituents_raise() -> None:
    """Every constituent must be represented in the joint covariance."""
    covar = pd.DataFrame([[0.04]], index=['Equity'], columns=['Equity'])

    with pytest.raises(KeyError, match='benchmark constituents missing'):
        benchmark_beta.compute_benchmark_beta_loadings_from_covar(
            covar=covar,
            benchmark_weights=BENCHMARK_WEIGHTS,
            asset_tickers=['Equity'],
        )


@pytest.mark.parametrize('benchmark_variance', [0.0, -0.01])
def test_nonpositive_benchmark_variance_raises(benchmark_variance: float) -> None:
    """A zero or negative benchmark variance cannot define beta loadings."""
    covar = pd.DataFrame(
        [[0.04, 0.0], [0.0, benchmark_variance]],
        index=['Asset', 'Benchmark'],
        columns=['Asset', 'Benchmark'],
    )

    with pytest.raises(ValueError, match='benchmark variance must be positive'):
        benchmark_beta.compute_benchmark_beta_loadings_from_covar(
            covar=covar,
            benchmark_weights=pd.Series([1.0], index=['Benchmark']),
            asset_tickers=['Asset'],
        )


@pytest.mark.parametrize('benchmark_variance', [np.nan, np.inf])
def test_nonfinite_benchmark_variance_raises(benchmark_variance: float) -> None:
    """A non-finite benchmark variance cannot define beta loadings."""
    covar = pd.DataFrame(
        [[0.04, 0.0], [0.0, benchmark_variance]],
        index=['Asset', 'Benchmark'],
        columns=['Asset', 'Benchmark'],
    )

    with pytest.raises(ValueError, match='benchmark variance must be finite and positive'):
        benchmark_beta.compute_benchmark_beta_loadings_from_covar(
            covar=covar,
            benchmark_weights=pd.Series([1.0], index=['Benchmark']),
            asset_tickers=['Asset'],
        )


def test_nonfinite_covariance_loading_raises() -> None:
    """A finite denominator must not conceal a non-finite asset cross-covariance."""
    covar = pd.DataFrame(
        [[0.04, np.nan], [np.nan, 0.09]],
        index=['Asset', 'Benchmark'],
        columns=['Asset', 'Benchmark'],
    )

    with pytest.raises(ValueError, match='benchmark beta loadings must be finite'):
        benchmark_beta.compute_benchmark_beta_loadings_from_covar(
            covar=covar,
            benchmark_weights=pd.Series([1.0], index=['Benchmark']),
            asset_tickers=['Asset'],
        )


def test_nonfinite_loading_panel_raises_instead_of_understating_beta() -> None:
    """NaN loadings must not disappear through pandas' skip-NaN row sum."""
    loadings = pd.DataFrame(
        [[0.8, np.nan]], index=[DATES[0]], columns=CONSTITUENTS)
    weights = pd.DataFrame(
        [[0.6, 0.4]], index=[DATES[0]], columns=CONSTITUENTS)

    with pytest.raises(ValueError, match='beta_loadings must contain only finite values'):
        benchmark_beta.compute_ex_ante_beta_ts(weights, loadings)


def test_joint_covariance_loadings_and_beta_match_qis_risk_model() -> None:
    """Match the canonical QIS loadings and portfolio beta on a seeded covariance."""
    rng = np.random.default_rng(20260824)
    universe = pd.Index([f'asset_{idx}' for idx in range(8)])
    root = rng.normal(scale=0.15, size=(len(universe), len(universe)))
    values = root @ root.T + np.diag(rng.uniform(0.002, 0.006, len(universe)))
    covar = pd.DataFrame(values, index=universe, columns=universe)
    benchmark_weights = pd.Series(
        [0.55, 0.30, 0.15], index=['asset_1', 'asset_4', 'asset_6'])
    asset_tickers = ['asset_0', 'asset_2', 'asset_3', 'asset_5', 'asset_7']
    portfolio_weights = pd.Series(
        [0.80, -0.25, 0.35, -0.15, 0.25], index=asset_tickers)
    date = pd.Timestamp('2026-06-30')

    actual = benchmark_beta.compute_benchmark_beta_loadings_from_covar(
        covar=covar,
        benchmark_weights=benchmark_weights,
        asset_tickers=asset_tickers,
    )
    full_benchmark = pd.Series(0.0, index=universe)
    full_benchmark.loc[benchmark_weights.index] = benchmark_weights
    full_portfolio = pd.Series(0.0, index=universe)
    full_portfolio.loc[portfolio_weights.index] = portfolio_weights
    risk_model = qis.RiskModel(covar={date: covar})
    expected = risk_model.compute_benchmark_beta_loadings_at_date(
        benchmark_weights=full_benchmark, date=date).loc[asset_tickers]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(
        float(actual @ portfolio_weights),
        risk_model.compute_benchmark_beta_at_date(
            benchmark_weights=full_benchmark,
            portfolio_weights=full_portfolio,
            date=date,
        ),
        rtol=1e-12,
        atol=1e-16,
    )


@pytest.mark.parametrize(
    'name',
    [
        'compute_benchmark_beta_loadings',
        'compute_benchmark_beta_loadings_from_covar',
        'compute_benchmark_beta_loadings_ts',
        'compute_ex_ante_beta_ts',
    ],
)
def test_utility_exports_preserve_function_identity(name: str) -> None:
    """The module, utility namespace, and package root expose one function object."""
    expected = getattr(benchmark_beta, name)
    assert getattr(utils_api, name) is expected
    assert getattr(op, name) is expected


@pytest.mark.parametrize(
    'name',
    ['compute_benchmark_beta_loadings', 'compute_benchmark_beta_loadings_from_covar'],
)
def test_legacy_constraint_exports_preserve_function_identity(name: str) -> None:
    """Historical constraints imports remain identity aliases after relocation."""
    assert getattr(constraints, name) is getattr(benchmark_beta, name)
