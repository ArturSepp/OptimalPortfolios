"""Canonical script of docs/covariance_estimators.md.

The page's six Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: an explicit EWMA weighted sum, exact fractions, a scalar sum over the factor
components, independent prefix-only refits and perturbations of later inputs. The script runs
offline after ``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.covariance_estimators
"""
from dataclasses import fields, replace
from fractions import Fraction
from importlib.metadata import version
import inspect
import re

import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, LassoModel, LassoModelType, VarianceColumns

import optimalportfolios as opt

ASSETS = ['Equity', 'Bonds', 'Balanced']
# The page's table of the three-observation example, in annual units.
SMALL_TABLE = [[0.006750, -0.002100], [-0.002100, 0.001500]]
EWMA_KEYS = ['2022-04-06', '2022-07-06', '2022-10-05']
FACTOR_KEYS = ['2022-12-31', '2023-03-31', '2023-06-30']
# Beta spans of the empirical-residual illustration: monthly and quarterly buckets.
SPANS = {'ME': 36, 'QE': 12}


def installed_version(package: str) -> tuple:
    """Return the first three numeric parts of an installed distribution's version."""
    return tuple(int(part) for part in re.findall(r'\d+', version(package))[:3])


def weighted_reference(prices: pd.DataFrame, span: int, annualization: float,
                       demean: bool) -> np.ndarray:
    """Final EWMA covariance as a finite weighted sum, without the qis return or EWMA helpers."""
    returns = pd.DataFrame(np.diff(np.log(prices.to_numpy()), axis=0))
    if demean:
        returns = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1 - 2 / (span + 1)
    weights = (1 - decay) * decay ** np.arange(len(returns) - 1, -1, -1)
    return annualization * np.einsum('t,ti,tj->ij', weights, returns, returns)


def component_reference(betas: pd.DataFrame, factor_covar: pd.DataFrame,
                        residual_vars: pd.Series, weight: float) -> np.ndarray:
    """Sum each asset and factor pair explicitly, then add the weighted residual diagonal."""
    expected = np.empty((len(betas), len(betas)))
    for i, left in enumerate(betas.index):
        for j, right in enumerate(betas.index):
            expected[i, j] = sum(betas.at[left, f] * factor_covar.at[f, g] * betas.at[right, g]
                                 for f in betas.columns for g in betas.columns)
        expected[i, i] += weight * residual_vars.loc[left]
    return expected


def off_diagonal(matrix: pd.DataFrame) -> np.ndarray:
    """Return the matrix with its diagonal set to zero."""
    values = np.array(matrix, dtype=float)
    np.fill_diagonal(values, 0.0)
    return values


def perturbed_after(prices: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Scale the Equity prices after ``cutoff`` by a factor rising from 1.1 to 3.0."""
    changed = prices.copy()
    later = changed.index > cutoff
    changed.loc[later, 'Equity'] *= np.linspace(1.1, 3.0, later.sum())
    return changed


def empirical_estimator() -> opt.FactorCovarEstimator:
    """A LASSO factor estimator preparing empirical residual correlation, with fresh state."""
    return opt.FactorCovarEstimator(
        lasso_model=LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5,
                               span_freq_dict=SPANS, warmup_period=8),
        factor_returns_freq='ME', factor_covar_span=24, residual_type='empirical',
    )


def check_empirical_residuals(factor_prices: pd.DataFrame, asset_returns: pd.DataFrame,
                              as_of: pd.Timestamp, period: qis.TimePeriod) -> None:
    """Assert the common-frequency empirical residual statements on monthly and quarterly data.

    The page's Balanced returns are summed into complete calendar quarters, so the universe has
    a monthly bucket with beta span 36 and a quarterly bucket with beta span 12.
    """
    buckets = {'ME': asset_returns[['Equity', 'Bonds']],
               'QE': asset_returns[['Balanced']].resample('QE').sum()}
    estimator = empirical_estimator()
    data = estimator.fit_current_factor_covars(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, estimation_date=as_of)
    prepared = data.residual_correlation
    # The default grid is the lowest native frequency with its beta span: QE and 12. The
    # estimate records its last complete period and its fit date.
    assert (prepared.frequency, prepared.span) == ('QE', 12)
    assert prepared.observation_date == prepared.estimation_date == as_of
    # The stored residual multiplier s_i equals the native observations per year A_i.
    metadata = prepared.asset_metadata.loc[ASSETS]
    assert metadata['residual_scale'].tolist() == metadata['annualisation_factor'].tolist()
    assert metadata['annualisation_factor'].tolist() == [12.0, 12.0, 4.0]
    # The fit leaves the final bucket's state on the supplied LassoModel.
    assert estimator.lasso_model.estimated_betas.index.tolist() == ['Balanced']
    # Empirical current fits truncate every input at estimation_date.
    sliced = empirical_estimator().fit_current_factor_covars(
        risk_factor_prices=factor_prices.loc[:as_of],
        asset_returns_dict={freq: returns.loc[:as_of] for freq, returns in buckets.items()},
        estimation_date=as_of)
    np.testing.assert_allclose(sliced.get_y_covar(residual_type='empirical'),
                               data.get_y_covar(residual_type='empirical'), rtol=1e-10, atol=1e-14)
    # Decomposition getters default to orthogonal residuals, while the shared estimator method
    # uses the configured empirical type.
    pd.testing.assert_frame_equal(data.get_y_covar(), data.get_y_covar(residual_type='orthogonal'))
    shared = empirical_estimator().fit_current_covar(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, estimation_date=as_of)
    np.testing.assert_allclose(shared, data.get_y_covar(residual_type='empirical'),
                               rtol=1e-10, atol=1e-14)
    # D = S[(1 - rho) I + rho C]S with S the current annual residual volatilities: the diagonal
    # never moves, rho = 0 is the orthogonal model, and rho = 0.5 keeps half the off-diagonal.
    vols = np.sqrt(data.y_variances[VarianceColumns.RESIDUAL_VARS.value].to_numpy())
    orthogonal = data.get_y_covar().to_numpy()
    for rho in (0.0, 0.5, 1.0):
        blend = (1 - rho) * np.eye(len(vols)) + rho * prepared.correlation.to_numpy()
        expected = orthogonal - np.diag(vols ** 2) + vols[:, None] * blend * vols[None, :]
        np.testing.assert_allclose(
            data.get_y_covar(residual_type='empirical', residual_corr_weight=rho), expected,
            rtol=1e-10, atol=1e-14)
    # residual_var_weight multiplies the whole residual block, off-diagonal entries included.
    block = data.get_residual_covar(residual_type='empirical')
    assert np.abs(off_diagonal(block)).max() > 0
    np.testing.assert_allclose(
        data.get_residual_covar(residual_type='empirical', residual_var_weight=0.35),
        0.35 * block, rtol=1e-12, atol=1e-16)
    # Monthly-only assets keep the monthly grid and its span.
    monthly = empirical_estimator().fit_current_factor_covars(
        risk_factor_prices=factor_prices, asset_returns_dict={'ME': asset_returns},
        estimation_date=as_of).residual_correlation
    assert (monthly.frequency, monthly.span) == ('ME', SPANS['ME'])
    # The residual grid is separate from the factor covariance, whose defaults stay weekly
    # returns with span 52.
    default = opt.FactorCovarEstimator()
    assert (default.factor_returns_freq, default.factor_covar_span) == ('W-WED', 52)
    # The retention weight defaults to 1 and must lie in [0, 1].
    assert estimator.residual_corr_weight == default.residual_corr_weight == 1.0
    try:
        replace(estimator, residual_corr_weight=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError('residual_corr_weight=1.5 was accepted')
    # Rolling getters: half retention halves the empirical off-diagonal dependence on every
    # date, and correlation vintages are dimensionless and keyed by fit dates.
    rolling = empirical_estimator().fit_rolling_factor_covars(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, time_period=period)
    orthogonal_covars = rolling.get_y_covars()
    half = rolling.get_y_covars(residual_type='empirical', residual_corr_weight=0.5)
    full = rolling.get_y_covars(residual_type='empirical')
    assert list(orthogonal_covars) == list(half) == list(full) == list(pd.to_datetime(FACTOR_KEYS))
    for date, covar in orthogonal_covars.items():
        np.testing.assert_allclose(half[date] - covar, 0.5 * (full[date] - covar),
                                   rtol=1e-10, atol=1e-16)
    vintages = rolling.get_residual_correlations()
    assert vintages and set(vintages) <= set(orthogonal_covars)
    for correlation in vintages.values():
        np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=0, atol=1e-12)


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2021-01-06", periods=160, freq="W-WED")
    steps = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {"Equity": 100.0 * np.exp(0.002 * steps + 0.05 * np.sin(steps / 5)),
         "Bonds": 100.0 * np.exp(0.0007 * steps + 0.02 * np.cos(steps / 7))},
        index=dates,
    )

    # 160 weekly Wednesdays of positive prices for two assets.
    assert len(prices) == 160 and list(prices.columns) == ["Equity", "Bonds"]
    assert (prices > 0).all().all() and (prices.index.dayofweek == 2).all()

    import optimalportfolios as opt

    estimator = opt.EwmaCovarEstimator(
        returns_freq="W-WED",
        span=52,
        rebalancing_freq="QE",
        demean=True,
    )
    current_covar = estimator.fit_current_covar(prices=prices)

    # Class defaults: weekly Wednesday returns, span 52, quarter-end dates, demeaning on and the
    # ordinary kernel.
    default = opt.EwmaCovarEstimator()
    assert (default.returns_freq, default.span, default.rebalancing_freq) == ("W-WED", 52, "QE")
    assert default.demean is True and default.is_apply_vol_normalised_returns is False
    # Span 52 means decay 51/53 and a half-life of about 18 observations, not 52.
    decay = 1 - 2 / (52 + 1)
    assert round(np.log(0.5) / np.log(decay)) == 18
    # The current fit is the zero-seeded weighted sum of demeaned weekly log returns, times 52.
    expected = weighted_reference(prices, span=52, annualization=52, demean=True)
    np.testing.assert_allclose(current_covar, expected, rtol=1e-10, atol=1e-14)
    assert current_covar.shape == (2, 2) and current_covar.index.equals(current_covar.columns)
    assert current_covar.columns.tolist() == ["Equity", "Bonds"]
    # No hard lookback: changing the first of 160 prices still moves the current matrix.
    earlier = prices.copy()
    earlier.iloc[0] *= 1.1
    assert np.abs(estimator.fit_current_covar(prices=earlier) - current_covar).max().max() > 1e-6
    # rebalancing_freq only selects rolling output dates; the current fit ignores it.
    pd.testing.assert_frame_equal(
        replace(estimator, rebalancing_freq="ME").fit_current_covar(prices=prices), current_covar)
    # estimate_current_ewma_covar is the function behind the current fit; without the
    # annualization factor it returns weekly units, one 52nd of the annual matrix.
    pd.testing.assert_frame_equal(
        opt.estimate_current_ewma_covar(prices, returns_freq="W-WED", span=52), current_covar)
    weekly = opt.estimate_current_ewma_covar(prices, returns_freq="W-WED", span=52,
                                             apply_an_factor=False)
    np.testing.assert_allclose(52 * weekly, current_covar, rtol=1e-12, atol=0)

    small_returns = np.array([[0.01, 0.02], [-0.02, 0.01], [0.03, -0.01]])
    small_prices = pd.DataFrame(
        100.0 * np.exp(np.vstack([np.zeros(2), np.cumsum(small_returns, axis=0)])),
        index=pd.date_range("2023-12-31", periods=4, freq="ME"),
        columns=["A", "B"],
    )
    small_estimator = opt.EwmaCovarEstimator(returns_freq="ME", span=3, demean=False)
    small_covar = small_estimator.fit_current_covar(prices=small_prices)

    # Span 3 gives decay 1/2: zero-seeded updates weight the three returns 1/8, 1/4 and 1/2 in
    # chronological order, and the weights sum to 7/8.
    observations = [
        [Fraction(1, 100), Fraction(2, 100)],
        [Fraction(-2, 100), Fraction(1, 100)],
        [Fraction(3, 100), Fraction(-1, 100)],
    ]
    half_decay = 1 - Fraction(2, 3 + 1)
    weights = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)]
    assert weights == [(1 - half_decay) * half_decay ** age for age in (2, 1, 0)]
    assert sum(weights) == Fraction(7, 8)
    np.testing.assert_allclose(small_returns, np.array(observations, dtype=float),
                               rtol=0, atol=1e-15)
    # Twelve times the weighted second moments, in exact arithmetic, and the page's table.
    exact = [[float(12 * sum(weight * row[i] * row[j]
                             for weight, row in zip(weights, observations)))
              for j in range(2)] for i in range(2)]
    np.testing.assert_allclose(small_covar, exact, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(small_covar, SMALL_TABLE, rtol=0, atol=0.5e-6)
    assert small_covar.columns.tolist() == ["A", "B"]

    monthly_dates = pd.date_range("2018-12-31", periods=73, freq="ME")
    m = np.arange(72, dtype=float)
    factor_returns = np.column_stack([
        0.003 + 0.025 * np.sin(m / 3),
        0.001 + 0.018 * np.cos(m / 5),
    ])
    factor_prices = pd.DataFrame(
        100.0 * np.exp(np.vstack([np.zeros(2), np.cumsum(factor_returns, axis=0)])),
        index=monthly_dates, columns=["Growth", "Rates"],
    )
    asset_values = (
        factor_returns @ np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.4]]).T
        + 0.004 * np.column_stack([np.cos(m / 2), np.sin(m / 4), np.cos(m / 6)])
    )
    asset_returns = pd.DataFrame(
        asset_values, index=monthly_dates[1:], columns=["Equity", "Bonds", "Balanced"],
    )

    # 73 month-end factor prices whose log differences are the factor returns, and asset log
    # returns on the 72 later month ends.
    assert factor_prices.shape == (73, 2) and asset_returns.shape == (72, 3)
    np.testing.assert_allclose(np.log(factor_prices).diff().iloc[1:], factor_returns,
                               rtol=0, atol=1e-14)
    assert asset_returns.index.equals(factor_prices.index[1:])

    from factorlasso import LassoModel, LassoModelType

    factor_estimator = opt.FactorCovarEstimator(
        lasso_model=LassoModel(
            model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-5, span=24, warmup_period=12, n_clusters=2,
        ),
        factor_returns_freq="ME", factor_covar_span=24, rebalancing_freq="QE",
    )
    as_of = monthly_dates[48]
    factor_data = factor_estimator.fit_current_factor_covars(
        risk_factor_prices=factor_prices.loc[:as_of],
        asset_returns_dict={"ME": asset_returns.loc[:as_of]},
        estimation_date=as_of,
    )
    factor_covar = factor_data.get_y_covar()
    factor_only = factor_data.get_y_covar(residual_var_weight=0.0)
    scaled_residual_covar = factor_data.get_y_covar(residual_var_weight=0.35)

    # The cutoff is 31 December 2022; three asset rows, two factor columns; CLARABEL solves.
    assert as_of == pd.Timestamp("2022-12-31") and factor_data.estimation_date == as_of
    assert factor_data.y_betas.shape == (3, 2)
    assert factor_data.y_betas.columns.tolist() == ["Growth", "Rates"]
    assert factor_estimator.lasso_model.solver == "CLARABEL"
    # Each assembled matrix equals an explicit sum over asset and factor pairs plus the weighted
    # residual diagonal: weight 1.0, 0.0 (factor only) and 0.35. The default weight is 1.0.
    residual_vars = factor_data.y_variances[VarianceColumns.RESIDUAL_VARS.value]
    assert (residual_vars > 0).all()
    for matrix, weight in ((factor_covar, 1.0), (factor_only, 0.0),
                           (scaled_residual_covar, 0.35)):
        reference = component_reference(factor_data.y_betas, factor_data.x_covar,
                                        residual_vars, weight)
        np.testing.assert_allclose(matrix, reference, rtol=1e-10, atol=1e-14)
        assert matrix.index.tolist() == ASSETS and matrix.columns.tolist() == ASSETS
    signature = inspect.signature(CurrentFactorCovarData.get_y_covar)
    assert signature.parameters["residual_var_weight"].default == 1.0
    # Moving the residual weight from 1.0 to 0.35 changes only the diagonal.
    np.testing.assert_allclose(off_diagonal(scaled_residual_covar), off_diagonal(factor_covar),
                               rtol=0, atol=1e-16)
    assert (np.diag(scaled_residual_covar) < np.diag(factor_covar)).all()
    # The factor covariance is a separate monthly demeaned EWMA sum at span 24, times 12.
    np.testing.assert_allclose(
        factor_data.x_covar,
        weighted_reference(factor_prices.loc[:as_of], span=24, annualization=12, demean=True),
        rtol=1e-10, atol=1e-14)
    # Cluster identifiers carry their frequency bucket; the fit leaves its state on the model.
    assert factor_data.clusters.notna().all()
    assert all(value.startswith("ME:") for value in factor_data.clusters)
    np.testing.assert_allclose(factor_estimator.lasso_model.estimated_betas.loc[ASSETS],
                               factor_data.y_betas.loc[ASSETS], rtol=0, atol=0)

    def fresh_factor(**overrides) -> opt.FactorCovarEstimator:
        """The page's factor estimator with a new LassoModel, so no fitted state is shared."""
        model = LassoModel(**factor_estimator.lasso_model.get_params())
        return replace(factor_estimator, lasso_model=model, **overrides)

    history = {"risk_factor_prices": factor_prices.loc[:as_of],
               "asset_returns_dict": {"ME": asset_returns.loc[:as_of]},
               "estimation_date": as_of}
    # The page's fit equals an independently invoked fit on the same prefixes.
    np.testing.assert_allclose(factor_covar, fresh_factor().fit_current_covar(**history),
                               rtol=1e-10, atol=1e-14)
    # estimation_date alone does not truncate an ordinary current fit: the whole history
    # labelled 31 December 2022 gives a different matrix, with the opposite sign of the
    # Equity-Bonds covariance.
    whole = fresh_factor().fit_current_covar(
        risk_factor_prices=factor_prices, asset_returns_dict={"ME": asset_returns},
        estimation_date=as_of)
    assert np.max(np.abs(whole - factor_covar).to_numpy()) > 1e-5
    assert whole.at["Equity", "Bonds"] > 0 > factor_covar.at["Equity", "Bonds"]
    # The top-level demean field is not read: demean=False reproduces the fit.
    undemeaned = fresh_factor(demean=False).fit_current_factor_covars(**history)
    np.testing.assert_allclose(undemeaned.x_covar, factor_data.x_covar, rtol=0, atol=1e-14)
    np.testing.assert_allclose(undemeaned.y_betas, factor_data.y_betas, rtol=0, atol=1e-12)
    # A supplied annual factor covariance is used as given, not multiplied by 12 again.
    supplied = factor_data.x_covar * 1.7
    pd.testing.assert_frame_equal(
        fresh_factor().fit_current_factor_covars(**history, x_covar=supplied).x_covar, supplied)
    # The reported residuals are 12 times asset returns minus the factor contribution, with no
    # intercept subtracted; the first row has no preceding factor price.
    factors = factor_prices.loc[:as_of]
    factor_log_returns = pd.DataFrame(np.diff(np.log(factors), axis=0),
                                      index=factors.index[1:], columns=factors.columns)
    scaled = 12 * (asset_returns.loc[:as_of] - factor_log_returns @ factor_data.y_betas.T)
    pd.testing.assert_index_equal(factor_data.residuals.index, scaled.index)
    np.testing.assert_allclose(factor_data.residuals.iloc[1:], scaled.iloc[1:], rtol=0,
                               atol=1e-14)
    assert factor_data.residuals.iloc[0].isna().all()

    import qis

    ewma_period = qis.TimePeriod(dates[60], dates[100])
    rolling_covars = estimator.fit_rolling_covars(prices=prices, time_period=ewma_period)
    factor_period = qis.TimePeriod(monthly_dates[47], monthly_dates[54])
    rolling_factor_covars = factor_estimator.fit_rolling_covars(
        risk_factor_prices=factor_prices,
        asset_returns_dict={"ME": asset_returns},
        time_period=factor_period,
    )

    # EWMA keys lie on the weekly grid, each the first Wednesday after a quarter end; factor
    # keys are calendar quarter ends.
    assert list(rolling_covars) == list(pd.to_datetime(EWMA_KEYS))
    for key in rolling_covars:
        assert key.dayofweek == 2 and (key - (key - pd.offsets.QuarterEnd())).days < 7
    assert list(rolling_factor_covars) == list(pd.to_datetime(FACTOR_KEYS))
    assert all(key.is_quarter_end for key in rolling_factor_covars)
    # Point in time with either kernel: each rolling EWMA matrix equals a current fit on the
    # prices through its date, and rescaled later prices change none of them.
    assert installed_version("qis") >= (5, 31, 0)
    later_prices = perturbed_after(prices, max(rolling_covars))
    normalised = replace(estimator, is_apply_vol_normalised_returns=True)
    normalised_covars = normalised.fit_rolling_covars(prices=prices, time_period=ewma_period)
    for kernel, covars in ((estimator, rolling_covars), (normalised, normalised_covars)):
        changed = kernel.fit_rolling_covars(prices=later_prices, time_period=ewma_period)
        assert list(changed) == list(covars)
        for date, covar in covars.items():
            pd.testing.assert_frame_equal(covar, changed[date], check_exact=False,
                                          rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(covar, kernel.fit_current_covar(prices.loc[:date]),
                                       rtol=1e-10, atol=1e-14)
    # The normalized kernel is a different estimate, and qis seeds each of its volatilities
    # with the column's first squared return rather than a full-array statistic.
    assert not np.allclose(normalised_covars[max(normalised_covars)],
                           rolling_covars[max(rolling_covars)], rtol=1e-6, atol=0)
    log_returns = np.log(prices).diff().iloc[1:].to_numpy()
    _, _, ewm_vols = qis.compute_ewm_covar_tensor_vol_norm_returns(a=log_returns, span=52)
    np.testing.assert_allclose(ewm_vols[0], np.abs(log_returns[0]), rtol=1e-12, atol=0)
    # Rolling factor fits truncate every input: later factor prices and asset returns change
    # no earlier matrix.
    factor_cutoff = max(rolling_factor_covars)
    later_factors, later_returns = factor_prices.copy(), asset_returns.copy()
    later_factors.loc[later_factors.index > factor_cutoff, "Growth"] *= 2.0
    later_returns.loc[later_returns.index > factor_cutoff, "Equity"] += 0.10
    changed_factor_covars = fresh_factor().fit_rolling_covars(
        risk_factor_prices=later_factors, asset_returns_dict={"ME": later_returns},
        time_period=factor_period,
    )
    for date, covar in rolling_factor_covars.items():
        np.testing.assert_allclose(covar, changed_factor_covars[date], rtol=1e-10, atol=1e-14)

    # The legacy estimate_rolling_ewma_covar is the qis function itself.
    assert opt.estimate_rolling_ewma_covar is qis.estimate_rolling_ewma_covar
    # Empirical residual correlation needs FactorLasso 0.19.0 or newer; older releases serve
    # only the orthogonal default.
    supported = "residual_correlation" in {field.name for field in fields(CurrentFactorCovarData)}
    assert supported == (installed_version("factorlasso") >= (0, 19, 0))
    if supported:
        check_empirical_residuals(factor_prices, asset_returns, as_of, factor_period)
    print("covariance_estimators: all page statements verified.")


if __name__ == '__main__':
    main()
