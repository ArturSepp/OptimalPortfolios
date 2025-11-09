"""
implementation of maximum sharpe ratio portfolios
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis
from typing import Tuple, List, Optional
from enum import Enum
from qis import TimePeriod

from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.covar_estimation.utils import squeeze_covariance_matrix


def rolling_maximize_portfolio_sharpe(prices: pd.DataFrame,
                                      constraints: Constraints,
                                      time_period: qis.TimePeriod,  # when we start building portfolios
                                      returns_freq: str = 'W-WED',
                                      rebalancing_freq: str = 'QE',
                                      span: int = 52,  # 1y
                                      roll_window: int = 20,  # defined on number of periods in rebalancing_freq
                                      solver: str = 'ECOS_BB',
                                      squeeze_factor: Optional[float] = None,  # for squeezing covar matrix
                                      print_inputs: bool = False
                                      ) -> pd.DataFrame:
    """
    maximise portfolio alpha subject to constraint on tracking tracking error
    """
    means, covars = estimate_rolling_means_covar(prices=prices,
                                                 returns_freq=returns_freq,
                                                 rebalancing_freq=rebalancing_freq,
                                                 roll_window=roll_window,
                                                 annualize=True,
                                                 span=span)

    tickers = prices.columns.to_list()
    weights = {}
    weights_0 = None
    for date, covar in zip(means.index, covars):
        if date >= time_period.start:
            pd_covar = pd.DataFrame(covar, index=tickers, columns=tickers)
            # call optimiser
            if print_inputs:
                print(f"date={date}")
                print(f"pd_covar=\n{pd_covar}")

            weights_ = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                         means=means.loc[date, :],
                                                         constraints=constraints,
                                                         weights_0=weights_0,
                                                         squeeze_factor=squeeze_factor,
                                                         solver=solver)

            weights_0 = weights_  # update for next rebalancing
            weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers).fillna(0.0)  # align with tickers
    return weights


def wrapper_maximize_portfolio_sharpe(pd_covar: pd.DataFrame,
                                      means: pd.Series,
                                      constraints: Constraints,
                                      weights_0: pd.Series = None,
                                      squeeze_factor: Optional[float] = None,  # for squeezing covar matrix
                                      solver: str = 'ECOS_BB'
                                      ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = dict(means=means)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if squeeze_factor is not None and squeeze_factor > 0.0:
        clean_covar = squeeze_covariance_matrix(clean_covar, squeeze_factor=squeeze_factor)

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0)

    weights = cvx_maximize_portfolio_sharpe(covar=clean_covar.to_numpy(),
                                            means=good_vectors['means'].to_numpy(),
                                            constraints=constraints1,
                                            solver=solver)
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers

    return weights


def cvx_maximize_portfolio_sharpe(covar: np.ndarray,
                                  means: np.ndarray,
                                  constraints: Constraints,
                                  verbose: bool = False,
                                  solver: str = 'ECOS_BB'
                                  ) -> np.ndarray:
    """
    max means^t*w / sqrt(w^t @ covar @ w)
    subject to
     1. weight_min <= w <= weight_max
    """
    # set up problem
    n = covar.shape[0]
    z = cvx.Variable(n+1)
    w = z[:n]
    k = z[n]
    objective = cvx.Minimize(cvx.quad_form(w, covar))

    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar, exposure_scaler=k)

    # add scaling constraints
    constraints_ += [means.T @ w == constraints.max_exposure]

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = z.value

    if optimal_weights is not None:
        optimal_weights = optimal_weights[:n] / optimal_weights[n]  # apply rescaling
    else:
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zeroweights")

    return optimal_weights


def estimate_rolling_means_covar(prices: pd.DataFrame,
                                 returns_freq: str = 'W-WED',
                                 rebalancing_freq: str = 'QE',
                                 roll_window: int = 20,  # defined on number of periods in rebalancing_freq
                                 span: int = 52,
                                 annualize: bool = True,
                                 is_regularize: bool = True,
                                 is_ewm_covar: bool = True
                                 ) -> Tuple[pd.DataFrame, List[np.ndarray]]:

    """
    inputs for solvers portfolios
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    # generate rebalancing dates on the returns index
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    if annualize:
        _, scaler = qis.get_period_days(freq=returns_freq)
    else:
        scaler = 1.0
    means = {}
    covars = []
    covar0 = np.zeros((len(prices.columns), len(prices.columns)))
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if idx >= roll_window-1 and value:
            period = TimePeriod(rebalancing_schedule.index[idx - roll_window + 1], date)
            # period.print()
            rets_ = period.locate(returns).to_numpy()
            means[date] = scaler*pd.Series(np.nanmean(rets_, axis=0), index=prices.columns)
            if is_ewm_covar:
                covar = qis.compute_ewm_covar(a=rets_, span=span, covar0=covar0)
                covar0 = covar
            else:
                covar = qis.compute_masked_covar_corr(data=rets_, bias=True)

            if is_regularize:
                covar = qis.matrix_regularization(covar=covar, cut=1e-5)

            covars.append(scaler * covar)
    means = pd.DataFrame.from_dict(means, orient="index")
    return means, covars


class LocalTests(Enum):
    ROLLING_MEANS_COVAR = 1
    SHARPE = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]  # need 5 years for max sharpe and max carra methods

    if local_test == LocalTests.ROLLING_MEANS_COVAR:
        # prices = prices[['SPY', 'TLT']].dropna()

        means, covars = estimate_rolling_means_covar(prices=prices, rebalancing_freq='QE', roll_window=20)
        #  = estimate_rolling_data(prices=prices, rebalancing_freq='ME', roll_window=60)

        vols = {}
        covs = {}
        for index, covar in zip(means.index, covars):
            vols[index] = pd.Series(np.sqrt(np.diag(covar)))
            covs[index] = pd.Series(np.extract(1 - np.eye(2), covar))
        vols = pd.DataFrame.from_dict(vols, orient='index')
        covs = pd.DataFrame.from_dict(covs, orient='index')
        print(vols)
        print(covs)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(3, 1, figsize=(7, 12))
            qis.plot_time_series(df=means,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=axs[0])
            qis.plot_time_series(df=vols,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=axs[1])
            qis.plot_time_series(df=covs,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=axs[2])

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ROLLING_MEANS_COVAR)
