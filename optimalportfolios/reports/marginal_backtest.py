"""
implement marginal backtest to stress marginal contribution from adding one asset to the investable universe
"""

# packages
import pandas as pd
import qis as qis
from typing import NamedTuple, Dict, Tuple, Any, Optional
from enum import Enum
from qis import PortfolioData, TimePeriod

import optimalportfolios as opt


class OptimisationType(str, Enum):
    EW = 'EqualWeight'
    ERC = 'ERC'
    MAX_DIV = 'MaxDiv'
    MAX_SHARPE = 'MaxSharpe'
    MIXTURE = 'CARA-3'


class OptimisationParams(NamedTuple):
    """
    holder for optimization params
    """
    marginal_asset_ew_weight: float = 0.02  # allocation for equal weight
    first_asset_target_weight: float = 0.75  # first asset is the benchmark
    rebalancing_freq: str = 'QE'  # when portfolio weigths are updated
    roll_window: int = 6*12  # hw many periods of returns_freq are used estimation of mv returns and mixture, default 6y
    returns_freq: str = 'ME'  # frequency of returns
    span: int = 24   # for ewma window in terms of returns freq
    carra: float = 0.5  # carra parameter
    n_mixures: int = 3
    rebalancing_costs: float = 0.0010  # 10 bp
    weight_implementation_lag: Optional[int] = 1  # for daily prices, t day weight is implemented at t+1 day

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()


def backtest_marginal_optimal_portfolios(prices: pd.DataFrame,  # for inclusion to backtest portfolio
                                         marginal_asset: str,  # this is asset we test for inclusion
                                         time_period: TimePeriod = None,  # for computing portfolio weights
                                         perf_time_period: TimePeriod = None,  # for reporting portfolio weights
                                         is_alternatives: bool = True,  #
                                         optimisation_type: OptimisationType = OptimisationType.MIXTURE,
                                         marginal_asset_ew_weight: float = 0.02,  # allocation for equal weight
                                         first_asset_target_weight: float = 0.75,  # first asset is the benchmark
                                         rebalancing_freq: str = 'QE',
                                         roll_window: int = 20,  # hw many rebalancing_freq periods are used for roll_window
                                         returns_freq: str = 'ME',
                                         span: int = 24,
                                         carra: float = 0.5,
                                         n_mixures: int = 3,
                                         rebalancing_costs: float = 0.0010,  # 10 bp
                                         weight_implementation_lag: Optional[int] = 1,
                                         **kwargs
                                         ) -> Tuple[PortfolioData, PortfolioData]:

    """
    test marginal inclusion of an asset to the optimal portfolio mix
    is_unconstrained = False, defines an asset with a fixed weight
    False: defines max/min weight for the marginal asset
    True: define the fixed allocation to the first (benchmark asset), then remaining and the marginal are flexible
    """

    prices_with_asset = prices
    prices_without_asset = prices_with_asset.drop(marginal_asset, axis=1)

    if is_alternatives:
        weight_min_with = None
        weight_min_wo = None
        weight_max_with = None
        weight_max_wo = None

        # ew
        ew_weights_wo = pd.Series(len(prices_without_asset.columns) / len(prices_without_asset.columns), index=prices_without_asset.columns)
        ew_weight = (1.0 - marginal_asset_ew_weight) / (len(prices_with_asset.columns) - 1)
        ew_weights_with = pd.Series(ew_weight, index=prices_with_asset.columns)
        ew_weights_with.iloc[0] = marginal_asset_ew_weight

        # erc
        budget_with = pd.Series(1.0, index=prices_with_asset.columns) / len(prices_with_asset.columns)
        budget_wo = pd.Series(1.0, index=prices_without_asset.columns) / len(prices_without_asset.columns)

    else:
        # for mvo
        weight_min_with = pd.Series(0.0, index=prices_with_asset.columns)
        weight_min_with.iloc[0] = first_asset_target_weight
        weight_min_wo = pd.Series(0.0, index=prices_without_asset.columns)
        weight_min_wo.iloc[0] = first_asset_target_weight

        weight_max_with = pd.Series(1.0, index=prices_with_asset.columns)
        weight_max_with.iloc[0] = first_asset_target_weight
        weight_max_wo = pd.Series(1.0, index=prices_without_asset.columns)
        weight_max_wo.iloc[0] = first_asset_target_weight

        # ew
        ew_weight = (1.0 - first_asset_target_weight) / (len(prices_without_asset.columns) - 1)
        ew_weights_wo = pd.Series(ew_weight, index=prices_without_asset.columns)
        ew_weights_wo.iloc[0] = first_asset_target_weight

        ew_weight = (1.0 - marginal_asset_ew_weight * (1.0 - first_asset_target_weight)) / (len(prices_with_asset.columns) - 2)
        ew_weights_with = pd.Series(ew_weight, index=prices_with_asset.columns)
        ew_weights_with.iloc[0] = first_asset_target_weight
        ew_weights_with.iloc[1] = marginal_asset_ew_weight * (1.0 - first_asset_target_weight)

        # erc
        budget_with = (1.0 - first_asset_target_weight) * pd.Series(1.0, index=prices_with_asset.columns) / (len(prices_with_asset.columns) - 1)
        budget_wo = (1.0 - first_asset_target_weight) * pd.Series(1.0, index=prices_without_asset.columns) / (len(prices_without_asset.columns) - 1)
        budget_with.iloc[0] = first_asset_target_weight
        budget_wo.iloc[0] = first_asset_target_weight

    # set ticker now
    ticker_wo = f"{optimisation_type.value} w/o {marginal_asset}"
    ticker_with = f"{optimisation_type.value} with {marginal_asset}"

    # default ewma estimator
    covar_estimator = opt.CovarEstimator(returns_freq=returns_freq, rebalancing_freq=rebalancing_freq, span=span)

    if optimisation_type == OptimisationType.EW:
        weights_wo = ew_weights_wo
        weights_with = ew_weights_with
        ticker_with = f"{optimisation_type.value} with {marginal_asset} {marginal_asset_ew_weight: 0.0%}"

    elif optimisation_type == OptimisationType.ERC:
        constraints0 = opt.Constraints()
        weights_wo = opt.rolling_risk_budgeting(prices=prices_without_asset,
                                                constraints0=constraints0,
                                                time_period=time_period,
                                                risk_budget=budget_wo,
                                                covar_estimator=covar_estimator)
        weights_with = opt.rolling_risk_budgeting(prices=prices_with_asset,
                                                  constraints0=constraints0,
                                                  time_period=time_period,
                                                  risk_budget=budget_with,
                                                  covar_estimator=covar_estimator)

    elif optimisation_type == OptimisationType.MAX_DIV:
        weights_wo = opt.rolling_maximise_diversification(prices=prices_without_asset,
                                                          constraints0=opt.Constraints(min_weights=weight_min_wo, max_weights=weight_max_wo),
                                                          time_period=time_period,
                                                          covar_estimator=covar_estimator)
        weights_with = opt.rolling_maximise_diversification(prices=prices_with_asset,
                                                            constraints0=opt.Constraints(min_weights=weight_min_with, max_weights=weight_max_with),
                                                            time_period=time_period,
                                                            covar_estimator=covar_estimator)

    elif optimisation_type == OptimisationType.MAX_SHARPE:
        weights_wo = opt.rolling_maximize_portfolio_sharpe(prices=prices_without_asset,
                                                           constraints0=opt.Constraints(min_weights=weight_min_wo, max_weights=weight_max_wo),
                                                           time_period=time_period,
                                                           returns_freq=returns_freq,
                                                           rebalancing_freq=rebalancing_freq,
                                                           span=span,
                                                           roll_window=roll_window)
        weights_with = opt.rolling_maximize_portfolio_sharpe(prices=prices_with_asset,
                                                             constraints0=opt.Constraints(min_weights=weight_min_with, max_weights=weight_max_with),
                                                             time_period=time_period,
                                                             returns_freq=returns_freq,
                                                             rebalancing_freq=rebalancing_freq,
                                                             span=span,
                                                             roll_window=roll_window)

    elif optimisation_type == OptimisationType.MIXTURE:
        weights_wo = opt.rolling_maximize_cara_mixture(prices=prices_without_asset,
                                                       constraints0=opt.Constraints(min_weights=weight_min_wo, max_weights=weight_max_wo),
                                                       time_period=time_period,
                                                       returns_freq=returns_freq,
                                                       rebalancing_freq=rebalancing_freq,
                                                       carra=carra,
                                                       n_components=n_mixures,
                                                       roll_window=roll_window)

        weights_with = opt.rolling_maximize_cara_mixture(prices=prices_with_asset,
                                                         constraints0=opt.Constraints(min_weights=weight_min_with, max_weights=weight_max_with),
                                                         time_period=time_period,
                                                         returns_freq=returns_freq,
                                                         rebalancing_freq=rebalancing_freq,
                                                         carra=carra,
                                                         n_components=n_mixures,
                                                         roll_window=roll_window)

    else:
        raise NotImplementedError

    if perf_time_period is not None:
        weights_wo = perf_time_period.locate(weights_wo)
        weights_with = perf_time_period.locate(weights_with)

    portfolio_wo = qis.backtest_model_portfolio(prices=qis.truncate_prior_to_start(df=prices_without_asset, start=weights_wo.index[0]),
                                                weights=weights_wo,
                                                rebalance_freq=rebalancing_freq,
                                                is_rebalanced_at_first_date=True,
                                                rebalancing_costs=rebalancing_costs,
                                                weight_implementation_lag=weight_implementation_lag,
                                                ticker=ticker_wo)

    portfolio_with = qis.backtest_model_portfolio(prices=qis.truncate_prior_to_start(df=prices_with_asset, start=weights_with.index[0]),
                                                  weights=weights_with,
                                                  rebalance_freq=rebalancing_freq,
                                                  is_rebalanced_at_first_date=True,
                                                  rebalancing_costs=rebalancing_costs,
                                                  weight_implementation_lag=weight_implementation_lag,
                                                  ticker=ticker_with)

    return portfolio_wo, portfolio_with
