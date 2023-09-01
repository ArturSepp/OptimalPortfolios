"""
implement marginal backtest to stress marginal contribution from adding one asset to the investable universe
"""

# packages
import pandas as pd
import numpy as np
from typing import NamedTuple, Dict, Tuple, Any
from enum import Enum

# qis
import qis
from qis import PortfolioData, TimePeriod

# project
import optimalportfolios.optimization.rolling.max_utility_sharpe as ms
import optimalportfolios.optimization.rolling.risk_based as rb
import optimalportfolios.optimization.rolling.max_mixture_carra as rp
from optimalportfolios.optimization.engine import PortfolioObjective


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
    rebalancing_freq: str = 'Q'  # when portfolio weigths are aupdate  
    roll_window: int = 20  # hw many periods are used for rolling estimation of mv returns and mixure
    returns_freq: str = 'M'  # frequency of returns
    span: int = 24   # for ewma window
    is_log_returns: bool = True  # use logreturns
    carra: float = 0.5  # carra parameter
    n_mixures: int = 3
    rebalancing_costs: float = 0.0010,  # 10 bp

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()


def backtest_marginal_optimal_portfolios(prices: pd.DataFrame,  # for inclusion to backtest portfolio
                                         marginal_asset: str,  # this is asset we test for inclusion
                                         time_period: TimePeriod = None,  # for reporting portfolio weights
                                         is_alternatives: bool = True,  #
                                         optimisation_type: OptimisationType = OptimisationType.MIXTURE,
                                         marginal_asset_ew_weight: float = 0.02,  # allocation for equal weight
                                         first_asset_target_weight: float = 0.75,  # first asset is the benchmark
                                         rebalancing_freq: str = 'Q',
                                         roll_window: int = 20,  # hw many periods are used for roll_window
                                         returns_freq: str = 'M',
                                         span: int = 24,
                                         is_log_returns: bool = True,
                                         carra: float = 0.5,
                                         n_mixures: int = 3,
                                         rebalancing_costs: float = 0.0010,  # 10 bp
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
        ew_weights_with[0] = marginal_asset_ew_weight

        # erc
        budget_with = np.ones(len(prices_with_asset.columns)) / len(prices_with_asset.columns)
        budget_wo = np.ones(len(prices_without_asset.columns)) / len(prices_without_asset.columns)

    else:
        # for mvo
        weight_min_with = pd.Series(0.0, index=prices_with_asset.columns)
        weight_min_with[0] = first_asset_target_weight
        weight_min_wo = pd.Series(0.0, index=prices_without_asset.columns)
        weight_min_wo[0] = first_asset_target_weight

        weight_max_with = pd.Series(1.0, index=prices_with_asset.columns)
        weight_max_with[0] = first_asset_target_weight
        weight_max_wo = pd.Series(1.0, index=prices_without_asset.columns)
        weight_max_wo[0] = first_asset_target_weight

        # ew
        ew_weight = (1.0 - first_asset_target_weight) / (len(prices_without_asset.columns) - 1)
        ew_weights_wo = pd.Series(ew_weight, index=prices_without_asset.columns)
        ew_weights_wo[0] = first_asset_target_weight

        ew_weight = (1.0 - marginal_asset_ew_weight * (1.0 - first_asset_target_weight)) / (len(prices_with_asset.columns) - 2)
        ew_weights_with = pd.Series(ew_weight, index=prices_with_asset.columns)
        ew_weights_with[0] = first_asset_target_weight
        ew_weights_with[1] = marginal_asset_ew_weight * (1.0 - first_asset_target_weight)

        # erc
        budget_with = (1.0 - first_asset_target_weight) * pd.Series(1.0, index=prices_with_asset.columns) / (len(prices_with_asset.columns) - 1)
        budget_wo = (1.0 - first_asset_target_weight) * pd.Series(1.0, index=prices_without_asset.columns) / (len(prices_without_asset.columns) - 1)
        budget_with[0] = first_asset_target_weight
        budget_wo[0] = first_asset_target_weight

    if optimisation_type == OptimisationType.EW:

        portfolio_wo = qis.backtest_model_portfolio(prices=prices_without_asset,
                                                    weights=ew_weights_wo.to_dict(),
                                                    rebalance_freq=rebalancing_freq,
                                                    is_rebalanced_at_first_date=True,
                                                    rebalancing_costs=rebalancing_costs,
                                                    ticker=f"{optimisation_type.value} w/o {marginal_asset}",
                                                    is_output_portfolio_data=True)

        portfolio_with = qis.backtest_model_portfolio(prices=prices_with_asset,
                                                      weights=ew_weights_with.to_dict(),
                                                      rebalance_freq=rebalancing_freq,
                                                      is_rebalanced_at_first_date=True,
                                                      rebalancing_costs=rebalancing_costs,
                                                      ticker=f"{optimisation_type.value} with {marginal_asset} {marginal_asset_ew_weight: 0.0%}",
                                                      is_output_portfolio_data=True)

    elif optimisation_type in [OptimisationType.ERC, OptimisationType.MAX_DIV]:
        if optimisation_type == OptimisationType.ERC:
            portfolio_objective = PortfolioObjective.EQUAL_RISK_CONTRIBUTION
            weight_min_wo = weight_max_wo = None
            weight_min_with = weight_max_with = None
        else:
            portfolio_objective = PortfolioObjective.MAX_DIVERSIFICATION

        portfolio_wo = rb.backtest_rolling_ewma_risk_based_portfolio(prices=prices_without_asset,
                                                                     time_period=time_period,
                                                                     rebalancing_freq=rebalancing_freq,
                                                                     returns_freq=returns_freq,
                                                                     span=span,
                                                                     is_log_returns=is_log_returns,
                                                                     portfolio_objective=portfolio_objective,
                                                                     budget=budget_wo,
                                                                     min_weights=weight_min_wo,
                                                                     max_weights=weight_max_wo,
                                                                     ticker=f"{optimisation_type.value} w/o {marginal_asset}",
                                                                     rebalancing_costs=rebalancing_costs)
        portfolio_with = rb.backtest_rolling_ewma_risk_based_portfolio(prices=prices_with_asset,
                                                                       time_period=time_period,
                                                                       rebalancing_freq=rebalancing_freq,
                                                                       returns_freq=returns_freq,
                                                                       span=span,
                                                                       is_log_returns=is_log_returns,
                                                                       portfolio_objective=portfolio_objective,
                                                                       budget=budget_with,
                                                                       min_weights=weight_min_with,
                                                                       max_weights=weight_max_with,
                                                                       ticker=f"{optimisation_type.value} with {marginal_asset}",
                                                                       rebalancing_costs=rebalancing_costs)

    elif optimisation_type == OptimisationType.MAX_SHARPE:
        portfolio_wo = ms.backtest_rolling_max_utility_sharpe_portfolios(prices=prices_without_asset,
                                                                         rebalancing_freq=rebalancing_freq,
                                                                         roll_window=roll_window,
                                                                         returns_freq=returns_freq,
                                                                         is_log_returns=is_log_returns,
                                                                         span=span,
                                                                         carra=0.0,
                                                                         min_weights=weight_min_wo,
                                                                         max_weights=weight_max_wo,
                                                                         ticker=f"{optimisation_type.value} w/o {marginal_asset}",
                                                                         rebalancing_costs=rebalancing_costs)
        portfolio_with = ms.backtest_rolling_max_utility_sharpe_portfolios(prices=prices_with_asset,
                                                                           rebalancing_freq=rebalancing_freq,
                                                                           roll_window=roll_window,
                                                                           returns_freq=returns_freq,
                                                                           is_log_returns=is_log_returns,
                                                                           span=span,
                                                                           carra=0.0,
                                                                           min_weights=weight_min_with,
                                                                           max_weights=weight_max_with,
                                                                           ticker=f"{optimisation_type.value} with {marginal_asset}",
                                                                           rebalancing_costs=rebalancing_costs)

    elif optimisation_type == OptimisationType.MIXTURE:
        portfolio_wo = rp.backtest_rolling_mixure_portfolio(prices=prices_without_asset,
                                                            time_period=time_period,
                                                            rebalancing_freq=rebalancing_freq,
                                                            roll_window=roll_window,
                                                            returns_freq=returns_freq,
                                                            is_log_returns=is_log_returns,
                                                            n_components=n_mixures,
                                                            carra=carra,
                                                            min_weights=weight_min_wo,
                                                            max_weights=weight_max_wo,
                                                            ticker=f"{optimisation_type.value} w/o {marginal_asset}",
                                                            rebalancing_costs=rebalancing_costs)
        portfolio_with = rp.backtest_rolling_mixure_portfolio(prices=prices_with_asset,
                                                              time_period=time_period,
                                                              rebalancing_freq=rebalancing_freq,
                                                              roll_window=roll_window,
                                                              returns_freq=returns_freq,
                                                              is_log_returns=is_log_returns,
                                                              n_components=n_mixures,
                                                              carra=carra,
                                                              min_weights=weight_min_with,
                                                              max_weights=weight_max_with,
                                                              ticker=f"{optimisation_type.value} with {marginal_asset}",
                                                              rebalancing_costs=rebalancing_costs)

    else:
        raise NotImplementedError

    return portfolio_wo, portfolio_with
