"""
example of minimization of tracking error
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
import yfinance as yf


from optimalportfolios import (Constraints,
                               wrapper_maximise_diversification,
                               estimate_current_ewma_covar,
                               compute_portfolio_vol,
                               local_path)


def compute_benchamark_portfolio_risk_contributions(w_portfolio: pd.Series,
                                                    w_benchmark: pd.Series,
                                                    covar: pd.DataFrame,
                                                    is_independent_risk: bool = False
                                                    ) -> pd.Series:
    """
    Compute per-asset tracking error contributions of portfolio vs benchmark.

    Two modes:
    - is_independent_risk=False: marginal TE contributions (sum = total TE)
    - is_independent_risk=True: independent (diagonal) TE contributions
    """
    active_weights = w_portfolio - w_benchmark
    covar_np = covar.to_numpy()
    active_np = active_weights.to_numpy()

    if is_independent_risk:
        # independent: use only diagonal of covar (asset vols), no cross terms
        diag_var = np.diag(covar_np)
        contributions = active_np ** 2 * diag_var
        contributions = np.sign(active_np) * np.sqrt(np.abs(contributions))
    else:
        # marginal TE contributions: w_active @ covar / TE
        portfolio_var = active_np @ covar_np @ active_np
        te = np.sqrt(np.maximum(portfolio_var, 0.0))
        if te < 1e-16:
            contributions = np.zeros_like(active_np)
        else:
            # marginal contribution of each asset to TE
            marginal = covar_np @ active_np
            contributions = active_np * marginal / te

    return pd.Series(contributions, index=w_portfolio.index)


def create_stocks_data():
    dow_30_tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'JPM', 'WMT', 'V', 'JNJ', 'PG', 'HD', 'KO', 'CSCO', 'IBM',
                      'CVX', 'UNH', 'CRM', 'DIS', 'AXP', 'MCD', 'GS', 'MRK', 'CAT', 'VZ', 'BA', 'AMGN', 'HON', 'NKE',
                      'SHW', 'MMM', 'TRV']
    prices = yf.download(tickers=dow_30_tickers, start="2003-12-31", end="2025-07-18", ignore_tz=True, auto_adjust=True)['Close'][dow_30_tickers]
    qis.save_df_to_csv(df=prices, file_name='dow30_prices', local_path=local_path.get_resource_path())

# create_stocks_data()
prices = qis.load_df_from_csv(file_name='dow30_prices', local_path=local_path.get_resource_path())
print(prices)
# create bench
benchmark_weights = qis.df_to_weight_allocation_sum1(df=prices.iloc[-1, :])

# prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()
time_period = qis.TimePeriod(start='31Dec2009', end=prices.index[-1])
perf_time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])  # backtest report

covar_matrix = estimate_current_ewma_covar(prices=prices, span=3*52)
print(covar_matrix)
qis.plot_corr_matrix_from_covar(covar=covar_matrix)


# portfolio_weights = wrapper_risk_budgeting(pd_covar=covar_matrix, constraints=Constraints(is_long_only=True))
# portfolio_weights = wrapper_quadratic_optimisation(pd_covar=covar_matrix, constraints=Constraints(is_long_only=True))
portfolio_weights = wrapper_maximise_diversification(pd_covar=covar_matrix, constraints=Constraints(is_long_only=True))

print(f"benchmark_vol={compute_portfolio_vol(covar_matrix, benchmark_weights):.2%}, "
      f"portfolio_vol={compute_portfolio_vol(covar_matrix, portfolio_weights):.2%},"
      f"tracking_error={compute_portfolio_vol(covar_matrix, benchmark_weights-portfolio_weights):.2%},"
      f"tracking_error1={np.nansum(compute_benchamark_portfolio_risk_contributions(w_portfolio=portfolio_weights, w_benchmark=benchmark_weights, covar=covar_matrix)):.2%}, "
      f"tracking_error ind={np.nansum(compute_benchamark_portfolio_risk_contributions(w_portfolio=portfolio_weights, w_benchmark=benchmark_weights, covar=covar_matrix, is_independent_risk=True)):.2%}")


risk_contributions = qis.compute_portfolio_risk_contributions(w=portfolio_weights, covar=covar_matrix)
risk_contributions_rel = risk_contributions / np.nansum(risk_contributions)

tre_contributions = qis.compute_portfolio_risk_contributions(w=(portfolio_weights-benchmark_weights), covar=covar_matrix)
tre_contributions_rel = tre_contributions / np.nansum(tre_contributions)

tre_contributions1 = compute_benchamark_portfolio_risk_contributions(w_portfolio=portfolio_weights, w_benchmark=benchmark_weights, covar=covar_matrix)
tre_contributions_rel1 = tre_contributions1 / np.nansum(tre_contributions1)

tre_contributions_ind = compute_benchamark_portfolio_risk_contributions(w_portfolio=portfolio_weights, w_benchmark=benchmark_weights, covar=covar_matrix, is_independent_risk=True)
tre_contributions_ind_rel = tre_contributions_ind / np.nansum(tre_contributions_ind)


df = pd.concat([benchmark_weights.rename('benchmark'),
                portfolio_weights.rename('portfolio'),
                risk_contributions.rename('risk-contribs bp'),
                risk_contributions_rel.rename('risk-contribs %'),
                tre_contributions.rename('tre contribs bp'),
                tre_contributions_rel.rename('tre contribs %'),
                tre_contributions1.rename('tre contribs1 bp'),
                tre_contributions_rel1.rename('tre contribs1 %'),
                tre_contributions_ind.rename('tre contribs ind bp'),
                tre_contributions_ind_rel.rename('tre contribs ind %'),
                ], axis=1).sort_values(by='portfolio', ascending=False)
df.loc['total', :] = df.sum(axis=0)
qis.plot_df_table(df=df, var_format='{:.2%}')

plt.show()


