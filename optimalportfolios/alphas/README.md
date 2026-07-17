# Alphas Module

Alpha signal construction and profiling for the `optimalportfolios` package.

This module computes cross-sectional alpha signals (momentum, low-beta, carry, residual momentum,
residual reversal, and their cluster variants), profiles them with a rank-based backtester, and
reports signal diagnostics. Signal construction and signal evaluation are kept separate: the signal
functions in `signals/` produce score panels, and the `profile/` submodule evaluates any score panel
without knowing how it was built.

## Architecture

```
alphas/
├── signals/                    # signal construction: prices → alpha score panel
│   ├── momentum.py             #   compute_momentum_alpha, compute_momentum_cluster_alpha
│   ├── low_beta.py             #   compute_low_beta_alpha, compute_low_beta_cluster_alpha
│   ├── carry.py                #   compute_ra_carry_alpha (risk-adjusted carry)
│   ├── residual_momentum.py    #   compute_residual_momentum_alpha (+ cluster)
│   ├── residual_reversal.py    #   compute_residual_reversal_alpha (+ cluster)
│   ├── managers_alpha.py       #   compute_managers_alpha
│   ├── rolling_ewma_mean.py    #   estimate_rolling_ewma_means
│   └── utils.py                #   extract_rolling_clusters, score_within_clusters
├── profile/                    # signal evaluation: score panel → MultiPortfolioData
│   ├── core.py                 #   THE rank-based profiler (signal-agnostic)
│   └── signal_profilers.py     #   per-signal wrappers (compute a panel, call the core)
├── alpha_data.py               # AlphasData container
└── signal_diagnostics.py       # IC/IR and score diagnostics
```

## The profiler

`profile/core.py` is the single backtester for alpha signals. It ranks the assets by a score panel,
holds the top quantile equal-weighted, and compares that basket to an equal-weight-all benchmark. It
runs no optimiser and no covariance, so the result is the selection power of the signal on its own.

```python
from optimalportfolios.alphas import backtest_alpha_rank_portfolio, compute_alpha_rank_analysis_table

# alpha_scores is a T x N panel (or a Dict[str, DataFrame] of named panels)
multi_portfolio_data = backtest_alpha_rank_portfolio(prices=prices,
                                                     alpha_scores=alpha_scores,
                                                     quantile=1.0 / 3.0,
                                                     rebalancing_freq='QE',
                                                     time_period=time_period)
table = compute_alpha_rank_analysis_table(multi_portfolio_data, time_period=time_period)
print(table)  # Return p.a., Vol, Sharpe, Max DD, Turnover p.a. per leg
```

Pass a `Dict[str, pd.DataFrame]` to profile several signals as legs of one `MultiPortfolioData`; the
equal-weight benchmark is appended last.

Three functions make up the core:

- `backtest_alpha_rank_portfolio(prices, alpha_scores, quantile=1/3, rebalancing_freq='QE', ...)` —
  backtest one panel or a dict of named panels. Returns `qis.MultiPortfolioData`.
- `compute_top_quantile_equal_weights(alpha_scores, prices, quantile=1/3)` — the rank-and-select
  weighting rule (availability-aware; `quantile=1.0` gives equal-weight-all).
- `compute_alpha_rank_analysis_table(multi_portfolio_data, time_period, perf_params)` — performance
  columns plus annualised two-sided turnover, one row per leg.

To save the full multi-strategy factsheet, pass the profiled `MultiPortfolioData` to
`generate_alpha_profile_report`:

```python
from optimalportfolios.alphas import generate_alpha_profile_report

generate_alpha_profile_report(multi_portfolio_data=multi_portfolio_data,
                              time_period=time_period,
                              group_data=group_data,
                              backtest_name='Alpha Signal Profile',
                              file_name='alpha_profile_report')  # writes {file_name}.pdf
```

It renders the multi-portfolio factsheet (every signal leg against the equal-weight benchmark) and
writes it to `{local_path}/{file_name}.pdf`. Reporting is kept separate from
`backtest_alpha_rank_portfolio` so the backtester stays a pure data producer.

## Per-signal profilers

`profile/signal_profilers.py` wraps each signal: the function takes the signal's parameters, computes
its score panel via the canonical `signals/` function, and calls the core. Use these when you want to
profile a standard signal without assembling the panel yourself.

```python
from optimalportfolios.alphas import profile_carry, profile_alpha_signals, ProfileSignal

# one signal, tuned parameters:
multi_portfolio_data = profile_carry(prices=prices, carry=yields, vol_span=13,
                                     quantile=1.0 / 3.0, rebalancing_freq='QE')

# several signals at once (default parameters), one MultiPortfolioData:
multi_portfolio_data = profile_alpha_signals(prices=prices,
                                             signals=[ProfileSignal.CARRY,
                                                      ProfileSignal.LOW_BETA,
                                                      ProfileSignal.MOMENTUM],
                                             benchmark_price=benchmark_price,
                                             carry=yields)
```

- `profile_momentum`, `profile_low_beta`, `profile_residual_momentum` take `benchmark_price`.
- `profile_carry` takes the `carry` (yield) panel; prices are used only for the volatility
  normalisation.
- `profile_alpha_signals(prices, signals, benchmark_price=, carry=, ...)` builds each requested
  signal with default parameters and profiles them jointly. It raises `ValueError` if a signal is
  requested without the input it needs (`benchmark_price` for momentum/low-beta/residual,
  `carry` for carry).

For a signal that has no wrapper, or for non-default parameters on a joint run, compute the panel
yourself and call `backtest_alpha_rank_portfolio` directly.

## Example

`examples/alphas/profile_alpha_signals.py` fetches a bond-ETF universe, profiles carry, low-beta and
momentum jointly, and sweeps carry across top-quantiles. Run it top to bottom.
