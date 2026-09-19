---
myst:
  html_meta:
    description: >-
      Contributor guide to OptimalPortfolios alpha signals: module ownership,
      named-panel profiling, offline checks, diagnostics, and report output.
---

# Alphas Module

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-03-15](https://github.com/ArturSepp/OptimalPortfolios/commit/ce4d4c3469216d807079af2c52f5a85775d9c572)*

Contributor documentation for [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

The alpha layer constructs score panels, evaluates ranked portfolios and connects signal
diagnostics to [QIS](https://github.com/ArturSepp/QuantInvestStrats). Signal construction and
evaluation have separate interfaces: constructors produce signals and scores; profilers consume
scores and delegate holdings simulation to QIS.

The [alpha methodology article](../../../docs/alphas_module_readme.md) is authoritative for
definitions, return conventions, sampling, scoring and timing limitations. This README explains
where contributors should work and how to exercise the existing interfaces.

## Architecture

| Area | Responsibility |
|---|---|
| [Alpha exports](./__init__.py) | Public alpha-layer imports, including profiling and diagnostics. |
| [Signal exports](./signals/__init__.py) | Constructors and scoring helpers; the subpackage has its own import surface. |
| [EWMA momentum](./signals/momentum.py) and [classic momentum](./signals/classic_momentum.py) | EWMA filters and fixed-window momentum are separate implementations. |
| [Low beta](./signals/low_beta.py), [residual momentum](./signals/residual_momentum.py) and [reversal](./signals/residual_reversal.py) | Benchmark-relative signal families and their cluster variants. |
| [Carry](./signals/carry.py) and [managers alpha](./signals/managers_alpha.py) | Supplied carry inputs and factor-model residual inputs. |
| [Rolling means](./signals/rolling_ewma_mean.py) and [signal utilities](./signals/utils.py) | Expected-return estimation, cadence/group preparation and cluster-score alignment. |
| [Profile core](./profile/core.py) and [profiler adapters](./profile/signal_profilers.py) | Rank-selection weights, QIS backtests, analysis tables and report orchestration. |
| [AlphasData](./alpha_data.py) and [diagnostics](./signal_diagnostics.py) | Supplied combined scores, optional components and adapters to QIS diagnostics. |
| [Additional backtest helpers](./backtest_alphas.py) | A separate module interface for signal/composite experiments; these helpers also use QIS execution. |

Use the owning module and its adjacent tests when making a change. Generic factor fitting and
cluster discovery belong to [FactorLasso](https://github.com/ArturSepp/FactorLasso); shared
statistics, holdings simulation and report rendering belong to QIS. The
[software design guide](../../../docs/software_design.md) describes the wider package boundaries.

Check the import level before using a name. In particular,
`signals.compute_ra_carry_alpha` returns a `(score, raw)` pair, while the legacy
`alphas.compute_ra_carry_alphas` returns only a score panel. The singular constructor is
exported by `optimalportfolios.alphas.signals`, not by its parent alpha package.
The methodology article's [signal matrix](../../../docs/alphas_module_readme.md#signal-matrix)
lists the constructor families without duplicating their formulas here.

## The profiler

[profile/core.py](./profile/core.py) builds target weights from precomputed scores and sends
each strategy and its equal-weight benchmark to `qis.backtest_model_portfolio`. It returns
`qis.MultiPortfolioData`, with the benchmark last. QIS holds units between rebalances, so
realized weights drift with prices.

| Entry point | Contract |
|---|---|
| `compute_top_quantile_equal_weights` | Rank/select on each score date; return a weight DataFrame. It does not run a backtest. |
| `backtest_alpha_rank_portfolio` | Accept one score DataFrame or a dictionary of named panels; create strategy legs plus the benchmark. |
| `profile_alpha_signals` | Accept a nonempty dictionary through `alpha_scores=`; delegate those panels to the profile core. |
| `compute_alpha_rank_analysis_table` | Return performance columns and annualized two-sided turnover for each leg. |
| `generate_alpha_profile_report` | Render QIS figures, save a PDF and return the figures. |

Higher scores are preferred. The selector checks non-missing prices and scores; it does not
fully validate positivity or finiteness. Ties follow price-column order. A fraction of `1.0`
selects all assets eligible under that score panel, which need not equal the benchmark universe.
See the methodology article's
[limitations](../../../docs/alphas_module_readme.md#interpretation-and-limitations).

Choose the price sample explicitly before profiling. In the profile core, `time_period`
filters target rows rather than ending the price history; simulated NAV can continue beyond
its end date. In `compute_alpha_rank_analysis_table`, that argument filters turnover while
performance uses the supplied NAV history. Use a consistently selected sample instead of
assuming that one argument makes every column share a reporting window.

The analysis table defaults to monthly performance statistics; its `Sharpe` column selects
the QIS zero-rate Sharpe. Cost inputs are passed through to QIS as fractional rates on traded
notional: `0.001` means 10 basis points. The
[transaction-cost contract](../../../docs/turnover_and_transaction_costs.md) and QIS
implementation define the units.

### Offline profiling workflow

Run this complete example after installing the core package. It reads the fixed
[monthly test fixture](../tests/data/multiasset.py), selects 2010–2019, computes two classic
momentum score panels with a one-month skip, and rebalances quarterly without transaction costs.
The fixture's source attribution remains incomplete in its loader; this is an integration
check, not evidence of investment performance.

```python
import qis
import optimalportfolios.alphas as alphas
from optimalportfolios.alphas import signals
from optimalportfolios.tests.data.multiasset import load_multiasset_data

data = load_multiasset_data()
prices = data.prices.loc["2010":"2019"].copy()

score_6m, _ = signals.compute_classic_momentum_alpha(
    prices=prices, returns_freq="ME", lookback_periods=6, skip_periods=1,
)
score_12m, _ = signals.compute_classic_momentum_alpha(
    prices=prices, returns_freq="ME", lookback_periods=12, skip_periods=1,
)
multi_portfolio_data = alphas.profile_alpha_signals(
    prices=prices,
    alpha_scores={"classic_6m": score_6m, "classic_12m": score_12m},
    quantile=1.0 / 3.0,
    rebalancing_freq="QE",
    rebalancing_costs=None,
)
table = alphas.compute_alpha_rank_analysis_table(
    multi_portfolio_data, perf_params=qis.PerfParams(freq="ME"),
)
print([portfolio.ticker for portfolio in multi_portfolio_data.portfolio_datas])
print(table.columns.tolist())
```

Expected structural output:

```text
['classic_6m', 'classic_12m', 'Equal Weight']
['Return p.a.', 'Vol', 'Sharpe', 'Max DD', 'Turnover p.a.']
```

The selected monthly prices are built from decimal total returns. Formation dates, warmup
and the skipped return belong to signal construction; quarterly rebalancing belongs to the
profiler. The example retains the profiler's existing QIS execution defaults. Use the
[rolling-backtest guide](../../../docs/rolling_backtests.md) when an explicit implementation
lag is needed.

### Optional PDF output

This fragment continues the example. Set `report_directory` to an existing absolute output
directory outside the checkout, with a trailing path separator. On the maintainer's Windows
host it must be in the task's C-local output area configured by [AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md).
This is an explicit file-writing step.

```python +SKIP
figures = alphas.generate_alpha_profile_report(
    multi_portfolio_data=multi_portfolio_data,
    group_data=data.group_data,
    backtest_name="Classic Momentum Profile",
    file_name="alpha_profile_report",
    local_path=report_directory,
    add_current_date=False,
)
```

With `add_current_date=False`, the PDF is named `alpha_profile_report.pdf`; the default is
to append a date. Omitting `local_path` uses the QIS default output location. Close the returned
matplotlib figures after use. Do not commit these reports or turn them into documentation
previews without the [analytics provenance process](../../../docs/documentation_standard.md).

## Per-signal profilers

[profile/signal_profilers.py](./profile/signal_profilers.py) provides
`profile_momentum`, `profile_classic_momentum`, `profile_low_beta`,
`profile_residual_momentum` and `profile_carry`. Each constructs its own score panel
and delegates to the profile core.

Momentum, low-beta and residual momentum require `benchmark_price`. Classic momentum
uses a fixed lookback and skip without a benchmark input. Carry requires a supplied annual
yield panel; prices support both volatility normalization and portfolio execution.

The joint `profile_alpha_signals` function consumes already computed panels; it does not
construct signals from `ProfileSignal` members or accept `signals=`, `carry=` or
`benchmark_price=`. The enum remains useful for labels through `ProfileSignal.CARRY.value`.
An empty `alpha_scores` dictionary raises `ValueError`.

Illustrative carry usage, continuing the offline setup above after the caller supplies a
`carry` DataFrame aligned with `prices`. Carry values are annual decimals known at their
formation dates; for example, `0.04` represents 4% per year.

```python +SKIP
carry_scores, _ = signals.compute_ra_carry_alpha(
    prices=prices, carry=carry, returns_freq="ME", vol_span=13,
)
single_carry = alphas.profile_carry(
    prices=prices, carry=carry, returns_freq="ME", vol_span=13,
    quantile=1.0 / 3.0, rebalancing_freq="QE",
)
joint = alphas.profile_alpha_signals(
    prices=prices,
    alpha_scores={
        alphas.ProfileSignal.CARRY.value: carry_scores,
        "classic_12m": score_12m,
    },
    rebalancing_freq="QE",
)
```

For custom parameters or a signal without a profiler adapter, compute its panel first and
pass it to the profile core. The
[default beta-initialization limitation](../../../docs/alphas_module_readme.md#low-beta)
also affects the low-beta and residual-momentum profiler paths; their adapters do not expose
`mean_adj_type`. Handle any alternative construction explicitly and verify its timing.

## Example

The repository's [bond-ETF profile example](../../../examples/alphas/profile_alpha_signals.py)
downloads prices and dividend histories with the `data` extra. Its `Locals` choices are
`JOINT_PROFILE`, `SINGLE_CARRY` and `QUANTILE_SWEEP`; the default joint case also writes a
PDF through the default report path and opens plots. Review that output configuration before
running it. It is a network example, distinct from the offline workflow above.

The [examples guide](../../../docs/examples_readme.md) describes its prerequisites and the
repository-only example layout. These scripts and component development runners are excluded
from distributions; use a source checkout to run them.

## Development workflow

Use the owning test directory:

| Change | Existing test location |
|---|---|
| A signal constructor or alignment rule | [signals/tests](./signals/tests/__init__.py) |
| Ranking, profiler adapters or report orchestration | [profile/tests](./profile/tests/__init__.py) |
| Containers, diagnostics or additional backtest helpers | [alphas/tests](./tests/__init__.py) |
| Published signal definitions, examples and timing | [Article contracts](../tests/alpha_signals_documentation_test.py) |

The [component runner](./signals/run_local/signals_run.py) uses `Locals` and
`run_local(local=...)` for manual diagnostics. It reads the configured local ETF CSV through
the [development data helper](../run_local/data/etf_prices.py) and displays plots. That local
resource is separate from the shipped monthly test fixture, so the runner is not an unattended
offline test.

From a source checkout, use the repository's configured interpreter and test dependencies.
On the maintainer's Windows host, follow [AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md), run from a C-local
source export and use `C:\Python\OptimalPortfolios312\Scripts\python.exe`.

```text
python tools/check_docs.py --files src/optimalportfolios/alphas/README.md
python -m pytest src/optimalportfolios/alphas/profile/tests
python -m pytest src/optimalportfolios/alphas/signals/tests
python -m pytest src/optimalportfolios/alphas/tests
python -m pytest src/optimalportfolios/tests/alpha_signals_documentation_test.py
```

The article contracts retain strict expected failures for the separately recorded default-beta
timing defect. A change to those outcomes requires review of the numerical fix and documentation.
Keep tests deterministic and offline; preserve the frozen fixture and existing seeds. Add public
imports at their owning `__init__.py`, and verify the root re-export contract when it is affected.

## References

- [Alpha methodology and verified method references](../../../docs/alphas_module_readme.md).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
