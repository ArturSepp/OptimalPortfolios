---
myst:
  html_meta:
    description: >-
      The convexity premium of portfolio overlays: a public introduction and an offline
      60/40 overlay optimization example using OptimalPortfolios and qis.
---

# The Convexity Premium of Portfolio Overlays

*Companion code: [Artur Sepp](https://github.com/ArturSepp)*

**Paper:** Sepp, A., and Kastenholz, M. (2026). *The Convexity Premium of Portfolio
Overlays*. Forthcoming in the Journal of Investment Management.

This folder introduces the paper's approach to smart diversification and runs the full
pipeline on synthetic data with [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios)
and [qis](https://github.com/ArturSepp/QuantInvestStrats). The paper's empirical results
use licensed fund and bank index data, which are not redistributed, so the manuscript and
the empirical replication are not part of this folder.

## Main concept

A diversifying overlay should be assessed by what it adds when the core portfolio
loses money, alongside its effect on the total Sharpe ratio. The paper decomposes
an overlay's arithmetic Sharpe ratio into contributions from the core portfolio's
Bear, Normal and Bull regimes. These contributions add up to the total Sharpe ratio.

The **convexity premium** is the Bear contribution beyond that implied by a Gaussian
model with the same standalone Sharpe ratio and benchmark correlation. It separates
defensive behavior attributable to correlation from additional tail behavior.
Smart-diversification curves show how adding an overlay changes both the total
Sharpe ratio and its Bear contribution.

The allocation step keeps the core exposure fixed and chooses overlay weights to
maximize the stacked portfolio's Sharpe ratio. A coverage floor specifies how much
of the core's average Bear-regime loss the overlay sleeve must offset. This is a
constraint on a sample return contribution, not a guarantee for an individual crisis.

The implementations and conventions are documented in:

- [qis: The convexity premium and smart diversification](https://quantinveststrats.readthedocs.io/en/latest/convexity_premium.html),
  including `qis.regimes` and `qis.SmartDiversificationReport`.
- [qis: Regime-conditional performance](https://quantinveststrats.readthedocs.io/en/latest/regime_conditional_performance.html),
  including additive Sharpe contributions and arithmetic versus per-annum conventions.
- [OP: Fixed-core overlays and linear tail floors](https://optimalportfolios.readthedocs.io/en/latest/overlay_tail_floor.html),
  section *The coverage floor*, using `LinearConstraints`, `Constraints` and
  `cvx_maximize_portfolio_sharpe`.

## Paper-to-code map

Section, equation and exhibit numbers refer to the accepted manuscript.

| Paper | Implementation in this example |
|---|---|
| Definition 1: Bear, Normal and Bull regimes at the core's 16% and 84% quantiles | `qis.regimes.create_sampled_returns_with_regime_id` with `q=(0, 0.16, 0.84, 1)` |
| Proposition 1 and equation (1): regime Sharpe contributions; Definition 2 and equation (4): the convexity premium; equation (6): the benchmark-adjusted premium | `qis.regimes.compute_regime_premium_table`, columns `bear_sharpe`, `convexity_premium` and `cp_star` |
| Definition 4: smart diversifiers, and the smart-diversification curves | `qis.SmartDiversificationReport.plot_smart_diversification_curve` |
| Equation (9): regime betas and idiosyncratic volatility | `qis.regimes.compute_regime_betas` |
| Equation (10) and Appendix B: the regime-mixture covariance | `qis.regimes.compute_regime_mixture_covar_from_sample` |
| Section II, program (11): maximum Sharpe ratio with a fixed core and a Bear-regime coverage floor | `solve_overlay`: an `optimalportfolios.LinearConstraints` row `bear_coverage` in `Constraints`, solved by `optimalportfolios.cvx_maximize_portfolio_sharpe` |
| Table 2: the estimation inventory | `OverlayInputs.input_sheet`, exported as `allocator_input_sheet.csv` |
| Table 4: allocations along the coverage-floor frontier | `OverlayResult.table` of `run_coverage_floor_frontier` |
| Figure 4: the overlay allocation frontier | `qis.plot_overlay_allocation_frontier` |

## Run the 60/40 example

The runnable source is [run_overlay_example.py](run_overlay_example.py).
It creates 360 months of synthetic simple excess returns with seed 17. The core holds
60% equities and 40% bonds, rebalanced monthly. Equity and bond annual excess-return
assumptions are 6.5% and 2.0%, annual volatilities are 18% and 7%, and their generating
correlation is -0.15. The four stylized funds use these annual assumptions:

| Synthetic overlay | Excess return | Volatility | Payoff shape |
|---|---:|---:|---|
| Trend | 4.5% | 10% | Convex exposure to large core moves plus independent noise |
| Equity L/S | 5.5% | 12% | Positive core exposure plus independent noise |
| Market neutral | 3.5% | 7% | Small core exposure, mostly independent noise |
| Tail hedge | -4.0% | 12% | Negative core exposure and extra downside convexity |

These are illustrative parameters and simulated fund observations, not actual fund
histories. The source exports the generated data and assumptions with its results.
Edit `ExampleParams` and `create_example_data` to explore other assumptions.

### Select a workflow

The example requires optimalportfolios 7.10.0 and qis 5.33.1 or newer, and was last
tested with exactly those versions. It reads two column-name constants from their
defining qis modules, `REGIME_COLUMN` and `EFFECTIVE_CAPITAL_COUNT`, which qis does not
re-export; a later qis release may move them. Use your OP Python environment with these
packages installed. The paper example is a repository file; it is not included in the
installed package.

```console
python -m pip install "optimalportfolios>=7.10.0" "qis>=5.33.1"
```

Open [run_overlay_example.py](run_overlay_example.py) and select one `Locals` mode
in the call under `__main__` at the bottom. For example, to run the coverage frontier:

<!-- fragment -->
```python
if __name__ == '__main__':
    run_local(local=Locals.RUN_COVERAGE_FLOOR_FRONTIER)
```

| `Locals` mode | Workflow |
|---|---|
| `ESTIMATE_INPUTS` | Runs the Table 2 estimation pipeline through qis and exports the numerical input sheet, benchmark regime moments and covariance; does not optimise |
| `RUN_SINGLE_OPTIMISATION` | Estimates inputs, solves one allocation at the chosen coverage floor and draws the qis illustrations |
| `RUN_COVERAGE_FLOOR_FRONTIER` | Estimates inputs once, solves the theta grid and prints a Table 4-style comparison with benchmark, equal-weight and no-floor references |

Each mode is self-contained: you can run the single case or frontier without first
running `ESTIMATE_INPUTS`. Mode selection uses the source code, with no CLI flags.

### Set parameters and run

Edit these settings at the start of `run_local`:

<!-- fragment -->
```python
params = ExampleParams(coverage=0.40, overlay_budget=1.0)
theta_grid = DEFAULT_THETA_GRID
```

- `coverage=0.40` requests 40% coverage of the core's average Bear-regime loss for
  the single optimisation.
- `overlay_budget=1.0` allocates 100% of capital to overlays on top of the core's
  fixed 100% exposure. It applies to both optimisation modes.
- `theta_grid` controls the frontier's coverage floors instead of `params.coverage`.
  Its default is `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`; replace it with your chosen grid.
  An infeasible floor raises an error.

Run the file in your IDE with the OP interpreter, or run this from the repository root:

```console
python -m papers.smart_diversification_joim_2026.run_overlay_example
```

From this paper folder, the equivalent command is `python run_overlay_example.py`.
To change modes, edit the call under `__main__` and run again.

### Find and interpret the results

The script prints its output directory. It uses OP's
`optimalportfolios.local_path.get_output_path()` and adds
`joim-overlay-example/<mode name in lowercase>/`. Before the first run, set
`OUTPUT_PATH` in [settings.yaml](../../src/optimalportfolios/settings.yaml) to an absolute
directory outside the repository and outside any synchronised folder. With the shipped
placeholder the default can be the current working directory, so a run from this paper
folder would write its outputs into the checkout. Rerunning a mode overwrites its exported
files.

| Mode | Main CSV output | Figures |
|---|---|---|
| `ESTIMATE_INPUTS` | `allocator_input_sheet.csv`, `regime_moments.csv`, `regime_covariance.csv` | None |
| `RUN_SINGLE_OPTIMISATION` | `single_optimisation.csv`, `single_optimisation_weights.csv`, `single_optimisation_statistics.csv` | Smart diversification, regime contributions, overlay weights and the Figure 4 allocation view |
| `RUN_COVERAGE_FLOOR_FRONTIER` | `coverage_floor_frontier.csv`, `coverage_floor_frontier_weights.csv`, `coverage_floor_frontier_statistics.csv` | Coverage frontier, allocations by floor and the Figure 4 allocation view |

Both optimisation modes save `overlay_allocation_frontier.png` using
`qis.plot_overlay_allocation_frontier`, the same plotting function as the paper's
Figure 4. Each fund point represents the fixed core plus that fund at the selected
overlay budget. The frontier mode adds the solved coverage-floor curve, equal weight,
the no-floor optimum and the allocation at `params.coverage` when that floor is in
the grid. The single-case mode shows its one selected allocation without solving an
additional frontier. All coordinates use realised arithmetic excess Sharpe contributions
from the same sample and core regimes. The supplied points and frontier statistics are
exported as `overlay_allocation_points.csv` and `overlay_allocation_frontier.csv`.

Every mode also exports the simulated returns, assumptions and estimation inputs.
`allocator_input_sheet.csv` retains full precision in decimal annual units;
`allocator_input_sheet_display.csv` is rounded for presentation. Allocation rows in
the single-case and frontier tables are percentages of the overlay budget. Their
volatility, Bear contribution and realised coverage rows are percentages; Sharpe
contributions and effective number of overlays are unscaled. The separate weights
CSVs contain raw exposures relative to capital, including the fixed core.

For interactive reuse, `estimate_inputs(panel)` returns a frozen `OverlayInputs`
dataclass. Pass that same object to `run_single_optimisation(inputs, coverage=0.40)`
or `run_coverage_floor_frontier(inputs, theta_grid=[0.0, 0.4, 0.8])` to change the
allocation policy without re-estimating. Use `result.weights`, `result.statistics`
and `result.table` to inspect either result; these calculation functions return
data without exporting files or drawing figures.

## What the example does

1. Uses qis to classify monthly returns into the 16% / 68% / 16% regimes and estimate
   arithmetic excess Sharpe contributions, convexity premiums and regime betas.
2. Calls `qis.regimes.compute_regime_mixture_covar_from_sample` for an annual covariance
   with empirical regime probabilities and diagonal residual risk, and estimates annual
   expected excess returns from the same complete sample.
3. Solves the maximum-Sharpe allocation with OP. The core stays at 100% of capital and
   long-only overlays sum to the overlay budget. Weights represent exposures on one capital
   base and must not be normalized to sum to one. The coverage floor is a named signed
   `LinearConstraints` row, and OP handles its solver transformation.
4. Prints the allocations in the layout of the paper's allocation table. The single
   optimisation draws `smart_diversification.png`, `regime_contributions.png` and
   `overlay_weights.png`, and the frontier draws `coverage_floor_frontier.png` and
   `coverage_floor_weights.png`. The smart-diversification curves use monthly
   rebalancing, an arithmetic Sharpe convention and a fixed core.

CSV exports contain the simulated returns, assumptions, regime statistics, model
covariance, weights and stacked-portfolio statistics.

All estimates and illustrations use the full synthetic sample. They are descriptive
and do not establish out-of-sample performance. Returns are already excess of cash.
Funding spreads, transaction costs and margin requirements are not modeled.
The payoff shapes illustrate exposures rather than executable trading strategies.

The public checks run without private inputs or network access, from the repository root:

```console
python -m pytest papers/smart_diversification_joim_2026/replication/tests/overlay_example_test.py -q
```

## Reproducing the paper

The paper's figures and tables use net-of-fee returns of fifteen funds and bank QIS
indices under commercial licenses. These data and the empirical replication code are not
distributed with this repository. The synthetic example calls the same qis and
optimalportfolios functions on simulated data.

## References

- Sepp, A., and Kastenholz, M. (2026). *The Convexity Premium of Portfolio Overlays*.
  Forthcoming in the Journal of Investment Management.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
