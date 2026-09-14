---
myst:
  html_meta:
    description: >-
      Compare portfolio-library workflows, estimation, constraints, evaluation and reporting,
      with a dated release snapshot and explicit documentation-evidence limits.
---

# Choosing a portfolio optimization library

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Portfolio optimization libraries connect estimates and constraints to allocations, trading
decisions and evaluation. OptimalPortfolios, PyPortfolioOpt, Riskfolio-Lib, skfolio and
cvxportfolio cover overlapping parts of that workflow. This guide compares their documented
interfaces and the work a caller must supply.

Start with [Choose by workflow](#choose-by-workflow), then use the capability tables to check
the assumptions that matter for your application. The comparison provides no performance ranking.

## Version snapshot

The original comparison recorded the following release snapshot on **21 August 2026**.
These version numbers are retained as historical context; they are not a list of today's
latest releases.

| Package | Version in the original snapshot | Release record |
|---|---|---|
| OptimalPortfolios | 6.21.2 | [PyPI](https://pypi.org/project/optimalportfolios/6.21.2/) |
| PyPortfolioOpt | 1.6.0 | [PyPI](https://pypi.org/project/pyportfolioopt/1.6.0/) |
| Riskfolio-Lib | 7.3.0 | [PyPI](https://pypi.org/project/riskfolio-lib/7.3.0/) |
| skfolio | 0.20.2 | [PyPI](https://pypi.org/project/skfolio/0.20.2/) |
| cvxportfolio | 1.5.1 | [PyPI](https://pypi.org/project/cvxportfolio/1.5.1/) |

The release records and linked capability documentation were reviewed again on
**14 September 2026**. This confirms the existence of those releases and the documented
features described below; it does not recreate the original review's "latest stable" status.

The OptimalPortfolios table describes the working source declaring **7.6.0**, as recorded in
[packaging metadata](../pyproject.toml), with qualifications from the current methodology
guides. Peer-package tables describe the linked official documentation reviewed on the later
date. They do not certify that every documented feature exists in the historical release above.
See [How this comparison was made](#how-this-comparison-was-made) for documentation-version mismatches.

## Capability matrix

The original twelve comparison categories are retained in one table per package so that the
descriptions and citations remain readable at article width. cvxportfolio retains its separate
workflow-level discussion.

**Not assessed** identifies a limit of this review. It is not a statement that the package lacks
the capability, or that a caller cannot assemble it from available components. A shared feature
name also does not establish identical units, constraints, timing or execution semantics.

### OptimalPortfolios

| Capability | Implementation and qualification |
|---|---|
| Single-date optimization | Labeled wrappers and numerical solvers have distinct validation and fallback contracts. [Optimization guide](optimization_module_readme.md). |
| Rolling point-in-time evaluation | Dated estimates feed rolling target construction, followed by QIS execution. Information cutoffs and estimator initialization still require review. [Rolling backtests](rolling_backtests.md). |
| Expected-return and risk estimation | Alpha signals, annualized EWMA covariance and factor covariance through FactorLasso. [Signals](alphas_module_readme.md), [covariance estimators](covariance_estimators.md). |
| Risk-measure breadth | Variance, diversification, Sharpe, risk budgeting, benchmark-relative and CARA-mixture routes; dispatcher coverage differs from the full public surface. [Optimization guide](optimization_module_readme.md). |
| Black-Litterman | **Not assessed** as a dedicated documented workflow. [API entry](api.rst). |
| Tracking error | Single-date and rolling minimum covariance tracking error; benchmark alignment and other benchmark-relative routes are documented separately. [Tracking error](minimum_tracking_error.md). |
| Turnover and transaction costs | Selected solvers use drifted prior targets as decision references; QIS charges costs on simulated trades. [Turnover and costs](turnover_and_transaction_costs.md). |
| Mixed-frequency data | Asset-return, factor and decision cadences can differ. Eligibility/dealing inputs and supported options depend on the entry point. [Mixed-frequency data](mixed_frequency_data.md). |
| Incomplete histories | Late starts, gaps, eligibility, frozen positions and unpriced trades have different contracts. [Incomplete histories](incomplete_histories.md). |
| Model selection and cross-validation | Rolling evaluation is documented; a generic hyperparameter cross-validation framework is **not assessed**. [Rolling backtests](rolling_backtests.md). |
| Reporting | The convenience backtest returns `qis.PortfolioData`; QIS owns simulation analytics and reports. [Software design](software_design.md). |
| Primary design emphasis | Dated multi-asset construction, shared constraints and explicit delegation to FactorLasso and QIS. [Architecture](software_design.md). |

OptimalPortfolios supplies the financial input preparation and construction layer.
[FactorLasso](https://github.com/ArturSepp/FactorLasso) supplies generic factor estimation;
[QIS](https://github.com/ArturSepp/QuantInvestStrats) supplies analytics, unit-based simulation
and reporting. Prior targets drift into a decision-date allocation reference. The convenience
backtest does not automatically feed executed QIS holdings back into each optimizer decision.

### PyPortfolioOpt

| Capability | Documented scope and primary source |
|---|---|
| Single-date optimization | Efficient-frontier objects with objectives and constraints. [Mean-variance guide][ppo-mv]. |
| Rolling point-in-time evaluation | **Not assessed** as an integrated execution engine. The guide covers allocation construction and reusable inputs. [User guide][ppo-user]. |
| Expected-return and risk estimation | Historical, exponential and CAPM means; sample, semi-, exponential and shrinkage covariance. [Returns][ppo-returns], [risk models][ppo-risk]. |
| Risk-measure breadth | Mean-variance, semivariance, conditional value at risk (CVaR) and conditional drawdown at risk (CDaR). [Efficient-frontier families][ppo-frontier]. |
| Black-Litterman | Priors and views produce posterior return/covariance inputs. [Black-Litterman guide][ppo-bl]. |
| Tracking error | Ex-ante and ex-post objective functions. [Objective functions][ppo-objectives]. |
| Turnover and transaction costs | Proportional `transaction_cost` objective using previous weights. [Objective functions][ppo-objectives]. |
| Mixed-frequency data | **Not assessed** for per-asset estimation and dealing calendars. A `frequency` parameter alone does not establish that workflow. [Returns][ppo-returns]. |
| Incomplete histories | **Not assessed** for an integrated eligibility/frozen-position execution contract. [Returns][ppo-returns]. |
| Model selection and cross-validation | **Not assessed** as a generic selection framework. [User guide][ppo-user]. |
| Reporting | `portfolio_performance` and plotting functions describe optimizer output. [Performance][ppo-mv], [plotting][ppo-plots]. |
| Primary design emphasis | Modular estimates, allocation objectives and weight post-processing. [User guide][ppo-user]. |

### Riskfolio-Lib

| Capability | Documented scope and primary source |
|---|---|
| Single-date optimization | `Portfolio` exposes mean-risk, risk-parity and factor formulations. [Portfolio API][rf-portfolio]. |
| Rolling point-in-time evaluation | Examples connect weights to Backtrader/vectorbt; the Backtrader example carries a compatibility warning. [Backtesting examples][rf-backtesting]. |
| Expected-return and risk estimation | Historical, Black-Litterman and factor-model parameterization. [Portfolio API][rf-portfolio]. |
| Risk-measure breadth | The convex portfolio guide lists 24 risk measures, including tail and drawdown families. [Portfolio API][rf-portfolio]. |
| Black-Litterman | Historical, factor, Bayesian and augmented variants appear in the example catalogue. [Examples][rf-examples]. |
| Tracking error | Benchmark-relative constraints through `allowTE` and `TE`. [Portfolio API][rf-portfolio]. |
| Turnover and transaction costs | `allowTO` and `turnover` constrain allocation changes; execution examples use external engines. [Portfolio API][rf-portfolio], [backtesting][rf-backtesting]. |
| Mixed-frequency data | **Not assessed** for per-asset estimation/dealing calendars. [Portfolio API][rf-portfolio]. |
| Incomplete histories | **Not assessed** for a complete eligibility/frozen-position execution contract. [Portfolio API][rf-portfolio]. |
| Model selection and cross-validation | **Not assessed** as a generic selection framework. [Examples][rf-examples]. |
| Reporting | Jupyter and Excel reporting functions. [Reports][rf-reports]. |
| Primary design emphasis | Exploration across risk measures and portfolio families, including hierarchical and factor models. [Examples][rf-examples]. |

### skfolio

| Capability | Documented scope and primary source |
|---|---|
| Single-date optimization | Estimators expose `fit` and `predict`. [Optimization guide][sk-opt]. |
| Rolling point-in-time evaluation | `WalkForward` and `cross_val_predict` generate out-of-sample portfolio results. Sequential prior-weight propagation is documented. [Model selection][sk-selection], [MeanRisk API][sk-meanrisk]. |
| Expected-return and risk estimation | Composable mean, covariance, prior and factor estimators. [Prior models][sk-prior]. |
| Risk-measure breadth | Variance, downside, tail and drawdown measures, with availability depending on the optimizer. [Optimization guide][sk-opt]. |
| Black-Litterman | A composable `BlackLitterman` prior estimator. [Prior models][sk-prior]. |
| Tracking error | Return-based constraints, target-weight formulations and `BenchmarkTracker`. [Tracking-error guide][sk-te]. |
| Turnover and transaction costs | Linear costs, previous weights and maximum turnover; `weight_drift` controls drift during portfolio evaluation. [MeanRisk API][sk-meanrisk], [portfolio guide][sk-portfolio]. |
| Mixed-frequency data | **Not assessed** for a full per-asset estimation/dealing-calendar contract. [Preprocessing API][sk-prices]. |
| Incomplete histories | Inception-NaN handling, missing-row thresholds and optional forward filling are documented; frozen-position execution is **not assessed** here. [Preprocessing API][sk-prices]. |
| Model selection and cross-validation | Scikit-learn selection, walk-forward, purged combinatorial and randomized CV, plus online evaluation. [Model selection][sk-selection]. |
| Reporting | `Portfolio` and `MultiPeriodPortfolio` summaries, measures, composition and plots. [Portfolio guide][sk-portfolio]. |
| Primary design emphasis | Portfolio estimators composed with model-selection and evaluation tools. [Model selection][sk-selection]. |

## Choose by workflow

These are starting points inferred from the documented interfaces, not exclusive feature
assignments. Follow the linked capability sections before selecting a package.

| Main requirement | Starting point and reason |
|---|---|
| Dated multi-asset construction with QIS evaluation | [OptimalPortfolios](#optimalportfolios): examine estimator timing, constraint support and execution assumptions for the selected route. |
| Classical allocation with modular estimates and post-processing | [PyPortfolioOpt](#pyportfolioopt): efficient-frontier and Black-Litterman workflows with direct allocation objects. |
| Exploration across risk measures and portfolio families | [Riskfolio-Lib](#riskfolio-lib): broad formulation and reporting catalogue. |
| Scikit-learn pipelines and out-of-sample model selection | [skfolio](#skfolio): composable estimators and cross-validation/evaluation tools. |
| Trading policies that optimize forecast returns, risk and costs over a planning horizon | [cvxportfolio](#cvxportfolio): single- and multi-period policies evaluated by its market simulator. |

Before comparing outputs, align the data cutoff, return convention, covariance annualization,
benchmark, constraints, transaction-cost definition, holdings drift and implementation timing.
An objective penalty, a turnover constraint and a cash deduction during simulation measure
different parts of the process. Out-of-sample folds also need an explicit holdings and execution
contract; a common label does not make two result series interchangeable.

For an executable OptimalPortfolios starting point, use the [offline quickstart](quickstart.md)
and [example catalogue](examples_readme.md). Peer examples linked here were not executed in this review.

## cvxportfolio

[cvxportfolio](https://github.com/cvxgrp/cvxportfolio) implements the trading framework described
by Boyd, Busseti, Diamond, Kahn, Koh, Nystrup and Speth (2017),
[*Multi-Period Trading via Convex Optimization*](https://stanford.edu/~boyd/papers/cvx_portfolio.html).
Its [official documentation](https://www.cvxportfolio.com/en/1.5.0/) describes single- and
multi-period policies, transaction and holding costs, and a market simulator.

The multi-period formulation plans a sequence of trades and executes the first decision before
replanning with updated information. This is a different abstraction from merely repeating a
single-date allocation over a historical sample. Forecast quality and information availability
remain inputs to the experiment. [Original paper](https://stanford.edu/~boyd/papers/cvx_portfolio.html).

The architectural comparison is that cvxportfolio exposes policy optimization and simulation
within its own framework, while OptimalPortfolios builds dated targets and delegates execution
and reporting to QIS. Inspect [OptimalPortfolios' state boundary](software_design.md#holdings-as-state)
when deciding how current holdings enter a construction decision. This comparison does not
establish that either package is restricted to one asset class or that their execution models
are equivalent. cvxportfolio remains outside the twelve-category tables.

## How this comparison was made

The original page recorded a manual documentation review and PyPI JSON lookup on
21 August 2026. The September revision preserves its five release identifiers and twelve
comparison categories, checks the release-specific PyPI records again, and reviews the
official documentation linked beside the retained claims.

There are three different kinds of evidence:

1. **Release identity:** a version-specific PyPI record establishes that a release exists.
   It does not establish which version a mutable documentation page describes.
2. **Documented capability:** official guides and API references support the descriptions here.
   A documentation review does not prove numerical correctness, completeness or version parity.
3. **Local implementation:** the OptimalPortfolios source and its linked methodology guides
   describe the working implementation. This is separate from the historical 6.21.2 release.

On 14 September 2026, the [PyPortfolioOpt guide][ppo-mv] still displayed **1.5.4** in its
documentation title, while the retained release record is **1.6.0**. The
[cvxportfolio documentation homepage](https://www.cvxportfolio.com/) redirected to **1.5.0**
documentation, while its retained release is **1.5.1**. The page does not infer undocumented
release changes from either mismatch. Unversioned skfolio and Riskfolio-Lib `latest` links
can also change independently of the snapshot.

Peer packages were not installed or executed. No speed, numerical-quality, dependency-size,
popularity or investment-performance benchmark was performed. The review's unresolved categories
remain **not assessed**. For a later adoption decision, inspect the versions and exact workflows
you intend to run and use a controlled example with matching conventions.

## See also

- [Software design and package boundaries](software_design.md)
- [Optimization interfaces and fallback contracts](optimization_module_readme.md)
- [Mixed-frequency data](mixed_frequency_data.md) and [incomplete histories](incomplete_histories.md)
- [Rolling backtests](rolling_backtests.md) and [turnover and costs](turnover_and_transaction_costs.md)

## References

Primary capability references are linked beside the relevant statements. The dated release
records appear in [Version snapshot](#version-snapshot).

- Boyd, S., Busseti, E., Diamond, S., Kahn, R., Koh, K., Nystrup, P., and Speth, J. (2017).
  [Multi-Period Trading via Convex Optimization](https://stanford.edu/~boyd/papers/cvx_portfolio.html).
  *Foundations and Trends in Optimization*, 3(1), 1–76.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff)
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)

[ppo-mv]: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html
[ppo-user]: https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html
[ppo-returns]: https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html
[ppo-risk]: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html
[ppo-frontier]: https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html
[ppo-bl]: https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html
[ppo-objectives]: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#module-pypfopt.objective_functions
[ppo-plots]: https://pyportfolioopt.readthedocs.io/en/latest/Plotting.html
[rf-portfolio]: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html
[rf-examples]: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/examples.html
[rf-backtesting]: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/examples.html#backtesting
[rf-reports]: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/reports.html
[sk-opt]: https://skfolio.org/user_guide/optimization.html
[sk-selection]: https://skfolio.org/user_guide/model_selection.html
[sk-prior]: https://skfolio.org/user_guide/prior.html
[sk-meanrisk]: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html
[sk-portfolio]: https://skfolio.org/user_guide/portfolio.html
[sk-prices]: https://skfolio.org/generated/skfolio.preprocessing.prices_to_returns.html
[sk-te]: https://skfolio.org/user_guide/optimization.html#tracking-error-optimization
