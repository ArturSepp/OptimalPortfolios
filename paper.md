---
title: 'optimalportfolios: point-in-time multi-asset portfolio construction and rolling backtesting in Python'
tags:
  - Python
  - quantitative finance
  - portfolio optimization
  - asset allocation
  - rolling backtesting
  - covariance estimation
authors:
  - name: Artur Sepp
    orcid: 0000-0002-7038-1748
    email: artursepp@gmail.com
    corresponding: true
    affiliation: 1
affiliations:
  - name: LGT Bank
    index: 1
date: 21 August 2026
bibliography: paper.bib
---

# Summary

`optimalportfolios` is an open-source Python package for deciding how to divide capital across
assets and for testing those decisions over time. Written for researchers and practitioners who
construct and evaluate multi-asset portfolios, it connects dated covariance and alpha estimates to
constrained portfolio decisions, rolling rebalancing, transaction costs, and performance reports. The package
supports strategic and tactical asset allocation, risk budgeting, minimum variance, maximum
diversification, maximum Sharpe ratio, and alpha-versus-tracking-error objectives.

The central object is not an isolated optimizer call but a sequence of decisions. At each date, the
software combines the information then available with the current holdings, eligibility rules,
asset-specific dealing schedules, and portfolio constraints. It produces target weights that can be
passed to a unit-based portfolio simulation. This structure allows one workflow to represent liquid
and illiquid assets, incomplete histories, and different return-estimation or rebalancing frequencies
without replacing missing data with invented observations.

# Statement of need

Portfolio optimization begins with the allocation problem introduced by Markowitz
[@markowitz1952], but a research backtest requires more than a covariance matrix and an objective.
Estimates must be aligned to decision dates; weights chosen at a date must be implemented only over
the subsequent holding period; and transaction costs must be charged on the trade from actual
pre-trade holdings. A sequence of mathematically valid single-date solutions can therefore form an
invalid experiment if it uses future data, resets holdings at every rebalance, or silently changes
the eligible universe.

These problems are acute in heterogeneous multi-asset panels. A liquid equity index may supply daily
prices and rebalance monthly, while a private or alternative asset may be observed and traded only
quarterly. Assets can enter after the start of a sample, have interior gaps, or be temporarily frozen.
Constraints on asset weights, groups, turnover, tracking error, return, and volatility must remain
coherent when the eligible set changes. Hand-built research scripts commonly distribute these rules
between data cleaning, optimizer setup, and backtest code, making it difficult to determine which
information was available and which holdings were tradable at a particular date.

The cost of the missing state is measurable rather than notional. On the wheel-shipped fixture —
19 asset-class indices, quarterly rebalancing, 95 rebalances — a minimum-variance backtest with a
hard 3% per-rebalance turnover budget behaves differently under the two conventions. With turnover
measured against the previous target weights, the optimizer reports compliance at every rebalance
while the executed trades breach the budget at 71% of rebalances, by up to 2.4 times the budget;
measured against drifted holdings, the package default, executed trades never exceed it. The
installed test suite pins these numbers, with executed turnover verified independently against the
`qis` simulator.

`optimalportfolios` makes that state explicit. Covariance and alpha inputs are dated, mixed-frequency
estimation is part of the public interface, previous targets drift with realized prices before the
next decision, and rebalancing indicators can hold an asset fixed while the liquid portion is
re-optimized. The same constraint objects pass through numerical, data-aware, and rolling layers.
Researchers can therefore compare objectives while reusing identical estimates and implementation
assumptions. An offline quickstart and wheel-shipped fixture provide a complete first experiment
without network or proprietary data.

# State of the field

Several mature Python projects address portfolio construction, but their documented design centers
differ. PyPortfolioOpt presents a compact interface to classical expected-return, covariance,
efficient-frontier, Black--Litterman, downside-risk, and hierarchical allocation workflows
[@martin2021]. Riskfolio-Lib develops a broad catalog of convex risk measures, risk-parity,
factor, hierarchical, network-constrained, and reporting models [@cajas2026riskfolio]. skfolio makes
portfolio estimators participate in the scikit-learn fit/predict, pipeline, model-selection, and
cross-validation ecosystem [@nicolini2025skfolio]. cvxportfolio implements single- and
multi-period trading policies with transaction- and holding-cost models, executed against its own
market simulator [@boyd2017multiperiod; @busseti2026cvxportfolio]. These are complementary
abstractions rather than a quality ranking.

The specialization of `optimalportfolios` is the transition between successive multi-asset
decisions: dated estimates become constrained targets; prior targets become drifted current
holdings; eligibility and dealing windows determine which positions can change; and the implemented
portfolio incurs costs on the resulting trades. cvxportfolio is the closest design center, and the
boundary is drawn there: it optimizes a trading policy from return forecasts at a single frequency
against its own simulator, while `optimalportfolios` centers on heterogeneous panels — mixed
estimation and dealing cadences, incomplete histories, frozen positions, group and tracking-error
constraints, risk-budgeting and allocation objectives — and delegates simulation and reporting to
`qis`. Contributing one more optimizer objective to an existing project would not supply that
top-level state model or the package's integration boundaries.
The versioned public comparison documents adjacent capabilities and records unassessed cells rather
than interpreting missing documentation as an absent feature.

# Software design

The implementation separates three layers. The mathematical layer takes clean arrays and formulates
one optimization problem. The wrapper layer aligns labeled pandas objects, filters unavailable
assets, transforms constraints to the valid universe, restores zero weights for excluded assets, and
validates the result. The rolling layer selects point-in-time inputs, drifts prior holdings, applies
date-specific eligibility or rebalancing indicators, and dispatches the same single-date solver at
each decision. This separation lets numerical tests exercise the optimizer independently while
rolling tests check time alignment and state transitions.

Verification follows the same boundaries. Closed-form and independently computed cases test solver
semantics; rolling tests assert decision-to-holding timing, frozen-position behavior, and changing
universes; and an artifact test installs the built wheel in a clean environment before running the
shipped offline suite. Cross-platform automation covers the supported Python versions on Linux,
macOS, and Windows, so package discovery and path behavior are checked alongside numerical results.

A shared `Constraints` data structure represents bounds on instruments and groups, gross and net
exposure, tracking error, turnover, return and volatility targets, benchmark weights, and frozen
positions. Keeping these semantics outside individual objectives prevents each solver from inventing
its own response to missing assets or dealing restrictions. Optimization problems are expressed with
CVXPY [@diamond2016cvxpy], with `quadprog` retained for direct quadratic-program formulations. A
small backend surface limits duplicated constraint translations and numerical diagnostics.

The package composes two other open-source projects instead of copying them. FactorLasso owns generic
sparse multi-output factor estimation, including the Hierarchical Clustering Group LASSO components
used for factor covariance models [@sepp2026factorlasso]. `optimalportfolios` adds finance-specific
frequencies, estimation schedules, annualization, and portfolio inputs. `qis` owns the unit-based
portfolio simulator, performance analytics, and reporting layer [@sepp2026qis]. It receives dated
target weights, implementation assumptions, and costs from `optimalportfolios`. This boundary keeps
one convention for holdings simulation and one implementation of generic factor estimation across
the author's open-source stack.

Network data clients and optional report renderers remain outside the core installation. The
numerical package and its installed-wheel tests therefore run offline, while `data` and `reports`
extras enable those integrations when requested. Non-quadratic risk measures such as conditional
value at risk, mean absolute deviation, and drawdown constraints, generic hyperparameter-search
frameworks, broker execution, and proprietary orchestration are intentionally outside scope.

# Research impact statement

The software has supported three research programs. OptimalPortfolios 1.0.x was created as
replication code for the cryptocurrency-allocation study published in *Risk* [@sepp2023crypto]. The
public package was subsequently used during the early-to-late-2025 research cycle for the Robust
Optimization of Strategic and Tactical Asset Allocation (ROSAA) framework, published in *The
Journal of Portfolio Management* [@sepp2026rosaa]. The ongoing MATF-CMA research on factor-consistent
capital-market assumptions uses both OptimalPortfolios and FactorLasso [@sepp2026matf].

The repository includes public methodological examples associated with the *Risk* and ROSAA papers.
They are not represented as exact exhibit rebuilds: historical environments are not fully recorded,
some market inputs are live or licensed, and no unrecorded release tag is inferred after the fact.
External engagement is recorded in the repository: in 2026, an independent engineer's audit of the
package produced 15 issues and 14 pull requests, each merged or closed with a recorded
disposition, and further public contributions corrected executable examples, expanded scientific
validation, identified a factor-covariance adapter defect, and strengthened cross-platform packaging
and documentation checks.

# AI usage disclosure

Generative AI tools assisted parts of the software's maintenance and this paper. Anthropic's
Claude (2026 models), through the Claude Code agentic interface, assisted implementation and
refactoring, tests, continuous integration and packaging, documentation, and research artifacts in
2026. OpenAI Codex (GPT-5 family; the exact deployed version was not retained in project metadata)
assisted repository audits, evidence collection, documentation, verification tooling, and paper
drafting. GitHub Copilot supplied review suggestions on some pull requests; its model version was not
recorded. The human author selected the research methods and architecture, reviewed and edited all
AI-assisted output, validated numerical work against analytic or independent references where
applicable, ran the automated software and documentation gates, and takes full responsibility for
the software and manuscript.

# Acknowledgements

The author thanks Thomas Schmelzer for substantial contributions to software validation, numerical
defect discovery, executable documentation, dependency and packaging design, cross-platform
continuous integration, wheel verification, and the independent JOSS-readiness audit. The author
thanks Mikko Ohtamaa and GitHub user ZhengGong-hub for public corrections and constraint
functionality, and Ivan
Ossa, Mika Kastenholz, and Emilie H. Hansen for the research collaborations that exercised and
informed the software.

This work received no dedicated external funding. The author's affiliation provided no sponsor role
in the software design, analysis, manuscript preparation, or decision to submit. The author declares
no competing interests.

# References
