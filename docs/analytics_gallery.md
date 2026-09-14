---
myst:
  html_meta:
    description: >-
      Reproducible OptimalPortfolios analytics: synthetic portfolio performance,
      allocation, trading costs, covariance spans, objectives and factor estimators.
---

# Analytics gallery

*[author / affiliation / date — placeholder]*

Examples from [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

These six teaching exhibits connect portfolio construction to risk and performance analytics.
OptimalPortfolios estimates risk and constructs targets,
[qis](https://github.com/ArturSepp/QuantInvestStrats) simulates holdings and computes analytics,
and [factorlasso](https://github.com/ArturSepp/FactorLasso) fits the sparse factor models.

All inputs are synthetic. The figures illustrate calculations and implementation conventions;
they do not establish historical performance or recommend an allocation or estimator.
The [shared provenance record](../examples/figures/analytics_manifest.json) identifies the
effective source, inputs, parameters, actual software environment, generation time and visual
review. A fixed sample ending in 2025 is separate from the date the image was generated.

## Choose an exhibit

| Question | Exhibit |
|---|---|
| How do growth and drawdowns compare with a synthetic benchmark? | [Performance](#portfolio-performance) |
| What are the final targets and their estimated risk contributions? | [Allocation and risk](#allocation-and-risk) |
| How do decided allocations and realized trading costs evolve? | [Allocation through time](#allocation-through-time) |
| How does covariance smoothing affect this example? | [Span sensitivity](#covariance-span-sensitivity) |
| How do three objectives behave with common risk inputs? | [Objective comparison](#portfolio-objectives) |
| How do estimates compare with a known simulated covariance? | [Covariance estimators](#covariance-estimators) |

Select a preview to open its full-resolution image. The
[offline quickstart](quickstart.md) provides a smaller executable introduction, and the
[examples guide](examples_readme.md) maps broader workflows and their data requirements.

## Samples and conventions

The first five exhibits use six assets from the fixed
[qis synthetic-universe generator](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/datasets/synthetic.py):
US and European equity, Treasuries, investment-grade bonds, gold and commodities.
Seed 20260725 and clean mode (`apply_quirks=False`) select complete business-day observations
from 4 January 2010 to 31 December 2025. Early history supplies estimation warmup;
the displayed performance window is 1 April 2015 to 31 December 2025.

The final exhibit uses the repository's
[known-factor simulator](../examples/covar_estimation/simulate_factor_returns.py), seed 42:
four factors, eight assets and 783 business-day observations from 2 January 2023 to
31 December 2025. Its displayed backtest starts on 3 January 2024 after weekly estimation warmup.

| Convention | Applied in these exhibits |
|---|---|
| Estimation returns | Weekly Wednesday log returns, with trailing EWMA demeaning |
| Covariance units | Annualized using 52 weekly observations per year |
| Construction | Fully invested, long-only target weights; 35% cap per asset |
| Decision and execution | Quarterly schedules mapped to observed Wednesdays; targets trade one business-day observation later |
| Holdings | qis holds units between trades, so realized weights drift |
| Trading costs | 10 basis points of gross traded notional, including entry |
| Other costs | No funding or management fees |
| Displayed growth | Portfolio NAV after costs; the 60/40 reference in the first panel is explicitly gross |

The last target decision is 1 October 2025. That target snapshot differs from holdings at the
sample end. Cost bars sum daily cash cost divided by that day's NAV, expressed in basis points;
they measure cost incidence rather than compounded performance drag.
The [refresh specification](documentation_standard.md#analytical-conventions-and-figures)
records detailed conventions, solver acceptance and independent numerical checks.

## Portfolio performance

The maximum-diversification portfolio is compared with the fixture's daily-rebalanced gross
60/40 benchmark. Both series are displayed as growth of 100; drawdown measures the fall from
each series' running peak over the report window.

[![Synthetic maximum-diversification growth and drawdowns against a gross 60/40 benchmark](../examples/figures/example_portfolio_factsheet1.PNG)](../examples/figures/example_portfolio_factsheet1.PNG)

**Sample:** 1 April 2015–31 December 2025. The portfolio is net of 10 bp trading costs;
the benchmark is gross. This illustrative comparison does not isolate an allocation advantage.
**Producer:** [portfolio reports](../tools/docs_analytics/portfolio_reports.py).
**Related methodology:** [rolling backtests](rolling_backtests.md).

## Allocation and risk

The upper panel shows decided maximum-diversification weights. The lower panel shows each
asset's contribution to estimated annualized portfolio volatility using the same decision's
covariance. Contributions are measured in volatility percentage points, not portfolio weights.

[![Synthetic target weights and contributions to annualized portfolio volatility](../examples/figures/example_portfolio_factsheet2.PNG)](../examples/figures/example_portfolio_factsheet2.PNG)

**Decision:** 1 October 2025. Weekly log returns, EWMA span 52, annualization factor 52.
The 35% asset cap can bind. The estimate describes targets and ex-ante risk, rather than
end-of-sample holdings or realized future volatility.
**Producer:** [portfolio reports](../tools/docs_analytics/portfolio_reports.py).
**Related methodology:** [covariance estimators](covariance_estimators.md).

## Allocation through time

The stacked panel extends each decided target until the following decision for display.
It does not plot drifted holdings. The lower panel aggregates realized trading costs by
calendar quarter, including the initial allocation.

[![Synthetic decided allocations through time and quarterly sums of trading costs](../examples/figures/example_customised_report.PNG)](../examples/figures/example_customised_report.PNG)

**Sample:** 1 April 2015–31 December 2025. Extending the last target to the report end adds
neither a new decision nor a trade. Quarterly cost sums use each day's NAV denominator.
**Producer:** [portfolio reports](../tools/docs_analytics/portfolio_reports.py).
**Related methodology:** [turnover and transaction costs](turnover_and_transaction_costs.md).

## Covariance span sensitivity

Five maximum-diversification backtests vary the EWMA span across **5, 13, 26, 52 and 104**
weekly observations. Each span affects both trailing demeaning and covariance smoothing.
A span is an EWMA parameter, not a half-life or a finite estimation window.

[![Synthetic net maximum-diversification growth and total trading costs across five EWMA spans](../examples/figures/max_diversification_span.PNG)](../examples/figures/max_diversification_span.PNG)

**Sample:** 1 April 2015–31 December 2025. Assets, constraints, costs and implementation dates
are shared. The lower panel sums daily cost/NAV over that common report period. This fixed
path illustrates parameter sensitivity; it does not select a preferred span.
**Producer:** [span sensitivity](../tools/docs_analytics/span_sensitivity.py).

## Portfolio objectives

Minimum variance, maximum diversification and equal risk budgets use the same covariance,
assets, weight constraints and trade dates. Expected-return estimation is outside this
comparison; it covers three objectives driven by covariance.

[![Synthetic net growth and trading costs for minimum variance, maximum diversification and equal risk budgets](../examples/figures/multi_optimisers_backtest.PNG)](../examples/figures/multi_optimisers_backtest.PNG)

**Sample:** 1 April 2015–31 December 2025. The 35% cap can prevent equal target risk budgets
from producing equal risk contributions at the decision date. Cost sums include entry. This single
synthetic path is not evidence of an objective's expected market performance.
**Producer:** [optimiser comparison](../tools/docs_analytics/optimiser_comparison.py).
**Related methodology:** [optimization guide](optimization_module_readme.md) and
[risk budgeting](risk_budgeting.md).

## Covariance estimators

Six estimates feed a common minimum-variance construction: EWMA, Lasso and Group Lasso,
each with its specified volatility-normalized variant. EWMA normalization acts on asset-return
covariance; factor variants normalize factor covariance only. Group Lasso uses four fixed
asset pairs, not clusters inferred from the known loadings.

[![Synthetic minimum-variance backtests and last-decision covariance errors for six estimators](../examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)](../examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)

**Backtest:** 3 January 2024–31 December 2025. **Error snapshot:** 1 October 2025.
The lower panel measures relative Frobenius error against the known simulated covariance:
the Euclidean size of all estimation errors divided by the size of the true matrix.
Estimated weekly covariance is annualized by 52; simulated daily truth by 260.
Known truth is used for descriptive evaluation and never supplied to portfolio estimation.

All cases fit only data available at their shared decision dates. The simulation has constant
loadings/covariance, no return premium and one fixed path. Closely overlapping performance
curves and one-date covariance errors do not establish an estimator ranking.
**Producer:** [covariance comparison](../tools/docs_analytics/covariance_comparison.py).
**Configuration:** [analytics registry](../tools/docs_analytics/registry.json).

## Reproduce and update

From a source checkout with the documented contributor environment and C-local setup, one
command regenerates all six previews and their supporting tables:

~~~console
python -m tools.docs_analytics.run --all --output-root <new-C-local-bundle>
~~~

Use a new directory below `AGENT_LOCAL_ROOT`, outside OneDrive and the source tree.
The runner executes offline and records the imported environment and effective source bytes.
Regeneration retains the fixed sample and seed; it does not fetch current market prices.
Distribution versions alone cannot identify uncommitted source edits.

Follow the [generation and publication workflow](documentation_standard.md#analytical-conventions-and-figures)
to compare repeated runs, validate the bundle, review each preview at full resolution and
article width, and publish the six images with their shared provenance record. Full tables
and intermediate output remain C-local. Preview paths are stable; the
[provenance file](../examples/figures/analytics_manifest.json) identifies their reviewed generation.

## See also

- [Documentation standard](documentation_standard.md): conventions, producer details and refresh workflow.
- [Constraints](constraints.md): units, alignment and backend support.
- [Software design](software_design.md): construction, estimation and analytics ownership.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff),
  for simulation, performance and risk analytics.
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff),
  for sparse factor-model fitting.
