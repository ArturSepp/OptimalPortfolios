---
myst:
  html_meta:
    description: >-
      Fit optional skfolio allocations and evaluate labelled targets through QIS with
      explicit information cutoffs, execution dates, universe eligibility and trade costs.
---

# Evaluate skfolio targets with QIS

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/OptimalPortfolios/commit/4b851495f5371b7b4dbefaab272a4f6e80220a79)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Maintained in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Accounting uses [QIS](https://github.com/ArturSepp/QuantInvestStrats)
([citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff));
allocation estimation uses [skfolio](https://skfolio.org/).

The [optional bridge example](../examples/integrations/skfolio_targets_local.py) fits
skfolio minimum-variance allocations and supplies labelled targets to
`qis.backtest_model_portfolio`. QIS owns units, cash, drifting weights, executed
trade costs and missing-quote freezes. The example prints gross and net NAV,
realised costs, versions, asset labels and the decision/execution calendar offline.
It changes no optimizer or accounting defaults in either package.

## Run the released-package workflow

The example source is repository-only; wheels do not ship `examples/`. Copy that
single script outside the checkout, or run it from the repository root. It uses
the monthly fixture already shipped in the OP wheel. No vendor or network data is
needed. Optional packages are needed only when fitting is requested; validation
and QIS evaluation work without skfolio.

The checked released set is `optimalportfolios==7.21.0`, `qis==5.37.0`,
`factorlasso==1.0.4`, `skfolio==1.9.3` and `scikit-learn==1.9.1`.
These are reproduction pins, not new runtime dependency declarations. Use the
[installation guide](installation.md) and the mandatory external environment and
C-local generated-state setup in AGENTS.md when working in this OneDrive checkout.

```console
python -m pip install optimalportfolios==7.21.0 qis==5.37.0 factorlasso==1.0.4 skfolio==1.9.3 scikit-learn==1.9.1
python examples/integrations/skfolio_targets_local.py
```

Installation can require the network; the example does not. Maintainer verification
also runs the example from a C-local directory with the pinned released wheels
first on the import path, proving the checkout does not supply their package code.
Complete transitive versions and wheel hashes belong in the execution evidence.
The established `_local.py` suffix records this optional prerequisite and excludes
it from unattended core-only example discovery; dedicated optional tests run it.
It is a broader workflow example, not a source-adjacent component runner.

## Contract at the boundary

| Item | Required convention |
|---|---|
| Asset identity | Construct a Series from `model.weights_` indexed by the actual fitted `model.feature_names_in_`. Reindex by price labels, never copy positional weights into a differently ordered panel. Duplicate or unknown assets fail. |
| Return basis | Declare `simple_fraction`; one percent is 0.01. Percent and log declarations fail. Training uses `qis.to_returns(..., is_log_returns=False, ffill_nans=False, drop_first=True)`. |
| Moments | Declare `per_observation`. The monthly example uses monthly moments, with no annualisation. The bridge receives targets, not a moment matrix; it cannot discover an incorrect upstream declaration. |
| Information and decision | Each target row is indexed by its decision timestamp; a datetime Series with exactly that index states the last observation used. Information cannot be later than decision. |
| Execution | A second datetime Series states the exact price observation used for implementation. Information must strictly precede execution, decision cannot be later than execution, and one later price observation is required. No calendar rounding, timezone conversion or hidden lag is applied. |
| Price universe | Every asset is labelled. Positive finite total-return marks use a common valuation currency and sampling grid. NaNs mean unquoted, not zero price. The caller owns FX, price adjustments and grid selection. |
| Exclusions | Assets omitted by the fitted universe require explicit `excluded_assets`; the bridge inserts zero targets only for those labels. Exclusions are zero targets, not a promise to liquidate an unquoted holding. No renormalisation occurs. |
| Late entrants | Explicit zero targets are permitted before the first quote. A nonzero target before the first quote fails; entry on the first quoted execution date is allowed. Fitting requires a deliberately chosen complete eligible training panel. |
| Missing quotes | A later missing quote passes unchanged to QIS, which freezes an unpriced holding and marks it under its canonical policy. Target weights can differ from actual holdings while a position is frozen. |
| Portfolio state | Long-only, sum-to-one targets and initial NAV 100, with no explicit leverage target or external flows. QIS holds units between target executions; its cost funding can leave a cash debit. Evaluation starts at the first execution, excluding estimation history. |
| Costs | Explicit nonnegative proportional QIS trade charge; 0.001 is ten basis points. Includes entry and uses actual trade-date turnover/cash funding. It does not translate a native skfolio periodic fee. |

The demonstration estimates at the 2018 and 2020 December closes, then executes
at the next monthly observation. This is a one-observation monthly delay, not a
claim that a monthly close quote supplies intramonth fills. Pre-decision data is
used for fitting only. Future test prices do not change fitted targets.

## Reconciliation and limitations

[The B0 benchmark guide](https://github.com/ArturSepp/OptimalPortfolios/blob/21f7581737d50a51a6afeff98f498b9bfd7c6e8d/tools/README_matched_benchmarks.md) defines the matched
six two-asset ledgers and native semantic differences. L1 feeds those same dated
targets through the bridge and reconciles NAV, units and trade costs against B0's
existing independent QIS cash ledger. Fully quoted gross paths match native
skfolio Portfolio evaluation when drift is enabled and identical execution
segments are supplied. Default fixed weights produce another path.

Native skfolio Portfolio costs are configured per observation; QIS charges actual
trades at execution. An unquoted-position freeze is also different from simply
forward-marking returns. The tests preserve and measure these cost/timing/freeze
differences instead of modifying fee periodicity or return data to force equality.
Use QIS evaluation for an explicit holdings/cash contract; the bridge does not
make skfolio native evaluation equivalent under every accounting convention.

The bridge is an example with deliberate validation, not broad automatic object
conversion or an added public API. It does not infer currencies, calendars,
trade feasibility, delisting treatment or missing characteristics. It does not
solve the L2 feedback problem of passing actual executed state to the next
optimizer decision. A FactorLasso-to-skfolio prior adapter remains deferred until
there is demonstrated demand and an agreed fitted-output contract.

## Verification

After the owning external interpreter and C-local setup, run:

```console
python -m pytest src/optimalportfolios/tests/skfolio_bridge_test.py src/optimalportfolios/tests/matched_benchmarks_test.py
python tools/check_docs.py --files docs/skfolio_interoperability.md docs/index.md
```

Core tests exercise labels, units, clocks, eligibility, exclusions and QIS accounting;
only actual peer fitting/native comparisons skip when skfolio is absent. The optional
lane must execute all of them. Recorded verification includes both registered OP CI
lanes, the pinned released-wheel workflow, alignment/timing/cost regression mutations,
source/link checks and strict Sphinx output. QIS and FactorLasso contracts are reused
unchanged. No additional figures or package dependencies are introduced.

## Primary sources

- [skfolio MeanRisk](https://skfolio.org/generated/skfolio.optimization.MeanRisk.html): fitted weights, feature names, variance objective and solver configuration.
- [skfolio Portfolio](https://skfolio.org/generated/skfolio.portfolio.Portfolio.html): native weight drift and return/cost evaluation.
- [Pinned skfolio release](https://pypi.org/project/skfolio/1.9.3/): tested optional peer snapshot.
- [QIS backtester](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py): canonical holdings and trade-cost evaluation.
