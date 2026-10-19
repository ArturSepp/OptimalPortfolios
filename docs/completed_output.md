---
myst:
  html_meta:
    description: The finite OptimalPortfolios output from supplied alphas, covariance and constraints, with solver status and a separate reporting handoff.
---

# The completed optimisation output

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

An allocation is complete when its target weights and solver outcome are available
and its feasibility has been inspected. The OP 8 development API accepts supplied
alphas, covariance and constraints; factor construction and fitting happen upstream.
General stress testing, holdings simulation and reporting use
[QIS](https://github.com/ArturSepp/QuantInvestStrats), with its
[software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
Neither step is required to inspect the allocation itself.

## Inputs and runnable example

The [canonical offline script](../examples/docs/completed_output.py) solves a
three-asset quadratic-utility allocation using the base installed OP package. Obtain
the script from the same development revision and run it from the checkout:

```console
python -m examples.docs.completed_output
```

No market data, TF or FL installation, network request or output directory is needed.
The script returns a `PortfolioOptimisationResult` and the separate
`OptimizationOutcome` from `wrapper_quadratic_optimisation`. It independently checks
the equality-constrained solution by the Lagrange multiplier equation; the synthetic
long-only bounds are inactive. Its Cholesky norm checks the QIS risk calculation.

| Input | Fixture and convention |
|---|---|
| Assets | Ordered labels Bonds, Credit, Equity |
| Alpha | Annual arithmetic decimals 0.04, 0.06, 0.08; common currency USD |
| Covariance | Annual decimal-return squared; diagonal 0.01, 0.0225, 0.04 |
| Decision and risk date | 2025-12-31; supplied risk must be available by this date |
| Objective and constraints | Quadratic utility, risk aversion 3, long-only and fully invested |
| Benchmark | One third per asset; only used for subsequent comparison |
| Solver | CLARABEL; accept only the documented accepted status and hard-constraint checks |

OP does not infer currency or annualise these inputs. Match alpha and covariance
units before solving. A covariance-only result provides total risk, but cannot
invent factor or residual attribution.

The accepted synthetic solution is approximately 21.8579% Bonds, 39.3443% Credit
and 38.7978% Equity. Annual volatility is 9.9909% and tracking error to the equal
benchmark is 1.8232%. These are illustrative results for the fixed inputs above;
the script checks the solution against independent identities with explicit tolerances.

## The completion point

Keep the solver outcome alongside the portfolio result: the latter does not contain
a `solver_status` field. A returned finite vector may be a fallback, so successful
construction of the result alone does not certify a successful solve.

| Actual output | Inspection and meaning |
|---|---|
| `weights` returned by the wrapper | Asset-indexed fractional target weights |
| `outcome.accepted`, `status`, `solver`, `reason`, `fallback_source` | Whether the solver iterate was accepted, raw status and explicit fallback provenance |
| `outcome.constraint_residuals`, `residuals_frame()`, `compliant` | Actual values, bounds, violation, tolerance, hard/soft flag and pass status |
| `result.get_weights(name)`, `to_weights_df()` | Named portfolio weights and benchmark/active comparisons |
| `result.optimisation_date`, `risk_model_date`, `reference_ccy` | Decision, selected risk snapshot and declared currency |
| `result.covariance`, `expected_return` | Selected supplied risk and expected-return context |
| `result.summary()` | Inspectable weight and optional risk/turnover summary |

Completion means that no further factor estimation, portfolio simulation or figure
rendering is needed to obtain these outputs. It does not mean that every solver
status is acceptable. The [solver outcome guide](solver_numerics_and_outcomes.md)
defines rejection, tolerated inaccurate solutions, compliance and fallback handling.
Other solver families and rolling functions have their own return contracts; retain
single-date outcomes when a persistent per-date audit is required.

## Mutability and explicit export

This is an inspectable Python result, not an immutable archive. Its constructor owns
copies of normalized weights, selected risk context and supplied diagnostic tables.
Public pandas objects and getter results remain mutable; copy them before editing.
Metadata, constraints and other auxiliary objects are not a universal deep-copy
contract. Reconstruct the result to change its consistent inputs rather than editing
an internal table and expecting other fields to be recalculated.

The example's `save_tables(result, outcome, output_dir)` writes weights, covariance,
summary and constraint CSVs plus a status/units/date JSON receipt. It requires a new
directory and refuses replacement. Calling `main()` performs no persistence. These
are review tables, not a serialized `PortfolioOptimisationResult`; OP promises no
universal loader or format compatibility for that example export. Preserve the
source inputs and their units, provenance and software version with a durable record.

## Separate downstream use

Pass the target-weight table, execution prices, implementation lag and transaction
cost assumptions to `qis.backtest_model_portfolio` when simulation is wanted; the
[quickstart](quickstart.md) demonstrates that separate handoff. Supplied QIS
`RiskModel` context enables risk and stress workflows without fitting a new model.
The result's convenience risk methods delegate to QIS.

For upstream factor inputs, start with the
[TF availability and supplied-risk integration guide](supplied_risk_inputs.md#tradablefactors-availability).
The full completed-model guide is `docs/completed_model.md` in an authorised TF
checkout. TF can supply dated covariance and expected-return arrays; OP does not
import TF or FL.
EWMA and signal utilities remain explicit optional preparation steps in OP.

## See also

- [API reference](api.rst) and [software boundaries](software_design.md).
- [Constraints](constraints.md) and [solver numerics and outcomes](solver_numerics_and_outcomes.md).
- [Canonical script and explicit export helper](../examples/docs/completed_output.py).
