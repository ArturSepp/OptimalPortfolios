---
myst:
  html_meta:
    description: >-
      Run synthetic execution examples: Exact primary with audited Ranked fallback,
      coordinated trades, and matched Exact, Ranked and Kernel comparisons with costs.
---

# Execution examples: Exact, Ranked and Kernel

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

The [`examples/execution/`](https://github.com/ArturSepp/OptimalPortfolios/tree/3f38f8a31b495b334d3991256250c89626340cb8/examples/execution) subpackage turns supplied
holdings and model targets into audited execution portfolios. Start with the production-policy
example, inspect why two trades may need to be selected together, then compare the three
search methods on exactly the same prepared inputs. All inputs are deterministic synthetic
arrays; no private mandates, market downloads or paper result files are used.

These are teaching examples of the methods in
[Execution-Constrained Portfolio Rebalancing: Trading Costs and Coordinated Trades](https://doi.org/10.5281/zenodo.24354159).
The public manuscript and replication bundle preserve the study's original experiments and
software identities. Running these small examples does not reproduce the paper's tables or
establish a performance ranking.

## Setup and commands

Use the repository root as the working directory with OP installed in your environment.
`examples/` is repository-only and is not shipped in the wheel. Use a source checkout matching
the installed package; the three-engine examples require the Kernel API introduced in OP 10.0.4.
Follow the [installation guide](installation.md) for environment setup, including the external
Python environment on Windows/OneDrive.

| Example | What it shows | Default requirements |
|---|---|---|
| [`policy.py`](../examples/execution/policy.py) | Exact primary; lazy, independently audited Ranked fallback; actual engine and reason | Core OP; native Exact is attempted when available |
| [`coordinated_trades.py`](../examples/execution/coordinated_trades.py) | One ticket cannot fund a switch; two tickets admit a sale and purchase together | Core OP with CLARABEL |
| [`compare_methods.py`](../examples/execution/compare_methods.py) | Same objective and domain across Ranked, Exact and Kernel; allocations and diagnostics | Core OP for Ranked; `--native` requires MOSEK |

```console
python -m examples.execution.policy
python -m examples.execution.coordinated_trades
python -m examples.execution.compare_methods
python -m examples.execution.compare_methods --tickets 2 --cost 0.0005
```

These commands print tables and write no files. The default comparison explicitly labels Exact
and Kernel as **NOT RUN**. To run all three methods, install the `exact` extra and supply a valid
MOSEK licence. The qualified native adapter requires CVXPY 1.9.2, pinned below; the
[prepared-input reference](execution_v2.md) documents that backend contract.

```console
python -m pip install "optimalportfolios[exact]" "cvxpy==1.9.2"
python -m examples.execution.compare_methods --native --seconds 10
python -m examples.execution.compare_methods --native --tickets 2 --cost 0.0005
```

The native comparison exits with an error if a method has no audited native candidate or
reports a native failure; it still prints the diagnostics and any valid retained portfolio.
This is different from the policy example, where audited fallback is an intended outcome.

## Supplied inputs and unchanged rules

The seven-name example has six noncash assets and settlement cash. The common starting
portfolio is feasible, so `base` equals `current`; there are no compulsory trades. Prepared
noncash corridors run from current to target, while cash retains its funding band. The containing
hard bounds are supplied separately for the FULL domain; no wrapper infers these corridors. All weights,
hard instrument limits, overlapping group exposures and group bands are supplied explicitly.
The covariance is a synthetic diagonal **annual** covariance. Current weights are the holdings
at execution time, not the previous rebalance's unadjusted targets.

| Rule | Teaching setting |
|---|---|
| Cash bounds | 4.9% to 5.1% NAV |
| Minimum noncash ticket | 2% NAV |
| Nonzero holding floor | 1% NAV |
| Total noncash ticket cap | 3 by default; `--tickets` changes only this cap |
| Mandatory trades and exemptions | None |
| Investment | 100% NAV, long-only |
| Movement and audit tolerance | Unmodified `ExecutionRules` defaults: `1e-8` and `1e-7` |

The independent `audit_execution` checks each delivered portfolio. Cash balances funding and
does not consume a ticket. The ticket cap counts trades, not portfolio holdings; zero movement
does not count as a trade. Printing rounded weights does not round the weights used by the
solver or audit.

The examples start **after common preparation**. Real consumers must resolve eligibility,
instructions, compulsory repairs and group membership before constructing this domain.
Preparation can itself require a mixed-integer solver: this example's feasible starting point
does not show that the entire production lifecycle can always run without one. Use
[`solve_execution_portfolio`](execution.md#production-policy-and-failures) for the higher-level
production adapter and the [execution integration guide](execution.md) for input ownership.

## What the methods share and what differs

| Method | Search | How to interpret its result |
|---|---|---|
| Exact | Matched mixed-integer optimisation on the complete CORRIDOR domain | An audited time-limited incumbent need not be optimal; inspect native status and bound consistency |
| Ranked | Bounded trade-support search with joint continuous sizing | Audited heuristic result; no global optimality certificate |
| Kernel | Relaxation-guided, restricted mixed-integer subproblems | Audited heuristic result; subproblem bounds are not full-domain certificates |
| Exact-primary policy | Attempts Exact, then runs Ranked only when primary acceptance fails | An operational fallback policy, not a best-of-two tournament |

All compared methods receive the **same immutable problem**: current/base/target, covariance,
cost coefficients, hard bounds, group rows, ticket cap, floor and audit rules. The comparison
uses CORRIDOR for every method and supplies no cross-method warm starts. Comparing FULL Exact
with CORRIDOR Ranked would also change the policy domain; it would not isolate search quality.

Each method receives the same requested search allowance. Exact uses one native thread and
relative gap `1e-4`; Kernel uses its implemented one-thread/gap policy. For this small teaching
universe, Kernel uses `minimum_kernel=2` and `kernel_per_ticket=1` to expose bucket stages.
Those are explicit example settings, not the library defaults of 24 and 2. The tables print
completed buckets. Preparation and in-progress work can exceed nominal allowances, and timings
from this short sequential run are descriptive, not a controlled speed benchmark.

## Read the comparison

The summary reports the **scored objective**, pure annual tracking error in basis points,
the proportional-cost term, noncash ticket count, gross noncash turnover as a percentage of
NAV, audit outcome and elapsed seconds. Per-asset tables show current, target and execution
weights and signed trades. Risk and objective metrics come from OP's result diagnostics;
the example does not reimplement portfolio risk analytics.

`--cost` supplies the same nonnegative coefficient for every noncash asset, with zero cash
cost. Its units are **annual variance per unit traded NAV**, not a quoted bid-ask spread in bps.
The objective is tracking variance minus twice the weight-score contribution plus twice
the proportional-cost contribution. Here weight scores are absent. Lower objective is better;
lower pure tracking error alone need not be better when trading has a cost. Compare methods
within one cost setting; different coefficients represent different objectives.

> **Pitfall.** An audited no-trade portfolio is not evidence of a successful native solve.
> Both native wrappers can retain the valid common starting portfolio. The comparison prints
> native status and candidate evidence separately and fails an incomplete native comparison.
> A passing audit establishes feasibility, not optimality.

In the coordinated-trades example, either one-sided minimum trade would exceed available cash
room. With one ticket, holding is the only feasible action. With two, the sale funds the purchase.
The script checks this funding argument independently and asserts the delivered trade counts
and weights. This illustrates why ticket-by-ticket greedy reasoning can miss feasible switches;
it does not claim that Ranked finds every combination.

## Further reading

- [Execution integration](execution.md): production policy, preparation and consumer inputs.
- [Prepared-input numerical reference](execution_v2.md): domain, objective and native backend.
- [Examples and recipes](examples_readme.md): all example tasks and execution lanes.
- [Public manuscript and replication](https://doi.org/10.5281/zenodo.24354159): frozen scientific study.
