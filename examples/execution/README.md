# Execution examples

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

See the [execution examples guide](../../docs/execution_examples.md) for the supplied-input
contract, objective units, methods and interpretation. All inputs are synthetic; scripts print
results without downloading data or writing files. These examples accompany the
[public execution manuscript and replication](https://doi.org/10.5281/zenodo.24354159),
but are not the paper's benchmark or a production configuration.

From the repository root, with the matching OP package installed:

```console
python -m examples.execution.policy
python -m examples.execution.coordinated_trades
python -m examples.execution.compare_methods
python -m examples.execution.compare_methods --tickets 2 --cost 0.0005
```

- `policy.py`: Exact primary with independently audited Ranked fallback; prints the engine
  actually used and the fallback reason.
- `coordinated_trades.py`: a tight-cash sale/purchase pair under one and two noncash tickets.
- `compare_methods.py`: matched CORRIDOR comparison, with risk, cost, trades and audit diagnostics.
  The default runs Ranked and labels Exact/Kernel as not run.
- `__init__.py`: reusable synthetic supplied inputs and common reporting helpers.

Native comparisons require the `exact` extra, the supported CVXPY/native adapter and a valid
MOSEK licence. The three-engine API is available from OP 10.0.4. The examples directory is not
included in wheels. Follow [installation](../../docs/installation.md) to prepare an environment;
on Windows/OneDrive, keep it outside the checkout.

```console
python -m pip install "optimalportfolios[exact]" "cvxpy==1.9.2"
python -m examples.execution.compare_methods --native --seconds 10
python -m examples.execution.compare_methods --native --tickets 2 --cost 0.0005
```

`--cost` is annual variance per unit of traded NAV, **not basis points of transaction cost**.
Kernel uses a small teaching kernel (2 names minimum, 1 per ticket); the library defaults are
unchanged. Native failures are reported as failures even if a feasible starting portfolio was
retained. Passing the independent audit proves feasibility, not optimality. These brief timings
are illustrative, not evidence that one method is faster.
