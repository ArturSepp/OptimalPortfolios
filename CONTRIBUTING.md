# Contributing to OptimalPortfolios

Thanks for your interest in `optimalportfolios`. `optimalportfolios` is the reference implementation of the ROSAA framework published in *The Journal of Portfolio Management*, so published results constrain what can change.

## Scope

In scope:

- Bug fixes in optimisation, constraints, covariance estimation, or the backtest engine
- New optimisation objectives or constraint types with a reference for the formulation
- Numerical robustness improvements, with a test that demonstrates the failure case
- Documentation, examples, and tests

Out of scope — these will be declined, so please open an issue to discuss before
writing code:

- A third optimisation backend. The package uses `cvxpy` and `quadprog`
- Reimplementations of analytics or plotting that belong in
  [`qis`](https://github.com/ArturSepp/QuantInvestStrats), or of factor estimation that
  belongs in [`factorlasso`](https://github.com/ArturSepp/factorlasso) — both are
  declared dependencies
- Silent changes to optimiser defaults, constraint semantics, or rebalancing conventions
- Changes to `paper_code/`, which accompanies the published papers
- Examples that require a paid data subscription to run

## Reporting a bug

Open an issue using the bug report template. A report needs the `optimalportfolios` version, your
Python version, a minimal self-contained reproducer, and the full traceback or the
incorrect numbers. Reproducers that depend on proprietary or licensed data cannot be
run, so please use generated or public data.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are
welcome; where a question is really about the published papers, please say which paper
and section you are reading.

## Development setup

```bash
git clone https://github.com/ArturSepp/OptimalPortfolios.git
cd OptimalPortfolios
pip install -e ".[dev]"
pytest optimalportfolios/   # tests live inside the package, not in a top-level tests/
ruff check optimalportfolios/
```

`AGENTS.md` in this repository documents the layout, commands, conventions, and
constraints in more detail — it is written for AI coding agents but is equally useful
to human contributors.

## Pull requests

- One topic per pull request. Unrelated changes in the same PR make review slower and
  are likely to be asked to split.
- Add or update tests for behaviour you change. A bug fix should come with a test that
  fails before the fix.
- Run the test suite and `ruff` before submitting.
- Do not bump the version in `pyproject.toml` or `CITATION.cff`; releases are cut
  separately.
- Do not commit generated output: figures, factsheets, backtest results, or data files.
- Keep the public API stable. If a change alters a public signature or default, say so
  explicitly in the PR description.

## Replication

`paper_code/` reproduces results from the published papers. If your change alters
optimiser behaviour, covariance estimation, or backtest mechanics, please re-run the
relevant scripts and confirm the published tables still reproduce. If they do not,
report the difference in the PR rather than updating the expected values.

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are
not.

## Licence

This project is MIT licensed. By contributing, you agree that your contributions are licensed under
the MIT licence of this project.
