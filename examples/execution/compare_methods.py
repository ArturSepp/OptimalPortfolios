"""Compare execution searches on one immutable prepared CORRIDOR domain.

Run: python -m examples.execution.compare_methods
All engines: python -m examples.execution.compare_methods --native
Costs: python -m examples.execution.compare_methods --native --cost 0.0005
Default runs Ranked only and explicitly marks Exact/Kernel as not requested.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import pandas as pd

from optimalportfolios.execution import (
    KernelConfig, RankedConfig, wrapper_exact_execution, wrapper_kernel_execution,
    wrapper_ranked_execution,
)
from examples.execution import delivery_row, native_evidence, portfolio_problem, trade_table


def main():
    """Print matched objectives, trades and native evidence; fail incomplete native runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="require licensed Exact and Kernel")
    parser.add_argument("--seconds", type=float, default=10., help="search allowance per method")
    parser.add_argument("--tickets", type=int, default=3, help="total noncash ticket cap")
    parser.add_argument("--cost", type=float, default=0.,
                        help="annual variance per unit traded NAV")
    args = parser.parse_args()
    problem = portfolio_problem(ticket_cap=args.tickets, cost=args.cost)
    methods = [("Ranked", lambda: wrapper_ranked_execution(
        problem, RankedConfig(time_limit_s=args.seconds)))]
    if args.native:
        methods += [
            ("Exact", lambda: wrapper_exact_execution(
                problem, "CORRIDOR", time_limit_s=args.seconds, threads=1, gap_rel=1e-4)),
            # Deliberately small teaching kernel exposes buckets on this small universe.
            # Library defaults are minimum_kernel=24 and kernel_per_ticket=2.
            ("Kernel", lambda: wrapper_kernel_execution(
                problem, KernelConfig(time_limit_s=args.seconds,
                                      minimum_kernel=2, kernel_per_ticket=1))),
        ]
    for package in ("optimalportfolios", "cvxpy", "mosek"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: not installed")
    print("Same current/base/target, covariance, costs, bounds, groups, ticket rules and audit.")
    print("No cross-method warm starts. One native thread; elapsed times are descriptive only.")
    rows, incomplete = [], []
    for method, solve in methods:
        started = perf_counter()
        solved = solve()
        rows.append(delivery_row(method, problem, solved, perf_counter() - started))
        print(f"\n{method}\n{trade_table(problem, solved).round(4).to_string()}")
        if method != "Ranked":
            evidence = native_evidence(method, solved)
            print(evidence)
            if not evidence["native_usable"]:
                incomplete.append(method)
    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format="%.6g"))
    if not args.native:
        print("Exact: NOT RUN. Kernel: NOT RUN. Use --native with MOSEK and a valid licence.")
    print("Objective includes twice the cost term; annual TE excludes cost. Lower is better.")
    print("No optimality claim: Exact's time limit and Kernel's restricted bounds matter.")
    if incomplete:
        raise RuntimeError(f"Native comparison incomplete: {incomplete}; inspect evidence above")


if __name__ == "__main__":
    main()
