"""Run Exact primary with audited Ranked fallback on supplied synthetic inputs.

Run: python -m examples.execution.policy
Native Exact needs the optional MOSEK backend and a valid licence. Without it,
this prepared-feasible example demonstrates the audited Ranked fallback.
"""

from time import perf_counter

from optimalportfolios.execution import MatchedExecutionPolicyConfig, wrapper_execution_policy
from examples.execution import delivery_row, portfolio_problem, trade_table


def main():
    """Print the actual selected engine, fallback reason and independently audited trades."""
    problem = portfolio_problem()
    # This wrapper consumes the constraints in problem; config controls search.
    # The lifecycle dataclass requires a class map even for a prepared problem.
    config = MatchedExecutionPolicyConfig(tre_weight_by_asset_class={"Risk": 1.})
    started = perf_counter()
    solved = wrapper_execution_policy(problem, config)
    row = delivery_row("Exact primary / Ranked fallback", problem, solved,
                       perf_counter() - started)
    print(f"Engine: {solved.diagnostics['engine']}")
    print(f"Fallback reason: {solved.diagnostics['fallback_reason']}")
    print(f"Audit: {row['audited']}; tickets: {row['tickets']}; "
          f"annual TE: {row['annual_te_bp']:.3f} bp")
    print(trade_table(problem, solved).round(4).to_string())
    print("Cash funds the trades and does not consume a ticket. No files written.")


if __name__ == "__main__":
    main()
