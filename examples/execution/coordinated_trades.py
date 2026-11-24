"""Demonstrate why a sale and purchase can need two tickets together.

Run: python -m examples.execution.coordinated_trades
Uses core OP with CLARABEL; no native mixed-integer solver or licence required.
"""

from time import perf_counter

import numpy as np
import pandas as pd

from optimalportfolios.execution import wrapper_ranked_execution
from examples.execution import delivery_row, pair_problem, trade_table


def main():
    """Check the funding argument independently and compare one versus two tickets."""
    rows = []
    for cap in (1, 2):
        problem = pair_problem(cap)
        # Either single trade is at least 2% NAV, exceeding cash's 0.1% room.
        cash_room = max(problem.upper[-1] - problem.base[-1],
                        problem.base[-1] - problem.lower[-1])
        assert cash_room < problem.rules.minimum_ticket
        started = perf_counter()
        solved = wrapper_ranked_execution(problem)
        row = delivery_row(f"Ranked K={cap}", problem, solved, perf_counter() - started)
        rows.append(row)
        assert row["tickets"] == (0 if cap == 1 else 2)
        expected = problem.current if cap == 1 else problem.target
        # Accuracy against the analytic minimiser is separate from feasibility;
        # delivery_row has already applied the unchanged independent audit.
        np.testing.assert_allclose(solved.weights, expected, atol=1e-6, rtol=0)
        print(f"\nTicket cap: {cap}")
        print(trade_table(problem, solved).round(4).to_string())
    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format="%.6g"))
    print("The extra ticket admits a coordinated switch; no limit or tolerance changed.")


if __name__ == "__main__":
    main()
