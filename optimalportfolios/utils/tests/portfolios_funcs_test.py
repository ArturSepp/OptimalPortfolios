
import pandas as pd
from enum import Enum
from optimalportfolios.utils.portfolio_funcs import round_weights_to_pct



class LocalTests(Enum):
    ROUND_WEIGHTS = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.ROUND_WEIGHTS:
        weights = pd.Series({
            'Precious Metals': 0.069859331,
            'Real Estate Participations': 0.090140669,
            'Bonds CHF Govt': 0.079284593,
            'Investment Grade Corporate Bonds World': 0.110715408,
            'Equities Emerging Markets World': 0.020343894,
            'Equities Europe ex UK, CH': 0.049,
            'Equities Japan': 0.035,
            'Equities North America': 0.122,
            'Equities Switzerland': 0.373656106,
            'Money Market': 0.05,
        })

        pct_weights = round_weights_to_pct(weights, decimals=2)

        print("Rounded weights (%):")
        print(pct_weights.to_string())
        print(f"\nSum: {pct_weights.sum():.2f}")


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ROUND_WEIGHTS)