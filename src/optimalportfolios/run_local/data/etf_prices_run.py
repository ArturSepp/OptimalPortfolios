"""Run local ETF price-data maintenance and loading scenarios."""

from enum import Enum

from optimalportfolios.run_local.data.etf_prices import load_test_data, update_test_prices


class Locals(Enum):
    """Available local ETF price-data scenarios."""

    UPDATE_TEST_PRICES = 1
    LOAD_TEST_PRICES = 2


def run_local(local: Locals) -> None:
    """Run the selected ETF price-data scenario."""
    if local == Locals.UPDATE_TEST_PRICES:
        prices = update_test_prices()
        print(prices)
    elif local == Locals.LOAD_TEST_PRICES:
        prices = load_test_data()
        print(prices)


if __name__ == "__main__":
    run_local(local=Locals.UPDATE_TEST_PRICES)
