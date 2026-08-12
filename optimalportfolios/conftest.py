"""Configure a non-interactive plotting backend for the shipped pytest suite."""

import os

if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")


def pytest_configure() -> None:
    """Keep pytest's conftest import out of the package's public namespace."""
    import optimalportfolios

    vars(optimalportfolios).pop("conftest", None)
