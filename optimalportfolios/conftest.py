"""Configure a non-interactive plotting backend for the shipped pytest suite."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure() -> None:
    """Keep pytest's conftest import out of the package's public namespace."""
    import optimalportfolios

    vars(optimalportfolios).pop("conftest", None)
