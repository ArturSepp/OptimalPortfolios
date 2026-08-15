"""Configure a non-interactive plotting backend for the shipped pytest suite.

The selection is a function rather than a bare module-level ``if`` so that both of its outcomes
can be tested. As inline code it was only ever exercised when ``MPLBACKEND`` happened to be unset
in the ambient environment, which made its coverage depend on where the suite ran: unset locally,
set by ``ci.yml`` on the runner. That also meant CI never executed the mechanism this file exists
to provide for users of ``pytest --pyargs optimalportfolios``.
"""

import logging
import os
from pathlib import Path
from typing import MutableMapping, Optional

import pytest


def configure_matplotlib_backend(
    environ: Optional[MutableMapping[str, str]] = None,
) -> Optional[str]:
    """Force a non-interactive backend unless one was chosen explicitly.

    Args:
        environ: Environment mapping to read and update. Defaults to ``os.environ``; tests pass a
            synthetic mapping so neither outcome depends on the ambient environment.

    Returns:
        The backend name that was forced, or None when an explicit choice was left in place.
    """
    environ = os.environ if environ is None else environ
    if environ.get("MPLBACKEND"):
        return None
    environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    return "Agg"


configure_matplotlib_backend()


def pytest_configure() -> None:
    """Keep pytest's conftest import out of the package's public namespace."""
    import optimalportfolios

    vars(optimalportfolios).pop("conftest", None)


def _find_root() -> Path | None:
    """Walk up from this file for the repository checkout, or None when installed.

    Keyed on ``pyproject.toml`` alone. Pairing it with ``README.md`` would conflate two different
    situations: no checkout at all, and a checkout whose README is missing. The second is then
    reported as the first, which is misleading -- so each file is checked where it is needed, and
    a checkout whose README is missing fails at the point the README is read rather than
    disappearing behind this function's skip.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Not covered by design: reached only when this package is imported from an installed wheel
    # rather than a checkout, and coverage is measured on the primary checkout cell. The `wheel`
    # job is what exercises this line, and that job deliberately does not measure coverage.
    return None  # pragma: no cover


@pytest.fixture(scope="session")
def root() -> Path:
    """The repository checkout root.

    Skips rather than fails when there is no checkout. The `wheel` job in ci.yml installs the
    built wheel into a clean environment and runs `pytest --pyargs optimalportfolios` from
    outside the repository, where no `pyproject.toml` is on the path at all, so a test that reads
    repository files has nothing to assert there. Skipping keeps that job green without weakening
    the checkout run, where this fixture always resolves.
    """
    found = _find_root()
    # Same reason as the `return None` above: the skip is the `wheel` job's path, not the
    # coverage cell's.
    if found is None:  # pragma: no cover
        pytest.skip("no repository checkout: running against an installed wheel")
    return found


@pytest.fixture
def logger(request: pytest.FixtureRequest) -> logging.Logger:
    """A per-test logger, surfaced by pytest's own capture at `--log-level`."""
    return logging.getLogger(request.node.name)
