"""Configure a non-interactive plotting backend for the shipped pytest suite.

The selection is a function rather than a bare module-level ``if`` so that both of its outcomes
can be tested. As inline code it was only ever exercised when ``MPLBACKEND`` happened to be unset
in the ambient environment, which made its coverage depend on where the suite ran: unset locally,
set by ``ci.yml`` on the runner. That also meant CI never executed the mechanism this file exists
to provide for users of ``pytest --pyargs optimalportfolios``.
"""

import os
from typing import MutableMapping, Optional


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
