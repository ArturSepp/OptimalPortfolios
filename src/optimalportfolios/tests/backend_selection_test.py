"""Tests for the shipped conftest's plotting-backend selection.

``pytest --pyargs optimalportfolios`` is the documented post-install check, and it runs wherever
the user is — including on a workstation whose default matplotlib backend wants a display. The
conftest forces ``Agg`` there, while leaving an explicitly chosen backend alone so that someone
debugging a plot interactively is not overridden.

Both outcomes are asserted against a synthetic mapping rather than the real environment. That is
the point of the function existing at all: read from ``os.environ`` and the result depends on
where the suite runs, which is how this branch came to be covered locally and uncovered in CI,
where ``ci.yml`` sets ``MPLBACKEND`` itself.
"""

import pytest

from optimalportfolios.conftest import configure_matplotlib_backend


def test_unset_backend_is_forced_to_agg() -> None:
    """With nothing chosen, the non-interactive backend is selected and recorded."""
    environ: dict = {}

    forced = configure_matplotlib_backend(environ=environ)

    assert forced == "Agg"
    assert environ["MPLBACKEND"] == "Agg"


def test_explicitly_chosen_backend_is_left_alone() -> None:
    """An explicit choice survives: the function reports no override and changes nothing."""
    environ = {"MPLBACKEND": "QtAgg"}

    forced = configure_matplotlib_backend(environ=environ)

    assert forced is None
    assert environ["MPLBACKEND"] == "QtAgg", "an explicit backend was overwritten"


def test_empty_backend_value_counts_as_unset() -> None:
    """An empty string is treated as no choice rather than as a backend named ''.

    ``MPLBACKEND=`` appears in shell environments that export the name without a value. Taking it
    literally hands matplotlib an unusable backend name instead of falling back to ``Agg``.
    """
    environ = {"MPLBACKEND": ""}

    forced = configure_matplotlib_backend(environ=environ)

    assert forced == "Agg"
    assert environ["MPLBACKEND"] == "Agg"


def test_the_real_environment_has_a_backend_by_the_time_tests_run() -> None:
    """The module-level call already ran, so the ambient environment is configured either way."""
    import os

    assert os.environ.get("MPLBACKEND"), "the shipped conftest left no backend selected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
