"""Tests for module docstrings using doctest.

Automatically discovers all packages under `src/` and runs doctests for each.

The pattern is adopted from jebel-quant/rhiza's `test_doctest.py`, like `readme_test.py` beside
it. This repository is not rhiza-managed, so the file is maintained here directly rather than
synced. Two departures from upstream:

- The upstream version reads `SOURCE_FOLDER` from `.rhiza/.env` via `python-dotenv`. There is no
  `.rhiza/` directory here and there never will be, so that read could only ever return its
  default. `src` is hardcoded instead, which also keeps `python-dotenv` out of the dependency
  tree -- it is currently neither a dependency nor in `uv.lock`.
- `root` is resolved by the shared fixture in `conftest.py`, which skips when there is no
  checkout. `src/` is not wheel content, so this test has nothing to discover in the `wheel` job.
- Upstream warns and continues on any `ImportError`, and skips when it finds no examples. Both
  paths let this gate report success while testing less than it claims: a newly broken first-party
  module would vanish from the run behind a warning nobody reads, and a tree whose examples were
  all removed or `+SKIP`-ed would still come back green. Here, only the modules listed in
  `OPTIONAL_EXTRA_MODULES` may fail to import, every other import failure fails the test, and zero
  discovered doctests is a failure rather than a skip.

Most `>>>` blocks in this package are illustrative -- they name a `prices` panel or a
`time_period` that the surrounding prose describes but the docstring never builds. Those carry
`# doctest: +SKIP`, so what remains executed is the examples that state a real expected output.
Add `+SKIP` to a new illustrative block; write a runnable example anywhere the output is cheap
and exact.
"""

from __future__ import annotations

import doctest
import importlib
from pathlib import Path

import pytest

# The package layout is a `src/` tree; see the module docstring on why this is not configurable.
SOURCE_FOLDER = "src"

# The only modules allowed to be missing from this gate, each because a module-level import of an
# optional extra makes them unimportable on the test-group install CI measures on. Everything else
# that fails to import is a broken first-party module and fails the test: warning and continuing
# would let a module drop out of the doctest gate entirely while the run still reported success.
#
# `reports/portfolio_result_pybloqs.py` is the documented exception to the module-level-optional-
# import rule (it is named in ruff's `per-file-ignores` for TID253 as a module dedicated to one
# optional backend, unreachable from `optimalportfolios/__init__.py`). Adding an entry here means
# accepting that its `>>>` examples are never executed on a core install -- justify it in review,
# and prefer a function-level import so the module stays in the gate.
OPTIONAL_EXTRA_MODULES = {
    "optimalportfolios.reports.portfolio_result_pybloqs": (
        "requires the `reports` extra (pybloqs), imported at module level by design"
    ),
}


def _iter_modules_from_path(logger, package_path: Path, src_path: Path):
    """Recursively find all Python modules in a directory.

    Raises whatever `ImportError` a module raises unless it is allowlisted in
    `OPTIONAL_EXTRA_MODULES`, so an unexpected import failure fails the doctest gate.
    """
    for path in package_path.rglob("*.py"):
        if path.name == "__init__.py":
            module_path = path.parent.relative_to(src_path)
        else:
            module_path = path.relative_to(src_path).with_suffix("")

        # Convert path to module name in an OS-independent way
        module_name = ".".join(module_path.parts)

        try:
            yield importlib.import_module(module_name)
        except ImportError as e:
            reason = OPTIONAL_EXTRA_MODULES.get(module_name)
            if reason is None:
                raise
            logger.info("Excluded from the doctest gate -- %s: %s (%s)", module_name, reason, e)


def _find_packages(src_path: Path):
    """Find all packages in the source path, including those nested under namespace packages."""
    for init_file in src_path.rglob("__init__.py"):
        package_dir = init_file.parent
        # Only yield top-level packages (those whose parent doesn't have __init__.py or is src_path)
        parent = package_dir.parent
        if parent == src_path or not (parent / "__init__.py").exists():
            yield package_dir


def test_doctests(
    logger, root, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Run doctests for each package directory."""
    src_path = root / SOURCE_FOLDER

    logger.info("Starting doctest discovery in: %s", src_path)
    if not src_path.exists():
        logger.info("Source directory not found: %s -- skipping doctests", src_path)
        pytest.skip(f"Source directory not found: {src_path}")

    # Add source path to sys.path with automatic cleanup
    monkeypatch.syspath_prepend(str(src_path))
    logger.debug("Prepended to sys.path: %s", src_path)

    total_tests = 0
    total_failures = 0
    failed_modules = []

    # Find all packages in the source path (supports namespace packages)
    for package_dir in _find_packages(src_path):
        if package_dir.is_dir() and (package_dir / "__init__.py").exists():
            # Import the package
            package_name = package_dir.name
            logger.info("Discovered package: %s", package_name)
            # No `except ImportError` around this loop: an unexpected import failure must reach
            # the test result. `_iter_modules_from_path` already excuses the allowlisted modules.
            modules = list(_iter_modules_from_path(logger, package_dir, src_path))
            logger.debug("%d module(s) found in package %s", len(modules), package_name)

            for module in modules:
                logger.debug("Running doctests for module: %s", module.__name__)
                # Disable pytest's stdout capture during doctest to avoid interference
                with capsys.disabled():
                    results = doctest.testmod(
                        module,
                        verbose=False,
                        optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE),
                    )
                total_tests += results.attempted

                if results.failed:
                    logger.warning(
                        "Doctests failed for %s: %d/%d failed",
                        module.__name__,
                        results.failed,
                        results.attempted,
                    )
                    total_failures += results.failed
                    failed_modules.append((module.__name__, results.failed, results.attempted))
                else:
                    logger.debug(
                        "Doctests passed for %s (%d test(s))",
                        module.__name__,
                        results.attempted,
                    )

    if failed_modules:
        formatted = "\n".join(
            f"  {name}: {failed}/{attempted} failed" for name, failed, attempted in failed_modules
        )
        msg = (
            f"Doctest summary: {total_tests} tests across {len(failed_modules)} module(s)\n"
            f"Failures: {total_failures}\n"
            f"Failed modules:\n{formatted}"
        )
        logger.error("%s", msg)
        assert total_failures == 0, msg
    else:
        logger.info("Doctest summary: %d tests, 0 failures", total_tests)

    # Fail closed rather than skip. Zero discovered examples means every `>>>` block was removed,
    # marked `+SKIP`, or lost to a discovery bug -- a gate with nothing left to run must not report
    # success. Kept as a single `assert` so the check itself is always executed.
    assert total_tests > 0, (
        f"No doctests were found under {src_path}. Discovery reached "
        f"{len(list(_find_packages(src_path)))} package(s); every `>>>` example is either absent "
        f"or marked `# doctest: +SKIP`."
    )


class TestImportFailuresFailClosed:
    """Tests that only allowlisted modules are allowed to drop out of the doctest gate."""

    @staticmethod
    def _write_package(tmp_path: Path, name: str, body: str) -> Path:
        """Create a single-module package under `tmp_path` and return its directory."""
        package_dir = tmp_path / name
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(body, encoding="utf-8")
        return package_dir

    def test_unexpected_import_error_propagates(self, logger, tmp_path, monkeypatch):
        """A first-party module that fails to import fails the test instead of warning."""
        package_dir = self._write_package(
            tmp_path, "unexpectedly_broken_pkg", "import a_module_that_does_not_exist\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ImportError):
            list(_iter_modules_from_path(logger, package_dir, tmp_path))

    def test_allowlisted_import_error_is_excluded(self, logger, tmp_path, monkeypatch):
        """A module named in `OPTIONAL_EXTRA_MODULES` is excluded rather than raised."""
        package_dir = self._write_package(
            tmp_path, "optional_extra_pkg", "import a_module_that_does_not_exist\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setitem(
            OPTIONAL_EXTRA_MODULES, "optional_extra_pkg", "synthetic entry for this test"
        )
        assert list(_iter_modules_from_path(logger, package_dir, tmp_path)) == []

    def test_allowlist_entries_are_first_party_and_carry_a_reason(self):
        """Each allowlisted entry names a module in this package and states why it is excused."""
        for module_name, reason in OPTIONAL_EXTRA_MODULES.items():
            assert module_name.startswith("optimalportfolios."), module_name
            assert reason.strip(), module_name
