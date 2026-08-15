"""Verify that the thin Colab notebook remains an output-free mirror of D6."""

from __future__ import annotations

import difflib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "getting_started" / "production_quickstart.py"
NOTEBOOK_PATH = SCRIPT_PATH.with_suffix(".ipynb")
SYNC_TAG = "d6-source"
INSTALL_COMMAND = "%pip install -q optimalportfolios"
REQUIRED_LINKS = (
    "https://optimalportfolios.readthedocs.io/en/latest/quickstart.html",
    "https://github.com/ArturSepp/OptimalPortfolios/blob/main/"
    "examples/getting_started/production_quickstart.py",
)


def _source(cell: dict) -> str:
    """Return a notebook cell's source as one normalized string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    return source.replace("\r\n", "\n").rstrip() + "\n"


def _fail(message: str) -> int:
    """Print one actionable failure and return a failing exit status."""
    print(f"quickstart notebook check failed: {message}")
    return 1


def main() -> int:
    """Check notebook structure, clean state, links, install, version, and source parity."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        return _fail("nbformat must be 4")

    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells):
        if cell.get("execution_count") is not None or cell.get("outputs"):
            return _fail(f"code cell {index} contains committed execution state")

    tagged = [
        cell for cell in code_cells if SYNC_TAG in cell.get("metadata", {}).get("tags", [])
    ]
    if len(tagged) != 1:
        return _fail(f"expected exactly one code cell tagged {SYNC_TAG!r}")

    expected = SCRIPT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip() + "\n"
    actual = _source(tagged[0])
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=str(SCRIPT_PATH.relative_to(REPO_ROOT)),
                tofile=f"{NOTEBOOK_PATH.relative_to(REPO_ROOT)}:{SYNC_TAG}",
            )
        )
        print(diff)
        return _fail("the tagged cell has drifted from the production quickstart")

    code = "\n".join(_source(cell) for cell in code_cells)
    if INSTALL_COMMAND not in code:
        return _fail(f"missing released-package install command {INSTALL_COMMAND!r}")
    if "optimalportfolios.__version__" not in code:
        return _fail("the installed package version is not displayed")

    markdown = "\n".join(_source(cell) for cell in cells if cell.get("cell_type") == "markdown")
    missing_links = [link for link in REQUIRED_LINKS if link not in markdown]
    if missing_links:
        return _fail(f"missing required link(s): {', '.join(missing_links)}")

    print(
        "quickstart notebook check passed: exact D6 source, released-package install, "
        "version display, required links, and no saved outputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
