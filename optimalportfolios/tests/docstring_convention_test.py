"""Repository-wide tests for the documented Google-style docstring convention."""

import ast
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
NUMPYDOC_SECTION = re.compile(
    r"(?m)^\s*(Parameters|Returns|Raises)\s*\n\s*-{3,}\s*$"
)
DOCSTRING_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _iter_docstrings(path: Path):
    """Yield each located docstring and its owning AST node from ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_NODES):
            docstring = ast.get_docstring(node, clean=False)
            if docstring is not None:
                yield node, docstring


def test_docstrings_do_not_use_numpydoc_section_headings() -> None:
    """All package docstrings use Google sections, never underlined numpydoc headings."""
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for node, docstring in _iter_docstrings(path):
            for match in NUMPYDOC_SECTION.finditer(docstring):
                relative = path.relative_to(PACKAGE_ROOT.parent)
                violations.append(f"{relative}:{getattr(node, 'lineno', 1)}: {match.group(1)}")

    assert not violations, "numpydoc-style sections found:\n" + "\n".join(violations)
