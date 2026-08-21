"""
Library modules do not import the top-level ``optimalportfolios`` package at module scope.

``src/optimalportfolios/__init__.py`` star-imports subpackages in sequence. A library module that
imports through that package root can therefore re-enter a partially initialized module, and
whether the requested name is already bound depends on the order of those star imports.

Examples and tests are excluded because they intentionally exercise the public package API.
Function- and method-local imports are also allowed: they execute only when the callable runs,
after normal package initialization. Imports nested in module-level control flow remain covered
because they still execute while the module is imported.
"""
# packages
import ast
from pathlib import Path
from typing import Iterator, List, Tuple
# optimalportfolios
import optimalportfolios

PACKAGE_ROOT: Path = Path(optimalportfolios.__file__).parent
PACKAGE_NAME: str = optimalportfolios.__name__
EXCLUDED_PARTS: Tuple[str, ...] = ('examples', 'tests', 'notebooks')


def _walk_module_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Yield nodes outside function, method, and class scopes."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_module_scope(child)


def find_root_package_imports() -> List[str]:
    """Return module-scope imports of the top-level package in library modules."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name.endswith(('_test.py', '_tests.py')) or path.name == '__init__.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        for node in _walk_module_scope(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == PACKAGE_NAME:
                names = ', '.join(alias.name for alias in node.names)
                offenders.append(f"{rel}:{node.lineno}: from {PACKAGE_NAME} import {names}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == PACKAGE_NAME:
                        as_part = f" as {alias.asname}" if alias.asname else ''
                        offenders.append(f"{rel}:{node.lineno}: import {PACKAGE_NAME}{as_part}")
    return offenders


def test_no_library_module_imports_the_root_package() -> None:
    """Module-scope root imports must not create initialization-order cycles."""
    offenders = find_root_package_imports()
    assert not offenders, (
        "library module imports the top-level package at module scope; import the defining "
        "module instead:\n" + '\n'.join(offenders)
    )
