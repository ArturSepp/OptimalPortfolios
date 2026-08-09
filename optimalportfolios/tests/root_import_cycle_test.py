"""
no library module imports the top-level ``optimalportfolios`` package.

``optimalportfolios/__init__.py`` star-imports the subpackages in sequence. A library module
that imports back through the package root therefore re-enters a *partially initialised*
module, and whether the name it wants is already bound depends on nothing but the order of
those star-imports.

This bit for real. ``optimization/taa/maximise_alpha_with_target_yield.py`` did
``from optimalportfolios import filter_covar_and_vectors_for_nans``. That symbol is defined in
``utils/filter_nans.py``, star-imported four lines above ``optimization`` - so the package
imported cleanly, and reversing those six lines produced
``ImportError: cannot import name 'filter_covar_and_vectors_for_nans' from partially
initialized module 'optimalportfolios' (most likely due to a circular import)``. Two further
modules, ``optimization/wrapper_rolling_portfolios.py`` and ``reports/marginal_backtest.py``,
did ``import optimalportfolios as opt``; those survived only because attribute lookup on a
module object is deferred to call time, which makes them the same latent cycle one step from
failing.

A `from` import is the failing form and a plain `import` the merely fragile one, so the check
covers both: a library module imports the module that *defines* what it needs, never the root.

Examples and tests are excluded. An example is read top to bottom by a user, for whom
``import optimalportfolios as opt`` is the documented entry point, and a test imports the
package the way a caller would.

To confirm this check can fail, put ``from optimalportfolios import Constraints`` at the top of
any module under ``optimization/``: the site is reported below by file and line. That was run
before this file was committed.
"""
# packages
import ast
from pathlib import Path
from typing import List, Tuple
# optimalportfolios
import optimalportfolios

PACKAGE_ROOT: Path = Path(optimalportfolios.__file__).parent
PACKAGE_NAME: str = optimalportfolios.__name__

# directories whose contents are scripts rather than library code: an example is read by a user
# and a test imports the package the way a caller would, so neither carries the convention
EXCLUDED_PARTS: Tuple[str, ...] = ('examples', 'tests', 'notebooks')


def find_root_package_imports() -> List[str]:
    """Return one line per library module importing the top-level package at module scope."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name.endswith(('_test.py', '_tests.py')) or path.name == '__init__.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        for node in ast.walk(tree):
            # `from optimalportfolios import x` -- fails outright on an unlucky ordering
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == PACKAGE_NAME:
                names = ', '.join(alias.name for alias in node.names)
                offenders.append(f"{rel}:{node.lineno}: from {PACKAGE_NAME} import {names}")
            # `import optimalportfolios [as opt]` -- the same cycle, deferred to call time
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == PACKAGE_NAME:
                        as_part = f" as {alias.asname}" if alias.asname else ''
                        offenders.append(f"{rel}:{node.lineno}: import {PACKAGE_NAME}{as_part}")
    return offenders


def test_no_library_module_imports_the_root_package() -> None:
    """a library module importing the package root re-enters it partially initialised"""
    offenders = find_root_package_imports()
    assert not offenders, (
            "library module imports the top-level package, creating an import cycle that survives "
            "only on the star-import order in __init__.py; import the defining module instead:\n"
            + '\n'.join(offenders))


if __name__ == '__main__':
    for offender in find_root_package_imports():
        print(offender)
