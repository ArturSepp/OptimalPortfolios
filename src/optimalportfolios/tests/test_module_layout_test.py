"""Repository contracts separating pytest modules from local diagnostics."""

import ast
import importlib.util
from pathlib import Path


PACKAGE_TEST_SUPPORT_MODULES = {
    "src/optimalportfolios/tests/data/multiasset.py",
    "src/optimalportfolios/tests/data_masks.py",
}
PACKAGE_RUN_LOCAL_SUPPORT_MODULES = {
    "src/optimalportfolios/run_local/data/etf_prices.py",
}


def _relative(path: Path, root: Path) -> str:
    """Return a repository-relative path with stable POSIX separators."""
    return path.relative_to(root).as_posix()


def _is_package_test_module(path: Path, package_root: Path) -> bool:
    """Return whether ``path`` is a Python module below a package ``tests`` directory."""
    return path.suffix == ".py" and "tests" in path.relative_to(package_root).parts


def _has_test_definition(path: Path) -> bool:
    """Return whether a module contains a pytest-collectable function or method name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def _is_main_comparison(node: ast.AST) -> bool:
    """Return whether an AST node compares ``__name__`` with ``__main__``."""
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "__main__"
    )


def _has_main_guard(path: Path) -> bool:
    """Return whether a module contains an executable ``__main__`` guard."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(node, ast.If) and _is_main_comparison(node.test) for node in tree.body)


def _defines_top_level(
    path: Path,
    node_type: type[ast.AST] | tuple[type[ast.AST], ...],
    name: str,
) -> bool:
    """Return whether a module defines a named top-level class or function."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, node_type) and getattr(node, "name", None) == name
        for node in tree.body
    )


def _imports_run_local(path: Path) -> bool:
    """Return whether a module imports a source-adjacent development runner package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any("run_local" in alias.name.split(".") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module_parts = (node.module or "").split(".")
            imports_package = any(alias.name == "run_local" for alias in node.names)
            if "run_local" in module_parts or imports_package:
                return True
    return False


def test_package_pytest_filename_classifier() -> None:
    """Only the trailing ``_test.py`` form denotes a package pytest module."""
    assert Path("weights_drift_test.py").name.endswith("_test.py")
    assert not Path("test_weights_drift.py").name.endswith("_test.py")
    assert not Path("weights_test_dev.py").name.endswith("_test.py")
    assert not Path("weights_local.py").name.endswith("_test.py")
    assert not Path("weights_run.py").name.endswith("_test.py")


def test_package_test_layout_is_unambiguous(root: Path) -> None:
    """Package tests are pytest-only and package diagnostics never use local suffixes."""
    package_root = root / "src" / "optimalportfolios"
    python_modules = sorted(package_root.rglob("*.py"))
    package_test_modules = [
        path for path in python_modules if _is_package_test_module(path, package_root)
    ]
    local_modules = {
        _relative(path, root) for path in python_modules if path.name.endswith("_local.py")
    }
    ambiguous_test_modules = {
        _relative(path, root)
        for path in package_test_modules
        if path.name != "__init__.py"
        and not path.name.endswith("_test.py")
    }
    pytest_modules = [path for path in package_test_modules if path.name.endswith("_test.py")]
    executable_test_modules = {
        _relative(path, root) for path in pytest_modules if _has_main_guard(path)
    }
    empty_test_modules = [
        _relative(path, root) for path in pytest_modules if not _has_test_definition(path)
    ]
    assert not local_modules, f"local diagnostics below the package: {local_modules}"
    assert ambiguous_test_modules == PACKAGE_TEST_SUPPORT_MODULES
    assert not executable_test_modules, (
        f"pytest modules with executable runners: {executable_test_modules}"
    )
    assert not empty_test_modules, f"pytest-shaped modules with no tests: {empty_test_modules}"


def test_package_development_runner_layout(root: Path) -> None:
    """Source-adjacent development runners follow one discoverable execution contract."""
    package_root = root / "src" / "optimalportfolios"
    python_modules = sorted(package_root.rglob("*.py"))
    run_local_modules = [
        path for path in python_modules
        if "run_local" in path.relative_to(package_root).parts and path.name != "__init__.py"
    ]
    runner_modules = [path for path in run_local_modules if path.name.endswith("_run.py")]
    support_modules = {
        _relative(path, root) for path in run_local_modules if not path.name.endswith("_run.py")
    }
    misplaced_runners = {
        _relative(path, root)
        for path in python_modules
        if path.name.endswith("_run.py") and path not in runner_modules
    }
    runners_without_main = {
        _relative(path, root) for path in runner_modules if not _has_main_guard(path)
    }
    runners_without_locals = {
        _relative(path, root)
        for path in runner_modules
        if not _defines_top_level(path, ast.ClassDef, "Locals")
    }
    runners_without_entrypoint = {
        _relative(path, root)
        for path in runner_modules
        if not _defines_top_level(path, (ast.FunctionDef, ast.AsyncFunctionDef), "run_local")
    }
    runners_with_tests = {
        _relative(path, root) for path in runner_modules if _has_test_definition(path)
    }
    production_imports = {
        _relative(path, root)
        for path in python_modules
        if "run_local" not in path.relative_to(package_root).parts
        and "tests" not in path.relative_to(package_root).parts
        and _imports_run_local(path)
    }

    assert runner_modules, "expected source-adjacent development runners"
    assert support_modules == PACKAGE_RUN_LOCAL_SUPPORT_MODULES
    assert not misplaced_runners, f"runner modules outside run_local/: {misplaced_runners}"
    assert not runners_without_main, f"runners without __main__: {runners_without_main}"
    assert not runners_without_locals, f"runners without Locals: {runners_without_locals}"
    assert not runners_without_entrypoint, (
        f"runners without run_local(local): {runners_without_entrypoint}"
    )
    assert not runners_with_tests, f"development runners contain pytest tests: {runners_with_tests}"
    assert not production_imports, f"production modules importing run_local: {production_imports}"


def test_example_discovery_excludes_local_diagnostics(root: Path) -> None:
    """Local diagnostics remain manual and are excluded from unattended example lanes."""
    script = root / ".github" / "scripts" / "run_examples.py"
    spec = importlib.util.spec_from_file_location("run_examples_layout_check", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    local_modules = sorted((root / "examples").rglob("*_local.py"))
    discovered = set(module.discover())
    local_modules_with_tests = [
        _relative(path, root) for path in local_modules if _has_test_definition(path)
    ]

    assert local_modules, "expected the manual diagnostics below examples/"
    assert not discovered.intersection(local_modules)
    assert not local_modules_with_tests, (
        f"local diagnostics contain pytest test definitions: {local_modules_with_tests}"
    )
