"""Load legacy replication modules whose source was lost in the 2026-08-14 incident.

F7 reconstructed ``configs`` and ``run_backtests`` as readable source. The remaining
historical E-stage entry points still use their surviving CPython 3.12 artifacts until a
separate reconstruction is commissioned; ordinary source always takes precedence.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path


class RelocatedLoader(importlib.machinery.SourcelessFileLoader):
    """Execute bytecode while exposing its original source-relative location."""

    def exec_module(self, module) -> None:
        """Set the lost source location before executing path-relative constants."""
        name = module.__name__.rsplit(".", 1)[-1].removeprefix("_executed_")
        module.__file__ = str(Path(self.path).parent.parent / f"{name}.py")
        super().exec_module(module)


class StudyFinder(importlib.abc.MetaPathFinder):
    """Resolve a missing replication module from surviving CPython 3.12 bytecode."""

    def find_spec(self, fullname, path=None, target=None):
        """Return a bytecode loader only when ordinary source is absent."""
        prefix = "papers.cluster_lineage_2026.replication."
        if not fullname.startswith(prefix):
            return None
        name = fullname.removeprefix(prefix)
        root = Path(__file__).resolve().parent
        if "." in name or (root / f"{name}.py").exists():
            return None
        matches = sorted((root / "recovery_bytecode").glob(f"{name}.cpython-312.pyc"))
        if not matches:
            return None
        loader = RelocatedLoader(fullname, str(matches[-1]))
        spec = importlib.machinery.ModuleSpec(fullname, loader, origin=str(matches[-1]))
        spec.has_location = True
        return spec


def install() -> None:
    """Install the narrowly scoped legacy finder once."""
    if not any(isinstance(item, StudyFinder) for item in sys.meta_path):
        sys.meta_path.insert(0, StudyFinder())


def load_executed(name: str):
    """Load one exact legacy module under a private recovery name."""
    install()
    path = Path(__file__).resolve().parent / "recovery_bytecode" / f"{name}.cpython-312.pyc"
    if not path.exists():
        raise FileNotFoundError(path)
    fullname = f"papers.cluster_lineage_2026.replication._executed_{name}"
    loader = RelocatedLoader(fullname, str(path))
    spec = importlib.util.spec_from_loader(fullname, loader)
    if spec is None:
        raise ImportError(f"cannot load executed module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    try:
        loader.exec_module(module)
    except Exception:
        sys.modules.pop(fullname, None)
        raise
    return module
