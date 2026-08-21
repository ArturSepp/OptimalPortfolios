"""External output paths for the cluster-lineage empirical harness.

All generated artifacts and caches live outside the repository. Set
``CLUSTER_LINEAGE_OUTPUT_DIR`` to reproduce a run under a different root.
"""
from __future__ import annotations

import os
from pathlib import Path

OUTPUT_ENV = "CLUSTER_LINEAGE_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR = Path.home() / "OneDrive" / "analytics" / "outputs" / "cluster_lineage_2026"


def get_output_root(*, create: bool = False) -> Path:
    """Return the configured external output root, optionally creating it."""
    configured = os.environ.get(OUTPUT_ENV)
    root = Path(configured).expanduser() if configured else DEFAULT_OUTPUT_DIR
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def get_output_path(*parts: str, create: bool = False) -> Path:
    """Return a path below the configured output root.

    When ``create`` is true, the returned path is treated as a directory and created.
    """
    path = get_output_root(create=create and not parts).joinpath(*parts)
    if create and parts:
        path.mkdir(parents=True, exist_ok=True)
    return path
