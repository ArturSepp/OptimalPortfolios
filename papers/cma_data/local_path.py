"""
Path resolution for the shared paper-data layer, optimalportfolios style.

Paths resolve from an OPTIONAL flat settings.yaml, searched in this order:
the current working directory, this folder, the papers/ folder above it.
Recognized keys: SNAPSHOTS_PATH. When no settings.yaml exists (the normal
case for a fresh clone), every path resolves relative to this file, so the
package works with zero configuration. settings.yaml is naturally untracked
(the repository ignores *.yaml) and exists only to override locations on
machines with a nonstandard layout.

No sys.path mutation anywhere: consumers import this package by file
location (see the per-paper local_path.py modules).
"""
# packages
from pathlib import Path
from typing import Dict, Optional

CMA_DATA_PATH = Path(__file__).resolve().parent


def _read_flat_yaml(file_path: Path) -> Dict[str, str]:
    """read a flat key: value yaml without a yaml dependency; ignores comments and nesting."""
    settings = {}
    for line in file_path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if ':' in line:
            key, value = line.split(':', 1)
            if key.strip() and value.strip():
                settings[key.strip()] = value.strip().strip("'\"")
    return settings


def load_settings() -> Dict[str, str]:
    """merged settings from the search path; later locations do not override earlier ones."""
    merged: Dict[str, str] = {}
    for folder in (Path.cwd(), CMA_DATA_PATH, CMA_DATA_PATH.parent):
        candidate = folder / 'settings.yaml'
        if candidate.exists():
            for key, value in _read_flat_yaml(candidate).items():
                merged.setdefault(key, value)
    return merged


def get_snapshots_path(settings: Optional[Dict[str, str]] = None) -> Path:
    """root folder of the versioned data snapshots."""
    settings = load_settings() if settings is None else settings
    if 'SNAPSHOTS_PATH' in settings:
        return Path(settings['SNAPSHOTS_PATH'])
    return CMA_DATA_PATH / 'snapshots'
