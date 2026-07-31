"""
Path resolution for the matf_cma_jpm_2026 replication package.

optimalportfolios-style local_path: locations resolve from an OPTIONAL flat
settings.yaml (searched in this folder, then the paper folder, then papers/),
with working defaults relative to this file so a fresh clone runs with zero
configuration. Recognized keys: CMA_DATA_PATH, OUTPUT_PATH.

load_cma_data() imports the shared papers/cma_data package by file location
through importlib (registered once in sys.modules) — no sys.path mutation.
"""
# packages
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional

REPLICATION_PATH = Path(__file__).resolve().parent


def _read_flat_yaml(file_path: Path) -> Dict[str, str]:
    """flat key: value yaml without a yaml dependency; comments and nesting ignored."""
    settings = {}
    for line in file_path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if ':' in line:
            key, value = line.split(':', 1)
            if key.strip() and value.strip():
                settings[key.strip()] = value.strip().strip("'\"")
    return settings


def load_settings() -> Dict[str, str]:
    """merged optional settings.yaml from the search path; earlier locations win."""
    merged: Dict[str, str] = {}
    for folder in (REPLICATION_PATH, REPLICATION_PATH.parent, REPLICATION_PATH.parents[1]):
        candidate = folder / 'settings.yaml'
        if candidate.exists():
            for key, value in _read_flat_yaml(candidate).items():
                merged.setdefault(key, value)
    return merged


def get_cma_data_path(settings: Optional[Dict[str, str]] = None) -> Path:
    """location of the shared papers/cma_data package."""
    settings = load_settings() if settings is None else settings
    if 'CMA_DATA_PATH' in settings:
        return Path(settings['CMA_DATA_PATH'])
    return REPLICATION_PATH.parents[1] / 'cma_data'


def get_output_path(settings: Optional[Dict[str, str]] = None) -> Path:
    """output folder for exhibit files."""
    settings = load_settings() if settings is None else settings
    if 'OUTPUT_PATH' in settings:
        return Path(settings['OUTPUT_PATH'])
    return REPLICATION_PATH / 'figures'


def load_cma_data():
    """import the shared cma_data package by file location and return the module."""
    if 'cma_data' in sys.modules:
        return sys.modules['cma_data']
    package_path = get_cma_data_path()
    init_file = package_path / '__init__.py'
    if not init_file.exists():
        raise ValueError(f"cma_data package not found, got {str(package_path)!r}; "
                         f"set CMA_DATA_PATH in settings.yaml")
    spec = importlib.util.spec_from_file_location('cma_data', init_file,
                                                  submodule_search_locations=[str(package_path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['cma_data'] = module
    spec.loader.exec_module(module)
    return module
