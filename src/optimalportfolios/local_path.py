"""Resolve configured resource and output directories portably.

``settings.yaml`` may override either directory with an absolute or relative path.  A missing
file, an empty value, or the shipped parent-directory placeholder uses checkout-aware defaults:
the repository root for resources and ``<repository>/outputs`` for generated files.  A checkout
is the src layout, ``<repository>/src/optimalportfolios`` beside ``<repository>/pyproject.toml``.
Installed packages fall back to the current working directory because no checkout root is
available.

A non-empty ``OPTIMALPORTFOLIOS_OUTPUT_PATH`` environment variable takes precedence over
``settings.yaml`` and the defaults for the output directory, which is then created if missing.
Set it to keep generated files out of a synchronised or shared checkout without editing the
tracked settings file.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml


_PACKAGE_DIR = Path(__file__).resolve().parent
_SETTINGS_PATH = _PACKAGE_DIR / 'settings.yaml'
_OUTPUT_PATH_ENV = 'OPTIMALPORTFOLIOS_OUTPUT_PATH'


def _checkout_root() -> Path | None:
    """Return the repository root when this module is running from a src-layout checkout."""
    source_dir = _PACKAGE_DIR.parent
    candidate = source_dir.parent
    if source_dir.name == 'src' and (candidate / 'pyproject.toml').is_file():
        return candidate
    return None


def _as_portable_string(path: Path) -> str:
    """Return an absolute path with forward separators on every platform."""
    return path.expanduser().resolve().as_posix()


def _is_placeholder(value: object) -> bool:
    """Return whether a YAML value is empty or the shipped parent-directory placeholder."""
    if value is None:
        return True
    normalized = str(value).strip().replace(chr(92), '/').rstrip('/')
    return normalized in {'', '..'}


def _default_resource_path() -> Path:
    """Return the checkout root, or the working directory for an installed package."""
    return _checkout_root() or Path.cwd()


def _default_output_path() -> Path:
    """Create and return the first writable default output directory."""
    checkout_root = _checkout_root()
    candidates = [checkout_root / 'outputs', Path.cwd()] if checkout_root else [Path.cwd()]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    raise OSError('no writable default output directory is available')


@lru_cache(maxsize=1)
def get_paths() -> Dict[str, object]:
    """Read the path settings once; call ``cache_clear`` to force a re-read."""
    if not _SETTINGS_PATH.is_file():
        return {}
    with _SETTINGS_PATH.open(encoding='utf-8') as settings:
        settings_data = yaml.safe_load(settings)
    if settings_data is None:
        return {}
    if not isinstance(settings_data, dict):
        raise TypeError('settings.yaml must contain a mapping')
    return settings_data


def _configured_path(key: str, default_factory) -> str:
    """Resolve one configured path, preserving missing-key errors in an existing file."""
    paths = get_paths()
    if not _SETTINGS_PATH.is_file():
        return _as_portable_string(default_factory())
    value = paths[key]
    if _is_placeholder(value):
        return _as_portable_string(default_factory())
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _SETTINGS_PATH.parent / path
    return _as_portable_string(path)


def get_resource_path() -> str:
    """Return the configured resource directory or its checkout-aware default."""
    return _configured_path('RESOURCE_PATH', _default_resource_path)


def get_output_path() -> str:
    """Return the environment override, the configured output directory, or a writable default.

    An override that cannot be created raises ``OSError`` rather than falling back to a default.
    """
    override = os.environ.get(_OUTPUT_PATH_ENV, '').strip()
    if override:
        path = Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return _as_portable_string(path)
    return _configured_path('OUTPUT_PATH', _default_output_path)
