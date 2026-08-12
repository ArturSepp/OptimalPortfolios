"""
the settings.yaml path accessors.

Three lines of code with one property worth pinning: ``get_paths`` is ``lru_cache``d, so the
YAML is read once per process and every later caller gets the first read back. That is the
documented behaviour -- the docstring points at ``cache_clear`` -- but it also means a test
that monkeypatches the settings file has to clear the cache on both sides or it leaks a fake
path into every test that runs after it. The fixture below does exactly that, and the caching
itself is asserted rather than assumed.
"""
# packages
from pathlib import Path
import pytest
import yaml
# optimalportfolios
from optimalportfolios import local_path


@pytest.fixture(autouse=True)
def clear_path_cache():
    """Drop the cached settings before and after each test so none leaks into the next."""
    local_path.get_paths.cache_clear()
    yield
    local_path.get_paths.cache_clear()


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch) -> Path:
    """Point the module at a temporary settings.yaml with both documented keys."""
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': '/resources/', 'OUTPUT_PATH': '/outputs/'}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)
    return path


def test_the_shipped_settings_file_carries_both_keys() -> None:
    """The real settings.yaml travels with the package and defines both paths."""
    paths = local_path.get_paths()
    assert {'RESOURCE_PATH', 'OUTPUT_PATH'} <= set(paths)


def test_the_resource_and_output_paths_are_read_from_the_yaml(settings_file: Path) -> None:
    """Both accessors are thin lookups into the parsed settings."""
    assert local_path.get_resource_path() == '/resources/'
    assert local_path.get_output_path() == '/outputs/'


def test_the_settings_are_read_once_and_then_cached(settings_file: Path) -> None:
    """A later edit to the file is not picked up until the cache is cleared."""
    assert local_path.get_resource_path() == '/resources/'
    settings_file.write_text(yaml.safe_dump({'RESOURCE_PATH': '/changed/',
                                             'OUTPUT_PATH': '/outputs/'}))
    assert local_path.get_resource_path() == '/resources/'      # still the cached read
    local_path.get_paths.cache_clear()
    assert local_path.get_resource_path() == '/changed/'


def test_a_missing_key_raises_rather_than_returning_none(tmp_path: Path, monkeypatch) -> None:
    """A settings file without OUTPUT_PATH is a configuration error, not a None path."""
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': '/resources/'}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)
    with pytest.raises(KeyError, match='OUTPUT_PATH'):
        local_path.get_output_path()
