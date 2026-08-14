"""
the settings.yaml path accessors.

``get_paths`` is ``lru_cache``d, so the YAML is read once per process and every later caller
gets the first read back. That is documented -- the docstring points at ``cache_clear`` -- but
it also means a test that monkeypatches the settings file must clear the cache on both sides or
it leaks a fake path into every test that runs afterwards. The autouse fixture below does
exactly that, and the caching itself is asserted rather than assumed.

The resolution rules matter more than they look. A path may be absent, empty, the shipped ``..``
placeholder, relative, or absolute, and each resolves differently: the first three fall back to
checkout-aware defaults, a relative path is anchored to ``settings.yaml``'s own directory rather
than the working directory, and only an absolute path is taken as given. Anchoring a relative
path to the CWD instead would still produce a plausible directory -- one that moves depending on
where the process was started.

Every returned path is normalised to forward slashes, so a Windows-authored settings file and a
POSIX one agree. That is asserted directly, since a backslash surviving into a path is exactly
the defect this module was rewritten to fix (issue #43).
"""
# packages
import os
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
    resource_path = tmp_path / 'resources'
    output_path = tmp_path / 'outputs'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': str(resource_path),
                                    'OUTPUT_PATH': str(output_path)}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)
    return path


def test_the_shipped_settings_file_carries_both_keys() -> None:
    """The real settings.yaml travels with the package and defines both paths."""
    paths = local_path.get_paths()
    assert {'RESOURCE_PATH', 'OUTPUT_PATH'} <= set(paths)


def test_the_resource_and_output_paths_are_read_from_the_yaml(settings_file: Path) -> None:
    """Both accessors are thin lookups into the parsed settings."""
    paths = yaml.safe_load(settings_file.read_text())
    assert local_path.get_resource_path() == Path(paths['RESOURCE_PATH']).as_posix()
    assert local_path.get_output_path() == Path(paths['OUTPUT_PATH']).as_posix()


def test_the_settings_are_read_once_and_then_cached(settings_file: Path) -> None:
    """A later edit to the file is not picked up until the cache is cleared."""
    initial = Path(yaml.safe_load(settings_file.read_text())['RESOURCE_PATH']).as_posix()
    changed = settings_file.parent / 'changed'
    assert local_path.get_resource_path() == initial
    settings_file.write_text(yaml.safe_dump({'RESOURCE_PATH': str(changed),
                                             'OUTPUT_PATH': str(settings_file.parent / 'outputs')}))
    assert local_path.get_resource_path() == initial      # still the cached read
    local_path.get_paths.cache_clear()
    assert local_path.get_resource_path() == changed.as_posix()


def test_a_missing_key_raises_rather_than_returning_none(tmp_path: Path, monkeypatch) -> None:
    """A settings file without OUTPUT_PATH is a configuration error, not a None path."""
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': '/resources/'}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)
    with pytest.raises(KeyError, match='OUTPUT_PATH'):
        local_path.get_output_path()


@pytest.mark.parametrize('settings_value', [None, '..'])
def test_placeholder_output_falls_back_to_a_writable_checkout_directory(
        settings_value, tmp_path: Path, monkeypatch) -> None:
    """A placeholder output resolves to a writable checkout or installed-package default."""
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': settings_value,
                                    'OUTPUT_PATH': settings_value}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    output_path = Path(local_path.get_output_path())
    checkout_root = local_path._checkout_root()
    expected_output = checkout_root / 'outputs' if checkout_root else Path.cwd()

    assert output_path == expected_output
    assert output_path.is_dir()
    assert os.access(output_path, os.W_OK)
    assert chr(92) not in local_path.get_output_path()


def test_absent_settings_file_uses_portable_checkout_defaults(tmp_path: Path, monkeypatch) -> None:
    """A missing YAML file uses portable checkout or installed-package defaults."""
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', tmp_path / 'absent.yaml')
    checkout_root = local_path._checkout_root()
    expected_resource = checkout_root or Path.cwd()
    expected_output = checkout_root / 'outputs' if checkout_root else Path.cwd()

    assert local_path.get_resource_path() == expected_resource.as_posix()
    assert local_path.get_output_path() == expected_output.as_posix()
    assert chr(92) not in local_path.get_resource_path()
    assert chr(92) not in local_path.get_output_path()


# --------------------------------------------------------------------------- #
# how a configured value is resolved
# --------------------------------------------------------------------------- #
def test_a_relative_path_is_anchored_to_the_settings_file_not_the_working_directory(
        tmp_path: Path, monkeypatch) -> None:
    """A relative RESOURCE_PATH resolves against ``settings.yaml``'s own directory.

    Anchoring to the CWD instead would still yield a plausible directory, but one that moves
    with wherever the process happened to be started from.
    """
    settings_dir = tmp_path / 'config'
    settings_dir.mkdir()
    path = settings_dir / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': 'data', 'OUTPUT_PATH': 'out'}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)
    monkeypatch.chdir(tmp_path)                       # a CWD that is *not* the settings dir

    assert local_path.get_resource_path() == (settings_dir / 'data').resolve().as_posix()
    assert local_path.get_output_path() == (settings_dir / 'out').resolve().as_posix()


def test_an_absolute_path_is_taken_as_given(tmp_path: Path, monkeypatch) -> None:
    """An absolute value is used unchanged, only normalised to forward slashes."""
    target = tmp_path / 'absolute-resources'
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': str(target),
                                    'OUTPUT_PATH': str(target)}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    assert local_path.get_resource_path() == target.resolve().as_posix()
    assert chr(92) not in local_path.get_resource_path()


def test_a_user_home_prefix_is_expanded(tmp_path: Path, monkeypatch) -> None:
    """``~`` is expanded rather than treated as a literal directory name."""
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': '~/resources',
                                    'OUTPUT_PATH': '~/outputs'}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    resolved = local_path.get_resource_path()
    assert resolved == (Path.home() / 'resources').resolve().as_posix()
    assert '~' not in resolved


# --------------------------------------------------------------------------- #
# malformed and empty settings files
# --------------------------------------------------------------------------- #
def test_an_empty_yaml_file_parses_to_no_settings_and_then_raises(tmp_path: Path,
                                                                  monkeypatch) -> None:
    """``yaml.safe_load`` gives None for an empty file, which ``get_paths`` maps to ``{}``.

    An *absent* file falls back to the checkout defaults, but a file that exists and defines no
    keys does not: it takes the same path as any file missing the key and raises ``KeyError``.
    Worth stating explicitly, because "missing file" and "empty file" read as the same situation
    and resolve differently -- the fallback is keyed on the file's existence, not its contents.
    """
    path = tmp_path / 'settings.yaml'
    path.write_text('')
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    assert local_path.get_paths() == {}
    with pytest.raises(KeyError, match='RESOURCE_PATH'):
        local_path.get_resource_path()


def test_a_yaml_file_that_is_not_a_mapping_raises(tmp_path: Path, monkeypatch) -> None:
    """A list or scalar document would fail later with a confusing index error."""
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump(['RESOURCE_PATH', 'OUTPUT_PATH']))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    with pytest.raises(TypeError, match='must contain a mapping'):
        local_path.get_paths()


def test_an_unwritable_default_output_directory_raises(tmp_path: Path, monkeypatch) -> None:
    """When no candidate directory can be created, the failure is explicit.

    ``mkdir`` raising OSError on every candidate is the only way to exhaust the list, so it is
    forced here; silently returning an unwritable path would fail later at the first save.
    """
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': None, 'OUTPUT_PATH': None}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    def refuse(self, *args, **kwargs):
        """Stand in for a filesystem that refuses every directory creation."""
        raise OSError('read-only filesystem')

    monkeypatch.setattr(Path, 'mkdir', refuse)
    with pytest.raises(OSError, match='no writable default output directory'):
        local_path.get_output_path()
