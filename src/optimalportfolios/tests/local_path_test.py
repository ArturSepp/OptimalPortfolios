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

A checkout is recognised by its src layout: ``<repository>/src/optimalportfolios`` beside
``<repository>/pyproject.toml``. The detection once looked for ``pyproject.toml`` one level up,
in ``src/``, so after the move to the src layout every checkout was treated as an installed
package and output went to the working directory instead of ``<repository>/outputs``. The
defaults are therefore tested on built directory layouts, and the running checkout is asserted
separately.

``OPTIMALPORTFOLIOS_OUTPUT_PATH`` overrides both the settings file and the defaults for the output
directory. An autouse fixture unsets it, so a developer's own override cannot change a result.
"""
# packages
import os
from pathlib import Path
import pytest
import yaml
# optimalportfolios
from optimalportfolios import local_path

OUTPUT_OVERRIDE = 'OPTIMALPORTFOLIOS_OUTPUT_PATH'


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


@pytest.fixture(autouse=True)
def no_output_override(monkeypatch) -> None:
    """Unset the output environment override so the caller's own setting cannot leak in."""
    monkeypatch.delenv(OUTPUT_OVERRIDE, raising=False)


def _place_package(tmp_path: Path, monkeypatch, *parts: str) -> Path:
    """Point the module at a package directory built under ``tmp_path`` and move the CWD away.

    The working directory is a separate empty folder, so a default that silently falls back to
    the CWD cannot coincide with the checkout it should have found.
    """
    package_dir = tmp_path.joinpath(*parts)
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(local_path, '_PACKAGE_DIR', package_dir)
    working_dir = tmp_path / 'working-directory'
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    return package_dir


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch) -> Path:
    """Build a src-layout checkout, point the module at it, and return the repository root."""
    repository = tmp_path / 'repository'
    _place_package(tmp_path, monkeypatch, 'repository', 'src', 'optimalportfolios')
    (repository / 'pyproject.toml').write_text('[project]\nname = "optimalportfolios"\n')
    return repository


@pytest.fixture
def installed(tmp_path: Path, monkeypatch) -> Path:
    """Place the module in site-packages with no checkout around it; return the CWD."""
    _place_package(tmp_path, monkeypatch, 'environment', 'Lib', 'site-packages',
                   'optimalportfolios')
    return Path.cwd()


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


# --------------------------------------------------------------------------- #
# checkout detection
# --------------------------------------------------------------------------- #
def test_the_running_checkout_is_detected(root: Path) -> None:
    """The editable checkout this suite runs from is recognised as one.

    Regression: detection looked for ``pyproject.toml`` in ``src/`` and returned None here.
    The ``root`` fixture skips when the suite runs from an installed wheel.
    """
    assert local_path._checkout_root() == root


def test_a_src_layout_checkout_is_detected(checkout: Path) -> None:
    """``<repository>/src/optimalportfolios`` beside ``pyproject.toml`` is a checkout."""
    assert local_path._checkout_root() == checkout


@pytest.mark.parametrize('package_parts, pyproject_parts', [
    (('environment', 'Lib', 'site-packages', 'optimalportfolios'), None),
    (('project', 'site-packages', 'optimalportfolios'), ('project',)),
], ids=['installed-wheel', 'pyproject-above-a-non-src-parent'])
def test_a_package_outside_a_src_layout_checkout_has_no_root(
        package_parts, pyproject_parts, tmp_path: Path, monkeypatch) -> None:
    """Only the src layout counts; any other ``pyproject.toml`` above the package does not.

    A wheel installed into another project's virtual environment has that project's
    ``pyproject.toml`` among its ancestors, so walking every parent would put this package's
    output in someone else's repository.
    """
    _place_package(tmp_path, monkeypatch, *package_parts)
    if pyproject_parts is not None:
        tmp_path.joinpath(*pyproject_parts, 'pyproject.toml').write_text('')

    assert local_path._checkout_root() is None


# --------------------------------------------------------------------------- #
# defaults for the shipped placeholder and an absent settings file
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('settings_value', [None, '..', '..\\'])
def test_placeholder_output_falls_back_to_the_checkout_outputs_directory(
        settings_value, checkout: Path, tmp_path: Path, monkeypatch) -> None:
    """A placeholder output resolves to a created, writable ``<repository>/outputs``.

    ``'..\\'`` is the literal value shipped in ``settings.yaml``.
    """
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'RESOURCE_PATH': settings_value,
                                    'OUTPUT_PATH': settings_value}))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)

    output_path = Path(local_path.get_output_path())

    assert output_path == checkout / 'outputs'
    assert output_path != Path.cwd()
    assert output_path.is_dir()
    assert os.access(output_path, os.W_OK)
    assert chr(92) not in local_path.get_output_path()


def test_absent_settings_file_uses_portable_checkout_defaults(checkout: Path, tmp_path: Path,
                                                              monkeypatch) -> None:
    """A missing YAML file in a checkout uses the repository root and its outputs directory."""
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', tmp_path / 'absent.yaml')

    assert local_path.get_resource_path() == checkout.as_posix()
    assert local_path.get_output_path() == (checkout / 'outputs').as_posix()
    assert chr(92) not in local_path.get_resource_path()
    assert chr(92) not in local_path.get_output_path()


def test_an_installed_package_falls_back_to_the_working_directory(installed: Path, tmp_path: Path,
                                                                  monkeypatch) -> None:
    """With no checkout around the package, both defaults are the current working directory."""
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', tmp_path / 'absent.yaml')

    assert local_path.get_resource_path() == installed.as_posix()
    assert local_path.get_output_path() == installed.as_posix()


# --------------------------------------------------------------------------- #
# the output environment override
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('settings', [
    None,
    {'RESOURCE_PATH': None, 'OUTPUT_PATH': None},
    {'RESOURCE_PATH': None, 'OUTPUT_PATH': 'configured'},
    {'RESOURCE_PATH': None},
], ids=['absent-file', 'placeholder', 'configured-value', 'missing-key'])
def test_the_output_override_wins_over_settings_and_defaults(
        settings, checkout: Path, tmp_path: Path, monkeypatch) -> None:
    """A set override is used and created whatever the settings file says.

    The environment outranks the file, as it does for TrendFollowing's ``TF_OUTPUT_PATH``, so a
    host-wide setting keeps output out of the checkout without editing the tracked settings.
    The checkout default, which ``get_output_path`` would otherwise create, is left untouched.
    """
    path = tmp_path / 'settings.yaml'
    if settings is not None:
        path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', path)
    override = tmp_path / 'local-drive' / 'outputs'
    monkeypatch.setenv(OUTPUT_OVERRIDE, str(override))

    assert local_path.get_output_path() == override.resolve().as_posix()
    assert override.is_dir()
    assert not (checkout / 'outputs').exists()


@pytest.mark.parametrize('value', ['', '   '])
def test_an_empty_output_override_is_ignored(value, checkout: Path, tmp_path: Path,
                                             monkeypatch) -> None:
    """An empty or blank override is treated as unset rather than as the working directory."""
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', tmp_path / 'absent.yaml')
    monkeypatch.setenv(OUTPUT_OVERRIDE, value)

    assert local_path.get_output_path() == (checkout / 'outputs').as_posix()


def test_an_unusable_output_override_raises_rather_than_falling_back(checkout: Path,
                                                                     tmp_path: Path,
                                                                     monkeypatch) -> None:
    """An override that cannot be a directory fails loudly instead of using the checkout.

    Falling back would write into exactly the location the override exists to avoid.
    """
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', tmp_path / 'absent.yaml')
    blocker = tmp_path / 'a-file'
    blocker.write_text('')
    monkeypatch.setenv(OUTPUT_OVERRIDE, str(blocker))

    with pytest.raises(OSError):
        local_path.get_output_path()
    assert not (checkout / 'outputs').exists()


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
