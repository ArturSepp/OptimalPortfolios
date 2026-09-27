"""Guard the Read the Docs build commands against failures that CI's docs job cannot see.

Read the Docs builds from ``.readthedocs.yaml``, while ``docs.yml`` builds in its own
environment, so a command that fails only on Read the Docs leaves every CI check green. Three
such failures shipped, one after another:

1. A bare ``uv`` was not on the build PATH (every build from 2026-09-14 to 2026-09-26).
2. A single quote inside a command ended the ``/bin/sh -c '<command>'`` wrapper that Read the
   Docs puts, unescaped, around each ``build.jobs`` command (2026-09-27).
3. ``python -m uv`` resolved to the project virtualenv's interpreter, which Read the Docs puts
   first on PATH once it exists, and that interpreter has no uv (2026-09-27).

The configuration therefore installs uv into its own directory and calls that binary by path.
These tests enforce the rules on every pull request.
"""

import shlex
from pathlib import Path
from typing import Optional

import pytest
import yaml


def job_commands(root: Path) -> list[tuple[str, str]]:
    """Return ``(job, command)`` for every custom build-job command of the checkout."""
    config = yaml.safe_load((root / ".readthedocs.yaml").read_text(encoding="utf-8"))
    return [
        (job, command)
        for job, commands in config["build"]["jobs"].items()
        for command in commands
    ]


def has_single_quote(command: str) -> bool:
    """Whether the command would end Read the Docs' single-quoted shell wrapper early."""
    return "'" in command


def uv_calls(command: str) -> list[str]:
    """Classify each uv call in a command as ``bare``, ``module`` or ``path``.

    ``bare`` is a ``uv`` looked up on PATH, ``module`` is ``<python> -m uv``, whose interpreter
    is looked up on PATH, and ``path`` is a uv executable called by an explicit path.
    """
    tokens = shlex.split(command)
    calls = []
    for index, token in enumerate(tokens):
        if token == "uv":
            calls.append("module" if index and tokens[index - 1] == "-m" else "bare")
        elif token.endswith("/bin/uv"):
            calls.append("path")
    return calls


def uv_target(command: str) -> Optional[str]:
    """Return the ``--target`` directory of a ``pip install`` of uv, if the command is one."""
    tokens = shlex.split(command)
    installs_uv = "pip" in tokens and "install" in tokens and any(
        token.startswith("uv==") for token in tokens
    )
    if installs_uv and "--target" in tokens:
        return tokens[tokens.index("--target") + 1]
    return None


@pytest.fixture
def commands(root: Path) -> list[tuple[str, str]]:
    """The custom build-job commands; an empty list would make the guards pass vacuously."""
    found = job_commands(root)
    assert found, "no build.jobs commands in .readthedocs.yaml"
    return found


def test_no_command_contains_a_single_quote(commands):
    """Every build-job command survives Read the Docs' unescaped ``sh -c '...'`` wrapper."""
    assert [pair for pair in commands if has_single_quote(pair[1])] == []


def test_uv_is_called_only_by_its_installed_path(commands):
    """No uv call depends on PATH, and every call uses the directory pip installed uv into."""
    targets = {target for _, command in commands if (target := uv_target(command))}
    assert len(targets) == 1, f"expected one pip --target install of uv, found {targets}"
    binary = f"{targets.pop()}/bin/uv"
    calls = [(job, command) for job, command in commands if uv_calls(command)]
    assert calls, "no uv call found; the guard would pass vacuously"
    for job, command in calls:
        assert set(uv_calls(command)) == {"path"}, (job, command)
        assert binary in shlex.split(command), (job, command)


@pytest.mark.parametrize(
    "command, quoted, calls",
    [
        # Failed from 2026-09-14 to 2026-09-26 with "uv: not found".
        ('uv venv "$READTHEDOCS_VIRTUALENV_PATH"', False, ["bare"]),
        # Failed on 2026-09-27 with a shell syntax error.
        ("python -m uv venv --python \"$(python -c 'import sys; print(sys.executable)')\" "
         '"$READTHEDOCS_VIRTUALENV_PATH"', True, ["module"]),
        # Failed on 2026-09-27 with "No module named uv".
        ('UV_PROJECT_ENVIRONMENT="$READTHEDOCS_VIRTUALENV_PATH" python -m uv sync --locked',
         False, ["module"]),
        # The corrected forms.
        ('"$READTHEDOCS_VIRTUALENV_PATH.uv/bin/uv" venv --python 3.12 '
         '"$READTHEDOCS_VIRTUALENV_PATH"', False, ["path"]),
        ('UV_PROJECT_ENVIRONMENT="$READTHEDOCS_VIRTUALENV_PATH" '
         '"$READTHEDOCS_VIRTUALENV_PATH.uv/bin/uv" sync --locked --extra docs', False, ["path"]),
    ],
)
def test_guards_reject_the_commands_that_failed(command, quoted, calls):
    """The guards flag the three historical failures and pass their corrected forms."""
    assert has_single_quote(command) is quoted
    assert uv_calls(command) == calls


def test_uv_target_reads_the_install_directory():
    """The pip install of uv names the directory whose ``bin/uv`` every call must use."""
    command = 'python -m pip install --target "$READTHEDOCS_VIRTUALENV_PATH.uv" uv==0.12.13'
    assert uv_target(command) == "$READTHEDOCS_VIRTUALENV_PATH.uv"
    assert uv_target("python -m pip install uv==0.12.13") is None
