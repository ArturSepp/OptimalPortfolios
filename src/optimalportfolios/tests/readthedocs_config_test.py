"""Guard the Read the Docs build commands against failures that CI's docs job cannot see.

Read the Docs builds from ``.readthedocs.yaml``, while ``docs.yml`` builds in its own
environment, so a command that fails only on Read the Docs leaves every CI check green. Two such
failures shipped. A bare ``uv``, whose entry point pip had not put on the build PATH, failed every
build from 2026-09-14 to 2026-09-26. The fix then put a single quote inside a command, which ends
the ``/bin/sh -c '<command>'`` wrapper that Read the Docs puts, unescaped, around each
``build.jobs`` command (2026-09-27). Both rules are checked here on every pull request.
"""

import shlex
from pathlib import Path

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


def runs_bare_uv(command: str) -> bool:
    """Whether the command calls a ``uv`` executable instead of ``python -m uv``."""
    tokens = shlex.split(command)
    return any(
        token == "uv" and (index == 0 or tokens[index - 1] != "-m")
        for index, token in enumerate(tokens)
    )


@pytest.fixture
def commands(root: Path) -> list[tuple[str, str]]:
    """The custom build-job commands; an empty list would make the guards pass vacuously."""
    found = job_commands(root)
    assert found, "no build.jobs commands in .readthedocs.yaml"
    return found


def test_no_command_contains_a_single_quote(commands):
    """Every build-job command survives Read the Docs' unescaped ``sh -c '...'`` wrapper."""
    assert [pair for pair in commands if has_single_quote(pair[1])] == []


def test_uv_runs_as_a_module(commands):
    """Every uv call goes through ``python -m uv``, which does not depend on the build PATH."""
    assert [pair for pair in commands if runs_bare_uv(pair[1])] == []


@pytest.mark.parametrize(
    "command, quoted, bare",
    [
        # The command that failed from 2026-09-14 to 2026-09-26 with "uv: not found".
        ('uv venv "$READTHEDOCS_VIRTUALENV_PATH"', False, True),
        ('UV_PROJECT_ENVIRONMENT="$READTHEDOCS_VIRTUALENV_PATH" uv sync --locked --extra docs',
         False, True),
        # The command that failed on 2026-09-27 with a shell syntax error.
        ("python -m uv venv --python \"$(python -c 'import sys; print(sys.executable)')\" "
         '"$READTHEDOCS_VIRTUALENV_PATH"', True, False),
        ('python -m uv venv --python 3.12 "$READTHEDOCS_VIRTUALENV_PATH"', False, False),
        ('UV_PROJECT_ENVIRONMENT="$READTHEDOCS_VIRTUALENV_PATH" python -m uv sync --locked',
         False, False),
    ],
)
def test_guards_reject_the_commands_that_failed(command, quoted, bare):
    """Both guards flag the two historical failures and pass their corrected forms."""
    assert has_single_quote(command) is quoted
    assert runs_bare_uv(command) is bare
