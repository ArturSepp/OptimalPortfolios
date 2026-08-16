"""Hold the Makefile recipes to the commands CONTRIBUTING.md documents.

The Makefile is a convenience layer: every recipe is meant to be one of the ``uv`` invocations
written out in CONTRIBUTING.md, unchanged. That claim is the whole basis for having the aliases
at all -- the prose explains why each flag is there (``--locked``, ``--only-group lint``, the
explicit ``--select``, ``--python 3.12``), and a recipe that quietly drifts from it teaches a
contributor a command CI does not run.

Nothing enforced that. This module does, in the spirit of ``readme_test.py``: the repository
executes its documentation rather than trusting it, and a fourth restatement of the CI commands
should not be the exception.

The contract is one-directional. Every ``uv`` command in a Makefile recipe must appear in a fenced
block in CONTRIBUTING.md; the reverse is deliberately not required, because the document carries
commands that are intentionally not targets -- the PowerShell variants of the audit block, and the
``python -m pytest --pyargs optimalportfolios`` wheel check, which runs against a separately built
wheel in a clean environment rather than the project virtualenv.

Comparison is on the command text only. Trailing ``#`` comments in the documented block and
whitespace used for column alignment are not part of what gets run, so both are normalised away
before matching.
"""

import re
from pathlib import Path

import pytest

# Fenced blocks in CONTRIBUTING.md, any language. Taking every fence rather than only ```bash
# keeps this a superset of what the recipes may draw on, which is the safe direction for a
# containment check: a command can only go missing from the document, never sneak in.
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# The prefix that marks a line as a command this harness has an opinion about. Recipes also run
# `echo`, `rm`, `find` and shell conditionals; those are the Makefile's own business and are not
# claimed to come from the document.
COMMAND_PREFIX = "uv "


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, which is annotation rather than part of the command.

    CONTRIBUTING.md aligns explanatory comments into a column after several of its commands, e.g.
    ``uv sync --extra dev    # editable install, versions from uv.lock``. None of the documented
    commands contain a literal ``#`` themselves, so splitting on the first one is unambiguous.
    """
    return line.split("#", 1)[0]


def _normalise(line: str) -> str:
    """Reduce a command line to what is actually executed.

    Strips any trailing comment and collapses runs of whitespace, so a recipe and a documented
    command that differ only in column alignment compare equal, while a changed flag does not.
    """
    return " ".join(_strip_comment(line).split())


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations so a command split across lines is still seen as one.

    Without this a recipe wrapped for width would be read as several fragments, none of which
    starts with ``uv``, and the command would silently escape the check -- a false pass, which is
    the failure mode this module exists to prevent.
    """
    joined: list[str] = []
    pending = ""
    for line in text.splitlines():
        if line.endswith("\\"):
            pending += line[:-1]
        else:
            joined.append(pending + line)
            pending = ""
    if pending:
        joined.append(pending)
    return joined


def _recipe_commands(makefile_text: str) -> list[str]:
    """Every ``uv`` command run by a Makefile recipe, normalised.

    Recipe lines are the tab-indented ones. A leading ``@`` (suppress echo) or ``-`` (ignore
    failure) is a directive to make, not part of the command, so both are removed before matching.
    """
    commands = []
    for line in _logical_lines(makefile_text):
        if not line.startswith("\t"):
            continue
        stripped = line.lstrip("\t").lstrip("@-").strip()
        if stripped.startswith(COMMAND_PREFIX):
            commands.append(_normalise(stripped))
    return commands


def _documented_commands(contributing_text: str) -> set[str]:
    """Every ``uv`` command appearing in a fenced block of CONTRIBUTING.md, normalised."""
    documented = set()
    for block in FENCE.findall(contributing_text):
        for line in _logical_lines(block):
            normalised = _normalise(line.strip())
            if normalised.startswith(COMMAND_PREFIX):
                documented.add(normalised)
    return documented


@pytest.fixture
def makefile(root: Path) -> Path:
    """The repository Makefile, which must exist once a checkout has been found.

    Fails rather than skips, for the reason the ``readme`` fixture does: the installed-wheel case
    is handled one level up by ``root``, so reaching here means there is a checkout, and a checkout
    whose Makefile is gone while CONTRIBUTING.md still advertises the targets is a broken
    documentation contract rather than a case with nothing to assert.
    """
    path = root / "Makefile"
    assert path.is_file(), (
        f"no Makefile at {path}, but CONTRIBUTING.md documents its targets. Remove that paragraph "
        f"too, or restore the file -- a documented target that does not exist is worse than no "
        f"Makefile at all."
    )
    return path


@pytest.fixture
def contributing(root: Path) -> Path:
    """The repository CONTRIBUTING.md, which is the definition the recipes must match."""
    path = root / "CONTRIBUTING.md"
    assert path.is_file(), (
        f"no CONTRIBUTING.md at {path}. The Makefile recipes are claimed to repeat its commands "
        f"verbatim, so without it this contract has lost its subject."
    )
    return path


def test_every_recipe_command_is_documented(logger, makefile, contributing):
    """No Makefile recipe may run a ``uv`` command that CONTRIBUTING.md does not show."""
    recipes = _recipe_commands(makefile.read_text(encoding="utf-8"))
    documented = _documented_commands(contributing.read_text(encoding="utf-8"))

    logger.info(
        "Found %d uv command(s) in Makefile recipes and %d in CONTRIBUTING.md fences",
        len(recipes),
        len(documented),
    )

    # Fail closed on an empty parse, the same way the README harness does. A Makefile whose recipe
    # indentation was converted to spaces, or a CONTRIBUTING.md whose fences were renamed, would
    # otherwise yield an empty set that is trivially contained in another and report success while
    # comparing nothing.
    assert recipes, (
        f"No uv command found in any recipe of {makefile}. Recipe lines must be tab-indented; if "
        f"the file now uses spaces, make would not run it either."
    )
    assert documented, (
        f"No uv command found in any fenced block of {contributing}. This test compares recipes "
        f"against that document, so an empty parse asserts nothing."
    )

    undocumented = sorted(set(recipes) - documented)
    assert not undocumented, (
        "Makefile recipes run uv commands that CONTRIBUTING.md does not document:\n"
        + "\n".join(f"  {command}" for command in undocumented)
        + f"\n\nThe document is the definition and the recipes repeat it verbatim. Update "
        f"{contributing.name} if the command genuinely changed, or fix the recipe if it drifted."
    )


class TestNormalisation:
    """Tests for reducing a line to the command it actually runs."""

    def test_trailing_comment_is_not_part_of_the_command(self):
        """The aligned explanatory comments in CONTRIBUTING.md must not defeat matching."""
        commented = "uv sync --extra dev   # editable install"
        assert _strip_comment(commented) == "uv sync --extra dev   "
        assert _strip_comment("uv sync --extra dev") == "uv sync --extra dev"

    def test_column_alignment_is_collapsed(self):
        """A documented command padded into a column equals the recipe that runs it."""
        assert _normalise("uv run --locked pytest    # the full suite") == "uv run --locked pytest"
        assert _normalise("  uv   run  pytest  ") == "uv run pytest"

    def test_a_changed_flag_still_differs(self):
        """Normalisation must not be so aggressive that real drift compares equal."""
        assert _normalise("uv run --locked pytest") != _normalise("uv run pytest")


class TestLogicalLines:
    """Tests for backslash-continuation joining."""

    def test_continuation_is_joined_into_one_command(self):
        """A recipe wrapped for width is still seen as the single command it runs."""
        assert _logical_lines("uv run \\\n--locked pytest") == ["uv run --locked pytest"]

    def test_plain_lines_are_unchanged(self):
        """Text with no continuation passes through line by line."""
        assert _logical_lines("one\ntwo") == ["one", "two"]

    def test_trailing_continuation_at_end_of_text_is_kept(self):
        """A file ending mid-continuation still yields its fragment rather than dropping it."""
        assert _logical_lines("uv run \\\n") == ["uv run "]


class TestRecipeExtraction:
    """Tests for which Makefile lines count as commands this harness checks."""

    def test_only_tab_indented_lines_are_recipes(self):
        """A target line or a variable assignment is not a recipe, however it reads."""
        assert _recipe_commands("uv run pytest\n") == []
        assert _recipe_commands("\tuv run pytest\n") == ["uv run pytest"]

    def test_echo_suppression_and_error_tolerance_are_stripped(self):
        """A leading @ or - is a directive to make, not part of the command."""
        assert _recipe_commands("\t@uv run pytest\n") == ["uv run pytest"]
        assert _recipe_commands("\t-uv run pytest\n") == ["uv run pytest"]

    def test_non_uv_recipe_lines_are_ignored(self):
        """Shell built-ins and conditionals in recipes are the Makefile's own business."""
        assert _recipe_commands('\techo "hello"\n\trm -rf build\n\tif command -v uv; then\n') == []

    def test_a_wrapped_recipe_command_is_still_found(self):
        """The false pass this harness must not allow: a uv command split across lines."""
        assert _recipe_commands("\tuv run \\\n\t--locked pytest\n") == ["uv run --locked pytest"]


class TestDocumentedExtraction:
    """Tests for reading the documented command set out of CONTRIBUTING.md."""

    def test_commands_are_taken_from_fenced_blocks(self):
        """Only fenced content counts; prose that mentions a command does not."""
        text = "Run `uv run nothing` inline.\n\n```bash\nuv run --locked pytest\n```\n"
        assert _documented_commands(text) == {"uv run --locked pytest"}

    def test_every_fence_language_is_read(self):
        """Taking all fences keeps the documented set a superset, the safe direction here."""
        text = (
            "```bash\nuv sync --extra dev\n```\n"
            '```powershell\nuv export --locked -o "$env:TEMP\\x"\n```\n'
        )
        assert _documented_commands(text) == {
            "uv sync --extra dev",
            'uv export --locked -o "$env:TEMP\\x"',
        }

    def test_non_command_lines_in_a_fence_are_ignored(self):
        """A fence may hold output or other tools without polluting the command set."""
        assert _documented_commands("```bash\npython -m pytest --pyargs x\n```\n") == set()
