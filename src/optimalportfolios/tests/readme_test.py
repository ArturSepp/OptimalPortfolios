"""Tests for executable Python examples in the README.

This module extracts Python code and expected result blocks from README.md,
executes the code, and verifies the output matches the documented result.

The pattern is adopted from jebel-quant/rhiza's ``test_readme.py``. This repository is not
rhiza-managed -- there is no ``.rhiza/`` pointer and nothing syncs this file -- so it is
maintained here directly, and the language-neutral half of the upstream pair (that the README
exists, and that its ``bash`` fences parse) is not present. What this file does is the part that
only means something in a Python project: running a ``python`` fence and diffing it against the
following ``result`` block.

Every ``python`` fence in README.md except the quick-start carries ``+SKIP``. Those blocks
are illustrative fragments -- they reference names defined nowhere (``benchmark``,
``asset_class_groups``), reproduce a dataclass definition, or fetch from the network and write a
PDF. Merging them into the executed script would only assert that undefined names raise.
"""

import re
import subprocess  # nosec B404
import sys

import pytest

# Regex for Python code blocks -- captures optional flags (e.g. "+SKIP") and the code body.
CODE_BLOCK = re.compile(r"```python([^\n]*)\n(.*?)```", re.DOTALL)

RESULT = re.compile(r"```result\n(.*?)```", re.DOTALL)

# Flag that marks a code block as intentionally excluded from readme tests.
# Usage: add the flag after the language identifier on the opening fence line,
# e.g. ```python +SKIP  or  ```bash +SKIP
SKIP_FLAG = "+SKIP"


def _should_skip(flags: str) -> bool:
    """Return True if the fence flags string contains the +SKIP marker."""
    return SKIP_FLAG in flags


def test_readme_runs(logger, root):
    """Execute README code blocks and compare output to documented results."""
    readme = root / "README.md"
    logger.info("Reading README from %s", readme)
    readme_text = readme.read_text(encoding="utf-8")
    all_code_blocks = CODE_BLOCK.findall(readme_text)
    result_blocks = RESULT.findall(readme_text)

    code_blocks = []
    for i, (flags, code) in enumerate(all_code_blocks):
        if _should_skip(flags):
            logger.info("Skipping Python code block %d (%s flag)", i, SKIP_FLAG)
        else:
            code_blocks.append(code)

    logger.info(
        "Found %d code block(s) (%d skipped) and %d result block(s) in README",
        len(all_code_blocks),
        len(all_code_blocks) - len(code_blocks),
        len(result_blocks),
    )

    code = "".join(code_blocks)  # merged code
    expected = "".join(result_blocks)  # merged results

    # Trust boundary: we execute Python snippets sourced from README.md in this repo.
    # The README is part of the trusted repository content and reviewed in PRs.
    logger.debug("Executing README code via %s -c ...", sys.executable)
    result = subprocess.run(  # nosec
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=root
    )

    stdout = result.stdout
    logger.debug("Execution finished with return code %d", result.returncode)
    if result.stderr:
        logger.debug("Stderr from README code:\n%s", result.stderr)
    logger.debug("Stdout from README code:\n%s", stdout)

    assert result.returncode == 0, (
        f"README code exited with {result.returncode}. Stderr:\n{result.stderr}"
    )
    logger.info("README code executed successfully; comparing output to expected result")
    assert stdout.strip() == expected.strip()
    logger.info("README code output matches expected result")


class TestReadmeTestEdgeCases:
    """Edge cases for README code block testing."""

    def test_readme_code_is_syntactically_valid(self, root):
        """Python code blocks in README should be syntactically valid (skipped blocks excluded)."""
        readme = root / "README.md"
        content = readme.read_text(encoding="utf-8")
        all_code_blocks = CODE_BLOCK.findall(content)

        for i, (flags, code) in enumerate(all_code_blocks):
            if _should_skip(flags):
                continue
            try:
                compile(code, f"<readme_block_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Code block {i} has syntax error: {e}")


class TestSkipFlag:
    """Tests for the +SKIP flag as it applies to Python fences."""

    def test_should_skip_returns_true_for_skip_flag(self):
        """+SKIP in flags string should cause _should_skip to return True."""
        assert _should_skip(" +SKIP") is True
        assert _should_skip("+SKIP") is True
        assert _should_skip(" +SKIP other-flag") is True

    def test_should_skip_returns_false_without_flag(self):
        """Absence of +SKIP should cause _should_skip to return False."""
        assert _should_skip("") is False
        assert _should_skip(" ") is False
        assert _should_skip("other-flag") is False

    def test_python_block_with_skip_flag_is_excluded(self, tmp_path):
        """A ```python +SKIP block should not appear in the list of blocks to execute."""
        readme = tmp_path / "README.md"
        readme.write_text(
            '```python +SKIP\nraise RuntimeError("should not run")\n```\n'
            "```python\nprint('hello')\n```\n"
            "```result\nhello\n```\n",
            encoding="utf-8",
        )
        content = readme.read_text(encoding="utf-8")
        all_blocks = CODE_BLOCK.findall(content)
        assert len(all_blocks) == 2
        executed = [code for flags, code in all_blocks if not _should_skip(flags)]
        assert len(executed) == 1
        assert "raise RuntimeError" not in executed[0]
