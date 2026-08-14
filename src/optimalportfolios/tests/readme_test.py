"""Tests for executable Python examples in the README.

This module extracts Python code and expected result blocks from README.md,
executes the code, and verifies the output matches the documented result.

The pattern is adopted from jebel-quant/rhiza's ``test_readme.py``. This repository is not
rhiza-managed -- there is no ``.rhiza/`` pointer and nothing syncs this file -- so it is
maintained here directly, and the language-neutral half of the upstream pair (that the README
exists, and that its ``bash`` fences parse) is not present. What this file does is the part that
only means something in a Python project: running a ``python`` fence and diffing it against the
following ``result`` block.

The harness fails closed. An empty parse -- no executable ``python`` fence, or no ``result`` fence
-- is an error rather than a vacuous ``"" == ""`` pass, blocks are joined with an explicit newline
so an unterminated body cannot splice onto the next one, and execution is bounded by
``EXECUTION_TIMEOUT_SECONDS`` so a hanging quick-start fails here instead of holding the CI job
open to its outer limit.

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

# Wall-clock ceiling for the merged README script. The quick-start measures ~7s locally; the
# margin covers a cold cvxpy/qis import on the slowest matrix cell. The point of any bound at all
# is that a snippet which blocks -- on stdin, on a socket, on a solver that will not converge --
# fails this test in bounded time instead of holding the CI job open until its outer timeout.
EXECUTION_TIMEOUT_SECONDS = 300


def _should_skip(flags: str) -> bool:
    """Return True if the fence flags string contains the +SKIP marker."""
    return SKIP_FLAG in flags


def _join_blocks(blocks: list[str]) -> str:
    """Join fence bodies with exactly one newline between them.

    Explicit rather than `"".join(...)`: a fence body that does not end in a newline would
    otherwise splice its last line onto the first line of the next block, which silently changes
    both the executed script and the expected output. Each body is stripped of its own leading and
    trailing blank lines so the separator is exactly one newline regardless of fence spacing.
    """
    return "\n".join(block.strip("\n") for block in blocks)


def _as_text(stream) -> str:
    """Render a possibly-absent, possibly-bytes captured stream as text.

    `TimeoutExpired` does not guarantee the text-mode conversion `run()` applies on a clean exit,
    and what it carries differs between Windows and POSIX, so both cases are handled here.
    """
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")


def _run_readme_script(code: str, cwd, timeout: float = EXECUTION_TIMEOUT_SECONDS):
    """Execute the merged README script, failing the test if it does not finish in `timeout`."""
    # Trust boundary: we execute Python snippets sourced from README.md in this repo.
    # The README is part of the trusted repository content and reviewed in PRs.
    try:
        return subprocess.run(  # nosec
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = _as_text(exc.stdout)
        partial_stderr = _as_text(exc.stderr)
        pytest.fail(
            f"README code did not finish within {timeout:g}s and was killed. A quick-start that "
            f"hangs is a broken quick-start.\n"
            f"Partial stdout:\n{partial_stdout}\nPartial stderr:\n{partial_stderr}"
        )


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

    # Fail closed on an empty parse. Without these two guards a README whose fences were renamed,
    # reflowed or accidentally all marked +SKIP yields empty code and an empty expectation, and
    # `"" == ""` reports success while executing nothing at all.
    assert code_blocks, (
        f"No executable ```python fence found in {readme}: {len(all_code_blocks)} python fence(s) "
        f"present, all carrying {SKIP_FLAG}. At least one block must be executed, or this test "
        f"asserts nothing."
    )
    assert result_blocks, (
        f"No ```result fence found in {readme}. The executed output needs something to be diffed "
        f"against, or this test asserts nothing."
    )

    code = _join_blocks(code_blocks)  # merged code
    expected = _join_blocks(result_blocks)  # merged results

    logger.debug("Executing README code via %s -c ...", sys.executable)
    result = _run_readme_script(code, root)

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


class TestBlockJoining:
    """Tests that merging blocks keeps them separated."""

    def test_unterminated_body_does_not_splice_onto_the_next_block(self):
        """A body with no trailing newline must not concatenate onto the following one."""
        assert _join_blocks(["print(1)", "print(2)\n"]) == "print(1)\nprint(2)"

    def test_merged_unterminated_blocks_remain_two_statements(self):
        """Two unterminated bodies merge into a script that still parses as two statements."""
        merged = _join_blocks(["x = 1", "print(x)"])
        assert merged == "x = 1\nprint(x)"
        compile(merged, "<merged>", "exec")

    def test_single_block_joins_to_itself(self):
        """One block is unchanged apart from surrounding blank lines."""
        assert _join_blocks(["\nprint(1)\n"]) == "print(1)"


class TestExecutionTimeout:
    """Tests for the wall-clock bound on the executed script."""

    def test_hanging_script_fails_with_a_clear_message(self, tmp_path):
        """A snippet that does not finish fails this test rather than the outer CI job."""
        with pytest.raises(pytest.fail.Exception, match="did not finish within"):
            _run_readme_script("import time; time.sleep(30)", tmp_path, timeout=0.5)

    def test_partial_capture_is_rendered_for_any_stream_type(self):
        """Partial output is reportable whether it arrives as None, bytes or str."""
        assert _as_text(None) == ""
        assert _as_text(b"partial") == "partial"
        assert _as_text("partial") == "partial"


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

    def test_all_skipped_readme_leaves_nothing_to_execute(self, tmp_path):
        """The condition the empty-parse guard in `test_readme_runs` refuses to let pass."""
        readme = tmp_path / "README.md"
        readme.write_text("```python +SKIP\nprint('x')\n```\n", encoding="utf-8")
        content = readme.read_text(encoding="utf-8")
        blocks = CODE_BLOCK.findall(content)
        assert [code for flags, code in blocks if not _should_skip(flags)] == []
        assert RESULT.findall(content) == []
