"""Exercise private-output containment, failure propagation and workflow wiring."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import private_dependency as guard


ROOT = Path(__file__).resolve().parents[2]
SENTINEL = "PRIVATE_SOURCE_SENTINEL"


def job_text(filename, job):
    """Select one YAML job without adding a YAML parser dependency."""
    source = (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
    block = source.split(f"\n  {job}:\n", 1)[1]
    import re
    return re.split(r"\n  [a-z][a-z-]*:\n", block, maxsplit=1)[0]


class PrivateDependencyTests(unittest.TestCase):
    """Verify real child output and public GitHub command files remain private."""

    def setUp(self):
        """Create a temporary runner root and controlled environment."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.environment = patch.dict(os.environ, {
            "RUNNER_TEMP": self.temporary.name,
            "GITHUB_WORKSPACE": str(Path(self.temporary.name) / "checkout"),
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def invoke(self, code, label="probe"):
        """Run real Python child code and return its controlled public output."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            status = guard.run_private(label, [sys.executable, "-c", code])
        return status, stream.getvalue()

    def test_stdout_stderr_and_traceback_are_private(self):
        """A failing process preserves its status without publishing its content."""
        code = (
            "import sys; print('PRIVATE_SOURCE_SENTINEL'); "
            "print('::error::PRIVATE_SOURCE_SENTINEL', file=sys.stderr); "
            "raise RuntimeError('PRIVATE_SOURCE_SENTINEL')"
        )
        status, public = self.invoke(code)
        self.assertEqual(status, 1)
        self.assertNotIn(SENTINEL, public)
        self.assertIn("FAILED (exit 1)", public)
        private = guard.private_directory() / "probe.log"
        self.assertIn(SENTINEL, private.read_text())
        self.assertIn("Traceback", private.read_text())
        record = json.loads(private.with_suffix(".status.json").read_text())
        self.assertEqual(record, {"check": "probe", "result": "failed", "exit_code": 1})

    def test_success_and_exact_nonzero_status_are_preserved(self):
        """Neither success output nor a nonstandard failure is misreported."""
        for code in (0, 17):
            with self.subTest(code=code):
                status, public = self.invoke(
                    f"import sys; print('{SENTINEL}'); sys.exit({code})", f"exit-{code}",
                )
                self.assertEqual(status, code)
                self.assertNotIn(SENTINEL, public)

    def test_child_command_files_are_redirected(self):
        """Children cannot write sensitive text to public summary/output files."""
        public_files = {key: str(Path(self.temporary.name) / key) for key in guard.COMMAND_FILES}
        with patch.dict(os.environ, public_files):
            status, public = self.invoke(
                "import os,pathlib; "
                f"keys={guard.COMMAND_FILES!r}; "
                "[pathlib.Path(os.environ[k]).write_text('PRIVATE_SOURCE_SENTINEL') for k in keys]"
            )
        self.assertEqual(status, 0)
        self.assertNotIn(SENTINEL, public)
        self.assertTrue(all(not Path(path).exists() for path in public_files.values()))
        for key in guard.COMMAND_FILES:
            self.assertEqual(
                (guard.private_directory() / f"probe.{key.lower()}").read_text(), SENTINEL,
            )

    def test_child_does_not_inherit_read_credential(self):
        """The private checkout token is not handed to build/test subprocesses."""
        with patch.dict(os.environ, {"FM_READ_TOKEN": SENTINEL, "TRADABLEFACTORS_READ_TOKEN": SENTINEL, "FACTORMODELS_READ_SSH_KEY": SENTINEL}):
            status, _ = self.invoke(
                "import os; assert 'FM_READ_TOKEN' not in os.environ; "
                "assert 'TRADABLEFACTORS_READ_TOKEN' not in os.environ; "
                "assert 'FACTORMODELS_READ_SSH_KEY' not in os.environ"
            )
        self.assertEqual(status, 0)

    def test_arguments_are_not_reparsed_by_a_shell(self):
        """Spaces, quotes and metacharacters reach direct commands unchanged."""
        argument = "space 'quote' \"double\" ; $(not-executed)"
        code = "import sys; assert sys.argv[1] == " + repr(argument)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(guard.run_private("arguments", [sys.executable, "-c", code, argument]), 0)

    def test_bash_script_failure_and_quoting_are_preserved(self):
        """Actual Bash preserves quoted text and stops at a failing command."""
        bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path("/bin/bash")
        script = b"printf '%s\\n' 'PRIVATE_SOURCE_SENTINEL ; $literal'\nexit 23\nprintf LEAK\n"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = guard.run_private("script", [str(bash), "-e", "-o", "pipefail"], script)
        self.assertEqual(status, 23)
        self.assertNotIn(SENTINEL, stream.getvalue())
        content = (guard.private_directory() / "script.log").read_text()
        self.assertIn(SENTINEL + " ; $literal", content)
        self.assertNotIn("LEAK", content)

    def test_configuration_rejects_missing_token_and_mutable_refs(self):
        """Configuration is fail closed and never echoes credentials or refs."""
        for ref, token, expected in (
            ("a" * 40, "true", 0), ("main", "true", 1),
            ("migration/tradablefactors-stage-4", "true", 1),
            ("a" * 40, "", 1), ("a" * 40, "false", 1), (SENTINEL, SENTINEL, 1),
        ):
            with self.subTest(ref=ref, present=bool(token)):
                with patch.dict(os.environ, {"FM_SOURCE_SHA": ref, "FM_READ_CREDENTIAL_PRESENT": token}):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(guard.check_configuration(), expected)
                    self.assertNotIn(SENTINEL, output.getvalue())

    def test_missing_command_and_bad_invocations_fail_without_details(self):
        """Startup and parser failures cannot expose command data."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = guard.run_private("missing", [str(Path(self.temporary.name) / SENTINEL)])
            malformed = guard.main(["run", SENTINEL, "--", "anything"])
        self.assertEqual((status, malformed), (1, 1))
        self.assertNotIn(SENTINEL, output.getvalue())

    def test_cli_returns_child_failure_without_output(self):
        """The real command-line entry point preserves failure and hides output."""
        result = subprocess.run(
            [sys.executable, str(Path(guard.__file__)), "run", "cli", "--",
             sys.executable, "-c", "print('PRIVATE_SOURCE_SENTINEL'); raise SystemExit(19)"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 19)
        self.assertNotIn(SENTINEL, result.stdout + result.stderr)
        self.assertIn("FAILED (exit 19)", result.stdout)

    def test_script_entry_point_forwards_exact_stdin(self):
        """The CLI passes the script unchanged to fail-fast Bash without evaluation."""
        content = b"printf '%s' 'PRIVATE_SOURCE_SENTINEL $literal'\nexit 23\n"
        stream = io.TextIOWrapper(io.BytesIO(content))
        with patch.object(sys, "stdin", stream), patch.object(guard, "run_private", return_value=23) as run:
            self.assertEqual(guard.main(["run-script", "probe"]), 23)
        run.assert_called_once_with(
            "probe", ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail"], content,
        )

    def test_runner_directory_must_be_outside_checkout(self):
        """Private logs cannot silently fall back into the repository."""
        for root in ("", "relative", str(Path(self.temporary.name) / "checkout")):
            with self.subTest(root=root), patch.dict(os.environ, {"RUNNER_TEMP": root}):
                with self.assertRaises(ValueError):
                    guard.private_directory()


class WorkflowPrivacyTests(unittest.TestCase):
    """Keep public jobs wired to the checked output/cache boundary."""

    def test_all_consumers_require_pin_token_and_disable_cache(self):
        """Every FM consumer validates configuration before its private checkout."""
        for file, job in (("ci.yml", "factor-integration"), ("docs.yml", "docs"), ("examples.yml", "network")):
            with self.subTest(file=file):
                block = job_text(file, job)
                self.assertIn("enable-cache: false", block)
                self.assertIn("save-cache: false", block)
                self.assertIn("persist-credentials: false", block)
                self.assertIn("ref: ${{ vars.TRADABLEFACTORS_REF }}", block)
                self.assertIn("token: ${{ secrets.TRADABLEFACTORS_READ_TOKEN || github.token }}", block)
                self.assertIn("ssh-key: ${{ secrets.FACTORMODELS_READ_SSH_KEY }}", block)
                self.assertIn("FM_READ_CREDENTIAL_PRESENT:", block)
                self.assertNotIn("FM_READ_TOKEN:", block)
                self.assertNotIn("migration/tradablefactors-stage-4", block)
                self.assertLess(block.index("check-config"), block.index("repository: ArturSepp/FactorModels"))
                self.assertIn("private_dependency_test.py", block)
                self.assertIn("private_dependency.py run checkout-identity --", block)
                self.assertIn("rev-parse", block)
                self.assertLess(block.index("repository: ArturSepp/FactorModels"), block.index("run checkout-identity"))

    def test_read_credentials_are_forwarded_and_tests_are_independent(self):
        """The reusable jobs receive SSH/PAT while regression tests receive neither."""
        required = (ROOT / ".github/workflows/required.yml").read_text()
        self.assertEqual(required.count("FACTORMODELS_READ_SSH_KEY: ${{ secrets.FACTORMODELS_READ_SSH_KEY }}"), 3)
        for file, job in (("ci.yml", "factor-integration"), ("docs.yml", "docs"), ("examples.yml", "network")):
            block = job_text(file, job)
            test_step = block.split("- name: Test private dependency containment", 1)[1].split("- name:", 1)[0]
            self.assertNotIn("env:", test_step)
            self.assertNotIn("secrets.", test_step)
            self.assertIn("private_dependency_test.py", test_step)

    def test_composed_gates_remain_nonempty_and_zero_skip(self):
        """Output capture leaves both test lanes and their fail-closed XML checks."""
        block = job_text("ci.yml", "factor-integration")
        self.assertEqual(block.count("assert cases"), 2)
        self.assertEqual(block.count("assert not list(s.iter('skipped'))"), 2)
        self.assertIn("-m factor_integration", block)
        self.assertIn("--pyargs optimalportfolios.tests.factor_integration_test", block)
        self.assertIn("--lane composed", block)
        self.assertEqual(block.count("private_dependency.py run-script"), 4)

    def test_only_fixed_link_status_is_uploaded(self):
        """Neither raw Sphinx records nor source/build artifacts enter uploads."""
        block = job_text("docs.yml", "docs")
        self.assertIn("private-factor-checks/check-links.status.json", block)
        self.assertNotIn("linkcheck/*.json", block)
        self.assertEqual(block.count("uses: actions/upload-artifact@"), 1)
        self.assertIn("if-no-files-found: error", block)

    def test_network_summary_never_replays_captured_output(self):
        """The advisory status remains visible without copying failure details."""
        block = job_text("examples.yml", "network")
        self.assertNotIn("tee ", block)
        self.assertNotIn("grep ", block)
        self.assertNotIn("network.log", block)
        self.assertIn("continue-on-error: true", block)
        self.assertIn("steps.run.outcome", block)


if __name__ == "__main__":
    unittest.main()
