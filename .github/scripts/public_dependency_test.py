"""Keep public FM verification reproducible, credential-free and fail closed."""

from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

import run_examples as runner


ROOT = Path(__file__).resolve().parents[2]
CONSUMERS = (("ci.yml", "factor-integration"), ("docs.yml", "docs"), ("examples.yml", "network"))


def workflow(filename):
    """Read the workflow whose dependency behavior is being verified."""
    return (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")


def job_text(filename, job):
    """Select one YAML job without introducing a workflow runtime dependency."""
    block = workflow(filename).split(f"\n  {job}:\n", 1)[1]
    return re.split(r"\n  [a-z][a-z-]*:\n", block, maxsplit=1)[0]


class PublicDependencyWorkflowTests(unittest.TestCase):
    """Reject private access regressions or silently weakened integration gates."""

    def test_public_consumers_need_no_private_checkout_or_credentials(self):
        """Public forks must be able to run all required dependency checks."""
        for filename in (*[name for name, _ in CONSUMERS], "required.yml"):
            with self.subTest(filename=filename):
                source = workflow(filename)
                for retired in ("TRADABLEFACTORS_REF", "TRADABLEFACTORS_READ_TOKEN",
                                "FACTORMODELS_READ_SSH_KEY", "private_dependency.py",
                                "private_dependency_test.py", ".integration/tradablefactors"):
                    self.assertNotIn(retired, source)

    def test_each_consumer_installs_the_exact_public_release(self):
        """Every composed environment uses the reviewed PyPI release, not a ref."""
        for filename, job in CONSUMERS:
            with self.subTest(filename=filename):
                block = job_text(filename, job)
                self.assertIn('"factormodels==0.3.0"', block)
                self.assertNotIn("repository: ArturSepp/FactorModels", block)
                self.assertNotIn("git+", block)
                self.assertNotIn(".integration/", block)
                self.assertIn("public_dependency_test.py", block)

    def test_composed_source_and_wheel_installations_are_independent(self):
        """The installed check must not inherit the source checkout environment."""
        block = job_text("ci.yml", "factor-integration")
        self.assertIn('uv venv "$RUNNER_TEMP/composed-source"', block)
        self.assertIn('uv venv "$RUNNER_TEMP/composed-wheel"', block)
        self.assertEqual(block.count('"factormodels==0.3.0"'), 2)
        self.assertIn('".[riskbudget,data]"', block)
        self.assertEqual(block.count('"qis>=5.44.5"'), 2)
        self.assertIn('cd "$RUNNER_TEMP"', block)
        self.assertIn('--pyargs optimalportfolios.tests.factor_integration_test', block)

    def test_both_composed_gates_are_nonempty_and_reject_skips(self):
        """A missing dependency cannot silently turn a composed lane green."""
        block = job_text("ci.yml", "factor-integration")
        for expression in ("assert cases", "assert not list(s.iter('skipped'))",
                           "assert not list(s.iter('failure'))", "assert not list(s.iter('error'))"):
            self.assertEqual(block.count(expression), 2)
        self.assertEqual(block.count('OP_REQUIRE_FACTOR_INTEGRATION: "1"'), 2)
        self.assertIn("-m factor_integration", block)
        self.assertIn("--lane composed", block)

    def test_base_wheel_keeps_factor_dependencies_absent(self):
        """Publishing FM does not add a factor dependency to the OP base wheel."""
        block = job_text("ci.yml", "wheel")
        self.assertIn("'factorlasso', 'factormodels', 'tradablefactors'", block)
        self.assertNotIn('"factormodels==0.3.0"', block)
        self.assertNotIn('"factormodels==0.3.0"', job_text("examples.yml", "offline"))

    def test_public_diagnostics_and_link_evidence_are_visible(self):
        """Dependency failures retain their actual public diagnostics."""
        for filename, job in CONSUMERS:
            self.assertNotIn("private_dependency", job_text(filename, job))
        block = job_text("docs.yml", "docs")
        self.assertIn("documentation/linkcheck/*.json", block)
        self.assertIn("if-no-files-found: error", block)

    def test_live_provider_lane_remains_advisory(self):
        """Public dependency availability does not make Yahoo availability a gate."""
        block = job_text("examples.yml", "network")
        self.assertIn("continue-on-error: true", block)
        self.assertIn("steps.run.outcome", block)
        self.assertIn("--lane network --jobs 4", block)


class PublicExampleSolverTests(unittest.TestCase):
    """The unattended portfolio example explicitly selects an available public solver."""

    def test_exact_extra_uses_updater_normalized_distribution_name(self):
        """Pinned solver names must match Dependabot's normalized update target."""
        import tomllib

        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requirement, = project["project"]["optional-dependencies"]["exact"]
        name, operator, version = requirement.partition("==")
        self.assertEqual(name, "mosek")
        self.assertEqual(operator, "==")
        self.assertTrue(version)

    def test_pipeline_command_selects_clarabel_without_changing_its_default(self):
        path = ROOT / "examples/oss_portfolio_pipeline/__main__.py"
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch.object(runner.subprocess, "run", return_value=completed) as execute:
            self.assertTrue(runner.run_one(path, 15)[1])
        self.assertEqual(execute.call_args.args[0], [
            sys.executable, "-m", "examples.oss_portfolio_pipeline.__main__", "--solver", "CLARABEL",
        ])
        execute.assert_called_once()
        import ast
        tree = ast.parse(path.read_text(encoding="utf-8"))
        solver = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                      and any(isinstance(arg, ast.Constant) and arg.value == "--solver" for arg in node.args))
        self.assertEqual(next(k.value.value for k in solver.keywords if k.arg == "default"), "MOSEK")

    def test_example_errors_are_fatal_and_other_commands_are_preserved(self):
        for module in ("examples.oss_portfolio_pipeline.__main__", "examples.solvers.tracking_error"):
            path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
            completed = subprocess.CompletedProcess([], 1, stdout="", stderr="RuntimeError: fixture")
            with self.subTest(module=module), patch.object(
                    runner.subprocess, "run", return_value=completed) as execute:
                result = runner.run_one(path, 15)
                self.assertFalse(result[1])
                self.assertIn("RuntimeError: fixture", result[-1])
                execute.assert_called_once()
                if module.endswith("tracking_error"):
                    self.assertEqual(execute.call_args.args[0], [sys.executable, "-m", module])


if __name__ == "__main__":
    unittest.main()
