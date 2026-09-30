"""Regression checks for private publication and staged-policy enforcement."""

from pathlib import Path
import io
import tarfile
import tempfile
import unittest
import zipfile

from check_paper_policy import check_artifacts, check_repository, git


class PaperPolicyTests(unittest.TestCase):
    """Exercise actual temporary Git indexes, including force-added private files."""

    def setUp(self) -> None:
        """Create a minimal public example with OP's proposed defaults."""
        self.temp = tempfile.TemporaryDirectory(prefix="op-paper-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")
        source = Path(__file__).resolve().parents[2]
        self.write(".gitignore", (source / ".gitignore").read_text(encoding="utf-8"))
        self.write("papers/AGENTS.md", "Paper contract\n")
        self.write("papers/smart_diversification_joim_2026/.gitignore",
                   (source / "papers/smart_diversification_joim_2026/.gitignore")
                   .read_text(encoding="utf-8"))
        self.write("papers/crypto_allocation_risk_2023/.gitignore", "!/paper/current.tex\n")
        self.write("papers/crypto_allocation_risk_2023/paper/current.tex", "Existing approved manuscript\n")
        self.write("papers/crypto_allocation_risk_2023/replication/reproduce.py", "print('example')\n")
        git(self.root, "add", ".")

    def write(self, name: str, text: str) -> None:
        """Write a fixture without depending on shell quoting or platform newlines."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_approved_bundle_passes(self) -> None:
        """Exact approved manuscript files and replication code are publishable."""
        self.assertEqual(check_repository(self.root), [])

    def test_joim_introduction_and_synthetic_example_are_public(self) -> None:
        """Ordinary staging exposes only the approved introduction, example and its test."""
        base = "papers/smart_diversification_joim_2026/"
        public = ["README.md", "run_overlay_example.py",
                  "replication/tests/overlay_example_test.py"]
        local = ["replication/reproduce_paper_figures.py", "replication/data/input.csv",
                 "replication/inputs/long_vol_qis_universe.py", "paper/paper.tex",
                 "replication/tests/runner_test.py", "replication/results/table.csv"]
        for name in public + local:
            self.write(base + name, "fixture\n")
        git(self.root, "add", ".")
        indexed = set(git(self.root, "ls-files", "-z").decode().split("\0"))
        self.assertTrue(all(base + name in indexed for name in public))
        self.assertFalse(any(base + name in indexed for name in local))
        self.assertEqual(check_repository(self.root), [])

    def test_joim_extra_exception_cannot_publish_empirical_code(self) -> None:
        """The narrow source allowlist also rejects a deliberate local ignore override."""
        base = "papers/smart_diversification_joim_2026/"
        name = base + "replication/reproduce_paper_figures.py"
        self.write(base + ".gitignore", "!/replication/\n!/replication/reproduce_paper_figures.py\n")
        self.write(name, "private empirical runner\n")
        git(self.root, "add", "-f", base + ".gitignore", name)
        self.assertTrue(any("protected material" in error and name in error
                            for error in check_repository(self.root)))

    def test_force_added_private_material_fails(self) -> None:
        """Ignore rules do not protect an already staged or force-added file."""
        for section in ("private", "drafts", "agents", "replication/data/local"):
            with self.subTest(section=section):
                name = f"papers/crypto_allocation_risk_2023/{section}/record.txt"
                self.write(name, "private\n")
                git(self.root, "add", "-f", name)
                self.assertTrue(any(name in error for error in check_repository(self.root)))
                git(self.root, "rm", "--cached", "-f", name)

    def test_nested_exception_cannot_publish_private_material(self) -> None:
        """Private section classification wins over deliberate ignore negations."""
        self.write("papers/crypto_allocation_risk_2023/private/.gitignore", "!record.txt\n")
        self.write("papers/crypto_allocation_risk_2023/private/record.txt", "private\n")
        git(self.root, "add", "-f", "papers/crypto_allocation_risk_2023/private")
        self.assertTrue(any("protected material" in error for error in check_repository(self.root)))

    def test_unapproved_pdf_fails(self) -> None:
        """Force-adding a manuscript PDF is not a publication decision."""
        name = "papers/crypto_allocation_risk_2023/paper/current.pdf"
        self.write(name, "pdf fixture\n")
        git(self.root, "add", "-f", name)
        self.assertTrue(any(name in error for error in check_repository(self.root)))

    def test_unstaged_exception_does_not_change_index_verdict(self) -> None:
        """The checker reads the staged .gitignore, not a more permissive worktree."""
        name = "papers/crypto_allocation_risk_2023/paper/current.pdf"
        self.write(name, "pdf fixture\n")
        git(self.root, "add", "-f", name)
        self.write("papers/crypto_allocation_risk_2023/.gitignore", "!/paper/current.tex\n!/paper/current.pdf\n")
        self.assertTrue(check_repository(self.root))
        self.assertEqual(check_repository(self.root, worktree=True), [])
        git(self.root, "add", "papers/crypto_allocation_risk_2023/.gitignore")
        self.assertEqual(check_repository(self.root), [])

    def test_missing_generic_protection_fails(self) -> None:
        """A policy regression is detected before a private file exists."""
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text().replace("/papers/**/private/\n", ""))
        git(self.root, "add", ".gitignore")
        self.assertTrue(any("missing default" in error for error in check_repository(self.root)))

    def test_wildcard_publication_exception_fails(self) -> None:
        """A wildcard must not silently approve future manuscript versions."""
        self.write("papers/crypto_allocation_risk_2023/.gitignore", "!/paper/*.tex\n")
        git(self.root, "add", "papers/crypto_allocation_risk_2023/.gitignore")
        self.assertTrue(any("exact publication" in error for error in check_repository(self.root)))

    def test_open_directory_does_not_approve_all_figures(self) -> None:
        """Opening a parent directory cannot implicitly approve its contents."""
        self.write("papers/crypto_allocation_risk_2023/.gitignore", "!/paper/current.tex\n!/paper/figures/\n")
        self.write("papers/crypto_allocation_risk_2023/paper/figures/unreviewed.png", "image\n")
        git(self.root, "add", "papers/crypto_allocation_risk_2023")
        self.assertTrue(any("exact exception" in error for error in check_repository(self.root)))

    def test_private_files_stay_out_of_normal_add(self) -> None:
        """Ordinary staging cannot expose local sections or a new paper workspace."""
        names = [
            "papers/crypto_allocation_risk_2023/private/report.txt",
            "papers/robust_optimisation_jpm_2026/drafts/v1.tex",
            "papers/robust_optimisation_jpm_2026/presentations/slides.tex",
            "papers/robust_optimisation_jpm_2026/agents/notes.md",
            "papers/crypto_allocation_risk_2023/replication/data/unapproved.csv",
            "papers/smart_diversification_joim_2026/paper/paper.tex",
            "papers/new_paper/replication/reproduce.py",
        ]
        for name in names:
            self.write(name, "local material\n")
        git(self.root, "add", ".")
        indexed = set(git(self.root, "ls-files", "-z").decode().split("\0"))
        self.assertFalse(indexed.intersection(names))
        self.assertEqual(check_repository(self.root), [])

    def test_whole_local_workspace_cannot_be_reopened(self) -> None:
        """A nested exception cannot publish an unapproved paper's replication."""
        self.write(".gitignore", (self.root / ".gitignore").read_text()
                   + "\n!/papers/new_paper/\n")
        name = "papers/new_paper/replication/reproduce.py"
        self.write(name, "private implementation\n")
        git(self.root, "add", ".")
        self.assertTrue(any("protected material" in error and name in error
                            for error in check_repository(self.root)))

    def test_staged_export_without_git(self) -> None:
        """The pre-commit export passes without Git metadata and rejects private files."""
        with tempfile.TemporaryDirectory(prefix="op-policy-export-") as directory:
            export = Path(directory)
            for name in git(self.root, "ls-files", "-z").decode().split("\0"):
                if name:
                    path = export / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(git(self.root, "show", f":{name}"))
            self.assertEqual(check_repository(export, source_export=True), [])
            private = export / "papers/crypto_allocation_risk_2023/private/report.txt"
            private.parent.mkdir(parents=True)
            private.write_text("private")
            self.assertTrue(any("protected material" in error for error in
                                check_repository(export, source_export=True)))

    def test_source_and_wheel_workspaces_fail(self) -> None:
        """Distribution checks inspect archive members, including source archives."""
        directory = self.root / "dist"
        directory.mkdir()
        with zipfile.ZipFile(directory / "example.whl", "w") as archive:
            archive.writestr("papers/crypto_allocation_risk_2023/private/report.txt", "private")
        with tarfile.open(directory / "example.tar.gz", "w:gz") as archive:
            entry = tarfile.TarInfo("example/agents/ROADMAP.md")
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
        self.assertEqual(len(check_artifacts(directory)), 2)


if __name__ == "__main__":
    unittest.main()
