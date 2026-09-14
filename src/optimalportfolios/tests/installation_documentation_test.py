"""Validate installation recipes against metadata and fresh core imports.

These checks never install packages or synchronize the environment. Checkout-only checks use
the shared root fixture so installed-wheel runs skip them when the guide is absent.
"""

import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pytest requires tomli on Python 3.10.


@pytest.fixture
def installation(root: Path):
    """Read the authored guide and its authoritative packaging metadata."""
    page = root / "docs/installation.md"
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return page, page.read_text(encoding="utf-8"), metadata


def command_blocks(source: str) -> str:
    """Extract shell fences without treating explanatory prose as commands."""
    blocks = re.findall(r"^\x60{3}(?:console|powershell)\n(.*?)^\x60{3}$",
                        source, flags=re.MULTILINE | re.DOTALL)
    assert blocks, "The installation guide must provide copyable commands."
    return "\n".join(blocks)


def test_installation_utility_and_source_links(root, installation):
    """Keep the migrated page in the inventory with a unique Sphinx basename."""
    page, source, _ = installation
    checker = runpy.run_path(str(root / "tools/check_docs.py"))
    assert not checker["check_document"](source, methodology=False)
    assert not checker["check_local_links"](source, page, root)
    assert not (root / "docs/installation.rst").exists()
    inventory = json.loads((root / "tools/docs_inventory.json").read_text(encoding="utf-8"))
    assert inventory["pages"]["docs/installation.md"]["form"] == "utility"
    assert "docs/installation.rst" not in inventory["pages"]


def test_pip_recipes_use_published_extras(installation):
    """Reject the historical all-extra install even though all is a valid group."""
    _, source, metadata = installation
    extras = set(metadata["project"]["optional-dependencies"])
    table = source.split("## Choose optional integrations", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^\| \x60([^\x60]+)\x60 \|", table, re.MULTILINE))
    assert documented == extras
    requirements = re.findall(
        r"^python -m pip install (?:-e )?\"?([^\s\"]+)\"?$",
        command_blocks(source), re.MULTILINE,
    )
    assert requirements
    project_requirements = [item for item in requirements
                            if item.startswith(("optimalportfolios", "."))]
    assert project_requirements, "Keep a project installation recipe."
    for requirement in project_requirements:
        match = re.fullmatch(r"(optimalportfolios|\.)(?:\[([a-z,-]+)\])?", requirement)
        assert match, requirement
        selected = set(match[2].split(",")) if match[2] else set()
        assert selected <= extras, f"Unpublished extra in {requirement}"


def test_uv_recipes_use_real_groups_and_an_external_environment(installation):
    """Check selections and prevent uv's OneDrive-local environment default."""
    _, source, metadata = installation
    groups = metadata["dependency-groups"]
    table = source.split("### Contributor groups and the lockfile", 1)[1].split("\n### ", 1)[0]
    assert set(re.findall(r"^\| \x60([^\x60]+)\x60 \|", table, re.MULTILINE)) == set(groups)
    commands = command_blocks(source)
    assignment = "$env:UV_PROJECT_ENVIRONMENT = 'C:\\Python\\OptimalPortfolios312'"
    assert assignment in commands
    assert commands.index(assignment) < commands.index("uv sync")
    syncs = [line for line in commands.splitlines() if line.startswith("uv sync ")]
    runs = [line for line in commands.splitlines() if line.startswith("uv run ")]
    assert syncs and runs
    for line in syncs:
        assert "--locked" in line, line
        assert set(re.findall(r"--group (\S+)", line)) <= set(groups), line
        extras = set(re.findall(r"--extra (\S+)", line))
        assert extras <= set(metadata["project"]["optional-dependencies"]), line
    assert all("--no-sync" in line for line in runs)


def test_python_requirement_matches_project(installation):
    """Keep the stated interpreter floor tied to the packaging contract."""
    _, source, metadata = installation
    requirement = re.search(r'requires-python = "([^"]+)"', source)
    assert requirement
    assert requirement[1] == metadata["project"]["requires-python"]


def test_import_example_runs_without_optional_integrations(installation, tmp_path):
    """Run the actual snippet in a fresh process with optional imports and sockets blocked."""
    _, source, _ = installation
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}$", source,
                        flags=re.MULTILINE | re.DOTALL)
    assert len(blocks) == 1 and blocks[0].strip()
    guard = r'''
import importlib.abc
import socket
import sys

class NoOptionalImports(importlib.abc.MetaPathFinder):
    """Make an installed optional dependency unavailable to this smoke check."""
    def find_spec(self, fullname, path=None, target=None):
        """Reject optional integration imports before delegating core imports."""
        if fullname.split(".")[0] in {
            "yfinance", "pandas_datareader", "pybloqs", "plotly", "pyarrow",
            "psycopg2", "sqlalchemy",
        }:
            raise ModuleNotFoundError(fullname)
        return None

def no_network(*args, **kwargs):
    """Fail if the import example opens a network connection."""
    raise AssertionError("The installation import example must work offline.")

sys.meta_path.insert(0, NoOptionalImports())
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
'''
    result = subprocess.run(
        [sys.executable, "-c", guard + "\n" + blocks[0]],
        cwd=tmp_path, capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Python:" in result.stdout
    assert "OptimalPortfolios source:" in result.stdout
    assert "Available extras:" in result.stdout


def test_documented_classifier_keeps_quickstart_offline(root, installation):
    """Check the first workflow against the actual example dependency classifier."""
    _, source, _ = installation
    commands = command_blocks(source)
    assert "python examples/getting_started/production_quickstart.py" in commands
    assert "python .github/scripts/run_examples.py --list" in commands
    result = subprocess.run(
        [sys.executable, ".github/scripts/run_examples.py", "--list"],
        cwd=root, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    listing = result.stdout.replace("\\", "/")
    assert re.search(r"^offline\s+examples/getting_started/production_quickstart.py$",
                     listing, re.MULTILINE)
