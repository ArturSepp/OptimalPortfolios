"""Keep the example catalogue aligned with runnable sources and execution prerequisites."""

import ast
from collections import Counter
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

import pytest


@pytest.fixture
def guide(root: Path):
    """Load the source catalogue without importing scripts that download or write files."""
    page = root / "docs/examples_readme.md"
    return page, page.read_text(encoding="utf-8")


def catalogue(source: str):
    """Return each source row with its execution lane and description."""
    rows = re.findall(
        r"^\| \[[^\]]+\]\(\.\./(examples/[^)]+\.py)\) \| "
        r"(Offline|Network|Local) \| (.+) \|$",
        source, re.MULTILINE,
    )
    assert rows, "The guide must map example sources to execution prerequisites."
    assert len(rows) == len({row[0] for row in rows}), "Duplicate catalogue source."
    return {path: (lane.lower(), description) for path, lane, description in rows}


def assignments(root: Path, path: str, name: str):
    """Read literal assignments without running an example's import-time side effects."""
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                values.append(ast.literal_eval(node.value))
    return values


def test_examples_utility_and_local_links(root, guide):
    """Validate the utility standard and real source links through the shared checker."""
    page, source = guide
    checker = runpy.run_path(str(root / "tools/check_docs.py"))
    assert not checker["check_document"](source, methodology=False)
    assert not checker["check_local_links"](source, page, root)
    inventory = json.loads((root / "tools/docs_inventory.json").read_text(encoding="utf-8"))
    assert inventory["pages"]["docs/examples_readme.md"]["form"] == "utility"


def test_catalogue_matches_classifier_and_local_exclusions(root, guide):
    """Catch missing examples, stale lane labels and incorrect inventory totals."""
    _, source = guide
    result = subprocess.run(
        [sys.executable, ".github/scripts/run_examples.py", "--list"],
        cwd=root, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {}
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"(offline|network)\s+(examples[\\/].+\.py)", line)
        if match:
            expected[match[2].replace("\\", "/")] = match[1]
    assert expected, result.stdout
    local = {p.relative_to(root).as_posix(): "local"
             for p in (root / "examples").rglob("*_local.py")}
    documented = {path: row[0] for path, row in catalogue(source).items()}
    assert documented == {**expected, **local}
    counts = Counter(expected.values())
    summary = f"{counts['offline']} offline, {counts['network']} network, {len(expected)}"
    assert summary + " unattended examples" in source
    assert f"{len(local)} local-data workflows" in source


@pytest.mark.parametrize("path,scenario", [
    ("examples/data/sp500_universe_local.py", "CREATE_UNIVERSE_DATA_WITH_BLOOMBERG"),
    ("examples/backtests/multiasset_saa.py", "OBJECTIVE_SWEEP"),
])
def test_documented_default_scenario_matches_main_guard(root, guide, path, scenario):
    """Prevent a guide from sending a user into an unmentioned default data workflow."""
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and "__main__" in ast.unparse(node.test)]
    assert len(guards) == 1
    selected = [keyword.value.attr for node in ast.walk(guards[0])
                if isinstance(node, ast.Call) for keyword in node.keywords
                if keyword.arg == "local" and isinstance(keyword.value, ast.Attribute)]
    assert selected == [scenario]
    assert scenario in catalogue(guide[1])[path][1]


def test_csv_catalogue_distinguishes_default_and_load_mode(root, guide):
    """Verify the actual parser default without fetching or requiring an input bundle."""
    path = "examples/covar_estimation/rolling_factor_covar_from_csv.py"
    module = runpy.run_path(str(root / path))
    calls = []
    main = module["main"]
    main.__globals__["fetch_and_save_yahoo_csvs"] = lambda **kw: calls.append(("fetch", kw))
    main.__globals__["fit_rolling_risk_model_from_csv"] = lambda **kw: calls.append(("load", kw))
    main([])
    assert [kind for kind, _ in calls] == ["fetch", "load"]
    calls.clear()
    main(["load", "--data-dir", "existing-csv-bundle"])
    assert calls == [("load", {"data_dir": Path("existing-csv-bundle")})]
    row = catalogue(guide[1])[path][1]
    assert "default \x60all\x60 mode" in row
    assert "\x60load\x60 mode uses an existing bundle without downloading" in row


def test_preview_writers_are_identified_in_output_guidance(root, guide):
    """Ensure every source-relative preview writer is disclosed before batch execution."""
    source = guide[1]
    section = source.split("### Output paths and preview images\n", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"\]\(\.\./(examples/[^)]+\.py)\)", section))
    writers = set()
    for path in (root / "examples").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "save_fig"
               and any(kw.arg == "local_path" and isinstance(kw.value, ast.Name)
                       and kw.value.id == "FIGURES_PATH" for kw in node.keywords)
               for node in ast.walk(tree)):
            writers.add(path.relative_to(root).as_posix())
    assert writers and documented == writers
    assert "\x60examples/figures/\x60" in section
    assert "overwrite existing documentation previews" in section
    assert "C-local source export" in section


@pytest.mark.parametrize("path", [
    "examples/comparisons/parameter_sensitivity.py",
    "examples/comparisons/sp500_minvar_spans_local.py",
])
def test_span_catalogue_uses_configured_observation_counts(root, guide, path):
    """Keep numerical parameter labels tied to the scripts without changing their values."""
    values, = assignments(root, path, "spans")
    spans = list(values.values()) if isinstance(values, dict) else values
    row = catalogue(guide[1])[path][1]
    assert "weekly ewma spans " + ", ".join(map(str, spans)) in row.lower()
    assert "not half-lives or fixed rolling windows" in guide[1]


def test_documented_module_commands_and_migration_destinations_exist(root, guide):
    """Resolve copyable module commands and every old-to-new destination without imports."""
    source = guide[1]
    commands = re.findall(r"^python -m ([\w.]+)$", source, re.MULTILINE)
    assert commands
    migration = source.split("## Migration note\n", 1)[1].split("\n## ", 1)[0]
    destinations = re.findall(r"^\| \x60[^\x60]+\x60 → \x60([^\x60]+)\x60 \|$",
                              migration, re.MULTILINE)
    assert destinations
    for module in commands + destinations:
        relative = Path(*module.split(".")).with_suffix(".py")
        assert (root / relative).is_file() or (root / "src" / relative).is_file(), module


def test_catalogued_solver_names_are_public(guide):
    """Reject stale public entry-point names without running any network examples."""
    import optimalportfolios as op

    section = guide[1].split("## \x60solvers/\x60", 1)[1].split("\n## ", 1)[0]
    names = re.findall(r"\x60((?:rolling_|wrapper_|compute_rolling_)\w+)\x60", section)
    assert names
    for name in names:
        assert callable(getattr(op, name, None)), name
