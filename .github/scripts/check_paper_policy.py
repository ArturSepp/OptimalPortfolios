"""Check indexed paper publication boundaries, or preview the working tree."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import zipfile


PUBLIC_WORKSPACES = {"cma_data", "crypto_allocation_risk_2023", "robust_optimisation_jpm_2026",
                     "smart_diversification_joim_2026"}

# This workspace's public scope is the companion, never the empirical replication tree.
LIMITED_PUBLIC_FILES = {
    "smart_diversification_joim_2026": {
        ".gitignore", "README.md", "run_overlay_example.py",
        "replication/tests/overlay_example_test.py",
    },
}

PROBES = (
    "papers/policy_probe/replication/reproduce.py",
    "papers/policy_probe/drafts/v1/paper.tex",
    "papers/policy_probe/private/referee.txt",
    "papers/policy_probe/agents/ROADMAP.md",
    "papers/policy_probe/paper/current.tex",
    "papers/policy_probe/paper/current.pdf",
    "papers/policy_probe/presentations/event/slides.pdf",
    "papers/policy_probe/replication/data/local/input.csv",
    "papers/policy_probe/replication/outputs/run.csv",
    "agents/ROADMAP.md",
)
PROBES += tuple(
    f"papers/{workspace}/{suffix}"
    for workspace in sorted(PUBLIC_WORKSPACES)
    for suffix in (
        "drafts/v1/paper.tex", "private/referee.txt", "agents/ROADMAP.md",
        "paper/unapproved.tex", "paper/unapproved.pdf", "presentations/event/slides.pdf",
        "replication/data/local/input.csv", "replication/data/unapproved.csv",
        "replication/outputs/run.csv",
    )
)


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    """Run read-only Git queries without relying on personal exclusions."""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-c", "core.excludesFile=",
         "-C", str(root), *args], input=data, capture_output=True, check=False,
    )
    if result.returncode not in (0, 1) or (result.returncode and args[0] != "check-ignore"):
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def selected_files(root: Path, worktree: bool) -> list[str]:
    """Select indexed paths, or tracked and visible new working-tree files."""
    names = set(git(root, "ls-files", "-z").decode("utf-8").split("\0")) - {""}
    if worktree:
        names.update(git(root, "ls-files", "--others", "--exclude-standard", "-z")
                     .decode("utf-8").split("\0"))
        names = {name for name in names if name and (root / name).is_file()}
    return sorted(names)


def protected(name: str) -> bool:
    """Identify private sections even if a nested ignore rule reopens them."""
    parts = PurePosixPath(name).parts
    lower = [part.casefold() for part in parts]
    if lower[0] == "agents":
        return True
    if lower[0] != "papers":
        return False
    if len(parts) >= 3 and lower[1] not in PUBLIC_WORKSPACES:
        return True
    if len(parts) >= 3 and lower[1] in LIMITED_PUBLIC_FILES:
        if PurePosixPath(*parts[2:]).as_posix() not in LIMITED_PUBLIC_FILES[lower[1]]:
            return True
    directories = lower[1:-1]
    return bool(set(directories) & {"private", "drafts", "agents"}) or any(
        directories[i:i + 2] == ["data", "local"] for i in range(len(directories))
    )


def check_repository(
    root: Path, worktree: bool = False, source_export: bool = False,
) -> list[str]:
    """Evaluate paths against an isolated copy of the selected ignore policy."""
    names = (sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file()) if source_export else selected_files(root, worktree))
    errors = [f"protected material is tracked: {name}" for name in names if protected(name)]
    paper_names = [name for name in names if name.casefold().startswith("papers/")]
    approved_files = set()
    if "papers/AGENTS.md" not in names:
        errors.append("papers/AGENTS.md is missing from the selected files")
    with tempfile.TemporaryDirectory(prefix="op-paper-policy-") as directory:
        fixture = Path(directory)
        git(fixture, "init", "-q")
        for name in names:
            if PurePosixPath(name).name != ".gitignore":
                continue
            contents = (root / name).read_bytes() if worktree or source_export else git(root, "show", f":{name}")
            destination = fixture / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            for line in contents.decode("utf-8-sig").splitlines():
                if line.startswith("!") and not line.endswith("/"):
                    if not any(character in line for character in "*?["):
                        approved_files.add((PurePosixPath(name).parent / line[1:].lstrip("/"))
                                           .as_posix())
                if line.startswith("!") and ("paper/" in line or "presentations/" in line):
                    if any(character in line for character in "*?["):
                        errors.append(f"use exact publication exceptions in {name}: {line}")
        candidates = paper_names + list(PROBES)
        data = ("\0".join(candidates) + "\0").encode("utf-8")
        ignored = set(git(fixture, "check-ignore", "--no-index", "-z", "--stdin", data=data)
                      .decode("utf-8").split("\0")) - {""}
        errors.extend(f"indexed paper file violates ignore policy: {name}"
                      for name in paper_names if name in ignored)
        errors.extend(f"missing default ignore protection: {name}"
                      for name in PROBES if name not in ignored)
    for name in paper_names:
        parts = PurePosixPath(name).parts
        if len(parts) >= 4 and parts[2].casefold() in {"paper", "presentations"}:
            if name not in approved_files:
                errors.append(f"publication file needs an exact exception: {name}")
        if len(parts) == 3 and parts[1] != "cma_data" and name.endswith(".py"):
            if (parts[-1] != "__init__.py"
                    and parts[-1] not in LIMITED_PUBLIC_FILES.get(parts[1], set())):
                errors.append(f"paper code belongs in replication/: {name}")
    return errors


def check_artifacts(directory: Path) -> list[str]:
    """Reject paper or agent workspaces in both wheel and source archives."""
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    errors = []
    if not wheels or not sources:
        errors.append("artifact check requires both a wheel and a .tar.gz source distribution")
    for path in wheels + sources:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(path) as archive:
                names = archive.getnames()
        for name in names:
            if {part.casefold() for part in PurePosixPath(name).parts} & {"papers", "agents"}:
                errors.append(f"research workspace shipped in {path.name}: {name}")
    return errors


def main() -> int:
    """Run the index, worktree-preview or archive check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", action="store_true", help="preview unstaged changes")
    parser.add_argument("--artifacts", type=Path, help="check wheels and source archives")
    parser.add_argument("--source-export", action="store_true",
                        help="validate all files in an isolated staged export without .git")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    # Preflight exports index blobs without .git. In a checkout, retain index semantics.
    source_export = args.source_export and not (root / ".git").exists()
    errors = (check_artifacts(args.artifacts) if args.artifacts
              else check_repository(root, args.worktree, source_export))
    if errors:
        print("Paper policy failed:\n- " + "\n- ".join(errors))
        return 1
    mode = "archives" if args.artifacts else "working tree" if args.worktree else "Git index"
    if source_export:
        mode = "staged source export"
    print(f"PASS: paper publication policy ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
