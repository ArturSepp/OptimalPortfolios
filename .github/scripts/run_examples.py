"""
run the example scripts under optimalportfolios/examples, in parallel, and report what broke.

Why this exists at all
----------------------
`examples/` is excluded from wheels, dropped by `[tool.coverage.run] omit` and never collected by
pytest, so nothing in the test suite executes it. Ruff does not help either: the rot these scripts
attract is attribute-level, not name-level. `LassoModelType.GROUP_LASSO_CLUSTERS` was renamed
upstream and sat broken in two examples, a shipped docstring and three documents, because an
enum member that no longer exists is a perfectly valid attribute access to a linter.

Why the split into lanes
------------------------
Of the 26 examples, 21 need live Yahoo Finance data -- 8 import `yfinance` directly and the rest
reach it through `examples/data/universe.py`, whose `fetch_benchmark_universe_data()` downloads 15
tickers back to 2003 (`sp500_universe.py` pulls the whole index). Gating a pull request on 63 live
downloads would fail on Yahoo's availability far more often than on this repository's code, and a
check that is usually red for reasons outside the diff is a check people learn to ignore.

So the network examples run on a schedule and advisory-only, while the offline ones gate. The
classification is *derived*, not listed: a hand-maintained list of five paths would drift the
first time someone adds an example, and the point of this file is to notice drift.

Classification
--------------
An example is `network` if `yfinance` is reachable from it -- directly, or through any chain of
intra-`examples` imports. Everything else is `offline`. The walk is over the AST, so it costs
nothing and needs no imports to succeed.

A note on `UnicodeEncodeError`
-----------------------------
Examples run here with their output captured, so the child's stdout is a pipe rather than a
console. On Windows that changes the encoding: Python writes to a console through the Unicode
console API, but falls back to the locale encoding (cp1252) for a pipe, so a `print()` containing
box-drawing characters or mathematical symbols raises where the same script is fine interactively.
That is a real defect rather than an artifact -- it is what any user gets from
`python example.py > out.txt` -- so the fix belongs in the example, and this runner deliberately
does not set `PYTHONIOENCODING` to paper over it. Keep printed output ASCII; docstrings and
comments are unconstrained, since they never reach the stream.

Usage:
    python .github/scripts/run_examples.py --lane offline [--jobs N] [--timeout SECONDS]
    python .github/scripts/run_examples.py --list
"""
# packages
import argparse
import ast
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "optimalportfolios" / "examples"

# The package path `examples` lives at, used to turn an import statement back into a file.
EXAMPLES_PACKAGE = "optimalportfolios.examples"

# The module whose presence anywhere in an example's import closure makes it network-bound. It is
# the only optional extra the examples reach for; if that changes, add to this set rather than
# reclassifying by hand.
NETWORK_MODULES: Set[str] = {"yfinance"}

# `*_local.py` are `run_local_test` diagnostic dispatchers that need local price data which is not
# in the repository, and `__init__.py` is not an example. Neither is a runnable example here.
def _is_example(path: Path) -> bool:
    """a runnable example, as opposed to a local dispatcher or a package marker."""
    return (path.suffix == ".py"
            and not path.name.endswith("_local.py")
            and path.name != "__init__.py")


def discover() -> List[Path]:
    """every runnable example, in a stable order."""
    return sorted(p for p in EXAMPLES_ROOT.rglob("*.py") if _is_example(p))


def _module_name(path: Path) -> str:
    """the dotted module name an example file is importable as."""
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _imports_of(path: Path) -> Tuple[Set[str], Set[str]]:
    """
    the top-level modules and the intra-examples modules a file imports.

    Returns (external_roots, internal_modules). `external_roots` is what a `NETWORK_MODULES`
    membership test runs against; `internal_modules` is what the closure walk follows.
    """
    external: Set[str] = set()
    internal: Set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if alias.name.startswith(EXAMPLES_PACKAGE):
                    internal.add(alias.name)
                else:
                    external.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import: resolve it against the importing file's package.
                pkg_parts = _module_name(path).split(".")[:-1]
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith(EXAMPLES_PACKAGE):
                internal.add(target)
                # `from ...data.universe import f` may name the module, or the package plus a
                # module-valued attribute; record both candidates and let resolution drop misses.
                for alias in node.names:
                    internal.add(f"{target}.{alias.name}")
            elif target:
                external.add(target.split(".")[0])

    return external, internal


def _resolve(module: str) -> Path | None:
    """the file a dotted intra-examples module refers to, if it is a module rather than a symbol."""
    candidate = REPO_ROOT / Path(*module.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package_init = REPO_ROOT / Path(*module.split(".")) / "__init__.py"
    return package_init if package_init.is_file() else None


def classify(examples: List[Path]) -> Dict[Path, str]:
    """
    label each example `network` or `offline` by walking its intra-examples import closure.

    Direct imports are not enough: thirteen of these scripts touch the network only through
    `examples/data/universe.py`, and treating them as offline is how a lane that promises to need
    no network starts calling Yahoo.
    """
    cache: Dict[Path, bool] = {}

    def reaches_network(path: Path, seen: Set[Path]) -> bool:
        if path in cache:
            return cache[path]
        if path in seen:
            return False  # an import cycle contributes nothing on its own
        seen.add(path)

        external, internal = _imports_of(path)
        if external & NETWORK_MODULES:
            cache[path] = True
            return True

        for module in internal:
            target = _resolve(module)
            if target is not None and reaches_network(target, seen):
                cache[path] = True
                return True

        cache[path] = False
        return False

    return {p: ("network" if reaches_network(p, set()) else "offline") for p in examples}


def run_one(path: Path, timeout: int) -> Tuple[Path, bool, float, str]:
    """execute one example in its own interpreter; never raise, always report."""
    started = time.time()
    env = dict(os.environ)
    # The examples call plt.show(); Agg makes that a no-op rather than a block on a headless
    # runner. conftest.py does this for the suite, but nothing imports conftest here.
    env.setdefault("MPLBACKEND", "Agg")
    try:
        # `errors="replace"` matters on Windows and is not cosmetic. `text=True` decodes the
        # child's output as UTF-8, but a child writing to a pipe there encodes with the locale
        # codec, so an em-dash arrives as the single byte 0x97 -- invalid UTF-8. Without this the
        # *runner* dies with a UnicodeDecodeError while the example it was reporting on had
        # succeeded, turning a green example into an unreadable harness traceback.
        proc = subprocess.run([sys.executable, str(path)], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, errors="replace", timeout=timeout)
        ok = proc.returncode == 0
        detail = "" if ok else _last_error_line(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        ok, detail = False, f"timed out after {timeout}s"
    return path, ok, time.time() - started, detail


def _last_error_line(output: str) -> str:
    """the most useful single line of a traceback, for a one-line-per-example report."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for line in reversed(lines):
        if any(marker in line for marker in ("Error", "Exception", "error:")):
            return line[:200]
    return lines[-1][:200] if lines else "no output"


def main() -> int:
    """run the selected lane and report one line per example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["offline", "network", "all"], default="offline")
    parser.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2) * 2))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--list", action="store_true", help="classify and print, run nothing")
    args = parser.parse_args()

    examples = discover()
    if not examples:
        print("::error::no examples discovered; the glob or the tree has moved", flush=True)
        return 1

    labels = classify(examples)
    selected = [p for p in examples if args.lane == "all" or labels[p] == args.lane]

    if args.list:
        for path in examples:
            print(f"{labels[path]:8} {path.relative_to(REPO_ROOT)}")
        print(f"\n{sum(v == 'offline' for v in labels.values())} offline, "
              f"{sum(v == 'network' for v in labels.values())} network, {len(examples)} total")
        return 0

    if not selected:
        # An empty lane is a classification bug, not a pass. The offline lane going quietly empty
        # would report success while running nothing at all.
        print(f"::error::lane '{args.lane}' selected no examples out of {len(examples)}", flush=True)
        return 1

    print(f"running {len(selected)} '{args.lane}' example(s) with {args.jobs} workers\n", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda p: run_one(p, args.timeout), selected))

    failed = []
    for path, ok, seconds, detail in sorted(results, key=lambda r: (r[1], str(r[0]))):
        rel = path.relative_to(REPO_ROOT)
        if ok:
            print(f"  PASS  {seconds:6.1f}s  {rel}", flush=True)
        else:
            failed.append((rel, detail))
            print(f"  FAIL  {seconds:6.1f}s  {rel}  ::  {detail}", flush=True)

    print(f"\n{len(results) - len(failed)}/{len(results)} passed "
          f"in {time.time() - started:.1f}s wall clock")

    for rel, detail in failed:
        # A workflow annotation, so the failure is visible without opening the log.
        print(f"::error file={rel}::{detail}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
