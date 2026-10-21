"""Keep private dependency diagnostics on the ephemeral runner, never in public output.

This is a log boundary for trusted verification commands, not a sandbox. Full
debugging remains in the FactorModels-hosted private verification workflow.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


COMMAND_FILES = ("GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH", "GITHUB_STEP_SUMMARY")
LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}")


def private_directory():
    """Require an absolute runner temporary directory outside the checkout."""
    value = os.environ.get("RUNNER_TEMP", "")
    if not value or not Path(value).is_absolute():
        raise ValueError("Runner temporary directory required")
    root = Path(value).resolve()
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace and root.is_relative_to(Path(workspace).resolve()):
        raise ValueError("Runner temporary directory must be outside the checkout")
    target = root / "private-factor-checks"
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    return target


def check_configuration():
    """Fail closed unless both a reviewed full SHA and explicit read access exist."""
    valid = re.fullmatch(r"[0-9a-fA-F]{40}", os.environ.get("FM_SOURCE_SHA", ""))
    if not valid or os.environ.get("FM_READ_CREDENTIAL_PRESENT") != "true":
        print("Private dependency configuration FAILED: reviewed SHA and read credential required.")
        return 1
    print("Private dependency configuration PASS.")
    return 0


def run_private(label, command, input_bytes=None):
    """Capture both output streams and command files while preserving failure."""
    if not LABEL.fullmatch(label) or not command:
        raise ValueError("Invalid private check")
    directory = private_directory()
    environment = os.environ.copy()
    for key in COMMAND_FILES:
        environment[key] = str(directory / f"{label}.{key.lower()}")
    for key in ("FM_READ_TOKEN", "TRADABLEFACTORS_READ_TOKEN", "FACTORMODELS_READ_SSH_KEY", "BASH_ENV", "ENV"):
        environment.pop(key, None)
    with (directory / f"{label}.log").open("wb") as output:
        try:
            result = subprocess.run(
                command, input=input_bytes, stdout=output, stderr=subprocess.STDOUT,
                env=environment, check=False,
            )
            code = result.returncode
            if code < 0:
                code = 128 - code
        except OSError:
            # Exception text can include a command or private filesystem content.
            output.write(b"Unable to start the verification command.\n")
            code = 1
    record = {"check": label, "result": "passed" if code == 0 else "failed", "exit_code": code}
    (directory / f"{label}.status.json").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"Private dependency check {label}: {record['result'].upper()} (exit {code}).")
    if code:
        print("Full diagnostics remain runner-local; use the private-host verification for debugging.")
    return code


def main(argv=None):
    """Dispatch fixed configuration validation or a captured verification command."""
    args = sys.argv[1:] if argv is None else argv
    try:
        if args == ["check-config"]:
            return check_configuration()
        if len(args) >= 4 and args[0] == "run" and args[2] == "--":
            return run_private(args[1], args[3:])
        if len(args) == 2 and args[0] == "run-script":
            return run_private(
                args[1], ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail"],
                sys.stdin.buffer.read(),
            )
    except Exception:
        print("Private dependency check FAILED: runner setup or invocation error.")
        return 1
    print("Private dependency check FAILED: invalid invocation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
