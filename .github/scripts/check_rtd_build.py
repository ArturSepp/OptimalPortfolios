#!/usr/bin/env python3
"""Fail when the most recent finished Read the Docs build of a version did not succeed.

Read the Docs builds from its own configuration (``.readthedocs.yaml``), not from the
environment that ``docs.yml`` checks on every pull request, so a hosting-only failure leaves
CI green. Every build from 2026-09-14 to 2026-09-26 failed that way, and the site kept serving
the last good build. The daily ``link-health.yml`` run calls this script to make such a failure
visible. It uses the public, unauthenticated Read the Docs API v3 and the standard library only.

Exit status: 0 when the newest finished build of every requested version succeeded, 1 when one
failed or none finished, and 2 when the API could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


API_URL = (
    "https://app.readthedocs.org/api/v3/projects/{project}/versions/{version}/builds/"
    "?limit={limit}"
)
BUILD_PAGE = "https://app.readthedocs.org/projects/{project}/builds/{build_id}/"


class BuildStatusError(RuntimeError):
    """Report a failed build, or a listing with no finished build to judge."""


def latest_finished_build(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the newest build whose state is ``finished`` in an API v3 build listing.

    Builds still queued, cloning, installing or building are skipped, so a run that starts
    while a build is in progress judges the previous finished build.
    """
    builds = payload.get("results")
    if not isinstance(builds, list):
        raise BuildStatusError("The build listing has no 'results' list.")
    finished = [
        build for build in builds
        if isinstance(build, dict) and (build.get("state") or {}).get("code") == "finished"
    ]
    if not finished:
        raise BuildStatusError(f"No finished build among the {len(builds)} listed.")
    return max(finished, key=lambda build: str(build.get("created") or ""))


def check_build(payload: dict[str, Any], project: str) -> str:
    """Return a summary of the newest finished build, or raise if it did not succeed."""
    build = latest_finished_build(payload)
    summary = (
        f"build {build.get('id')} of {build.get('version')!r} at commit "
        f"{str(build.get('commit') or '')[:7]}, created {build.get('created')}"
    )
    if build.get("success") is not True:
        error = str(build.get("error") or "").strip() or "no error message; see the log"
        log = BUILD_PAGE.format(project=project, build_id=build.get("id"))
        raise BuildStatusError(f"FAIL: {summary}: {error}. Log: {log}")
    return f"OK: {summary}"


def fetch_builds(project: str, version: str, limit: int, timeout: float) -> dict[str, Any]:
    """Download the most recent builds of one version from the public API."""
    url = API_URL.format(project=project, version=version, limit=limit)
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "optimalportfolios-ci"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    """Check each requested version and return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default="optimalportfolios")
    parser.add_argument(
        "--version", action="append", dest="versions",
        help="Read the Docs version slug; repeat for several (default: latest)",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    status = 0
    for version in args.versions or ["latest"]:
        try:
            payload = fetch_builds(args.project, version, args.limit, args.timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"ERROR: could not read the {version!r} builds: {exc}", file=sys.stderr)
            status = max(status, 2)
            continue
        try:
            print(check_build(payload, args.project))
        except BuildStatusError as exc:
            print(str(exc), file=sys.stderr)
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
