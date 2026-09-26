"""Prove that the Read the Docs status checker fails on the build states it must report."""

from __future__ import annotations

import contextlib
import io
import unittest
import urllib.error
from unittest import mock

import check_rtd_build
from check_rtd_build import BuildStatusError, check_build, latest_finished_build, main


def _build(build_id: int, created: str, *, code: str = "finished", success: bool | None = True,
           error: str = "") -> dict:
    """Return one build record shaped like the API v3 listing."""
    return {
        "id": build_id, "version": "latest", "commit": "5e0624cfa1ded8c6cedc32fcc878defec090d502",
        "created": created, "state": {"code": code}, "success": success, "error": error,
    }


FAILED = _build(34776275, "2026-09-26T16:21:40Z", success=False)
SUCCEEDED = _build(34545846, "2026-09-14T09:34:00Z")


class LatestFinishedBuildTest(unittest.TestCase):
    """Select the build that decides the status."""

    def test_newest_finished_build_wins_over_listing_order(self) -> None:
        """An older success listed first does not hide a newer failure."""
        payload = {"results": [SUCCEEDED, FAILED]}
        self.assertEqual(latest_finished_build(payload)["id"], FAILED["id"])

    def test_builds_in_progress_are_skipped(self) -> None:
        """A build still installing is not judged; the previous finished one is."""
        running = _build(34800000, "2026-09-27T06:00:00Z", code="installing", success=None)
        payload = {"results": [running, SUCCEEDED]}
        self.assertEqual(latest_finished_build(payload)["id"], SUCCEEDED["id"])

    def test_listing_without_finished_build_fails(self) -> None:
        """No finished build is a failure to report, not a pass."""
        running = _build(34800000, "2026-09-27T06:00:00Z", code="building", success=None)
        for payload in ({"results": []}, {"results": [running]}):
            with self.subTest(payload=payload), self.assertRaises(BuildStatusError):
                latest_finished_build(payload)

    def test_malformed_listing_fails(self) -> None:
        """A response without a results list cannot pass."""
        with self.assertRaises(BuildStatusError):
            latest_finished_build({"detail": "Not found."})


class CheckBuildTest(unittest.TestCase):
    """Turn the deciding build into a pass or a failure."""

    def test_success_is_reported(self) -> None:
        """A successful build returns a summary naming the build and commit."""
        summary = check_build({"results": [SUCCEEDED]}, "optimalportfolios")
        self.assertTrue(summary.startswith("OK: build 34545846"))
        self.assertIn("5e0624c", summary)

    def test_failure_links_the_build_log(self) -> None:
        """A failed build raises with a link to its log."""
        with self.assertRaises(BuildStatusError) as caught:
            check_build({"results": [FAILED]}, "optimalportfolios")
        self.assertIn("projects/optimalportfolios/builds/34776275/", str(caught.exception))

    def test_unknown_success_is_a_failure(self) -> None:
        """Only an explicit success passes; a missing flag does not."""
        with self.assertRaises(BuildStatusError):
            check_build({"results": [_build(1, "2026-09-26T00:00:00Z", success=None)]}, "p")


class MainTest(unittest.TestCase):
    """Map outcomes to exit statuses without touching the network."""

    def _run(self, side_effect) -> int:
        """Run main with a patched fetch and return its exit status."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(check_rtd_build, "fetch_builds", side_effect=side_effect)
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            return main(["--version", "latest"])

    def test_exit_status(self) -> None:
        """Success exits 0, a failed build 1, and an unreadable API 2."""
        self.assertEqual(self._run([{"results": [SUCCEEDED]}]), 0)
        self.assertEqual(self._run([{"results": [FAILED]}]), 1)
        self.assertEqual(self._run(urllib.error.URLError("offline")), 2)


if __name__ == "__main__":
    unittest.main()
