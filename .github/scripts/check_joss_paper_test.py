"""Prove that the JOSS paper checker rejects mandated defect classes."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from check_joss_paper import MAX_WORDS, PaperCheckError, check_paper


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper.md"
BIBLIOGRAPHY = ROOT / "paper.bib"


class JossPaperCheckTest(unittest.TestCase):
    """Exercise the valid paper and isolated mutations of temporary copies."""

    @classmethod
    def setUpClass(cls) -> None:
        """Read the approved manuscript once for temporary-copy mutations."""
        cls.paper_text = PAPER.read_text(encoding="utf-8")
        cls.bibliography_text = BIBLIOGRAPHY.read_text(encoding="utf-8")

    def _check_mutation(
        self,
        *,
        paper_text: str | None = None,
        bibliography_text: str | None = None,
    ) -> None:
        """Run the checker against one temporary manuscript/bibliography pair."""
        with tempfile.TemporaryDirectory(prefix="optimalportfolios-joss-") as directory:
            root = Path(directory)
            paper_path = root / "paper.md"
            bibliography_path = root / "paper.bib"
            paper_path.write_text(paper_text or self.paper_text, encoding="utf-8")
            bibliography_path.write_text(
                bibliography_text or self.bibliography_text, encoding="utf-8"
            )
            check_paper(paper_path, bibliography_path)

    def test_approved_paper_passes(self) -> None:
        """Accept the author-approved D5 manuscript."""
        result = check_paper(PAPER, BIBLIOGRAPHY)
        self.assertLessEqual(result.word_count, MAX_WORDS)
        self.assertEqual(result.citation_count, result.bibliography_count)
        self.assertEqual(result.allowed_placeholders, ("QIS_JOSS_DOI_PENDING",))

    def test_missing_required_heading_fails(self) -> None:
        """Reject removal of any required substantive heading."""
        mutated = self.paper_text.replace("# Software design\n", "", 1)
        with self.assertRaisesRegex(PaperCheckError, "required headings"):
            self._check_mutation(paper_text=mutated)

    def test_unresolved_citation_fails(self) -> None:
        """Reject a citation key absent from the bibliography."""
        mutated = self.paper_text.replace(
            "\n# References", "\nAn unresolved citation [@missing_reference].\n\n# References", 1
        )
        with self.assertRaisesRegex(PaperCheckError, "citation key.*missing from bibliography"):
            self._check_mutation(paper_text=mutated)

    def test_word_count_above_official_range_fails(self) -> None:
        """Reject a manuscript outside the official 750-1,750 word range."""
        padding = " validation" * (MAX_WORDS + 1)
        mutated = self.paper_text.replace("\n# References", f"{padding}\n\n# References", 1)
        with self.assertRaisesRegex(PaperCheckError, "outside JOSS range"):
            self._check_mutation(paper_text=mutated)

    def test_duplicate_bibliography_key_fails(self) -> None:
        """Reject duplicate BibTeX keys even when citations still resolve."""
        duplicate = "\n@software{markowitz1952,\n  title = {Duplicate key}\n}\n"
        with self.assertRaisesRegex(PaperCheckError, "duplicate bibliography key"):
            self._check_mutation(bibliography_text=self.bibliography_text + duplicate)

    def test_missing_metadata_fails(self) -> None:
        """Reject removal of a required JOSS metadata field."""
        mutated = re.sub(r"^title:.*\n", "", self.paper_text, count=1, flags=re.MULTILINE)
        with self.assertRaisesRegex(PaperCheckError, "metadata is missing required field.*title"):
            self._check_mutation(paper_text=mutated)


if __name__ == "__main__":
    unittest.main()
