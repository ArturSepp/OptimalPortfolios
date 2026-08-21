#!/usr/bin/env python3
"""Validate the repository's JOSS paper without third-party dependencies."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MIN_WORDS = 750
MAX_WORDS = 1750
MIN_SECTION_WORDS = 25

REQUIRED_HEADINGS = (
    "Summary",
    "Statement of need",
    "State of the field",
    "Software design",
    "Research impact statement",
    "AI usage disclosure",
    "Acknowledgements",
    "References",
)
REQUIRED_METADATA = ("title", "tags", "authors", "affiliations", "date", "bibliography")
ALLOWED_PLACEHOLDERS = {"QIS_JOSS_DOI_PENDING"}

TOP_LEVEL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?:\s*(.*))?$")
MAPPING_ITEM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?:\s*(.*))?$")
CITATION_RE = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_.:+-]*)")
BIB_ENTRY_RE = re.compile(r"^@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
PLACEHOLDER_TOKEN_RE = re.compile(
    r"\b(?:TODO|TBD|FIXME|XXX|[A-Z][A-Z0-9_]*(?:PENDING|PLACEHOLDER)[A-Z0-9_]*)\b"
)
BRACKET_PLACEHOLDER_RE = re.compile(
    r"\[(?:AUTHOR|AFFILIATION|ORCID|EMAIL|DATE|DOI)(?:[^\]]*)\]", re.IGNORECASE
)


class PaperCheckError(ValueError):
    """Report one or more manuscript validation failures."""


@dataclass(frozen=True)
class CheckResult:
    """Summarize a successful paper validation."""

    word_count: int
    citation_count: int
    bibliography_count: int
    allowed_placeholders: tuple[str, ...]


def _parse_scalar(value: str) -> Any:
    """Parse the small YAML scalar subset used by JOSS front matter."""
    value = value.strip()
    if not value:
        return ""
    if value[0] in "'\"" and value[-1] == value[0]:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise PaperCheckError(f"invalid quoted YAML scalar: {value}") from exc
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(item) for item in inner.split(",")]
    return value


def _parse_sequence(lines: list[str], field: str) -> list[Any]:
    """Parse a YAML sequence of scalars or flat mappings."""
    items: list[Any] = []
    current: dict[str, Any] | None = None
    item_kind: str | None = None

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise PaperCheckError(f"metadata field '{field}' uses a tab for indentation")

        if raw_line.startswith("  - "):
            if current is not None:
                items.append(current)
                current = None
            content = raw_line[4:].strip()
            mapping_match = MAPPING_ITEM_RE.fullmatch(content)
            if mapping_match:
                if item_kind == "scalar":
                    raise PaperCheckError(f"metadata field '{field}' mixes scalar and map items")
                item_kind = "mapping"
                current = {
                    mapping_match.group(1): _parse_scalar(mapping_match.group(2) or "")
                }
            else:
                if item_kind == "mapping":
                    raise PaperCheckError(f"metadata field '{field}' mixes scalar and map items")
                item_kind = "scalar"
                items.append(_parse_scalar(content))
            continue

        if raw_line.startswith("    ") and current is not None:
            mapping_match = MAPPING_ITEM_RE.fullmatch(raw_line[4:].strip())
            if not mapping_match:
                raise PaperCheckError(f"invalid mapping line in metadata field '{field}'")
            key = mapping_match.group(1)
            if key in current:
                raise PaperCheckError(f"duplicate metadata key '{key}' in field '{field}'")
            current[key] = _parse_scalar(mapping_match.group(2) or "")
            continue

        raise PaperCheckError(f"unsupported YAML structure in metadata field '{field}'")

    if current is not None:
        items.append(current)
    if not items:
        raise PaperCheckError(f"metadata field '{field}' must not be empty")
    return items


def _split_document(document: str) -> tuple[list[str], str]:
    """Split YAML front matter from the Markdown body."""
    lines = document.splitlines()
    if not lines or lines[0].strip() != "---":
        raise PaperCheckError("paper must start with YAML front matter delimited by '---'")
    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise PaperCheckError("paper has no closing YAML front-matter delimiter") from exc
    return lines[1:closing_index], "\n".join(lines[closing_index + 1 :])


def _parse_front_matter(lines: list[str]) -> dict[str, Any]:
    """Parse the constrained front-matter structure used by the manuscript."""
    metadata: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        if line.startswith((" ", "\t")):
            raise PaperCheckError(f"unexpected indentation in metadata line: {line}")
        match = TOP_LEVEL_RE.fullmatch(line)
        if not match:
            raise PaperCheckError(f"invalid top-level metadata line: {line}")
        key, scalar = match.group(1), match.group(2) or ""
        if key in metadata:
            raise PaperCheckError(f"duplicate top-level metadata field '{key}'")
        if scalar:
            metadata[key] = _parse_scalar(scalar)
            index += 1
            continue

        block_start = index + 1
        block_end = block_start
        while block_end < len(lines):
            candidate = lines[block_end]
            if candidate.strip() and not candidate.startswith((" ", "\t")):
                break
            block_end += 1
        metadata[key] = _parse_sequence(lines[block_start:block_end], key)
        index = block_end
    return metadata


def _as_affiliation_indices(value: Any, author_name: str) -> list[int]:
    """Normalize one author's affiliation field to integer indices."""
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, int) for item in values):
        raise PaperCheckError(
            f"author '{author_name}' must reference one or more integer affiliation indices"
        )
    return values


def _validate_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return all semantic errors in the parsed JOSS metadata."""
    errors: list[str] = []
    missing = [field for field in REQUIRED_METADATA if field not in metadata]
    if missing:
        errors.append(f"metadata is missing required field(s): {', '.join(missing)}")

    if not isinstance(metadata.get("title"), str) or not metadata.get("title", "").strip():
        errors.append("metadata title must be a non-empty string")

    tags = metadata.get("tags")
    if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) for tag in tags):
        errors.append("metadata tags must be a non-empty list of strings")

    date_value = metadata.get("date")
    if not isinstance(date_value, str):
        errors.append("metadata date must use the display form 'D Month YYYY'")
    else:
        try:
            parsed_date = datetime.strptime(date_value, "%d %B %Y")
            if parsed_date.strftime("%d %B %Y").lstrip("0") != date_value:
                errors.append("metadata date must use the display form 'D Month YYYY'")
        except ValueError:
            errors.append("metadata date must be a valid date in display form 'D Month YYYY'")

    bibliography = metadata.get("bibliography")
    if not isinstance(bibliography, str) or not bibliography.endswith(".bib"):
        errors.append("metadata bibliography must name a .bib file")

    affiliations = metadata.get("affiliations")
    affiliation_indices: set[int] = set()
    if not isinstance(affiliations, list) or not affiliations:
        errors.append("metadata affiliations must be a non-empty list")
    else:
        for position, affiliation in enumerate(affiliations, start=1):
            if not isinstance(affiliation, dict):
                errors.append(f"affiliation {position} must be a mapping")
                continue
            if not isinstance(affiliation.get("name"), str) or not affiliation["name"].strip():
                errors.append(f"affiliation {position} must have a non-empty name")
            affiliation_index = affiliation.get("index")
            if not isinstance(affiliation_index, int):
                errors.append(f"affiliation {position} must have an integer index")
            elif affiliation_index in affiliation_indices:
                errors.append(f"duplicate affiliation index {affiliation_index}")
            else:
                affiliation_indices.add(affiliation_index)

    authors = metadata.get("authors")
    corresponding_count = 0
    if not isinstance(authors, list) or not authors:
        errors.append("metadata authors must be a non-empty list")
    else:
        for position, author in enumerate(authors, start=1):
            if not isinstance(author, dict):
                errors.append(f"author {position} must be a mapping")
                continue
            name = author.get("name")
            display_name = name if isinstance(name, str) and name.strip() else str(position)
            if display_name == str(position):
                errors.append(f"author {position} must have a non-empty name")
            orcid = author.get("orcid")
            if not isinstance(orcid, str) or not ORCID_RE.fullmatch(orcid):
                errors.append(f"author '{display_name}' must have a valid ORCID")
            email = author.get("email")
            if not isinstance(email, str) or not re.fullmatch(r"[^\s@]+@[^\s@]+", email):
                errors.append(f"author '{display_name}' must have a valid email")
            if author.get("corresponding") is True:
                corresponding_count += 1
            try:
                author_indices = _as_affiliation_indices(author.get("affiliation"), display_name)
            except PaperCheckError as exc:
                errors.append(str(exc))
            else:
                unknown = sorted(set(author_indices) - affiliation_indices)
                if unknown:
                    errors.append(
                        f"author '{display_name}' references unknown affiliation(s): {unknown}"
                    )
    if corresponding_count != 1:
        errors.append("metadata must identify exactly one corresponding author")
    return errors


def _count_words(markdown: str) -> int:
    """Count manuscript prose while excluding headings, citations, code, and references."""
    without_references = markdown.split("\n# References", maxsplit=1)[0]
    text = re.sub(r"```.*?```", " ", without_references, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\[@[^\]]+\]", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", " ", text)
    return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text, flags=re.UNICODE))


def _validate_sections(body: str) -> tuple[list[str], int]:
    """Validate required section order, substance, and official JOSS word range."""
    errors: list[str] = []
    headings = tuple(re.findall(r"^# (.+)$", body, flags=re.MULTILINE))
    if headings != REQUIRED_HEADINGS:
        errors.append(
            "required headings must appear exactly once and in this order: "
            + "; ".join(REQUIRED_HEADINGS)
        )
    for heading in REQUIRED_HEADINGS[:-1]:
        match = re.search(
            rf"^# {re.escape(heading)}\s*$\n(?P<section>.*?)(?=^# |\Z)",
            body,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match and _count_words(match.group("section")) < MIN_SECTION_WORDS:
            errors.append(
                f"section '{heading}' must contain at least {MIN_SECTION_WORDS} words"
            )
    word_count = _count_words(body)
    if not MIN_WORDS <= word_count <= MAX_WORDS:
        errors.append(
            f"manuscript word count {word_count} is outside JOSS range "
            f"{MIN_WORDS}-{MAX_WORDS}"
        )
    return errors, word_count


def _validate_placeholders(document: str, bibliography: str) -> tuple[list[str], tuple[str, ...]]:
    """Reject unresolved placeholders except the explicit pre-submission QIS DOI gate."""
    errors: list[str] = []
    combined = f"{document}\n{bibliography}"
    found = Counter(PLACEHOLDER_TOKEN_RE.findall(combined))
    unresolved = sorted(token for token in found if token not in ALLOWED_PLACEHOLDERS)
    if unresolved:
        errors.append(f"unresolved placeholder token(s): {', '.join(unresolved)}")
    bracketed = sorted(set(BRACKET_PLACEHOLDER_RE.findall(combined)))
    if bracketed:
        errors.append(f"unresolved bracket placeholder(s): {', '.join(bracketed)}")
    repeated_allowed = sorted(
        token for token in ALLOWED_PLACEHOLDERS if found.get(token, 0) > 1
    )
    if repeated_allowed:
        errors.append(
            "allowed pre-submission placeholder must appear at most once: "
            + ", ".join(repeated_allowed)
        )
    allowed_found = tuple(sorted(token for token in ALLOWED_PLACEHOLDERS if found.get(token, 0)))
    return errors, allowed_found


def _validate_citations(body: str, bibliography: str) -> tuple[list[str], int, int]:
    """Validate citation resolution and uniqueness of bibliography keys."""
    errors: list[str] = []
    cited_keys = set(CITATION_RE.findall(body.split("\n# References", maxsplit=1)[0]))
    bibliography_keys = BIB_ENTRY_RE.findall(bibliography)
    counts = Counter(bibliography_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate bibliography key(s): {', '.join(duplicates)}")
    available = set(bibliography_keys)
    missing = sorted(cited_keys - available)
    if missing:
        errors.append(f"citation key(s) missing from bibliography: {', '.join(missing)}")
    uncited = sorted(available - cited_keys)
    if uncited:
        errors.append(f"uncited bibliography key(s): {', '.join(uncited)}")
    if not cited_keys:
        errors.append("paper must contain at least one citation")
    if not bibliography_keys:
        errors.append("bibliography must contain at least one entry")
    return errors, len(cited_keys), len(available)


def check_paper(paper_path: Path, bibliography_path: Path | None = None) -> CheckResult:
    """Validate one JOSS Markdown manuscript and its BibTeX bibliography."""
    paper_path = paper_path.resolve()
    if not paper_path.is_file():
        raise PaperCheckError(f"paper does not exist: {paper_path}")
    document = paper_path.read_text(encoding="utf-8")
    front_matter, body = _split_document(document)
    metadata = _parse_front_matter(front_matter)

    metadata_errors = _validate_metadata(metadata)
    bibliography_name = metadata.get("bibliography")
    if bibliography_path is None:
        if not isinstance(bibliography_name, str):
            bibliography_path = paper_path.with_name("paper.bib")
        else:
            bibliography_path = paper_path.parent / bibliography_name
    bibliography_path = bibliography_path.resolve()
    if not bibliography_path.is_file():
        metadata_errors.append(f"bibliography does not exist: {bibliography_path}")
        bibliography = ""
    else:
        bibliography = bibliography_path.read_text(encoding="utf-8")

    section_errors, word_count = _validate_sections(body)
    citation_errors, citation_count, bibliography_count = _validate_citations(body, bibliography)
    placeholder_errors, allowed_placeholders = _validate_placeholders(document, bibliography)
    errors = metadata_errors + section_errors + citation_errors + placeholder_errors
    if errors:
        raise PaperCheckError("\n".join(f"- {error}" for error in errors))
    return CheckResult(
        word_count=word_count,
        citation_count=citation_count,
        bibliography_count=bibliography_count,
        allowed_placeholders=allowed_placeholders,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line paper check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", nargs="?", type=Path, default=Path("paper/paper.md"))
    parser.add_argument("--bibliography", type=Path, help="override the bibliography path")
    args = parser.parse_args(argv)
    try:
        result = check_paper(args.paper, args.bibliography)
    except PaperCheckError as exc:
        print(f"JOSS paper check failed:\n{exc}", file=sys.stderr)
        return 1

    placeholder_summary = ", ".join(result.allowed_placeholders) or "none"
    print("JOSS paper check passed")
    print(f"  words: {result.word_count} ({MIN_WORDS}-{MAX_WORDS})")
    print(
        f"  citations: {result.citation_count}; "
        f"bibliography entries: {result.bibliography_count}"
    )
    print(f"  permitted pre-submission placeholders present: {placeholder_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
