"""
enforce that the three release version locations agree.

A release touches `pyproject.toml`, `CITATION.cff` and the software BibTeX entry
in `README.md`. They have drifted before — at 6.4.0 the three read 6.3.0, 6.2.0
and versionless respectively — so the agreement is a test rather than a
checklist line.

The test is skipped when the repository root is not on disk, which is the case
for an installed wheel: the three files are packaging metadata, not package
data, so there is nothing to compare against.
"""
# packages
import re
from pathlib import Path
from typing import Optional

import pytest
import yaml


def _repo_root() -> Optional[Path]:
    """return the first ancestor holding pyproject.toml, or None."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file():
            return parent
    return None


ROOT = _repo_root()
pytestmark = pytest.mark.skipif(ROOT is None,
                                reason='repository root not on disk (installed wheel)')


def _pyproject_version() -> str:
    """read [project] version from pyproject.toml without a TOML parser on 3.10."""
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']',
                      text.split('[project]', 1)[-1], flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"no [project] version in {ROOT / 'pyproject.toml'}")
    return match.group(1)


def _citation_version() -> str:
    """read the top-level software version from CITATION.cff."""
    data = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))
    version = data.get('version')
    if version is None:
        raise ValueError(f"no version key in {ROOT / 'CITATION.cff'}")
    return str(version)


def _readme_bibtex_version() -> str:
    """read version={...} from the @software entry in README.md."""
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    entry = re.search(r'@software\{.*?\n\}', text, flags=re.DOTALL)
    if entry is None:
        raise ValueError(f"no @software BibTeX entry in {ROOT / 'README.md'}")
    match = re.search(r'version\s*=\s*\{([^}]+)\}', entry.group(0))
    if match is None:
        raise ValueError(f"@software entry in {ROOT / 'README.md'} carries no version field")
    return match.group(1).strip()


def test_citation_cff_matches_pyproject():
    """``CITATION.cff`` and ``pyproject.toml`` declare the same version."""
    assert _citation_version() == _pyproject_version()


def test_readme_bibtex_matches_pyproject():
    """The README BibTeX entry and ``pyproject.toml`` declare the same version."""
    assert _readme_bibtex_version() == _pyproject_version()


def test_citation_cff_date_released_is_iso():
    """date-released must be an ISO date; a bare year silently sorts wrong on Zenodo."""
    data = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(data['date-released'])), \
        f"date-released must be YYYY-MM-DD, got {data['date-released']!r}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
