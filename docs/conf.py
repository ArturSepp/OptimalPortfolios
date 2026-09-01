"""Sphinx configuration for the OptimalPortfolios documentation."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "optimalportfolios"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autosummary_imported_members = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# SSRN returns HTTP 403 to automated link-check clients even when the public pages are live.
linkcheck_ignore = [
    r"https://(?:www\.)?ssrn\.com/.*",
    r"https://papers\.ssrn\.com/.*",
]

html_theme = "furo"
html_baseurl = "https://optimalportfolios.readthedocs.io/en/latest/"
html_title = "optimalportfolios - portfolio construction and rolling backtesting"
html_short_title = "optimalportfolios"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/OptimalPortfolios/",
    "source_branch": "main",
    "source_directory": "docs/",
}
