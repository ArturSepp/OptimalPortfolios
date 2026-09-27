"""Sphinx configuration for the OptimalPortfolios documentation.

As in the qis and FactorLasso documentation, the API reference is generated at build time rather
than maintained by hand. ``api.rst`` includes ``_generated/api_reference.rst``, which this file
writes from the package and the ownership recorded in ``tools/docs_inventory.json``:

1. one section per owning page, in the order of the sidebar, each with an autosummary table
   whose stub pages keep their published ``generated/optimalportfolios.<name>.html`` URLs;
2. the names re-exported from FactorLasso and qis, linked to the pages that explain them;
3. one table per mapped configuration dataclass, giving each field's default and owning page.

Autosummary finds stub entries by scanning source files and does not follow ``include``, so the
stubs of the generated body are written here, before Sphinx reads the sources. The generated
files are git-ignored, so the reference cannot drift from the package.
"""

import dataclasses
import enum
import inspect
import json
import os
import sys
from pathlib import Path

import tomllib

DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

project = "optimalportfolios"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
release = tomllib.loads(
    (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "optimalportfolios_indexing",
    "optimalportfolios_callouts",
]

# Dollar math keeps article source portable across MyST, GitHub and VS Code.
myst_enable_extensions = ["dollarmath"]
# Resolve ordinary Markdown section links with GitHub-compatible heading fragments.
myst_heading_anchors = 4
# A ```mermaid fence renders as a diagram in Sphinx and natively on GitHub.
myst_fence_as_directive = ["mermaid"]
# Keep each diagram's own aspect ratio; the extension's fixed 500px box shrinks tall diagrams.
mermaid_height = "auto"

# Stubs of the generated API body are written by _generate_api; autosummary still scans the
# ordinary sources, which contain no autosummary directive of their own.
autosummary_generate = True
autosummary_imported_members = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]

# SSRN returns HTTP 403 to automated link-check clients even when the public pages are live.
# artursepp.com answers HTTP 429 to the GitHub runner on a single request, for the same reason.
linkcheck_ignore = [
    r"https://(?:www\.)?ssrn\.com/.*",
    r"https://papers\.ssrn\.com/.*",
    r"https://(?:www\.)?artursepp\.com/.*",
]

GOOGLE_SITE_VERIFICATION = "cddUZk3Gsd1MySw42Rwuq_rMzUDcMNkJWekObx-QS9Y"

html_theme = "furo"
html_baseurl = (
    os.environ.get("READTHEDOCS_CANONICAL_URL")
    or "https://optimalportfolios.readthedocs.io/en/latest/"
)
html_title = "optimalportfolios - portfolio construction and rolling backtesting"
html_short_title = "optimalportfolios"
html_static_path = ["_static"]
html_css_files = ["optimalportfolios.css"]

# Packages whose versions the footer names, with their usual spelling.
BUILD_PACKAGES = {
    "qis": "qis",
    "factorlasso": "factorlasso",
    "cvxpy": "CVXPY",
    "numpy": "NumPy",
    "pandas": "pandas",
    "scipy": "SciPy",
}


def _build_versions() -> str:
    """Name the package versions of this build for the footer of every page.

    Pages state no hand-written version stamps, which went stale between releases; the build
    reads the versions of its own environment instead, which on Read the Docs is ``uv.lock``.
    The package itself is named by the source version in ``pyproject.toml``, because the pages
    document the checkout being built. A package missing from the environment is left out.
    """
    from importlib.metadata import PackageNotFoundError, version as installed_version

    names = []
    for distribution, label in BUILD_PACKAGES.items():
        try:
            names.append(f"{label} {installed_version(distribution)}")
        except PackageNotFoundError:
            continue
    built_with = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)
    return f"Built from optimalportfolios {release}" + (f" with {built_with}." if names else ".")


# The verification tag is emitted by _templates/base.html on every page, Markdown included;
# _templates/page.html adds the build versions to the footer.
html_context = {
    "google_site_verification": GOOGLE_SITE_VERIFICATION,
    "build_versions": _build_versions(),
}
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/OptimalPortfolios/",
    "source_branch": "main",
    "source_directory": "docs/",
}

GENERATED_DIR = DOCS_DIR / "_generated"
STUB_DIR = DOCS_DIR / "generated"
PACKAGE_MODULES = (
    "alphas", "config", "covar_estimation", "local_path", "optimization", "reports", "universe",
    "utils",
)
API_HEADER = """..
   Generated by docs/conf.py from the package and tools/docs_inventory.json.
   Do not edit by hand; the file is git-ignored.

"""
SELF_REFERENCE_STUB = """:orphan:

optimalportfolios.optimalportfolios
===================================

.. py:module:: optimalportfolios.optimalportfolios

Importing ``optimalportfolios.local_path`` binds the package name inside the package's own
namespace, so ``optimalportfolios.optimalportfolios`` is the package itself rather than a
separate module. Its public objects are listed in the :doc:`API reference </api>`.
"""


def _inventory() -> dict:
    """Read the documentation inventory that records page ownership."""
    return json.loads(
        (REPOSITORY_ROOT / "tools" / "docs_inventory.json").read_text(encoding="utf-8")
    )


def _page_title(relative: str, planned: dict) -> tuple[str, bool]:
    """Return the H1 title of an inventoried page, and whether the page exists.

    Args:
        relative: Repository-relative page path, such as ``docs/constraints.md``.
        planned: Planned pages of the inventory, which supply titles of pages not yet written.
    """
    path = REPOSITORY_ROOT / relative
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip(), True
    return planned.get(relative, {}).get("title", relative), False


def _page_reference(relative: str, planned: dict) -> str:
    """Return an RST link to an existing page, or the plain title of a planned one."""
    title, exists = _page_title(relative, planned)
    # Inline code in a page title would end the role's text early.
    title = title.replace("`", "")
    if exists:
        docname = Path(relative).relative_to("docs").with_suffix("").as_posix()
        return f":doc:`{title} </{docname}>`"
    return f"*{title}* (planned article)"


def _heading(title: str, underline: str) -> str:
    """Return an RST section heading."""
    return f"{title}\n{underline * len(title)}\n\n"


def _autosummary(names) -> str:
    """Return an autosummary table whose stubs live in ``generated/``."""
    entries = "".join(f"   optimalportfolios.{name}\n" for name in names)
    return f".. autosummary::\n   :toctree: generated\n\n{entries}\n"


def _format_default(field: dataclasses.Field) -> str:
    """Render a dataclass field default the way a reader would type it."""
    if field.default is not dataclasses.MISSING:
        value = field.default
        if isinstance(value, enum.Enum):
            return f"``{type(value).__name__}.{value.name}``"
        if inspect.isclass(value):
            return f"``{value.__name__}``"
        return f"``{value!r}``"
    if field.default_factory is not dataclasses.MISSING:
        return f"``{getattr(field.default_factory, '__name__', 'factory')}()``"
    return "required"


def _write_api_reference(target: Path = GENERATED_DIR) -> Path:
    """Write ``api_reference.rst`` from the package and the inventory ownership.

    Args:
        target: Output directory; tests pass a temporary directory.

    Returns:
        The written file.
    """
    import optimalportfolios

    inventory = _inventory()
    planned = inventory.get("planned", {})
    lines = [API_HEADER]
    lines.append(_heading("Objects by owning page", "~"))
    lines.append(
        "Each public object is explained on exactly one page. The sections follow the order of "
        "the sidebar; a page not yet written is marked as planned.\n\n"
    )
    for page, names in inventory["symbols"].items():
        title, _ = _page_title(page, planned)
        lines.append(_heading(title, "^"))
        lines.append(f"Explained in {_page_reference(page, planned)}.\n\n")
        lines.append(_autosummary(names))

    lines.append(_heading("Re-exported from FactorLasso and qis", "~"))
    lines.append(
        "These names are importable from ``optimalportfolios`` for compatibility. Their "
        "methodology and reference documentation belongs to the package that defines them.\n\n"
        ".. list-table::\n   :header-rows: 1\n   :widths: 34 16 50\n\n"
        "   * - Name\n     - Package\n     - Owning documentation\n"
    )
    for name, entry in inventory["external_symbols"].items():
        lines.append(
            f"   * - ``{name}``\n     - {entry['package']}\n"
            f"     - `Reference <{entry['url']}>`__, `article <{entry['article']}>`__\n"
        )
    lines.append("\n")
    lines.append(_autosummary(inventory["external_symbols"]))

    lines.append(_heading("Configuration fields by owning page", "~"))
    lines.append(
        "Each configuration field of these dataclasses is explained on exactly one page.\n\n"
    )
    for class_name, owners in inventory["parameters"].items():
        fields = {field.name: field
                  for field in dataclasses.fields(getattr(optimalportfolios, class_name))}
        owner_of = {name: page for page, names in owners.items() for name in names}
        lines.append(_heading(f"{class_name} fields", "^"))
        lines.append(
            f".. list-table:: Fields of :class:`~optimalportfolios.{class_name}`\n"
            "   :header-rows: 1\n   :widths: 34 26 40\n\n"
            "   * - Field\n     - Default\n     - Explained in\n"
        )
        for name, field in fields.items():
            lines.append(
                f"   * - ``{name}``\n     - {_format_default(field)}\n"
                f"     - {_page_reference(owner_of[name], planned)}\n"
            )
        lines.append("\n")

    lines.append(_heading("Subpackages", "~"))
    lines.append(
        "The objects above are re-exported at the package root; these are the subpackages that "
        "define them.\n\n"
    )
    lines.append(_autosummary(PACKAGE_MODULES))

    target.mkdir(parents=True, exist_ok=True)
    output = target / "api_reference.rst"
    output.write_text("".join(lines), encoding="utf-8")
    return output


def _generate_api(app) -> None:
    """Write the API body and its autosummary stubs before Sphinx reads the sources."""
    from sphinx.ext.autosummary.generate import generate_autosummary_docs

    body = _write_api_reference()
    generate_autosummary_docs(
        [str(body)], output_dir=STUB_DIR, suffix=".rst", base_path=app.srcdir,
        imported_members=True, app=app, overwrite=True,
    )
    # Keeps the published URL of the package's self-reference, which is not a subpackage.
    (STUB_DIR / "optimalportfolios.optimalportfolios.rst").write_text(
        SELF_REFERENCE_STUB, encoding="utf-8"
    )


def setup(app) -> None:
    """Generate the API reference once the builder, and so the template loader, exists."""
    app.connect("builder-inited", _generate_api)
