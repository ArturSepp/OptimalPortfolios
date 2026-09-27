"""Keep canonical HTML URLs and the public page sitemap consistent.

Adapted from ``QuantInvestStrats/docs/_ext/qis_indexing.py`` by way of
``FactorLasso/docs/_ext/factorlasso_indexing.py``. Copyright (c) Artur Sepp; the qis original is
distributed under the MIT License, as is this file as part of optimalportfolios.

Read the Docs serves the default version's sitemap at the domain root. Listing the actual pages
gives crawlers one preferred discovery path instead of only the version roots. The moving
``latest`` and ``stable`` aliases describe the same default documentation, so both use ``latest``
as their canonical URL, the version the site links to and the one that receives documentation
fixes between releases. Numbered releases keep their own canonical base URL, because their API
documentation can differ. qis consolidates onto ``stable`` instead; the choice is per package.
"""

from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
READTHEDOCS_ALIAS_PATHS = {"/en/latest", "/en/stable"}
CANONICAL_ALIAS_PATH = "/en/latest"
EXCLUDED_PAGES = {"search", "genindex", "py-modindex"}


def canonical_baseurl(baseurl: str) -> str:
    """Consolidate Read the Docs' moving aliases onto the ``latest`` URL.

    Args:
        baseurl: Configured ``html_baseurl``, with or without a trailing slash.

    Returns:
        Base URL with a trailing slash. Numbered release paths and deployments outside Read the
        Docs are returned unchanged apart from the slash.
    """
    parts = urlsplit(baseurl)
    path = parts.path.rstrip("/")
    if parts.netloc.endswith(".readthedocs.io") and path in READTHEDOCS_ALIAS_PATHS:
        path = CANONICAL_ALIAS_PATH
    return urlunsplit((parts.scheme, parts.netloc, path + "/", parts.query, parts.fragment))


def canonical_url(app: Any, pagename: str) -> str:
    """Return the same preferred URL for HTML metadata and sitemap entries.

    Args:
        app: Sphinx application with an HTML builder and a configured ``html_baseurl``.
        pagename: Sphinx document name, without the output suffix.

    Returns:
        Absolute URL, using the directory URL for the site's ``index.html``.
    """
    uri = app.builder.get_target_uri(pagename)
    if uri == "index.html":
        uri = ""
    return urljoin(canonical_baseurl(app.config.html_baseurl), quote(uri, safe="/%"))


def set_canonical_url(
    app: Any, pagename: str, templatename: str, context: dict, doctree: Any
) -> None:
    """Set the canonical URL consumed by the HTML theme, including the landing page.

    Args:
        app: Running Sphinx application.
        pagename: Current document name.
        templatename: HTML template selected by Sphinx; unchanged.
        context: Template variables; ``pageurl`` holds the canonical URL.
        doctree: Parsed document, or None for generated helper pages; unused.
    """
    if app.builder.name == "html" and app.config.html_baseurl:
        context["pageurl"] = canonical_url(app, pagename)


def write_sitemap(app: Any, exception: Optional[Exception]) -> None:
    """Write a deterministic sitemap only after a successful HTML build.

    Args:
        app: Sphinx application with the complete discovered document inventory.
        exception: Build failure, if any; a failed build must not publish a sitemap.
    """
    if exception is not None or app.builder.name != "html" or not app.config.html_baseurl:
        return
    namespace = SITEMAP_NAMESPACE
    root = ElementTree.Element(f"{{{namespace}}}urlset")
    pages = (
        name
        for name in app.env.found_docs
        if name not in EXCLUDED_PAGES and not name.startswith("_modules/")
    )
    for url in sorted({canonical_url(app, name) for name in pages}):
        entry = ElementTree.SubElement(root, f"{{{namespace}}}url")
        ElementTree.SubElement(entry, f"{{{namespace}}}loc").text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / "sitemap.xml",
        encoding="utf-8",
        xml_declaration=True,
        default_namespace=namespace,
    )


def setup(app: Any) -> dict:
    """Register the page-metadata and end-of-build sitemap callbacks.

    Args:
        app: Sphinx application being configured.

    Returns:
        Extension metadata; sitemap generation uses the complete merged inventory.
    """
    app.connect("html-page-context", set_canonical_url)
    app.connect("build-finished", write_sitemap)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
