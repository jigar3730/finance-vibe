"""Flask blueprint that renders the local docs/ tree as HTML."""
from __future__ import annotations

import os
import re
from typing import Any

import markdown
from flask import Blueprint, abort, render_template
from pygments.formatters import HtmlFormatter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DOCS_ROOT = os.path.abspath(os.path.join(BASE_DIR, "docs"))

DOC_SECTIONS = (
    ("Handbook", "handbook"),
    ("Architecture", "architecture"),
    ("Labs", "labs"),
)

MD_EXTENSIONS = ("fenced_code", "codehilite", "tables", "toc", "attr_list")
MD_EXTENSION_CONFIGS = {
    "codehilite": {
        "css_class": "codehilite",
        "guess_lang": True,
        "linenums": False,
    },
    "toc": {
        "permalink": False,
        "toc_depth": "2-3",
    },
}

_HREF_MD = re.compile(
    r'href="([^"]+?\.md)(#[^"]*)?"',
    re.IGNORECASE,
)

docs_bp = Blueprint("docs", __name__, url_prefix="/docs")

PYGMENTS_STYLE = "monokai"
PYGMENTS_CSS = HtmlFormatter(style=PYGMENTS_STYLE).get_style_defs(".codehilite")


def _is_under_docs(path: str) -> bool:
    """Return True if *path* resolves inside DOCS_ROOT (no traversal)."""
    try:
        return os.path.commonpath([os.path.abspath(path), DOCS_ROOT]) == DOCS_ROOT
    except ValueError:
        return False


def resolve_doc_file(doc_path: str | None) -> str:
    """Map a URL path to a markdown file under docs/, or abort 403/404.

    Accepts ``README.md``, a section folder (``labs`` → ``labs/README.md``),
    or a path without the ``.md`` suffix.
    """
    if not doc_path or doc_path in (".", "/"):
        rel = "README.md"
    else:
        rel = doc_path.replace("\\", "/").lstrip("/")
        if os.path.isabs(doc_path) or rel.startswith(".."):
            abort(403, "Access restricted outside the documentation tree.")

    candidate = os.path.abspath(os.path.join(DOCS_ROOT, rel))
    if not _is_under_docs(candidate):
        abort(403, "Access restricted outside the documentation tree.")

    if os.path.isdir(candidate):
        candidate = os.path.join(candidate, "README.md")
        if not _is_under_docs(candidate):
            abort(403, "Access restricted outside the documentation tree.")
    elif not candidate.endswith(".md"):
        with_ext = candidate + ".md"
        if os.path.isfile(with_ext) and _is_under_docs(with_ext):
            candidate = with_ext

    if not os.path.isfile(candidate) or not candidate.endswith(".md"):
        abort(404, "Documentation page not found.")

    return candidate


def relative_doc_path(abs_path: str) -> str:
    """Return the docs-relative POSIX path for a resolved markdown file."""
    rel = os.path.relpath(abs_path, DOCS_ROOT)
    return rel.replace(os.sep, "/")


def _first_heading(path: str, fallback: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()
    except OSError:
        pass
    stem = fallback.removesuffix(".md").replace("_", " ")
    return stem


def build_sidebar_nav(active_rel: str) -> list[dict[str, Any]]:
    """Build Handbook / Architecture / Labs sidebar links from the filesystem."""
    sections: list[dict[str, Any]] = [
        {
            "title": "Catalog",
            "links": [
                {
                    "title": "Documentation Home",
                    "href": "/docs/",
                    "active": active_rel in ("README.md", ""),
                }
            ],
        }
    ]

    for title, folder in DOC_SECTIONS:
        folder_path = os.path.join(DOCS_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        names = [
            name
            for name in os.listdir(folder_path)
            if name.endswith(".md") and not name.startswith(".")
        ]
        names.sort(key=lambda name: (name.lower() != "readme.md", name.lower()))
        links = []
        for name in names:
            rel = f"{folder}/{name}"
            links.append(
                {
                    "title": _first_heading(os.path.join(folder_path, name), name),
                    "href": f"/docs/{rel}",
                    "active": active_rel == rel,
                }
            )
        if links:
            sections.append({"title": title, "links": links})
    return sections


def _rewrite_internal_md_links(html: str, current_file: str) -> str:
    """Point relative ``.md`` hrefs at ``/docs/...`` when they stay in-tree."""
    current_dir = os.path.dirname(current_file)

    def _replace(match: re.Match[str]) -> str:
        href = match.group(1)
        fragment = match.group(2) or ""
        if href.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        target = os.path.abspath(os.path.join(current_dir, href))
        if not _is_under_docs(target):
            return match.group(0)
        rel = relative_doc_path(target)
        return f'href="/docs/{rel}{fragment}"'

    return _HREF_MD.sub(_replace, html)


def render_markdown(source: str, current_file: str) -> tuple[str, str]:
    """Convert markdown to HTML plus a TOC; rewrite in-tree .md links."""
    converter = markdown.Markdown(
        extensions=list(MD_EXTENSIONS),
        extension_configs=MD_EXTENSION_CONFIGS,
    )
    body = converter.convert(source)
    toc = getattr(converter, "toc", "") or ""
    return _rewrite_internal_md_links(body, current_file), toc


def _page_title(source: str, fallback: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _render_doc_page(abs_path: str) -> str:
    try:
        with open(abs_path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError:
        abort(404, "Documentation page not found.")

    rel = relative_doc_path(abs_path)
    html, toc = render_markdown(source, abs_path)
    return render_template(
        "docs.html",
        title=_page_title(source, rel),
        content_html=html,
        toc_html=toc,
        nav_sections=build_sidebar_nav(rel),
        pygments_css=PYGMENTS_CSS,
        doc_rel=rel,
    )


@docs_bp.route("/")
def index() -> str:
    """Render docs/README.md as the documentation catalog."""
    return _render_doc_page(resolve_doc_file("README.md"))


@docs_bp.route("/<path:doc_path>")
def show_doc(doc_path: str) -> str:
    """Render one markdown file from docs/, rejecting path traversal."""
    return _render_doc_page(resolve_doc_file(doc_path))
