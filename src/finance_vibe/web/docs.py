"""Operating docs served in the dashboard (`/docs`): Markdown files from the repo rendered as pages.

Only the files named in ``config.DOCS_PAGES`` are served, so a URL can never reach another
file. Links between served docs become ``/docs/<name>`` links; links to anything else (the
research notes, source files) keep their text and lose the link.
"""

from __future__ import annotations

import logging
import posixpath
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any

import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor

from finance_vibe import config

logger = logging.getLogger(__name__)

_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "toc"]
_H1 = re.compile(r"<h1\b[^>]*>.*?</h1>\s*", re.DOTALL)


def _path(name: str) -> Path:
    return Path(config.DOCS_DIR) / config.DOCS_PAGES[name][0]


def _doc_link(href: str, source: str) -> str | None:
    """``/docs/...`` for a link to a served doc, the href itself for an outside or in-page
    link, None for a link into the repo that is not served."""
    if href.startswith(("#", "http://", "https://", "mailto:", "/")):
        return href
    target, _, fragment = href.partition("#")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
    for name, (rel, _) in config.DOCS_PAGES.items():
        if rel == resolved:
            return f"/docs/{name}" + (f"#{fragment}" if fragment else "")
    return None


class _LinkRewriter(Treeprocessor):
    def __init__(self, md: markdown.Markdown, source: str) -> None:
        super().__init__(md)
        self.source = source

    def run(self, root: ET.Element) -> None:
        for parent in root.iter():
            for child in parent:
                if child.tag != "a":
                    continue
                href = _doc_link(child.get("href", ""), self.source)
                if href is not None:
                    child.set("href", href)
                    continue
                # Not served here: keep the link text as a plain span.
                child.tag = "span"
                child.attrib.clear()


class _DocLinks(Extension):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.treeprocessors.register(_LinkRewriter(md, self.source), "doc_links", 0)


@lru_cache(maxsize=32)
def _render(name: str, mtime_ns: int) -> dict[str, Any]:
    """Render one doc; cached per file version (``mtime_ns`` is part of the key)."""
    source = config.DOCS_PAGES[name][0]
    md = markdown.Markdown(extensions=[*_EXTENSIONS, _DocLinks(source)])
    html = md.convert(_path(name).read_text(encoding="utf-8"))
    # The page header shows the title, so drop the doc's own h1.
    html = _H1.sub("", html, count=1)
    tokens = getattr(md, "toc_tokens", [])
    title = tokens[0]["name"] if tokens and tokens[0]["level"] == 1 else name
    # Section links: every h1 after the title and every h2.
    sections = []
    for i, tok in enumerate(tokens):
        heads = tok["children"] if i == 0 and tok["level"] == 1 else [tok, *tok["children"]]
        sections += [{"id": h["id"], "name": h["name"]} for h in heads if h["level"] <= 2]
    return {"name": name, "title": title, "html": html, "sections": sections}


def page(name: str) -> dict[str, Any] | None:
    """The rendered doc, or None when the name is not served or its file is missing."""
    if name not in config.DOCS_PAGES:
        return None
    try:
        mtime = _path(name).stat().st_mtime_ns
    except OSError:
        logger.warning("doc %s missing: %s", name, _path(name))
        return None
    return _render(name, mtime)


def index() -> list[dict[str, Any]]:
    """Every served doc in ``config.DOCS_PAGES`` order: name, title, contents, missing."""
    rows = []
    for name, (_, contents) in config.DOCS_PAGES.items():
        doc = page(name)
        rows.append(
            {
                "name": name,
                "title": doc["title"] if doc else name,
                "contents": contents,
                "missing": doc is None,
            }
        )
    return rows
