"""/docs: the operating docs rendered in the dashboard (web.docs, config.DOCS_PAGES)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finance_vibe import config
from finance_vibe.web import app as app_module
from finance_vibe.web import docs
from finance_vibe.web.app import app

ROOT = Path(__file__).resolve().parents[1]
_HREF = re.compile(r'href="([^"]+)"')
_ID = re.compile(r'id="([^"]+)"')


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "LOG_DIRS", {"weekly": str(tmp_path), "daily": str(tmp_path)})
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_lists_every_doc_and_nav_links_it(client):
    html = client.get("/docs").get_data(as_text=True)
    assert 'class="nav-link active" href="/docs"' in html
    for name, (_, contents) in config.DOCS_PAGES.items():
        assert f'href="/docs/{name}"' in html
        assert contents in html
    assert "⚠ file missing" not in html


@pytest.mark.parametrize("name", list(config.DOCS_PAGES))
def test_every_doc_renders(client, name):
    resp = client.get(f"/docs/{name}")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    doc = docs.page(name)
    assert doc is not None
    assert f"<h1>📘 {doc['title']}" in html
    assert not doc["html"].lstrip().startswith("<h1")  # the title h1 moved to the page header
    assert ".md" not in " ".join(_HREF.findall(doc["html"]))  # no raw links to Markdown files


@pytest.mark.parametrize("name", ["nope", "..", "RESEARCH", "operations.md"])
def test_unknown_names_are_404(client, name):
    assert client.get(f"/docs/{name}").status_code == 404


def test_path_tricks_are_404(client):
    assert client.get("/docs/../config.py").status_code == 404
    assert client.get("/docs/%2e%2e%2fpyproject.toml").status_code == 404


@pytest.mark.parametrize(
    ("href", "source", "want"),
    [
        ("operations.md", "docs/runbook.md", "/docs/operations"),
        ("operations.md#scheduling", "docs/runbook.md", "/docs/operations#scheduling"),
        ("../README.md", "docs/operations.md", "/docs/readme"),
        ("docs/runbook.md", "README.md", "/docs/runbook"),
        ("#database", "docs/operations.md", "#database"),
        (
            "https://finviz.com/quote.ashx?t=SPY",
            "docs/runbook.md",
            "https://finviz.com/quote.ashx?t=SPY",
        ),
        ("../RESEARCH.md", "docs/operations.md", None),  # research notes are not served
        ("reviews/", "docs/README.md", None),
    ],
)
def test_doc_links(href, source, want):
    assert docs._doc_link(href, source) == want


def test_unserved_link_keeps_its_text(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(
        "# A\n\nSee [the research](../RESEARCH.md) and [B](b.md#part).\n"
    )
    (tmp_path / "docs" / "b.md").write_text("# B\n\n## Part\n")
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(
        config, "DOCS_PAGES", {"a": ("docs/a.md", "A doc"), "b": ("docs/b.md", "B doc")}
    )
    html = docs.page("a")["html"]
    assert "<span>the research</span>" in html
    assert '<a href="/docs/b#part">B</a>' in html


def test_missing_file_is_flagged_and_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DOCS_PAGES", {"gone": ("docs/gone.md", "Gone")})
    assert client.get("/docs/gone").status_code == 404
    assert "⚠ file missing" in client.get("/docs").get_data(as_text=True)


def test_links_between_served_docs_reach_real_sections():
    """Every /docs link in every served doc names a served doc and, with a #, one of its ids."""
    ids = {name: set(_ID.findall(docs.page(name)["html"])) for name in config.DOCS_PAGES}
    broken = []
    for name in config.DOCS_PAGES:
        for href in _HREF.findall(docs.page(name)["html"]):
            target, _, frag = href.partition("#")
            if href.startswith("#"):
                target = f"/docs/{name}"
            if not target.startswith("/docs/"):
                continue
            other = target.removeprefix("/docs/")
            if other not in ids or (frag and frag not in ids[other]):
                broken.append(f"{name}: {href}")
    assert broken == []


def test_image_ships_every_served_doc():
    """.dockerignore drops docs/ except the served files; a new served doc must be let in."""
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    assert "docs/*" in lines
    for path, _ in config.DOCS_PAGES.values():
        assert (ROOT / path).is_file(), path
        if path.startswith("docs/"):
            assert f"!{path}" in lines, f"{path} is not copied into the image"
