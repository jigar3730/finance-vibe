"""Docs blueprint: route registration, rendering, and path-safety checks."""
from __future__ import annotations

import os

import pytest
from werkzeug.exceptions import Forbidden, NotFound

from finance_vibe.app import app
from finance_vibe import docs_routes


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_docs_blueprint_registered():
    rules = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}
    assert "/docs/" in rules
    assert rules["/docs/"] == "docs.index"
    assert "/docs/<path:doc_path>" in rules
    assert rules["/docs/<path:doc_path>"] == "docs.show_doc"


def test_docs_index_renders_catalog(client):
    response = client.get("/docs/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Docs &amp; Labs" in body
    assert "Handbook" in body
    assert "Architecture" in body
    assert "Labs" in body
    assert "Finance Vibe documentation" in body


def test_docs_file_route_renders_markdown(client):
    response = client.get("/docs/handbook/scoring_logic.md")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Macro Vibe Score" in body
    assert 'class="docs-sidebar"' in body
    assert "<table>" in body


def test_docs_directory_falls_back_to_readme(client):
    response = client.get("/docs/labs")
    assert response.status_code == 200
    assert "Hands-on labs" in response.get_data(as_text=True)


def test_docs_missing_file_is_404(client):
    response = client.get("/docs/handbook/does_not_exist.md")
    assert response.status_code == 404


def test_docs_path_traversal_is_forbidden(client):
    response = client.get("/docs/../../requirements.txt")
    assert response.status_code == 403


def test_resolve_doc_file_rejects_escape():
    with pytest.raises(Forbidden):
        docs_routes.resolve_doc_file("../../requirements.txt")
    with pytest.raises(Forbidden):
        docs_routes.resolve_doc_file("/etc/passwd")
    with pytest.raises(NotFound):
        docs_routes.resolve_doc_file("does-not-exist.md")


def test_resolve_doc_file_stays_under_docs_root():
    resolved = docs_routes.resolve_doc_file("handbook/scoring_logic.md")
    assert os.path.commonpath([resolved, docs_routes.DOCS_ROOT]) == docs_routes.DOCS_ROOT
    assert resolved.endswith("handbook/scoring_logic.md")


def test_finviz_theme_stylesheet_is_served(client):
    response = client.get("/static/css/finviz-theme.css")
    assert response.status_code == 200
    css = response.get_data(as_text=True)
    assert "#131722" in css
    assert "#1e222d" in css
    assert "#089981" in css
    assert "#f23645" in css
    assert "Roboto Mono" in css or "finviz-font-mono" in css


def test_docs_pages_use_finviz_theme_and_monokai(client):
    index = client.get("/docs/").get_data(as_text=True)
    page = client.get("/docs/handbook/scoring_logic.md").get_data(as_text=True)
    dash = client.get("/").get_data(as_text=True)
    for body in (index, page, dash):
        assert 'href="/static/css/finviz-theme.css"' in body
    assert docs_routes.PYGMENTS_STYLE == "monokai"
    assert "#f92672" in page or "monokai" in page.lower() or "#272822" in page


def test_internal_md_links_rewrite_to_docs_routes():
    source = "See [swing](swing_setup.md) and [root](../README.md)."
    current = os.path.join(docs_routes.DOCS_ROOT, "handbook", "scoring_logic.md")
    html, _toc = docs_routes.render_markdown(source, current)
    assert 'href="/docs/handbook/swing_setup.md"' in html
    assert 'href="/docs/README.md"' in html
