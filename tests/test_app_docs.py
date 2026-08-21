"""Allowlisted documentation routes on the Flask dashboard."""

from pathlib import Path

import pytest

from finance_vibe.app import DOC_CATALOG, BASE_DIR, app, resolve_doc_path


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_docs_learn_returns_200(client):
    res = client.get("/docs/learn")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Learn Finance Vibe" in body or "Three tracks" in body or "Track A" in body


def test_learn_hub_returns_200(client):
    res = client.get("/learn")
    assert res.status_code == 200
    assert b"Track A" in res.data


def test_docs_index_lists_allowlist(client):
    res = client.get("/docs")
    assert res.status_code == 200
    assert b"MLOps runbook" in res.data
    assert b"Coiled Cobra rubric" in res.data


def test_unknown_slug_is_404(client):
    assert client.get("/docs/not-a-real-slug").status_code == 404


def test_traversal_slug_is_404(client):
    assert client.get("/docs/../.env").status_code == 404
    assert client.get("/docs/%2e%2e%2f.env").status_code == 404


def test_allowlist_files_exist():
    base = Path(BASE_DIR)
    for slug, spec in DOC_CATALOG.items():
        path = (base / spec["file"]).resolve()
        assert path.is_file(), f"missing {slug}: {path}"
        assert str(path).startswith(str(base.resolve()))


def test_resolve_doc_path_rejects_unknown():
    with pytest.raises(Exception):
        resolve_doc_path("nope")
