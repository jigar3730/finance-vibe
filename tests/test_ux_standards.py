"""docs/UX_STANDARDS.md: no verdict words in page text, no external CSS / JS / fonts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "finance_vibe" / "web"
TEMPLATES = sorted((WEB / "templates").glob("*.html"))
STYLESHEETS = sorted((WEB / "static" / "css").glob("*.css"))

# The Setups area may show raw scanner status / state names as data.
SETUPS_AREA = {"setups.html", "breakout_view.html", "cobra_view.html"}
VERDICT = re.compile(
    r"\b(bullish|bearish|strong|weak|elevated|fading|accelerat\w*|aligned|risk[\s-]?(?:on|off))\b",
    re.IGNORECASE,
)
# Known breaks, fixed by the home page redesign. Remove an entry once it is gone.
KNOWN_VIOLATIONS = {"market.html": {"strong"}}

_JINJA = re.compile(r"\{#.*?#\}|\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)
_ATTR_TEXT = re.compile(r"""\b(?:title|aria-label|alt|placeholder)\s*=\s*["']([^"']*)["']""")
_TAG = re.compile(r"<[^>]*>")
_STYLE_OR_SCRIPT = re.compile(r"<(style|script)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_EXTERNAL = [
    re.compile(r"<(?:link|script)\b[^>]*\b(?:href|src)\s*=\s*[\"'](?:https?:)?//", re.IGNORECASE),
    re.compile(r"@import", re.IGNORECASE),
    re.compile(r"url\(\s*[\"']?(?:https?:)?//", re.IGNORECASE),
    re.compile(r"fonts\.(?:googleapis|gstatic)\.com|use\.fontawesome|cdn\.", re.IGNORECASE),
]


def visible_text(html: str) -> str:
    """Page text a reader sees: tag-free text plus tooltip / label attributes."""
    html = _STYLE_OR_SCRIPT.sub(" ", _JINJA.sub(" ", html))
    attrs = " ".join(_ATTR_TEXT.findall(html))
    return f"{_TAG.sub(' ', html)} {attrs}"


def verdict_words(html: str) -> set[str]:
    return {m.group(1).lower() for m in VERDICT.finditer(visible_text(html))}


def external_links(text: str) -> list[str]:
    return [m.group(0) for pattern in _EXTERNAL for m in pattern.finditer(text)]


def test_helpers_catch_what_they_should():
    page = '<p title="Strong trend">VIX <strong>19.42</strong> Risk-Off {{ "bullish" }}</p>'
    assert verdict_words(page) == {"strong", "risk-off"}  # not the tag, not Jinja code
    assert verdict_words("<td>above its 20-day average</td>") == set()
    assert external_links('<link rel="stylesheet" href="https://cdn.x.com/t.css">')
    assert external_links('<script src="//unpkg.com/a.js"></script>')
    assert external_links("@font-face { src: url(https://fonts.gstatic.com/a.woff2); }")
    assert not external_links('<a href="https://finviz.com/quote.ashx?t=SPY">SPY</a>')
    assert not external_links("<link href=\"{{ url_for('static', filename='a.css') }}\">")


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_no_verdict_words_outside_setups(template):
    if template.name in SETUPS_AREA:
        pytest.skip("Setups area may show raw scanner names")
    found = verdict_words(template.read_text())
    known = KNOWN_VIOLATIONS.get(template.name, set())
    assert found - known == set(), f"verdict words in {template.name}: {sorted(found - known)}"
    assert known <= found, f"fixed, drop from KNOWN_VIOLATIONS: {sorted(known - found)}"


@pytest.mark.parametrize("path", TEMPLATES + STYLESHEETS, ids=lambda p: p.name)
def test_no_external_css_js_or_fonts(path):
    assert external_links(path.read_text()) == []
