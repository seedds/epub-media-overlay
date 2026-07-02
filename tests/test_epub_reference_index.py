"""Unit tests for epub_reference_index.build_reference_index and its helpers.

These verify that the reference index correctly identifies which class/id names are
"used" by the EPUB, since preprocess_remove_redundant_tags trusts these sets to
decide what is safe to delete. Key properties under test:

  - CSS class/id tokens are extracted from compound/descendant/grouped/pseudo/
    attribute selectors.
  - Rules whose declarations are empty or entirely inert (e.g. the Kobo no-op
    `.koboSpan { -webkit-text-combine: inherit; }`) do NOT count as references.
  - id fragments from href/src (including NCX <content src="...#id"/>) are captured.
  - JavaScript identifiers (getElementById('x'), etc.) are captured into both sets.

There is also an optional integration test against the real Foundation EPUB, which
is skipped when that file is not present.

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_epub_reference_index.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_epub_reference_index.py
"""

# Bootstrap: allow running this file directly.
# Source modules are imported by bare name below, so the repo root must be on
# sys.path before those imports. Under pytest, pyproject's pythonpath handles this.
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import posixpath
import zipfile

from bs4 import BeautifulSoup

import epub_reference_index as eri


# --- CSS selector extraction ----------------------------------------------


def _css_classes(css):
    classes, ids = set(), set()
    eri._collect_css_tokens(css, classes, ids)
    return classes, ids


def test_css_simple_class_and_id():
    classes, ids = _css_classes(".foo { color: red } #bar { color: blue }")
    assert "foo" in classes
    assert "bar" in ids


def test_css_compound_descendant_grouped_pseudo():
    classes, ids = _css_classes(
        ".chapter .extract, p.center > span:hover { margin: 0 }"
    )
    assert {"chapter", "extract", "center"} <= classes


def test_css_attribute_selector_values_captured():
    classes, ids = _css_classes('[class~="ttf"] .fn { font-style: italic }')
    # Attribute-selector values are added to both sets, conservatively.
    assert "ttf" in classes
    assert "fn" in classes


def test_inert_only_rule_is_not_a_reference():
    # The exact Kobo no-op rule must not protect `koboSpan`.
    classes, ids = _css_classes(".koboSpan { -webkit-text-combine: inherit; }")
    assert "koboSpan" not in classes


def test_empty_rule_is_not_a_reference():
    classes, ids = _css_classes(".empty { }")
    assert "empty" not in classes


def test_meaningful_rule_alongside_inert_still_references():
    # A class that is inert in one rule but real in another must be kept.
    classes, ids = _css_classes(
        ".koboSpan { -webkit-text-combine: inherit } .koboSpan { color: red }"
    )
    assert "koboSpan" in classes


def test_mixed_declarations_count_as_reference():
    classes, ids = _css_classes(
        ".x { -webkit-text-combine: inherit; color: red }"
    )
    assert "x" in classes


# --- at-rule (conditional group) recursion --------------------------------


def test_class_inside_media_query_is_referenced():
    # A class styled only inside @media must still be indexed, or the stripper
    # deletes it from the book. (Regression: at-rules were skipped entirely.)
    classes, ids = _css_classes(
        "@media screen and (min-width: 40em) { .highlight { background: yellow } }"
    )
    assert "highlight" in classes


def test_id_and_class_inside_supports_is_referenced():
    classes, ids = _css_classes(
        "@supports (display: grid) { #grid-note { color: red } .grid-only { display: grid } }"
    )
    assert "grid-only" in classes
    assert "grid-note" in ids


def test_nested_at_rules_recurse():
    classes, ids = _css_classes(
        "@media screen { @supports (gap: 1px) { .deep { gap: 1px } } }"
    )
    assert "deep" in classes


def test_inert_only_rule_inside_media_still_not_referenced():
    # The inert-rule filter must still apply inside a conditional block.
    classes, ids = _css_classes(
        "@media print { .koboSpan { -webkit-text-combine: inherit } }"
    )
    assert "koboSpan" not in classes


# --- idref attributes (aria-*, label for, td headers) ---------------------


def test_aria_and_label_and_headers_ids_are_referenced():
    # These reference element ids WITHOUT a leading '#', so the fragment scan misses
    # them; they must be collected from the attributes or the stripper deletes the
    # ids and leaves dangling accessibility references.
    html = (
        "<html><body>"
        '<h2 id="s1">Sec</h2><h3 id="s2">Sub</h3>'
        '<div aria-labelledby="s1 s2">x</div>'
        '<div aria-describedby="s2">y</div>'
        '<label for="fld">Name</label><input id="fld"/>'
        '<table><tr><th id="h1">H</th></tr>'
        '<tr><td headers="h1">v</td></tr></table>'
        "</body></html>"
    )
    classes, ids = set(), set()
    eri._collect_from_html(html, classes, ids)
    assert {"s1", "s2", "fld", "h1"} <= ids


# --- fragment id extraction -----------------------------------------------


def test_fragment_ids_from_href_and_src():
    ids = set()
    eri._collect_fragment_ids(
        '<a href="ch1.html#c39">x</a><img src="p.svg#glyph"/>', ids
    )
    assert {"c39", "glyph"} <= ids


def test_ncx_content_fragment_ids():
    ncx = '<content src="xhtml/ch07.html#c07"/>'
    ids = set()
    eri._collect_fragment_ids(ncx, ids)
    assert "c07" in ids


# --- JavaScript token extraction ------------------------------------------


def test_js_tokens_added_to_both_sets():
    classes, ids = set(), set()
    eri._collect_js_tokens(
        "document.getElementById('book-inner');"
        "document.querySelector('#nav-target');"
        "document.getElementsByClassName('highlight');",
        classes,
        ids,
    )
    for name in ("book-inner", "nav-target", "highlight"):
        assert name in classes
        assert name in ids


def test_js_single_char_tokens_ignored():
    # Length >= 2 keeps loop variables and other one-letter noise out of the sets.
    classes, ids = set(), set()
    eri._collect_js_tokens("for (var i = 0; i < n; i++) {}", classes, ids)
    assert "i" not in classes
    assert "i" not in ids


# --- HTML inline style/script ---------------------------------------------


def test_inline_style_and_fragment_and_script():
    html = (
        "<html><head>"
        "<style>.real { color: red } .koboSpan { -webkit-text-combine: inherit }</style>"
        "<script>var q = document.getElementById('scripted');</script>"
        "</head><body><a href='x.html#navid'>t</a></body></html>"
    )
    classes, ids = set(), set()
    eri._collect_from_html(html, classes, ids)
    assert "real" in classes
    assert "koboSpan" not in classes
    assert "navid" in ids
    assert "scripted" in ids


# --- integration against the real Foundation EPUB (skipped if absent) ------

_FOUNDATION_EPUB = (
    "/Users/f2pgod/Documents/Audiobooks/Foundation/Foundation [7a8727d0].epub"
)


def _build_foundation_index():
    with zipfile.ZipFile(_FOUNDATION_EPUB) as zip_file:
        opf_name = next(n for n in zip_file.namelist() if n.endswith(".opf"))
        opf_dir = posixpath.dirname(opf_name) or "."
        opf_soup = BeautifulSoup(zip_file.read(opf_name), "xml")
        return eri.build_reference_index(zip_file, opf_soup, opf_dir)


def test_foundation_reference_index_classifies_correctly():
    import os

    if not os.path.isfile(_FOUNDATION_EPUB):
        try:
            import pytest

            pytest.skip("Foundation EPUB not available")
        except ImportError:
            print("SKIP: Foundation EPUB not available")
            return

    index = _build_foundation_index()

    # koboSpan is only ever styled by an inert no-op rule -> not referenced.
    assert "koboSpan" not in index.referenced_classes
    # Real content classes from the external stylesheet -> referenced.
    assert {"chapter", "center", "extract", "fn"} <= index.referenced_classes
    # Chapter nav anchors targeted by nav/NCX/TOC -> referenced.
    assert {"c39", "c07"} <= index.referenced_ids
    # Ids used only by kobo.js -> referenced via the JS scan.
    assert {"book-columns", "book-inner"} <= index.referenced_ids
    # Individual kobo span ids are targeted by nothing.
    assert "kobo.1.1" not in index.referenced_ids


if __name__ == "__main__":
    from _runner import run_module_tests

    raise SystemExit(run_module_tests(globals()))
