"""Unit tests for mark_sentence.preprocess_remove_redundant_tags.

The cleaner removes markup/attributes that nothing in the EPUB references, driven by
two sets (referenced_classes, referenced_ids) from epub_reference_index. These tests
pin the three composed passes and, crucially, the safety controls that must never
fire:

  - unwrap now-bare wrapper spans (e.g. Kobo spans once their attrs are gone)
  - strip only *unreferenced* class values / ids, keeping referenced ones
  - remove empty attribute-free tags, but never void tags or <a>
  - never touch elements carrying inline style=

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_html_cleaner.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_html_cleaner.py
"""

# Bootstrap: allow running this file directly (`python tests/test_html_cleaner.py`).
# Source modules are imported by bare name below, so the repo root must be on
# sys.path before those imports. Under pytest, pyproject's pythonpath handles this.
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from bs4 import BeautifulSoup

from mark_sentence import preprocess_remove_redundant_tags


def _inner(html, referenced_classes=frozenset(), referenced_ids=frozenset()):
    """Clean `html` and return the <body> inner markup, whitespace-trimmed."""
    cleaned = preprocess_remove_redundant_tags(
        html, referenced_classes, referenced_ids
    )
    return BeautifulSoup(cleaned, "lxml").body.decode_contents().strip()


def _wrap(body):
    return f"<html><body>{body}</body></html>"


# --- pass 2: unwrap bare wrapper spans ------------------------------------


def test_kobo_span_unwrapped_when_unreferenced():
    out = _inner(_wrap('<p><span class="koboSpan" id="kobo.1.1">Hi</span> there</p>'))
    assert out == "<p>Hi there</p>"


def test_kobo_span_kept_when_class_referenced():
    out = _inner(
        _wrap('<p><span class="koboSpan" id="kobo.1.1">Hi</span></p>'),
        referenced_classes=frozenset({"koboSpan"}),
    )
    assert out == '<p><span class="koboSpan">Hi</span></p>'


# --- pass 1: strip unreferenced class values / ids ------------------------


def test_partial_class_list_trimmed():
    out = _inner(
        _wrap('<p class="chapter unused">t</p>'),
        referenced_classes=frozenset({"chapter"}),
    )
    assert out == '<p class="chapter">t</p>'


def test_span_with_only_unused_class_is_unwrapped():
    # strip class -> bare span -> unwrap.
    out = _inner(_wrap('<p><span class="unused">t</span></p>'))
    assert out == "<p>t</p>"


def test_non_span_element_keeps_itself_when_class_stripped():
    out = _inner(_wrap('<p class="unused">t</p>'))
    assert out == "<p>t</p>"


def test_unreferenced_id_removed_but_element_kept():
    out = _inner(_wrap('<p id="dead">t</p>'))
    assert out == "<p>t</p>"


def test_referenced_id_is_kept():
    out = _inner(_wrap('<p id="c39">t</p>'), referenced_ids=frozenset({"c39"}))
    assert out == '<p id="c39">t</p>'


# --- pass 3: remove empty attribute-free tags -----------------------------


def test_empty_unreferenced_span_removed():
    out = _inner(_wrap('<p>a<span id="stale"></span>b</p>'))
    assert out == "<p>ab</p>"


def test_empty_span_kept_when_id_referenced():
    # This is the load-bearing nav-anchor case (<span id="c39"></span>).
    out = _inner(
        _wrap('<p><span id="c39"></span>Chapter</p>'),
        referenced_ids=frozenset({"c39"}),
    )
    assert out == '<p><span id="c39"></span>Chapter</p>'


# --- safety controls: these must never be altered -------------------------


def test_inline_style_blocks_removal():
    out = _inner(_wrap('<p><span style="color:red">x</span></p>'))
    assert out == '<p><span style="color:red">x</span></p>'


def test_referenced_content_class_kept():
    out = _inner(
        _wrap('<p><span class="engra">x</span></p>'),
        referenced_classes=frozenset({"engra"}),
    )
    assert out == '<p><span class="engra">x</span></p>'


def test_empty_anchor_element_never_removed():
    # Even with an unreferenced id, the <a> element itself must survive (its id is
    # stripped, but empty anchors are load-bearing so the tag stays).
    out = _inner(_wrap('<p><a id="dead"></a>t</p>'))
    assert out == "<p><a></a>t</p>"


def test_empty_anchor_with_referenced_id_untouched():
    out = _inner(_wrap('<p><a id="fn1"></a>t</p>'), referenced_ids=frozenset({"fn1"}))
    assert out == '<p><a id="fn1"></a>t</p>'


def test_empty_table_cell_never_removed():
    # An empty <td> is positional: removing it shifts every later cell left a column.
    out = _inner(_wrap("<table><tr><td></td><td>A</td></tr></table>"))
    soup = BeautifulSoup(_wrap(out), "lxml")
    assert len(soup.find_all("td")) == 2


def test_empty_list_item_never_removed():
    # An empty <li> is positional: removing it renumbers an ordered list.
    out = _inner(_wrap("<ol><li></li><li>x</li></ol>"))
    soup = BeautifulSoup(_wrap(out), "lxml")
    assert len(soup.find_all("li")) == 2


def test_default_styled_inline_tag_kept():
    out = _inner(_wrap("<p><b>x</b></p>"))
    assert out == "<p><b>x</b></p>"


def test_html5_strikethrough_inline_tag_kept():
    # <s> is an inline formatting tag; it must not be dropped during reconstruction.
    out = _inner(_wrap("<p><s>x</s></p>"))
    assert out == "<p><s>x</s></p>"


def test_void_tags_kept():
    out = _inner(_wrap("<p>a<br/>b</p>"))
    assert out == "<p>a<br/>b</p>"


# --- structural invariants -------------------------------------------------


def test_head_is_preserved():
    html = (
        "<html><head><title>T</title></head>"
        '<body><p><span class="koboSpan" id="kobo.1.1">x</span></p></body></html>'
    )
    cleaned = preprocess_remove_redundant_tags(html)
    soup = BeautifulSoup(cleaned, "lxml")
    assert soup.head is not None
    assert soup.head.title is not None


def test_visible_text_is_unchanged():
    html = _wrap(
        '<p class="unused"><span class="koboSpan" id="kobo.1.1">Hello</span>'
        ' <span id="stale"></span>world</p>'
    )
    before = " ".join(BeautifulSoup(html, "lxml").get_text().split())
    after = " ".join(
        BeautifulSoup(preprocess_remove_redundant_tags(html), "lxml").get_text().split()
    )
    assert before == after == "Hello world"


if __name__ == "__main__":
    from _runner import run_module_tests

    raise SystemExit(run_module_tests(globals()))
