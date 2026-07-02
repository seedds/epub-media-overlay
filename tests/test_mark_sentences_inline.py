"""Round-trip tests for inline-formatting preservation in mark_sentences.

Segmentation wraps sentences in <span id="...-segmentN"> and rebuilds each block
from a character map that only re-emits tags listed in INLINE_TAGS. A missing inline
tag is walked transparently and silently dropped, changing rendering while leaving
visible text (and therefore validate_text_consistency) unchanged. These tests pin
that HTML5 inline formatting survives a segmentation round-trip.

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_mark_sentences_inline.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_mark_sentences_inline.py
"""

# Bootstrap: allow running this file directly (see conftest.py rationale).
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from bs4 import BeautifulSoup

import mark_sentence as ms


def _segmented(body):
    html = f"<html><body>{body}</body></html>"
    return ms.mark_sentences(html)


def _tag_count(markup, name):
    return len(BeautifulSoup(markup, "lxml").find_all(name))


def test_s_strikethrough_survives_segmentation():
    # <s> must not be dropped when a sentence containing it is re-emitted.
    out = _segmented("<p>Price: <s>$50</s> $30 available widely today for readers.</p>")
    assert _tag_count(out, "s") == 1


def test_bdi_survives_segmentation():
    out = _segmented("<p>User <bdi>a b c</bdi> posted a fairly long sentence here.</p>")
    assert _tag_count(out, "bdi") == 1


def test_data_survives_segmentation():
    out = _segmented(
        '<p>The product <data value="398">Mini</data> shipped to many buyers today.</p>'
    )
    assert _tag_count(out, "data") == 1


def test_output_survives_segmentation():
    out = _segmented(
        "<p>The total came to <output>42</output> after a rather long computation ran.</p>"
    )
    assert _tag_count(out, "output") == 1


def test_visible_text_preserved():
    # Sanity: the round-trip does not alter visible characters (this is why the
    # dropped-tag regression is invisible to text-only validation).
    body = "<p>Price: <s>$50</s> $30 available widely today for readers.</p>"
    out = _segmented(body)
    orig_text = BeautifulSoup(f"<html><body>{body}</body></html>", "lxml").get_text()
    new_text = BeautifulSoup(out, "lxml").get_text()
    assert orig_text == new_text


if __name__ == "__main__":
    from _runner import run_module_tests

    raise SystemExit(run_module_tests(globals()))
