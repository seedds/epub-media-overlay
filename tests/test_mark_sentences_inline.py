"""Round-trip tests for inline-formatting preservation in mark_sentences.

Segmentation wraps sentences in <span id="...-segmentN"> and rebuilds each block
from a character map that only re-emits tags listed in INLINE_TAGS. A missing inline
tag is walked transparently and silently dropped, changing rendering while leaving
visible text (and therefore validate_text_consistency) unchanged. These tests pin
that HTML5 inline formatting survives a segmentation round-trip.

Run:
  pytest tests/test_mark_sentences_inline.py -q
"""

import pytest
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



# --- reference-index semantics -------------------------------------------------
#
# The cleanup treats the reference sets as complete, so an EMPTY set strips every
# class/id. An UNKNOWN index (None) must therefore skip cleanup entirely; a failed
# index build used to fall back to empty sets and silently destroyed all markup.

_STYLED = '<html><body><p class="chapter" id="c1"><span class="real">x y z</span></p></body></html>'


def test_unknown_reference_index_leaves_classes_and_ids_alone():
    out = ms.mark_sentences(_STYLED, "x")
    p = BeautifulSoup(out, "lxml").p
    assert p["class"] == ["chapter"] and p["id"] == "c1"
    assert BeautifulSoup(out, "lxml").find("span", class_="real") is not None


def test_empty_reference_sets_strip_everything():
    out = ms.mark_sentences(
        _STYLED, "x", referenced_classes=frozenset(), referenced_ids=frozenset()
    )
    p = BeautifulSoup(out, "lxml").p
    assert "class" not in p.attrs and "id" not in p.attrs


def test_mark_segments_skips_cleanup_when_index_build_fails(tmp_path, monkeypatch):
    import zipfile

    import pipeline_core as pc

    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
        "<metadata/><manifest>"
        '<item id="c" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="c"/></spine></package>'
    )
    epub = tmp_path / "book.epub3"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/ch1.xhtml", _STYLED)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("indexer exploded")

    monkeypatch.setattr(pc, "build_reference_index", _boom)
    monkeypatch.chdir(tmp_path)  # mark_segments still writes loose copies to cwd
    book_info = {
        "folder_name": str(tmp_path),
        "out_file": str(epub),
        "opf_file": "OEBPS/content.opf",
        "opf_dir": "OEBPS",
        "language": "en",
        "matched_list": [{"json_file": "000.json", "html_file": "OEBPS/ch1.xhtml"}],
    }
    with pytest.warns(UserWarning, match="skipping redundant-markup cleanup"):
        pc.mark_segments(book_info)

    with zipfile.ZipFile(epub) as zf:
        html = zf.read("OEBPS/ch1.xhtml").decode("utf-8")
    p = BeautifulSoup(html, "lxml").p
    assert p["class"] == ["chapter"] and p["id"] == "c1"
    assert "-segment" in html  # segmentation itself still happened
