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


# --- cleanup must not alter visible text ---------------------------------------


def test_whitespace_only_inline_tag_keeps_the_space():
    html = "<html><body><p>foo<i> </i>bar and more words follow here.</p></body></html>"
    out = ms.mark_sentences(html, "x", referenced_classes=frozenset(), referenced_ids=frozenset())
    assert "foo bar" in BeautifulSoup(out, "lxml").get_text()


def test_empty_block_is_not_removed_by_cleanup():
    # An attribute-free empty <p> is a visible spacer; only inline leftovers go.
    out = ms.preprocess_remove_redundant_tags("<html><body><p>a</p><p></p><p>b</p></body></html>")
    assert len(BeautifulSoup(out, "lxml").find_all("p")) == 3


def test_text_altering_cleanup_is_detected(monkeypatch):
    def _drop_a_char(soup, classes, ids):
        node = soup.find(string=True)
        node.replace_with(node[1:])

    monkeypatch.setattr(ms, "_clean_soup", _drop_a_char)
    with pytest.raises(ValueError, match="Text integrity"):
        ms.mark_sentences(
            "<html><body><p>Hello there.</p></body></html>",
            "x",
            referenced_classes=frozenset(),
            referenced_ids=frozenset(),
        )


def test_css_link_added_once():
    html = "<html><head><title>t</title></head><body><p>Hello there.</p></body></html>"
    out = ms.mark_sentences(html, "x", css_href="../readaloud.css")
    out = ms.mark_sentences(out, "y", css_href="../readaloud.css")
    links = BeautifulSoup(out, "lxml").find_all("link", href="../readaloud.css")
    assert len(links) == 1


# --- zero-width (empty anchor) placement -------------------------------------


def _body(out):
    return BeautifulSoup(out, "lxml").body


def test_anchor_on_segment_boundary_emitted_once_without_empty_wrapper():
    # The boundary after "here. " lands exactly on the anchor. It used to be
    # appended to both neighbouring segments, leaving an empty <em></em> behind.
    html = (
        "<html><body><p><em>First sentence here. "
        '<a id="fn1"></a>Second sentence here.</em></p></body></html>'
    )
    body = _body(ms.mark_sentences(html, "x", referenced_ids=frozenset({"fn1"}), referenced_classes=frozenset()))
    assert len(body.find_all("a")) == 1
    assert not [e for e in body.find_all("em") if not e.get_text() and e.find(True) is None]


def test_anchor_at_end_of_block_survives():
    html = '<html><body><p>Only sentence here.<a id="end"></a></p></body></html>'
    body = _body(ms.mark_sentences(html, "x", referenced_ids=frozenset({"end"}), referenced_classes=frozenset()))
    anchor = body.find("a", id="end")
    assert anchor is not None
    assert anchor.find_parent("span")["id"].startswith("cx-segment")


def test_anchor_inside_segment_keeps_position():
    html = '<html><body><p>Alpha <a id="mid"></a>beta gamma delta.</p></body></html>'
    body = _body(ms.mark_sentences(html, "x", referenced_ids=frozenset({"mid"}), referenced_classes=frozenset()))
    text_after = body.find("a", id="mid").next_sibling
    assert str(text_after).startswith("beta")


# --- every container inside a leaf block is preserved ------------------------


def test_unlisted_inline_tag_survives_segmentation():
    out = _segmented("<p>An <acronym title='x'>ABC</acronym> appears in this long sentence here.</p>")
    assert _tag_count(out, "acronym") == 1


def test_table_cells_are_segmented():
    out = ms.mark_sentences(
        "<html><body><table><tr><td>First cell text.</td><th>Header</th></tr>"
        "<caption>A caption.</caption></table></body></html>",
        "x",
    )
    soup = BeautifulSoup(out, "lxml")
    assert soup.td.find("span", id=lambda v: v and "-segment" in v) is not None
    assert soup.th.find("span", id=lambda v: v and "-segment" in v) is not None
    assert soup.caption.find("span", id=lambda v: v and "-segment" in v) is not None
