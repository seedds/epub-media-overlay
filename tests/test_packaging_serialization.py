"""Tests for packaging-side serialization and audio helpers in pipeline_core.

Covers regressions:
  - convert_soup_to_html must not corrupt text/attributes containing the substring
    "opf:" (it previously did a blind str.replace("opf:", "") before a minidom
    pretty-print round-trip);
  - audio_media_type must reflect the configured extension, not a hardcoded
    "audio/mp4";
  - iter_audio_files / is_chunk_basename must select only NNN<ext> chunks, never
    the copied source audiobook;
  - get_audio_duration must raise (not silently return 0.0) when it cannot determine
    a duration.

Run:
  pytest tests/test_packaging_serialization.py -q
"""

import xml.dom.minidom as minidom

from bs4 import BeautifulSoup

import pipeline_core as pc


# --- convert_soup_to_html (F7) --------------------------------------------


def _opf_soup():
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">'
        '<metadata xmlns:opf="http://www.idpf.org/2007/opf">'
        '<dc:creator opf:role="aut" opf:file-as="Doe, John">John Doe</dc:creator>'
        "<dc:title>See opf:role docs</dc:title>"
        "</metadata></package>"
    )
    return BeautifulSoup(opf, "lxml-xml")


def test_convert_soup_preserves_opf_prefixed_attributes():
    out = pc.convert_soup_to_html(_opf_soup())
    assert "opf:role" in out
    assert "opf:file-as" in out


def test_convert_soup_preserves_text_containing_opf_substring():
    out = pc.convert_soup_to_html(_opf_soup())
    # The blind replace used to rewrite this to "See role docs".
    assert "See opf:role docs" in out


def test_convert_soup_output_is_well_formed_xml():
    # Whatever we emit must still parse as XML for downstream EPUB readers.
    out = pc.convert_soup_to_html(_opf_soup())
    minidom.parseString(out)  # raises on malformed XML


# --- audio_media_type / chunk selection (Part C + F2) ----------------------


def test_audio_media_type_by_extension():
    assert pc.audio_media_type({"audio_extension": ".m4a"}) == "audio/mp4"
    assert pc.audio_media_type({"audio_extension": ".mp3"}) == "audio/mpeg"
    assert pc.audio_media_type({"audio_extension": ".aac"}) == "audio/aac"
    assert pc.audio_media_type({"audio_extension": ".opus"}) == "audio/ogg"


def test_audio_media_type_unknown_falls_back():
    assert pc.audio_media_type({"audio_extension": ".weird"}) == "audio/mp4"


def test_is_chunk_basename_selects_only_numbered_chunks():
    assert pc.is_chunk_basename("000.m4a", ".m4a")
    assert pc.is_chunk_basename("123.m4a", ".m4a")
    # The copied source audiobook (arbitrary name) is not a chunk.
    assert not pc.is_chunk_basename("MyBook.m4a", ".m4a")
    # Wrong extension is not selected.
    assert not pc.is_chunk_basename("000.mp3", ".m4a")


def test_iter_audio_files_excludes_source(tmp_path):
    for name in ("000.m4a", "001.m4a", "MyBook.m4a", "notes.txt"):
        (tmp_path / name).write_bytes(b"\x00")
    chunks = pc.iter_audio_files({"folder_name": str(tmp_path), "audio_extension": ".m4a"})
    assert chunks == ["000.m4a", "001.m4a"]


# --- check_missing_transcripts honors audio_extension (F8) -----------------


def test_missing_transcripts_detects_non_m4a_extension():
    # With .aac chunks and no transcripts, the check must FAIL (not self-skip because
    # it only looked for *.m4a). skipped must be False and ok must be False.
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        for name in ("000.aac", "001.aac"):
            with open(os.path.join(folder, name), "wb") as handle:
                handle.write(b"\x00")
        book_info = {"folder_name": folder, "audio_extension": ".aac"}
        result = pc.check_missing_transcripts(pc.get_audio_inventory(book_info))
    assert result["skipped"] is False
    assert result["ok"] is False


def test_missing_transcripts_skips_only_when_no_chunks():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        book_info = {"folder_name": folder, "audio_extension": ".aac"}
        result = pc.check_missing_transcripts(pc.get_audio_inventory(book_info))
    assert result["skipped"] is True
    assert result["ok"] is True


# --- get_audio_duration (Part C) ------------------------------------------


def test_get_audio_duration_raises_on_probe_failure():
    # A nonexistent path makes ffprobe fail; this must raise rather than silently
    # returning 0.0 (which would drop a chunk's final segment).
    raised = False
    try:
        pc.get_audio_duration("/definitely/not/a/real/audio/file.m4a")
    except RuntimeError:
        raised = True
    assert raised



# --- folder-addressed engine (no cwd dependence) ----------------------------

_OPF_WITH_NAV = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="u">'
    '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title></metadata>'
    "<manifest>"
    '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
    '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
    '</manifest><spine><itemref idref="c1"/></spine></package>'
)


def _write_working_epub(path):
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("OEBPS/content.opf", _OPF_WITH_NAV)
        zf.writestr("OEBPS/nav.xhtml", "<html><body><nav/></body></html>")
        zf.writestr("OEBPS/ch1.xhtml", "<html><body><p>Hello there.</p></body></html>")


def test_preprocess_uses_book_folder_not_cwd(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    folder.mkdir()
    source = tmp_path / "Book.epub"
    _write_working_epub(source)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    book_info = {"folder_name": str(folder), "audio_file": "Book.m4b"}
    audio_file, epub_file, out_file = pc.preprocess(book_info, source)

    assert (audio_file, epub_file, out_file) == ("Book.m4b", "Book.epub", "Book.epub3")
    assert (folder / "Book.epub3").exists()
    assert not list(elsewhere.iterdir())
    assert book_info["opf_file"] == "OEBPS/content.opf"
    assert book_info["opf_dir"] == "OEBPS"


def test_merge_and_opf_rewrite_package_to_explicit_path(tmp_path, monkeypatch):
    import zipfile

    from bs4 import BeautifulSoup

    folder = tmp_path / "run"
    folder.mkdir()
    _write_working_epub(folder / "Book.epub3")
    (folder / "000.m4a").write_bytes(b"\x00\x01")
    (folder / "Book.m4b").write_bytes(b"\x00")  # copied source: must not be packaged
    smil_name = pc.make_overlay_basename("OEBPS/ch1.xhtml")
    (folder / smil_name).write_text(
        '<smil xmlns="http://www.w3.org/ns/SMIL" version="3.0"><body><seq>'
        '<par id="s1"><text src="../OEBPS/ch1.xhtml#s1"/>'
        '<audio src="../audio/000.m4a" clipBegin="0.000s" clipEnd="2.500s"/></par>'
        "</seq></body></smil>",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pc, "plan_audio_chunks",
        lambda book_info, audio_path=None: [
            {"id": 0, "start_time": "0", "end_time": "2.5", "output_name": "000.m4a"}
        ],
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    book_info = {
        "folder_name": str(folder),
        "audio_file": "Book.m4b",
        "out_file": "Book.epub3",
        "opf_dir": "OEBPS",
        "audio_extension": ".m4a",
        "matched_list": [{"json_file": "000.json", "html_file": "OEBPS/ch1.xhtml"}],
    }
    dest = tmp_path / "out" / "Book.epub"
    pc.merge_files(book_info)
    pc.post_processing_opf(book_info, dest)

    assert dest.exists()
    assert not (folder / "Book.epub3").exists()
    assert not list(elsewhere.iterdir())
    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        opf = BeautifulSoup(zf.read("OEBPS/content.opf"), "lxml-xml")
    assert {"audio/000.m4a", f"smil/{smil_name}", "OEBPS/readaloud.css"} <= names
    assert "audio/Book.m4b" not in names
    html_item = opf.find("item", attrs={"href": "ch1.xhtml"})
    assert html_item["media-overlay"] == pc.make_overlay_id("OEBPS/ch1.xhtml")
    durations = [m.get_text() for m in opf.find_all("meta", attrs={"property": "media:duration"})]
    assert durations == ["0:00:02.500", "0:00:02.500"]


# --- find_opf_path -----------------------------------------------------------


def test_find_opf_path_prefers_container_rootfile(tmp_path):
    import zipfile

    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("aaa_first.opf", "<package/>")  # sorts first; not the real one
        zf.writestr("OEBPS/content.opf", "<package/>")
        zf.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
    with zipfile.ZipFile(path) as zf:
        assert pc.find_opf_path(zf) == "OEBPS/content.opf"


def test_find_opf_path_falls_back_without_container(tmp_path):
    import zipfile

    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("OEBPS/content.opf", "<package/>")
        zf.writestr("OEBPS/ch1.xhtml", "<html/>")
    with zipfile.ZipFile(path) as zf:
        assert pc.find_opf_path(zf) == "OEBPS/content.opf"
    with zipfile.ZipFile(tmp_path / "empty.epub", "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
    with zipfile.ZipFile(tmp_path / "empty.epub") as zf:
        assert pc.find_opf_path(zf) is None


def test_inventory_records_unreadable_transcript(tmp_path):
    (tmp_path / "000.aac").write_bytes(b"\x00")
    (tmp_path / "000.json").write_text("{not json", encoding="utf-8")
    inventory = pc.get_audio_inventory({"folder_name": str(tmp_path), "audio_extension": ".aac"})
    assert inventory[0]["transcript_status"] == "unreadable_transcript"
    result = pc.check_missing_transcripts(inventory)
    assert result["ok"] is False and result["findings"][0]["issue"] == "unreadable_transcript"
