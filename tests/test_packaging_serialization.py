"""Tests for packaging-side serialization and audio helpers in pipeline_core.

Covers regressions:
  - convert_soup_to_html must not corrupt text/attributes containing the substring
    "opf:" (it previously did a blind str.replace("opf:", "") before a minidom
    pretty-print round-trip);
  - audio_media_type must reflect the configured extension, not a hardcoded
    "audio/mp4";
  - sorted_chunk_files / is_chunk_basename must select only NNN<ext> chunks, never
    the copied source audiobook;
  - get_audio_duration must raise (not silently return 0.0) when it cannot determine
    a duration and no fallback timing is available.

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_packaging_serialization.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_packaging_serialization.py
"""

# Bootstrap: allow running this file directly (see conftest.py rationale).
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

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


def test_sorted_chunk_files_excludes_source():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        for name in ("000.m4a", "001.m4a", "MyBook.m4a", "notes.txt"):
            with open(os.path.join(folder, name), "wb") as handle:
                handle.write(b"\x00")
        cwd = os.getcwd()
        try:
            os.chdir(folder)
            chunks = pc.sorted_chunk_files(".m4a")
        finally:
            os.chdir(cwd)
    assert chunks == ["000.m4a", "001.m4a"]


# --- test_missing_transcripts honors audio_extension (F8) ------------------


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
        result = pc.test_missing_transcripts(book_info)
    assert result["skipped"] is False
    assert result["ok"] is False


def test_missing_transcripts_skips_only_when_no_chunks():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        book_info = {"folder_name": folder, "audio_extension": ".aac"}
        result = pc.test_missing_transcripts(book_info)
    assert result["skipped"] is True
    assert result["ok"] is True


# --- get_audio_duration (Part C) ------------------------------------------


def test_get_audio_duration_raises_without_fallback_on_probe_failure():
    # A nonexistent path makes ffprobe fail; with no fallback matches this must raise
    # rather than silently returning 0.0 (which would drop a chunk's final segment).
    raised = False
    try:
        pc.get_audio_duration("/definitely/not/a/real/audio/file.m4a", [])
    except RuntimeError:
        raised = True
    assert raised


def test_get_audio_duration_uses_fallback_when_available():
    # With a fallback, the probe failure degrades gracefully (last end + 1.0).
    result = pc.get_audio_duration(
        "/definitely/not/a/real/audio/file.m4a", [{"end": 12.0}]
    )
    assert result == 13.0


if __name__ == "__main__":
    from _runner import run_module_tests

    raise SystemExit(run_module_tests(globals()))
