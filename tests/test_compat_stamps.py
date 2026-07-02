"""Unit tests for the artifact compatibility-stamp mechanism in pipeline_core.

The split and transcribe stages are expensive, so a config/signature change
preserves the run/ chunks and transcripts and relies on reconcile to re-validate
each one against the *current* config. These tests pin the stamp contract that makes
that re-validation real (closing the "existence/duration-only reconcile" holes):

  - a produced chunk/transcript carries a `.meta` stamp describing its config;
  - is_transcript_complete accepts a transcript only when its stamp matches the
    current model/language/backend AND its parent chunk stamp;
  - a stampless artifact (e.g. from a crash before the stamp was written, or from a
    pre-stamp run) is rejected so it gets re-derived;
  - atomic_write_json_local never leaves a truncated file at the final path.

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_compat_stamps.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_compat_stamps.py
"""

# Bootstrap: allow running this file directly (see conftest.py rationale).
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import contextlib
import json
import os
import tempfile

import pipeline_core as pc


# --- fixtures --------------------------------------------------------------


@contextlib.contextmanager
def _tmp():
    """Fresh temp dir usable both under pytest and the standalone __main__ runner."""
    with tempfile.TemporaryDirectory() as folder:
        yield folder


def _book_info(folder, **overrides):
    info = {
        "folder_name": str(folder),
        "audio_extension": ".m4a",
        "audio_codec": "aac",
        "audio_bitrate": "64k",
        "audio_sample_rate": 24000,
        "audio_channels": 1,
        "chunk_seconds": 600,
        "backend": "mlx",
        "model": "mlx-community/whisper-large-v3-mlx",
        "language": "en",
        "batch_size": 4,
    }
    info.update(overrides)
    return info


def _chunk_info(idx=0, start="0", end="600"):
    return {
        "id": idx,
        "start_time": start,
        "end_time": end,
        "output_name": f"{str(idx).zfill(3)}.m4a",
    }


def _make_chunk_with_stamp(folder, book_info, chunk_info):
    """Create a fake chunk file plus its compatibility stamp (as split would)."""
    chunk_path = os.path.join(str(folder), chunk_info["output_name"])
    with open(chunk_path, "wb") as handle:
        handle.write(b"\x00\x00")  # bytes are irrelevant to stamp logic
    pc.write_compatibility_stamp(
        chunk_path, pc.chunk_compatibility_stamp(book_info, chunk_info)
    )
    return chunk_path


def _make_transcript_with_stamp(folder, book_info, chunk_info):
    """Create a transcript + stamp derived from the chunk's stamp (as transcribe would)."""
    chunk_path = os.path.join(str(folder), chunk_info["output_name"])
    json_path = os.path.join(
        str(folder), chunk_info["output_name"].replace(".m4a", ".json")
    )
    pc.atomic_write_json_local(json_path, {"word_segments": [{"word": "hi"}]})
    stamp = pc.transcript_compatibility_stamp(
        book_info, pc.read_compatibility_stamp(chunk_path)
    )
    pc.write_compatibility_stamp(json_path, stamp)
    return json_path


# --- stamp identity --------------------------------------------------------


def test_chunk_stamp_reflects_audio_config():
    with _tmp() as folder:
        info = _book_info(folder)
        stamp = pc.chunk_compatibility_stamp(info, _chunk_info())
        assert stamp["audio_codec"] == "aac"
        assert stamp["audio_bitrate"] == "64k"
        assert stamp["audio_sample_rate"] == 24000
        assert stamp["chunk_seconds"] == 600


def test_stamp_roundtrip_matches():
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        path = _make_chunk_with_stamp(folder, info, chunk)
        assert pc.stamp_matches(path, pc.chunk_compatibility_stamp(info, chunk))


def test_missing_stamp_never_matches():
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        chunk_path = os.path.join(str(folder), chunk["output_name"])
        with open(chunk_path, "wb") as handle:
            handle.write(b"\x00")
        # No .meta written -> stampless artifact must not be accepted.
        assert not pc.stamp_matches(chunk_path, pc.chunk_compatibility_stamp(info, chunk))


# --- is_transcript_complete (transcribe-stage reconcile core) --------------


def test_transcript_complete_when_config_matches():
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        _make_chunk_with_stamp(folder, info, chunk)
        _make_transcript_with_stamp(folder, info, chunk)
        assert pc.is_transcript_complete(info, chunk["output_name"])


def test_transcript_stale_when_model_changes():
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        _make_chunk_with_stamp(folder, info, chunk)
        _make_transcript_with_stamp(folder, info, chunk)
        # A rerun with a different model must NOT reuse the old transcript.
        changed = _book_info(folder, model="mlx-community/whisper-small")
        assert not pc.is_transcript_complete(changed, chunk["output_name"])


def test_transcript_stale_when_language_changes():
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        _make_chunk_with_stamp(folder, info, chunk)
        _make_transcript_with_stamp(folder, info, chunk)
        changed = _book_info(folder, language="fr")
        assert not pc.is_transcript_complete(changed, chunk["output_name"])


def test_transcript_stale_when_parent_chunk_recut():
    # Re-cutting the chunk (e.g. new chunk_seconds) rewrites the chunk stamp; the old
    # transcript, whose stamp embeds the OLD chunk stamp, must be rejected.
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        _make_chunk_with_stamp(folder, info, chunk)
        _make_transcript_with_stamp(folder, info, chunk)

        recut = _book_info(folder, chunk_seconds=300)
        _make_chunk_with_stamp(folder, recut, chunk)  # rewrite chunk stamp only
        assert not pc.is_transcript_complete(recut, chunk["output_name"])


def test_transcript_incomplete_without_stamp():
    # Simulate a crash-truncated transcript: JSON present but no stamp (stamp is
    # written last). Reconcile must treat it as incomplete -> re-transcribe.
    with _tmp() as folder:
        info = _book_info(folder)
        chunk = _chunk_info()
        _make_chunk_with_stamp(folder, info, chunk)
        json_path = os.path.join(str(folder), "000.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write('{"word_segments": [')  # truncated, no stamp
        assert not pc.is_transcript_complete(info, chunk["output_name"])


# --- atomic write ----------------------------------------------------------


def test_atomic_write_json_local_no_partial_on_replace():
    with _tmp() as folder:
        target = os.path.join(folder, "007.json")
        pc.atomic_write_json_local(target, {"a": 1, "b": [1, 2, 3]})
        with open(target, encoding="utf-8") as handle:
            assert json.load(handle) == {"a": 1, "b": [1, 2, 3]}
        # No leftover temp file.
        assert not os.path.exists(target + ".tmp")


if __name__ == "__main__":
    from _runner import run_module_tests

    raise SystemExit(run_module_tests(globals()))
