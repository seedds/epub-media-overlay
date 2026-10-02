"""Tests for the orchestrator's derived-artifact model (generate_epub_overlay).

Pins three behaviors:

  - a signature reset removes only known derived paths (never a wildcard over the
    work-dir root, which may hold unrelated EPUBs) and does remove the final output
    and the working EPUB, both of which are derived;
  - executing a stage invalidates exactly the downstream derived artifacts;
  - the package reconcile reuses a packaged EPUB only when it is byte-identical to
    the one this state recorded, so an older output with the right entry names is
    not accepted after a reset.

Run:
  pytest tests/test_orchestrator_state.py -q
"""

import json
import zipfile
from pathlib import Path

import generate_epub_overlay as geo
import pipeline_core as pc


def _config_and_paths(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    audio = src_dir / "book.m4b"
    epub = src_dir / "book.epub"
    audio.write_bytes(b"\x00")
    epub.write_bytes(b"\x00")
    work = tmp_path / "work"
    config = geo.PipelineConfig(
        audio=audio,
        epub=epub,
        output_dir=src_dir,
        work_dir=work,
        backend="mlx",
        model="m",
        language="en",
        audio_extension=".m4a",
        audio_codec="copy",
        audio_bitrate=None,
        audio_sample_rate=None,
        audio_channels=None,
        split_jobs=1,
        chunk_seconds=600,
        batch_size=1,
        mlx_cache_gb=None,
    )
    paths = geo.build_paths(config)
    paths.run_dir.mkdir(parents=True)
    return config, paths


def _touch(*files: Path) -> None:
    for f in files:
        f.write_bytes(b"x")


def test_reset_removes_only_known_derived_paths(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    unrelated = paths.root / "other.epub"
    smil = paths.run_dir / "ch1.smil"
    _touch(
        unrelated,
        paths.matched_list_path,
        paths.segmented_snapshot_path,
        paths.validation_path,
        paths.packaged_epub_path,
        paths.output_path,
        paths.working_epub_path,
        smil,
        paths.run_dir / "000.m4a",
        paths.run_dir / "000.json",
    )

    geo.reset_derived_artifacts(paths, config.epub)

    assert unrelated.exists(), "an unrelated EPUB in the work root must survive"
    assert config.epub.exists()
    for gone in (
        paths.matched_list_path,
        paths.segmented_snapshot_path,
        paths.validation_path,
        paths.packaged_epub_path,
        paths.output_path,
        paths.working_epub_path,
        smil,
    ):
        assert not gone.exists(), gone
    assert (paths.run_dir / "000.m4a").exists()
    assert (paths.run_dir / "000.json").exists()


def test_reset_never_deletes_source_epub_sharing_a_derived_path(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    # --work-dir pointing at the source folder: packaged path == source path.
    paths = geo.RuntimePaths(**{**paths.__dict__, "packaged_epub_path": config.epub})
    geo.reset_derived_artifacts(paths, config.epub)
    assert config.epub.exists()


def test_invalidate_downstream_is_scoped(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    smil = paths.run_dir / "ch1.smil"
    _touch(
        paths.matched_list_path,
        paths.segmented_snapshot_path,
        paths.working_epub_path,
        smil,
        paths.packaged_epub_path,
        paths.output_path,
        paths.validation_path,
    )
    state = geo.default_state({}, config, paths)
    for stage in geo.STAGES:
        geo.set_stage_state(state, stage, "success")

    geo.invalidate_downstream("smil", config, paths, state)

    assert paths.matched_list_path.exists()
    assert paths.segmented_snapshot_path.exists()
    assert paths.working_epub_path.exists()
    assert smil.exists(), "the executing stage's own output is kept"
    assert not paths.packaged_epub_path.exists()
    assert not paths.output_path.exists()
    assert not paths.validation_path.exists()
    assert geo.stage_status(state, "smil") == "success"
    assert geo.stage_status(state, "package") == "pending"
    assert geo.stage_status(state, "validate") == "pending"


def test_invalidate_after_prepare_is_a_noop(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    _touch(paths.matched_list_path, paths.packaged_epub_path)
    state = geo.default_state({}, config, paths)
    geo.invalidate_downstream("prepare", config, paths, state)
    assert paths.matched_list_path.exists() and paths.packaged_epub_path.exists()


# --- package reconcile -----------------------------------------------------

_OPF = (
    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
    "<metadata/><manifest>"
    '<item id="c" href="ch1.xhtml" media-type="application/xhtml+xml" media-overlay="o"/>'
    '<item id="o" href="../smil/ch1.smil" media-type="application/smil+xml"/>'
    '</manifest><spine><itemref idref="c"/></spine></package>'
)


def _processed_epub(path: Path, marker: bytes = b"") -> None:
    smil_name = pc.make_overlay_basename("OEBPS/ch1.xhtml")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("OEBPS/content.opf", _OPF)
        zf.writestr("OEBPS/ch1.xhtml", "<html/>")
        zf.writestr("audio/000.m4a", b"a" + marker)
        zf.writestr(f"smil/{smil_name}", "<smil/>")


def _prepared_state(config, paths):
    # Enough prepare artifacts for ensure_prepare_state_from_artifacts to pass.
    (paths.run_dir / config.audio.name).write_bytes(b"\x00")
    (paths.run_dir / config.epub.name).write_bytes(b"\x00")
    with zipfile.ZipFile(paths.working_epub_path, "w") as zf:
        zf.writestr("OEBPS/content.opf", _OPF)
    state = geo.default_state({}, config, paths)
    state["book_info"] = geo.build_prepared_book_info_from_artifacts(config, paths)
    state["artifacts"]["audio_files"] = ["000.m4a"]
    paths.matched_list_path.write_text(
        json.dumps([{"json_file": "000.json", "html_file": "OEBPS/ch1.xhtml"}])
    )
    return state


def test_package_reconcile_rejects_unrecorded_package(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    state = _prepared_state(config, paths)
    _processed_epub(paths.output_path)  # looks complete, but this state never built it
    assert geo.reconcile_stage_from_artifacts("package", config, paths, state) is None


def test_package_reconcile_accepts_only_the_recorded_bytes(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    state = _prepared_state(config, paths)
    _processed_epub(paths.packaged_epub_path)
    fingerprint = geo.content_fingerprint(paths.packaged_epub_path)
    geo.set_stage_state(
        state, "package", "success", result={"packaged_fingerprint": fingerprint}
    )

    result = geo.reconcile_stage_from_artifacts("package", config, paths, state)
    assert result is not None and result["packaged_fingerprint"] == fingerprint
    assert paths.output_path.exists(), "missing output is backfilled from the packaged copy"

    # A rebuilt package with different bytes is not the recorded one.
    _processed_epub(paths.packaged_epub_path, marker=b"changed")
    _processed_epub(paths.output_path, marker=b"changed")
    assert geo.reconcile_stage_from_artifacts("package", config, paths, state) is None


# --- match reconcile -------------------------------------------------------


def _state_for_match(config, paths, matched, audio_files, recorded):
    state = _prepared_state(config, paths)
    state["artifacts"]["audio_files"] = audio_files
    if recorded is not None:
        state["artifacts"]["match_transcript_files"] = recorded
    paths.matched_list_path.write_text(json.dumps(matched))
    return state


def test_match_reconcile_accepts_unmatched_intro_transcript(tmp_path):
    # 000.json (an intro chunk) matched nothing; the match is still complete.
    config, paths = _config_and_paths(tmp_path)
    matched = [{"json_file": "001.json", "html_file": "OEBPS/ch1.xhtml"}]
    state = _state_for_match(
        config, paths, matched, ["000.m4a", "001.m4a"], ["000.json", "001.json"]
    )
    result = geo.reconcile_stage_from_artifacts("match", config, paths, state)
    assert result is not None and result["match_count"] == 1


def test_match_reconcile_rejects_when_transcript_set_changed(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    matched = [{"json_file": "001.json", "html_file": "OEBPS/ch1.xhtml"}]
    # A third transcript appeared since the match ran.
    state = _state_for_match(
        config, paths, matched, ["000.m4a", "001.m4a", "002.m4a"], ["000.json", "001.json"]
    )
    assert geo.reconcile_stage_from_artifacts("match", config, paths, state) is None


def test_match_reconcile_rejects_without_recorded_inputs(tmp_path):
    config, paths = _config_and_paths(tmp_path)
    matched = [{"json_file": "001.json", "html_file": "OEBPS/ch1.xhtml"}]
    state = _state_for_match(config, paths, matched, ["000.m4a", "001.m4a"], None)
    assert geo.reconcile_stage_from_artifacts("match", config, paths, state) is None


# --- split reconcile prunes stale chunks ---------------------------------------


def test_split_reconcile_prunes_chunks_outside_the_plan(tmp_path, monkeypatch):
    config, paths = _config_and_paths(tmp_path)
    state = _prepared_state(config, paths)
    plan = [{"id": 0, "start_time": "0", "end_time": "600", "output_name": "000.m4a"}]
    monkeypatch.setattr(pc, "plan_audio_chunks", lambda book_info, audio_path=None: plan)
    monkeypatch.setattr(pc, "is_audio_chunk_complete", lambda *a, **k: True)
    for name in ("000.m4a", "000.json", "099.m4a", "099.json", "099.json.meta"):
        (paths.run_dir / name).write_bytes(b"\x00")

    result = geo.reconcile_stage_from_artifacts("split", config, paths, state)

    assert result is not None and result["pruned_chunk_count"] == 1
    assert not (paths.run_dir / "099.m4a").exists()
    assert not (paths.run_dir / "099.json").exists()
    assert (paths.run_dir / "000.json").exists()


# --- the working EPUB survives stage invalidation --------------------------------


def test_invalidate_after_split_keeps_working_epub(tmp_path):
    # Match (and every later stage of the same run) still needs the working EPUB;
    # deleting it here made every fresh run fail at match.
    config, paths = _config_and_paths(tmp_path)
    _touch(paths.working_epub_path, paths.segmented_snapshot_path, paths.matched_list_path)
    state = geo.default_state({}, config, paths)
    geo.invalidate_downstream("split", config, paths, state)
    assert paths.working_epub_path.exists()
    assert not paths.segmented_snapshot_path.exists()
    assert not paths.matched_list_path.exists()


def test_segment_reconcile_ignores_ids_only_in_working_epub(tmp_path, monkeypatch):
    # Segment ids in the working EPUB alone may have been marked against an older
    # match; only the snapshot (deleted on every upstream re-run) is trusted.
    config, paths = _config_and_paths(tmp_path)
    state = _prepared_state(config, paths)
    monkeypatch.setattr(geo, "epub_contains_segment_ids", lambda path, html: path == paths.working_epub_path)
    assert geo.reconcile_stage_from_artifacts("segment", config, paths, state) is None


# --- end to end: the stage sequence with a stubbed engine -------------------------


_E2E_CONTAINER = (
    '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
    'version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)
_E2E_OPF = (
    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
    "<metadata/><manifest>"
    '<item id="c" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
    '</manifest><spine><itemref idref="c"/></spine></package>'
)


def _stub_engine(monkeypatch, calls):
    """Replace ffmpeg/ffprobe/ASR with stand-ins; everything else runs for real."""
    plan = [{"id": 0, "start_time": "0", "end_time": "50", "output_name": "000.m4a"}]

    def split(book_info):
        calls["split"] += 1
        Path(pc.resolve_book_path(book_info, "000.m4a")).write_bytes(b"a")
        return {"created_chunk_count": 1}

    def transcribe(book_info):
        calls["transcribe"] += 1
        words = [{"word": "word", "start": i, "end": i + 0.5} for i in range(50)]
        Path(pc.resolve_book_path(book_info, "000.json")).write_text(json.dumps({"word_segments": words}))

    real_link = pc.link_html_with_audio

    def link(book_info):
        calls["match"] += 1
        return real_link(book_info)

    monkeypatch.setattr(geo, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(geo, "release_models", lambda: None)
    monkeypatch.setattr(pc, "plan_audio_chunks", lambda book_info, audio_path=None: plan)
    monkeypatch.setattr(
        pc, "is_audio_chunk_complete",
        lambda book_info, chunk, tolerance=None: Path(pc.resolve_book_path(book_info, chunk["output_name"])).exists(),
    )
    monkeypatch.setattr(
        pc, "is_transcript_complete",
        lambda book_info, name: Path(pc.resolve_book_path(book_info, "000.json")).exists(),
    )
    monkeypatch.setattr(pc, "split_audio", split)
    monkeypatch.setattr(pc, "transcribe_audio", transcribe)
    monkeypatch.setattr(pc, "link_html_with_audio", link)
    monkeypatch.setattr(pc, "get_audio_duration", lambda path: 50.0)


def _e2e_config(tmp_path, **overrides):
    config, _paths = _config_and_paths(tmp_path)
    with zipfile.ZipFile(config.epub, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", _E2E_CONTAINER)
        zf.writestr("OEBPS/content.opf", _E2E_OPF)
        zf.writestr("OEBPS/ch1.xhtml", "<html><head></head><body><p>" + "word " * 50 + "</p></body></html>")
    return geo.PipelineConfig(**{**config.__dict__, "output_dir": tmp_path / "out", **overrides})


def _stage_statuses(config):
    state = json.loads((config.work_dir / "state.json").read_text())
    return {stage: state["stages"][stage]["status"] for stage in geo.STAGES}


def test_pipeline_first_run_resume_and_reset(tmp_path, monkeypatch):
    calls = {"split": 0, "transcribe": 0, "match": 0}
    _stub_engine(monkeypatch, calls)
    config = _e2e_config(tmp_path)
    output = geo.build_paths(config).output_path

    # First run: every stage executes, and match finds the working EPUB it reads.
    assert geo.run_pipeline(config) == 0
    assert set(_stage_statuses(config).values()) == {"success"}
    assert calls == {"split": 1, "transcribe": 1, "match": 1}
    assert output.exists()

    # Resume: split, transcribe and match are reused.
    assert geo.run_pipeline(config) == 0
    assert calls == {"split": 1, "transcribe": 1, "match": 1}

    # Signature reset (batch size changed): chunks/transcripts survive, match re-runs
    # against freshly prepared inputs and the run completes.
    reset = geo.PipelineConfig(**{**config.__dict__, "batch_size": 2})
    assert geo.run_pipeline(reset) == 0
    assert calls == {"split": 1, "transcribe": 1, "match": 2}
    assert set(_stage_statuses(reset).values()) == {"success"}
    assert output.exists()
