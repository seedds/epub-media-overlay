"""Tests for chunk planning and stale-chunk pruning in pipeline_core.

  - every post-split stage takes its chunk list from the plan, never from disk;
  - split removes NNN.* artifacts whose ordinal is not in the current plan (a
    plan change used to leave them to be re-transcribed, matched and packaged);
  - the copied source audiobook is never pruned, even with a chunk-like name;
  - chapter-based plans are snapped to cover the whole audio stream.

Run:
  pytest tests/test_split_plan.py -q
"""

import json

import pipeline_core as pc


def _book_info(folder, **overrides):
    info = {
        "folder_name": str(folder),
        "audio_file": "1984.m4a",  # chunk-like source name on purpose
        "audio_extension": ".m4a",
        "audio_codec": "copy",
        "chunk_seconds": 600,
    }
    info.update(overrides)
    return info


def _plan(*ordinals):
    return [
        {
            "id": i,
            "start_time": str(i * 600),
            "end_time": str((i + 1) * 600),
            "output_name": f"{ordinal}.m4a",
        }
        for i, ordinal in enumerate(ordinals)
    ]


def test_planned_chunk_basenames_ignores_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "plan_audio_chunks", lambda book_info, audio_path=None: _plan("000", "001"))
    (tmp_path / "007.m4a").write_bytes(b"\x00")
    assert pc.planned_chunk_basenames(_book_info(tmp_path)) == ["000.m4a", "001.m4a"]


def test_transcript_basename():
    assert pc.transcript_basename("012.m4a", ".m4a") == "012.json"
    assert pc.transcript_basename("012.aac", ".aac") == "012.json"


def test_split_prunes_stale_chunk_artifacts(tmp_path, monkeypatch):
    plan = _plan("000", "001")
    monkeypatch.setattr(pc, "plan_audio_chunks", lambda book_info, audio_path=None: plan)
    monkeypatch.setattr(pc, "is_audio_chunk_complete", lambda *a, **k: True)
    for name in (
        "000.m4a", "000.m4a.meta", "000.json", "000.json.meta",
        "001.m4a",
        "007.m4a", "007.m4a.meta", "007.json", "007.json.meta",
        "1984.m4a",  # the copied source audiobook
        "notes.txt",
    ):
        (tmp_path / name).write_bytes(b"\x00")

    stats = pc.split_audio(_book_info(tmp_path))

    assert stats["pruned_chunk_count"] == 1
    assert stats["reused_chunk_count"] == 2
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [
        "000.json", "000.json.meta", "000.m4a", "000.m4a.meta",
        "001.m4a", "1984.m4a", "notes.txt",
    ]


def test_prune_ignores_other_extensions(tmp_path, monkeypatch):
    # A stale chunk in a different extension is not ours to delete.
    (tmp_path / "007.mp3").write_bytes(b"\x00")
    removed = pc.prune_stale_chunk_artifacts(_book_info(tmp_path), _plan("000"))
    assert removed == 0 and (tmp_path / "007.mp3").exists()


def test_chapter_plan_snaps_to_stream_edges(monkeypatch):
    chapters = {
        "chapters": [
            {"start_time": "3.0", "end_time": "500.0"},
            {"start_time": "500.0", "end_time": "995.0"},
        ]
    }
    monkeypatch.setattr(pc.subprocess, "check_output", lambda *a, **k: json.dumps(chapters))
    monkeypatch.setattr(pc, "get_primary_audio_stream_duration", lambda path: 1000.0)

    plan = pc.plan_audio_chunks(_book_info("/nowhere"), "/nowhere/1984.m4a")

    assert plan[0]["start_time"] == "0.0"
    assert plan[-1]["end_time"] == "1000.0"
    assert plan[0]["end_time"] == "500.0"


def test_chapter_plan_keeps_sub_second_edges(monkeypatch):
    chapters = {"chapters": [{"start_time": "0.2", "end_time": "999.8"}]}
    monkeypatch.setattr(pc.subprocess, "check_output", lambda *a, **k: json.dumps(chapters))
    monkeypatch.setattr(pc, "get_primary_audio_stream_duration", lambda path: 1000.0)
    plan = pc.plan_audio_chunks(_book_info("/nowhere"), "/nowhere/1984.m4a")
    assert (plan[0]["start_time"], plan[0]["end_time"]) == ("0.2", "999.8")
