#!/usr/bin/env python3
"""Resumable EPUB media-overlay generation pipeline.

This script wraps the local pipeline and `mark_sentence.py` workflow in a
production-oriented CLI with explicit inputs, persistent state, and automatic
resume behavior after interruption.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mark_sentence import ensure_nltk_resources
from transcription_backend import (
    default_model_for_backend,
    detect_transcription_backend,
    required_module_for_backend,
)

STAGES = (
    "prepare",
    "split",
    "transcribe",
    "match",
    "segment",
    "smil",
    "package",
    "validate",
)

SEGMENT_ID_RE = re.compile(r'id="c[^"]+-segment\d+"')
DEFAULT_AAC_AUDIO_BITRATE = "64k"
DEFAULT_AAC_AUDIO_SAMPLE_RATE = 24000
DEFAULT_AAC_AUDIO_CHANNELS = 1


@dataclass(frozen=True)
class PipelineConfig:
    audio: Path
    epub: Path
    output_dir: Path
    output_path: Path
    work_dir: Path
    fresh: bool
    backend: str
    model: str
    language: str
    audio_extension: str
    audio_codec: str
    audio_bitrate: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    split_jobs: int
    chunk_seconds: int


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    run_dir: Path
    logs_dir: Path
    state_path: Path
    matched_list_path: Path
    segmented_snapshot_path: Path
    validation_path: Path
    packaged_epub_path: Path
    output_path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fingerprint_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f"{dest.name}.tmp")
    shutil.copy2(src, tmp_path)
    os.replace(tmp_path, dest)


def delete_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def default_state(signature: dict[str, Any], config: PipelineConfig, paths: RuntimePaths) -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "signature": signature,
        "inputs": {
            "audio": str(config.audio),
            "epub": str(config.epub),
        },
        "config": {
            "backend": config.backend,
            "model": config.model,
            "language": config.language,
            "audio_extension": config.audio_extension,
            "audio_codec": config.audio_codec,
            "audio_bitrate": config.audio_bitrate,
            "audio_sample_rate": config.audio_sample_rate,
            "audio_channels": config.audio_channels,
            "split_jobs": config.split_jobs,
            "chunk_seconds": config.chunk_seconds,
            "output_path": str(paths.output_path),
            "work_dir": str(paths.root),
        },
        "book_info": {},
        "artifacts": {},
        "stages": {
            stage: {
                "status": "pending",
                "updated_at": now_iso(),
            }
            for stage in STAGES
        },
    }


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(paths: RuntimePaths, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    atomic_write_json(paths.state_path, state)


def resolve_audio_bitrate(audio_codec: str, audio_bitrate: str | None) -> str | None:
    if audio_codec == "aac" and audio_bitrate is None:
        return DEFAULT_AAC_AUDIO_BITRATE
    return audio_bitrate


def resolve_audio_sample_rate(audio_codec: str, audio_sample_rate: int | None) -> int | None:
    if audio_codec == "aac" and audio_sample_rate is None:
        return DEFAULT_AAC_AUDIO_SAMPLE_RATE
    return audio_sample_rate


def resolve_audio_channels(audio_codec: str, audio_channels: int | None) -> int | None:
    if audio_codec == "aac" and audio_channels is None:
        return DEFAULT_AAC_AUDIO_CHANNELS
    return audio_channels


def resolve_split_jobs(audio_codec: str, split_jobs: int | None) -> int:
    cpu_count = os.cpu_count() or 1
    if split_jobs is None:
        split_jobs = max(1, cpu_count - 2) if audio_codec == "aac" else 1
    return max(1, min(split_jobs, cpu_count))


def stage_status(state: dict[str, Any], stage: str) -> str:
    return state.get("stages", {}).get(stage, {}).get("status", "pending")


def set_stage_state(
    state: dict[str, Any],
    stage: str,
    status: str,
    **extra: Any,
) -> None:
    record = state.setdefault("stages", {}).setdefault(stage, {})
    record.clear()
    record.update({"status": status, "updated_at": now_iso()})
    record.update(extra)


def build_signature(config: PipelineConfig) -> dict[str, Any]:
    return {
        "audio": fingerprint_file(config.audio),
        "epub": fingerprint_file(config.epub),
        "backend": config.backend,
        "model": config.model,
        "language": config.language,
        "audio_extension": config.audio_extension,
        "audio_codec": config.audio_codec,
        "audio_bitrate": config.audio_bitrate,
        "audio_sample_rate": config.audio_sample_rate,
        "audio_channels": config.audio_channels,
        "chunk_seconds": config.chunk_seconds,
    }


def ensure_command(command: str) -> None:
    if shutil.which(command):
        return
    raise RuntimeError(f"Required command not found in PATH: {command}")


def configure_logging(paths: RuntimePaths) -> logging.Logger:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_epub_overlay")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(paths.logs_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Generate a media-overlay EPUB from an audiobook and EPUB source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Input audiobook file")
    parser.add_argument("--epub", required=True, help="Input .epub ebook file")
    parser.add_argument(
        "--output-dir",
        help="Directory for the final EPUB. Defaults to the source EPUB folder",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory used for persistent state and intermediate artifacts",
    )
    parser.add_argument(
        "--model",
        help=(
            "Transcription model identifier. Defaults to "
            "mlx-community/whisper-turbo on Apple Silicon macOS and small elsewhere"
        ),
    )
    parser.add_argument("--language", default="en", help="Transcription language code")
    parser.add_argument(
        "--audio-extension",
        default=".m4a",
        help="Audio chunk extension produced during splitting",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("copy", "aac"),
        default="copy",
        help="Codec used for split audio chunks",
    )
    parser.add_argument(
        "--audio-bitrate",
        help=(
            "AAC bitrate for split audio chunks, such as 64k or 128k. "
            f"Defaults to {DEFAULT_AAC_AUDIO_BITRATE} with --audio-codec aac"
        ),
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        help=(
            "AAC sample rate for split audio chunks in Hz. "
            f"Defaults to {DEFAULT_AAC_AUDIO_SAMPLE_RATE} with --audio-codec aac"
        ),
    )
    parser.add_argument(
        "--audio-channels",
        type=int,
        help=(
            "AAC channel count for split audio chunks, such as 1 for mono or 2 for stereo. "
            f"Defaults to {DEFAULT_AAC_AUDIO_CHANNELS} with --audio-codec aac"
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=600,
        help="Fixed chunk length in seconds for audio files without chapters",
    )
    parser.add_argument(
        "--split-jobs",
        type=int,
        help=(
            "Parallel ffmpeg jobs used during audio splitting. Defaults to cpu_count - 2 "
            "with --audio-codec aac and 1 with --audio-codec copy, capped at CPU count"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing working state and restart from scratch instead of resuming automatically",
    )
    args = parser.parse_args()

    if args.audio_codec == "copy":
        copy_only_flags = []
        if args.audio_bitrate is not None:
            copy_only_flags.append("--audio-bitrate")
        if args.audio_sample_rate is not None:
            copy_only_flags.append("--audio-sample-rate")
        if args.audio_channels is not None:
            copy_only_flags.append("--audio-channels")
        if copy_only_flags:
            parser.error(f"{', '.join(copy_only_flags)} require --audio-codec aac")

    if args.audio_sample_rate is not None and args.audio_sample_rate <= 0:
        parser.error("--audio-sample-rate must be a positive integer")
    if args.audio_channels is not None and args.audio_channels <= 0:
        parser.error("--audio-channels must be a positive integer")
    if args.split_jobs is not None and args.split_jobs <= 0:
        parser.error("--split-jobs must be a positive integer")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be a positive integer")

    audio = Path(args.audio).expanduser().resolve()
    epub = Path(args.epub).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else epub.parent
    )
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output_dir / f".{epub.stem}.epubmo"
    )
    output_path = output_dir / f"{epub.stem}.media-overlay.epub"
    backend = detect_transcription_backend()
    model = args.model or default_model_for_backend(backend)
    audio_bitrate = resolve_audio_bitrate(args.audio_codec, args.audio_bitrate)
    audio_sample_rate = resolve_audio_sample_rate(args.audio_codec, args.audio_sample_rate)
    audio_channels = resolve_audio_channels(args.audio_codec, args.audio_channels)
    split_jobs = resolve_split_jobs(args.audio_codec, args.split_jobs)

    return PipelineConfig(
        audio=audio,
        epub=epub,
        output_dir=output_dir,
        output_path=output_path,
        work_dir=work_dir,
        fresh=args.fresh,
        backend=backend,
        model=model,
        language=args.language,
        audio_extension=args.audio_extension,
        audio_codec=args.audio_codec,
        audio_bitrate=audio_bitrate,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        split_jobs=split_jobs,
        chunk_seconds=args.chunk_seconds,
    )


def build_paths(config: PipelineConfig) -> RuntimePaths:
    run_dir = config.work_dir / "run"
    source_output_name = f"{config.epub.stem}.epub"
    return RuntimePaths(
        root=config.work_dir,
        run_dir=run_dir,
        logs_dir=config.work_dir / "logs",
        state_path=config.work_dir / "state.json",
        matched_list_path=config.work_dir / "matched_list.json",
        segmented_snapshot_path=config.work_dir / "segmented.epub3",
        validation_path=config.work_dir / "validation.json",
        packaged_epub_path=config.work_dir / source_output_name,
        output_path=config.output_path,
    )


def load_pipeline_module():
    try:
        return importlib.import_module("pipeline_core")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. Install the pipeline requirements before running."
        ) from exc


def preflight(config: PipelineConfig, logger: logging.Logger) -> None:
    if not config.audio.exists():
        raise FileNotFoundError(f"Input audio not found: {config.audio}")
    if not config.epub.exists():
        raise FileNotFoundError(f"Input EPUB not found: {config.epub}")
    if config.epub.suffix.lower() != ".epub":
        raise ValueError(f"Expected an .epub input, got: {config.epub}")
    ensure_command("ffprobe")
    ensure_command("ffmpeg")
    for module_name in ("bs4", "lxml", "tqdm", required_module_for_backend(config.backend)):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing Python dependency: {module_name}. Install the pipeline requirements before running."
            ) from exc

    ensure_nltk_resources(logger)


def initialize_state(config: PipelineConfig, paths: RuntimePaths) -> tuple[dict[str, Any], str]:
    signature = build_signature(config)
    existing_state = load_state(paths.state_path)
    mode = "new"

    if config.fresh and paths.root.exists():
        shutil.rmtree(paths.root)
        existing_state = None
        mode = "fresh_restart"

    if existing_state:
        if existing_state.get("signature") != signature:
            shutil.rmtree(paths.root, ignore_errors=True)
            existing_state = None
            mode = "signature_reset"
        else:
            mode = "resume"
    elif config.fresh:
        mode = "fresh_restart"

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    state = existing_state or default_state(signature, config, paths)
    save_state(paths, state)
    return state, mode


def describe_run_mode(mode: str) -> str:
    if mode == "resume":
        return "automatic resume from existing compatible work state"
    if mode == "fresh_restart":
        return "fresh restart requested with --fresh"
    if mode == "signature_reset":
        return "fresh restart because the saved work state does not match the current inputs"
    return "new run"


def log_run_header(
    logger: logging.Logger,
    config: PipelineConfig,
    paths: RuntimePaths,
    mode: str,
) -> None:
    logger.info("Starting EPUB media-overlay pipeline")
    logger.info("Run mode: %s", describe_run_mode(mode))
    logger.info("Input audio: %s", config.audio)
    logger.info("Input EPUB: %s", config.epub)
    logger.info("Transcription backend: %s", config.backend)
    logger.info("Transcription model: %s", config.model)
    logger.info("Transcription language: %s", config.language)
    logger.info("Fixed chunk length: %ss", config.chunk_seconds)
    logger.info("Split audio codec: %s", config.audio_codec)
    logger.info("Split jobs: %d (cpu count: %d)", config.split_jobs, os.cpu_count() or 1)
    if config.audio_codec == "aac":
        logger.info("Split audio bitrate: %s", config.audio_bitrate or "source default")
        logger.info(
            "Split audio sample rate: %s",
            config.audio_sample_rate or "source default",
        )
        logger.info(
            "Split audio channels: %s",
            config.audio_channels or "source default",
        )
    logger.info("Output EPUB: %s", config.output_path)
    logger.info("Work dir: %s", paths.root)
    logger.info("Detailed log: %s", paths.logs_dir / "pipeline.log")


def log_run_summary(
    logger: logging.Logger,
    config: PipelineConfig,
    paths: RuntimePaths,
    started_at: float,
    validation_ok: bool,
) -> None:
    logger.info("Run completed in %.1fs", time.monotonic() - started_at)
    logger.info("Final EPUB: %s", config.output_path)
    logger.info("Validation result: %s", "passed" if validation_ok else "failed")
    logger.info("Detailed log: %s", paths.logs_dir / "pipeline.log")


def build_book_info(state: dict[str, Any], paths: RuntimePaths, matched_list: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    book_info = dict(state.get("book_info", {}))
    if not book_info:
        raise RuntimeError("Working book metadata is missing. Run the prepare stage first.")
    book_info["folder_name"] = str(paths.run_dir)
    if matched_list is not None:
        book_info["matched_list"] = matched_list
    return book_info


def refresh_working_epub(legacy: Any, book_info: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    refreshed = {
        "folder_name": str(run_dir),
        "audio_file": book_info["audio_file"],
        "audio_extension": book_info["audio_extension"],
        "audio_codec": book_info.get("audio_codec", "copy"),
        "audio_bitrate": book_info.get("audio_bitrate"),
        "audio_sample_rate": book_info.get("audio_sample_rate"),
        "audio_channels": book_info.get("audio_channels"),
        "split_jobs": book_info.get("split_jobs", 1),
        "chunk_seconds": book_info.get("chunk_seconds", 600),
        "backend": book_info.get("backend"),
        "model": book_info.get("model"),
        "language": book_info.get("language", "en"),
    }
    with pushd(run_dir):
        audio_file, _source_epub, out_file, root_level = legacy.preprocess(refreshed)
    refreshed.update(
        {
            "audio_file": audio_file,
            "epub_file": out_file,
            "out_file": out_file,
            "root_level": root_level,
        }
    )
    return refreshed


def inspect_working_epub(working_epub_path: Path) -> tuple[str, int] | None:
    if not working_epub_path.exists():
        return None
    try:
        with zipfile.ZipFile(working_epub_path, "r") as zf:
            opf_file = next((name for name in zf.namelist() if name.endswith(".opf")), None)
    except (OSError, zipfile.BadZipFile):
        return None
    if not opf_file:
        return None
    return opf_file, len(list(Path(opf_file).parents))


def build_prepared_book_info_from_artifacts(
    config: PipelineConfig,
    paths: RuntimePaths,
) -> dict[str, Any] | None:
    copied_audio = paths.run_dir / config.audio.name
    copied_epub = paths.run_dir / config.epub.name
    working_epub = paths.run_dir / config.epub.with_suffix(".epub3").name
    opf_details = inspect_working_epub(working_epub)
    if not copied_audio.exists() or not copied_epub.exists() or opf_details is None:
        return None

    opf_file, root_level = opf_details
    opf_dir = str(Path(opf_file).parent)
    return {
        "folder_name": str(paths.run_dir),
        "audio_file": copied_audio.name,
        "audio_extension": config.audio_extension,
        "audio_codec": config.audio_codec,
        "audio_bitrate": config.audio_bitrate,
        "audio_sample_rate": config.audio_sample_rate,
        "audio_channels": config.audio_channels,
        "split_jobs": config.split_jobs,
        "chunk_seconds": config.chunk_seconds,
        "backend": config.backend,
        "model": config.model,
        "language": config.language,
        "epub_file": working_epub.name,
        "out_file": working_epub.name,
        "opf_file": opf_file,
        "opf_dir": opf_dir,
        "root_level": root_level,
    }


def ensure_prepare_state_from_artifacts(
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    existing = state.get("book_info")
    if existing:
        if existing.get("split_jobs") != config.split_jobs:
            existing = dict(existing)
            existing["split_jobs"] = config.split_jobs
            state["book_info"] = existing
        audio_path = paths.run_dir / existing.get("audio_file", "")
        working_epub_path = paths.run_dir / existing.get("out_file", "")
        copied_epub_path = paths.run_dir / config.epub.name
        if audio_path.exists() and copied_epub_path.exists() and inspect_working_epub(working_epub_path) is not None:
            return dict(existing)
    recovered = build_prepared_book_info_from_artifacts(config, paths)
    if recovered is None:
        return None
    state["book_info"] = recovered
    state.setdefault("artifacts", {})["prepared_epub"] = str(paths.run_dir / recovered["out_file"])
    return dict(recovered)


def expected_audio_files(legacy: Any, book_info: dict[str, Any], run_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    chunk_plan = legacy.plan_audio_chunks(book_info, run_dir / book_info["audio_file"])
    names = [chunk["output_name"] for chunk in chunk_plan]
    return names, chunk_plan


def list_transcripts(run_dir: Path) -> list[str]:
    return sorted(path.name for path in run_dir.glob("*.json"))


def load_matched_list(paths: RuntimePaths) -> list[dict[str, Any]]:
    if not paths.matched_list_path.exists():
        raise RuntimeError(f"Matched-list cache not found: {paths.matched_list_path}")
    return json.loads(paths.matched_list_path.read_text(encoding="utf-8"))


def expected_transcript_files(audio_files: list[str]) -> list[str]:
    return [Path(name).with_suffix(".json").name for name in audio_files]


def matched_json_files(matched_list: list[dict[str, Any]], legacy: Any) -> list[str]:
    seen = set()
    json_files = []
    for item in legacy.normalize_matched_list(matched_list):
        json_file = item["json_file"]
        if json_file in seen:
            continue
        seen.add(json_file)
        json_files.append(json_file)
    return json_files


def expected_smil_files(matched_list: list[dict[str, Any]], legacy: Any) -> list[str]:
    html_files = []
    seen = set()
    for item in legacy.normalize_matched_list(matched_list):
        html_file = item["html_file"]
        if html_file in seen:
            continue
        seen.add(html_file)
        html_files.append(html_file)
    return [legacy.make_overlay_basename(html_file) for html_file in html_files]


def matched_html_files(matched_list: list[dict[str, Any]], legacy: Any) -> list[str]:
    seen = set()
    html_files = []
    for item in legacy.normalize_matched_list(matched_list):
        html_file = item["html_file"]
        if html_file in seen:
            continue
        seen.add(html_file)
        html_files.append(html_file)
    return html_files


def epub_contains_segment_ids(epub_path: Path, html_files: list[str]) -> bool:
    if not epub_path.exists() or not html_files:
        return False
    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            for html_file in html_files:
                raw_html = zf.read(html_file)
                try:
                    html_content = raw_html.decode("utf-8")
                except UnicodeDecodeError:
                    html_content = raw_html.decode("latin-1", errors="replace")
                if not SEGMENT_ID_RE.search(html_content):
                    return False
    except (KeyError, OSError, zipfile.BadZipFile):
        return False
    return True


def restore_segmented_working_epub(state: dict[str, Any], paths: RuntimePaths) -> None:
    out_file = state.get("book_info", {}).get("out_file")
    if not out_file:
        raise RuntimeError("Cannot restore working EPUB without recorded out_file metadata")
    if not paths.segmented_snapshot_path.exists():
        raise RuntimeError("Segmented EPUB snapshot is missing")
    atomic_copy(paths.segmented_snapshot_path, paths.run_dir / out_file)


def reconcile_stage_from_artifacts(
    stage: str,
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
) -> dict[str, Any] | None:
    if stage == "prepare":
        book_info = ensure_prepare_state_from_artifacts(config, paths, state)
        if book_info is None:
            return None
        return {
            "audio_file": book_info["audio_file"],
            "source_epub": config.epub.name,
            "out_file": book_info["out_file"],
        }

    book_info = ensure_prepare_state_from_artifacts(config, paths, state)
    if book_info is None:
        return None

    if stage == "split":
        audio_files, chunks = expected_audio_files(legacy, book_info, paths.run_dir)
        if not audio_files:
            return None
        if any(not legacy.is_audio_chunk_complete(book_info, chunk) for chunk in chunks):
            return None
        state.setdefault("artifacts", {})["audio_files"] = audio_files
        return {
            "audio_files": audio_files,
            "chunk_count": len(chunks),
            "reused_chunk_count": len(audio_files),
            "regenerated_chunk_count": 0,
            "created_chunk_count": 0,
        }

    if stage == "transcribe":
        audio_files = state.get("artifacts", {}).get("audio_files")
        if not audio_files:
            audio_files, _chunks = expected_audio_files(legacy, book_info, paths.run_dir)
        if not audio_files:
            return None
        transcript_files = expected_transcript_files(audio_files)
        if any(not (paths.run_dir / name).exists() for name in transcript_files):
            return None
        state.setdefault("artifacts", {})["audio_files"] = audio_files
        state["artifacts"]["transcript_files"] = transcript_files
        return {"transcript_files": transcript_files}

    if stage == "match":
        matched_list = load_json_if_exists(paths.matched_list_path)
        if not matched_list:
            return None
        audio_files = state.get("artifacts", {}).get("audio_files")
        if not audio_files:
            audio_files, _chunks = expected_audio_files(legacy, book_info, paths.run_dir)
        transcript_files = expected_transcript_files(audio_files)
        if sorted(matched_json_files(matched_list, legacy)) != sorted(transcript_files):
            return None
        html_files = matched_html_files(matched_list, legacy)
        state.setdefault("artifacts", {})["audio_files"] = audio_files
        state["artifacts"]["transcript_files"] = transcript_files
        state["artifacts"]["matched_html_files"] = html_files
        return {"match_count": len(matched_list), "html_files": html_files}

    if stage == "segment":
        matched_list = load_json_if_exists(paths.matched_list_path)
        if not matched_list:
            return None
        html_files = matched_html_files(matched_list, legacy)
        if not html_files:
            return None
        working_epub_path = paths.run_dir / book_info["out_file"]
        snapshot_has_segments = epub_contains_segment_ids(paths.segmented_snapshot_path, html_files)
        working_has_segments = epub_contains_segment_ids(working_epub_path, html_files)
        if not snapshot_has_segments and not working_has_segments:
            return None
        if snapshot_has_segments and not working_has_segments:
            atomic_copy(paths.segmented_snapshot_path, working_epub_path)
        elif working_has_segments and not snapshot_has_segments:
            atomic_copy(working_epub_path, paths.segmented_snapshot_path)
        state.setdefault("artifacts", {})["matched_html_files"] = html_files
        return {"html_files": html_files, "snapshot": str(paths.segmented_snapshot_path)}

    if stage == "smil":
        matched_list = load_json_if_exists(paths.matched_list_path)
        if not matched_list:
            return None
        smil_files = expected_smil_files(matched_list, legacy)
        if not smil_files:
            return None
        if any(not (paths.run_dir / name).exists() for name in smil_files):
            return None
        return {"smil_files": smil_files}

    if stage == "package":
        matched_list = load_json_if_exists(paths.matched_list_path)
        if not matched_list:
            return None
        audio_files = state.get("artifacts", {}).get("audio_files")
        if not audio_files:
            audio_files, _chunks = expected_audio_files(legacy, book_info, paths.run_dir)
        smil_files = expected_smil_files(matched_list, legacy)
        package_candidates = [paths.output_path, paths.packaged_epub_path]
        for candidate in package_candidates:
            package_info = legacy.inspect_epub_package(candidate)
            if not package_info or not package_info.get("processed"):
                continue
            zip_names = package_info.get("zip_names", set())
            if any(f"audio/{name}" not in zip_names for name in audio_files):
                continue
            if any(f"smil/{name}" not in zip_names for name in smil_files):
                continue
            if not paths.output_path.exists() and candidate == paths.packaged_epub_path:
                atomic_copy(paths.packaged_epub_path, paths.output_path)
            if not paths.packaged_epub_path.exists() and candidate == paths.output_path:
                atomic_copy(paths.output_path, paths.packaged_epub_path)
            return {
                "packaged_epub": str(paths.packaged_epub_path),
                "output_epub": str(paths.output_path),
            }
        return None

    if stage == "validate":
        validation = load_json_if_exists(paths.validation_path)
        if not isinstance(validation, dict) or "ok" not in validation or "results" not in validation:
            return None
        return validation

    return None


def run_prepare_stage(
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    copied_audio = paths.run_dir / config.audio.name
    copied_epub = paths.run_dir / config.epub.name
    atomic_copy(config.audio, copied_audio)
    atomic_copy(config.epub, copied_epub)

    book_info = {
        "folder_name": str(paths.run_dir),
        "audio_file": copied_audio.name,
        "audio_extension": config.audio_extension,
        "audio_codec": config.audio_codec,
        "audio_bitrate": config.audio_bitrate,
        "audio_sample_rate": config.audio_sample_rate,
        "audio_channels": config.audio_channels,
        "split_jobs": config.split_jobs,
        "chunk_seconds": config.chunk_seconds,
        "backend": config.backend,
        "model": config.model,
        "language": config.language,
    }
    with pushd(paths.run_dir):
        audio_file, _epub_file, out_file, root_level = legacy.preprocess(book_info)
    book_info.update(
        {
            "audio_file": audio_file,
            "epub_file": out_file,
            "out_file": out_file,
            "root_level": root_level,
        }
    )

    state["book_info"] = book_info
    state["artifacts"]["prepared_epub"] = str(paths.run_dir / out_file)
    logger.info("Prepared working EPUB at %s", paths.run_dir / out_file)
    return {
        "audio_file": audio_file,
        "source_epub": copied_epub.name,
        "out_file": out_file,
    }


def run_split_stage(
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    book_info = build_book_info(state, paths)
    audio_files, chunks = expected_audio_files(legacy, book_info, paths.run_dir)
    with pushd(paths.run_dir):
        split_stats = legacy.split_audio(book_info)
    invalid = [chunk["output_name"] for chunk in chunks if not legacy.is_audio_chunk_complete(book_info, chunk)]
    if invalid:
        raise RuntimeError(f"Audio split stage did not produce all expected valid chunks: {invalid}")
    state["artifacts"]["audio_files"] = audio_files
    reused_chunk_count = split_stats.get("reused_chunk_count", 0)
    regenerated_chunk_count = split_stats.get("regenerated_chunk_count", 0)
    created_chunk_count = split_stats.get("created_chunk_count", 0)
    logger.info(
        "Audio split complete with %d chunk(s) (%d reused, %d regenerated, %d created)",
        len(audio_files),
        reused_chunk_count,
        regenerated_chunk_count,
        created_chunk_count,
    )
    return {
        "audio_files": audio_files,
        "chunk_count": len(chunks),
        "reused_chunk_count": reused_chunk_count,
        "regenerated_chunk_count": regenerated_chunk_count,
        "created_chunk_count": created_chunk_count,
    }


def run_transcribe_stage(
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    book_info = build_book_info(state, paths)
    book_info["backend"] = config.backend
    book_info["model"] = config.model
    book_info["language"] = config.language

    with pushd(paths.run_dir):
        legacy.transcribe_audio(book_info)

    audio_files = state.get("artifacts", {}).get("audio_files") or sorted(
        path.name for path in paths.run_dir.glob(f"*{config.audio_extension}")
    )
    transcript_files = [Path(name).with_suffix(".json").name for name in audio_files]
    missing = [name for name in transcript_files if not (paths.run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Transcription stage did not produce all expected JSON files: {missing}")
    state["artifacts"]["transcript_files"] = transcript_files
    logger.info("Transcription complete with %d transcript file(s)", len(transcript_files))
    return {"transcript_files": transcript_files}


def run_match_stage(
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    book_info = build_book_info(state, paths)
    with pushd(paths.run_dir):
        matched_list = legacy.link_html_with_audio(book_info)
    if not matched_list:
        raise RuntimeError("Matching stage produced an empty matched list")
    atomic_write_json(paths.matched_list_path, matched_list)
    html_files = matched_html_files(matched_list, legacy)
    state["artifacts"]["matched_html_files"] = html_files
    logger.info("Matching complete with %d transcript-to-HTML links", len(matched_list))
    return {"match_count": len(matched_list), "html_files": html_files}


def run_segment_stage(
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    matched_list = load_matched_list(paths)
    html_files = matched_html_files(matched_list, legacy)
    out_file = state["book_info"]["out_file"]
    working_epub_path = paths.run_dir / out_file

    if paths.segmented_snapshot_path.exists() and epub_contains_segment_ids(paths.segmented_snapshot_path, html_files):
        if not epub_contains_segment_ids(working_epub_path, html_files):
            restore_segmented_working_epub(state, paths)
        logger.info("Reusing existing segmented EPUB snapshot")
        return {"html_files": html_files, "snapshot": str(paths.segmented_snapshot_path)}

    if epub_contains_segment_ids(working_epub_path, html_files):
        atomic_copy(working_epub_path, paths.segmented_snapshot_path)
        logger.info("Recovered segmented EPUB snapshot from existing working EPUB")
        return {"html_files": html_files, "snapshot": str(paths.segmented_snapshot_path)}

    book_info = refresh_working_epub(legacy, build_book_info(state, paths, matched_list), paths.run_dir)
    book_info["matched_list"] = matched_list
    state["book_info"] = {k: v for k, v in book_info.items() if k != "matched_list"}

    with pushd(paths.run_dir):
        legacy.mark_segments(book_info)

    if not epub_contains_segment_ids(working_epub_path, html_files):
        raise RuntimeError("Segmented working EPUB did not contain the expected segment IDs")
    atomic_copy(working_epub_path, paths.segmented_snapshot_path)
    logger.info("Segment marking complete for %d HTML file(s)", len(html_files))
    return {"html_files": html_files, "snapshot": str(paths.segmented_snapshot_path)}


def run_smil_stage(
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    matched_list = load_matched_list(paths)
    expected_files = expected_smil_files(matched_list, legacy)
    if not expected_files:
        raise RuntimeError("No expected SMIL files could be derived from the matched list")
    html_files = matched_html_files(matched_list, legacy)
    if not epub_contains_segment_ids(paths.run_dir / state["book_info"]["out_file"], html_files):
        restore_segmented_working_epub(state, paths)

    book_info = build_book_info(state, paths, matched_list)
    with pushd(paths.run_dir):
        legacy.create_smil_files(book_info, skip=True)

    missing = [name for name in expected_files if not (paths.run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"SMIL generation did not produce all expected files: {missing}")
    logger.info("Generated %d SMIL file(s)", len(expected_files))
    return {"smil_files": expected_files}


def run_package_stage(
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    matched_list = load_matched_list(paths)
    restore_segmented_working_epub(state, paths)
    delete_path(paths.packaged_epub_path)

    book_info = build_book_info(state, paths, matched_list)
    with pushd(paths.run_dir):
        legacy.merge_files(book_info)
        legacy.post_processing_opf(book_info)

    if not paths.packaged_epub_path.exists():
        raise RuntimeError(f"Packaged EPUB not found after packaging stage: {paths.packaged_epub_path}")
    atomic_copy(paths.packaged_epub_path, config.output_path)
    logger.info("Packaged final EPUB at %s", config.output_path)
    return {
        "packaged_epub": str(paths.packaged_epub_path),
        "output_epub": str(config.output_path),
    }


def run_validate_stage(
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not paths.packaged_epub_path.exists() and paths.output_path.exists():
        atomic_copy(paths.output_path, paths.packaged_epub_path)
    matched_list = load_matched_list(paths) if paths.matched_list_path.exists() else None
    book_info = build_book_info(state, paths, matched_list)
    results = legacy.run_post_checks(book_info)
    atomic_write_json(paths.validation_path, results)
    logger.info("Validation summary: %s", results["summary"])
    return results


def execute_stage(
    stage: str,
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    if stage == "prepare":
        return run_prepare_stage(config, paths, state, legacy, logger)
    if stage == "split":
        return run_split_stage(paths, state, legacy, logger)
    if stage == "transcribe":
        return run_transcribe_stage(config, paths, state, legacy, logger)
    if stage == "match":
        return run_match_stage(paths, state, legacy, logger)
    if stage == "segment":
        return run_segment_stage(paths, state, legacy, logger)
    if stage == "smil":
        return run_smil_stage(paths, state, legacy, logger)
    if stage == "package":
        return run_package_stage(config, paths, state, legacy, logger)
    if stage == "validate":
        return run_validate_stage(paths, state, legacy, logger)
    raise RuntimeError(f"Unknown stage: {stage}")


def run_pipeline(config: PipelineConfig) -> int:
    paths = build_paths(config)
    logger = configure_logging(paths)
    preflight(config, logger)
    state, mode = initialize_state(config, paths)
    log_run_header(logger, config, paths, mode)
    started_at = time.monotonic()

    legacy = load_pipeline_module()

    for index, stage in enumerate(STAGES, start=1):
        reconciled_result = reconcile_stage_from_artifacts(stage, config, paths, state, legacy)
        if reconciled_result is not None:
            if stage_status(state, stage) != "success" or state["stages"].get(stage, {}).get("result") != reconciled_result:
                set_stage_state(state, stage, "success", result=reconciled_result)
                save_state(paths, state)
            logger.info("[%d/%d] Skipping %s (artifacts already complete)", index, len(STAGES), stage)
            continue

        logger.info("[%d/%d] Starting %s", index, len(STAGES), stage)
        set_stage_state(state, stage, "running")
        save_state(paths, state)
        stage_started_at = time.monotonic()

        try:
            result = execute_stage(stage, config, paths, state, legacy, logger)
        except Exception as exc:
            elapsed = time.monotonic() - stage_started_at
            set_stage_state(
                state,
                stage,
                "failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            save_state(paths, state)
            logger.exception("[%d/%d] %s failed after %.1fs", index, len(STAGES), stage, elapsed)
            logger.info("Detailed log: %s", paths.logs_dir / "pipeline.log")
            return 1

        set_stage_state(state, stage, "success", result=result)
        save_state(paths, state)
        logger.info(
            "[%d/%d] Completed %s in %.1fs",
            index,
            len(STAGES),
            stage,
            time.monotonic() - stage_started_at,
        )

    validation_status = stage_status(state, "validate")
    if validation_status == "success":
        validation_result = state["stages"]["validate"]["result"]
        validation_ok = validation_result.get("ok", False)
        log_run_summary(logger, config, paths, started_at, validation_ok)
        return 0 if validation_ok else 1

    log_run_summary(logger, config, paths, started_at, True)
    return 0


def main() -> int:
    try:
        config = parse_args()
        return run_pipeline(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
