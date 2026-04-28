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


@dataclass(frozen=True)
class PipelineConfig:
    m4b: Path
    epub: Path
    output_dir: Path
    output_path: Path
    work_dir: Path
    fresh: bool
    model: str
    language: str
    audio_extension: str


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


def default_state(signature: dict[str, Any], config: PipelineConfig, paths: RuntimePaths) -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "signature": signature,
        "inputs": {
            "m4b": str(config.m4b),
            "epub": str(config.epub),
        },
        "config": {
            "model": config.model,
            "language": config.language,
            "audio_extension": config.audio_extension,
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
        "m4b": fingerprint_file(config.m4b),
        "epub": fingerprint_file(config.epub),
        "model": config.model,
        "language": config.language,
        "audio_extension": config.audio_extension,
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
    parser.add_argument("--m4b", required=True, help="Input .m4b audiobook file")
    parser.add_argument("--epub", required=True, help="Input .epub ebook file")
    parser.add_argument("--output-dir", required=True, help="Directory for the final EPUB")
    parser.add_argument(
        "--work-dir",
        help="Working directory used for persistent state and intermediate artifacts",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        help="WhisperX model identifier",
    )
    parser.add_argument("--language", default="en", help="Transcription language code")
    parser.add_argument(
        "--audio-extension",
        default=".m4a",
        help="Audio chunk extension produced during splitting",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing working state and restart from scratch instead of resuming automatically",
    )
    args = parser.parse_args()

    m4b = Path(args.m4b).expanduser().resolve()
    epub = Path(args.epub).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output_dir / f".{epub.stem}.epubmo"
    )
    output_path = output_dir / f"{epub.stem}.media-overlay.epub"

    return PipelineConfig(
        m4b=m4b,
        epub=epub,
        output_dir=output_dir,
        output_path=output_path,
        work_dir=work_dir,
        fresh=args.fresh,
        model=args.model,
        language=args.language,
        audio_extension=args.audio_extension,
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


def preflight(config: PipelineConfig) -> None:
    if not config.m4b.exists():
        raise FileNotFoundError(f"Input M4B not found: {config.m4b}")
    if not config.epub.exists():
        raise FileNotFoundError(f"Input EPUB not found: {config.epub}")
    if config.m4b.suffix.lower() != ".m4b":
        raise ValueError(f"Expected an .m4b input, got: {config.m4b}")
    if config.epub.suffix.lower() != ".epub":
        raise ValueError(f"Expected an .epub input, got: {config.epub}")
    ensure_command("ffprobe")
    ensure_command("ffmpeg")
    for module_name in ("bs4", "lxml", "tqdm", "mlx_whisperx"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing Python dependency: {module_name}. Install the pipeline requirements before running."
            ) from exc

    ensure_nltk_resources()


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
    logger.info("Input M4B: %s", config.m4b)
    logger.info("Input EPUB: %s", config.epub)
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
    book_info["model"] = None
    if matched_list is not None:
        book_info["matched_list"] = matched_list
    return book_info


def refresh_working_epub(legacy: Any, book_info: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    refreshed = {
        "folder_name": str(run_dir),
        "audio_extension": book_info["audio_extension"],
        "model": None,
    }
    with pushd(run_dir):
        m4b_file, _source_epub, out_file, root_level = legacy.preprocess(refreshed)
    refreshed.update(
        {
            "m4b_file": m4b_file,
            "epub_file": out_file,
            "out_file": out_file,
            "root_level": root_level,
        }
    )
    return refreshed


def expected_audio_files(book_info: dict[str, Any], run_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    m4b_path = run_dir / book_info["m4b_file"]
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-i",
            str(m4b_path),
            "-show_chapters",
            "-print_format",
            "json",
        ],
        text=True,
    )
    chapters = json.loads(output).get("chapters", [])
    names = []
    filtered_chapters = []
    for chapter_info in chapters:
        if float(chapter_info["end_time"]) <= float(chapter_info["start_time"]):
            continue
        filtered_chapters.append(chapter_info)
        names.append(f"{str(chapter_info['id']).zfill(3)}{book_info['audio_extension']}")
    return names, filtered_chapters


def list_transcripts(run_dir: Path) -> list[str]:
    return sorted(path.name for path in run_dir.glob("*.json"))


def load_matched_list(paths: RuntimePaths) -> list[dict[str, Any]]:
    if not paths.matched_list_path.exists():
        raise RuntimeError(f"Matched-list cache not found: {paths.matched_list_path}")
    return json.loads(paths.matched_list_path.read_text(encoding="utf-8"))


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


def run_prepare_stage(
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    copied_m4b = paths.run_dir / config.m4b.name
    copied_epub = paths.run_dir / config.epub.name
    atomic_copy(config.m4b, copied_m4b)
    atomic_copy(config.epub, copied_epub)

    book_info = {
        "folder_name": str(paths.run_dir),
        "audio_extension": config.audio_extension,
        "model": None,
    }
    with pushd(paths.run_dir):
        m4b_file, _epub_file, out_file, root_level = legacy.preprocess(book_info)
    book_info.update(
        {
            "m4b_file": m4b_file,
            "epub_file": out_file,
            "out_file": out_file,
            "root_level": root_level,
        }
    )

    state["book_info"] = book_info
    state["artifacts"]["prepared_epub"] = str(paths.run_dir / out_file)
    logger.info("Prepared working EPUB at %s", paths.run_dir / out_file)
    return {
        "m4b_file": m4b_file,
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
    with pushd(paths.run_dir):
        legacy.split_m4b(book_info)
    audio_files, chapters = expected_audio_files(book_info, paths.run_dir)
    missing = [name for name in audio_files if not (paths.run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Audio split stage did not produce all expected chunks: {missing}")
    state["artifacts"]["audio_files"] = audio_files
    logger.info("Audio split complete with %d chunk(s)", len(audio_files))
    return {"audio_files": audio_files, "chapter_count": len(chapters)}


def run_transcribe_stage(
    config: PipelineConfig,
    paths: RuntimePaths,
    state: dict[str, Any],
    legacy: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    book_info = build_book_info(state, paths)
    original_transcribe = legacy.transcribe

    def configured_transcribe(file_name: str, model: str | None = None, language: str | None = None):
        return original_transcribe(file_name, model=config.model, language=config.language)

    legacy.transcribe = configured_transcribe
    try:
        with pushd(paths.run_dir):
            legacy.transcribe_audio(book_info)
    finally:
        legacy.transcribe = original_transcribe

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
    book_info = build_book_info(state, paths)
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
    preflight(config)
    state, mode = initialize_state(config, paths)
    log_run_header(logger, config, paths, mode)
    started_at = time.monotonic()

    legacy = load_pipeline_module()

    for index, stage in enumerate(STAGES, start=1):
        if stage_status(state, stage) == "success":
            logger.info("[%d/%d] Skipping %s (already completed)", index, len(STAGES), stage)
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
