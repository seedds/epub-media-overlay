"""
EPUB audiobook synchronization pipeline.

Purpose
-------

Transform one source audiobook container (`.m4b`) and one source ebook (`.epub`)
into a Media Overlay-capable EPUB package.

Primary responsibilities:

- create a working EPUB copy
- split the audiobook into numbered chunks
- transcribe each chunk
- match transcript chunks to EPUB HTML in reading order
- mark HTML with stable segment ids
- align transcript words to marked HTML segments
- generate one SMIL overlay per HTML file
- package audio/SMIL/CSS assets into the EPUB
- patch the OPF manifest/metadata for Media Overlays
- audit the resulting artifacts and package state

Operational constraints:

- audiobook chapter markers are only approximate input, not authoritative structure
- EPUB spine order is authoritative for reading order; filename order is not
- an audio chunk may start in the middle of an HTML file
- multiple audio chunks may map to the same HTML file
- the working `.epub3` and the final `.epub` are not always the same path
- validation must inspect both loose working artifacts and packaged EPUB contents

Pipeline overview
-----------------

1. Discover and prepare inputs.
   `preprocess()` finds the local `.m4b` and `.epub`, copies the source EPUB to a
   working `.epub3`, and records OPF-related metadata in `book_info`.

2. Split the audiobook into manageable chunks.
   `split_m4b()` uses `ffprobe` chapter metadata and `ffmpeg` to create numbered
   audio files (`000.m4a`, `001.m4a`, ...). These chapter markers are only a starting
   point; later matching can still place a chunk into the middle of an HTML file.

3. Transcribe each audio chunk.
   `transcribe_audio()` produces sibling WhisperX JSON files. These `.json` files are
   the canonical transcript artifacts for the matching and alignment stages.

4. Match audio chunks to HTML windows.
   `link_html_with_audio()` reads the EPUB spine in reading order, slices each HTML
   file into overlapping lexical windows, and then scores each transcript against the
   candidate windows. This stage is semantic/lexical matching, not final timestamp
   alignment.

5. Mark HTML with stable segment ids.
   `mark_segments()` runs `mark_sentence.mark_sentences()` on each matched HTML file
   once, injecting segment spans that later SMIL entries can target.

6. Align transcript words to HTML segment tokens.
   `create_smil_files()` performs fine-grained token alignment between transcript
   words and the marked HTML segments. This is where we turn transcript text into real
   timestamps per segment.

7. Package assets into the working EPUB.
   `merge_files()` adds audio, SMIL, and CSS assets into the `.epub3` zip.

8. Rewrite the OPF for Media Overlays.
   `post_processing_opf()` adds the overlay manifest items, duration metadata, and
   `media-overlay` references from HTML items to the per-HTML SMIL files, then moves
   the finished package to a final `.epub` path.

9. Run read-only post checks.
   The `test_*` helpers and `run_post_checks()` audit the loose artifacts and the
   packaged EPUB to catch missing transcripts, missing SMIL coverage, duplicate clip
   reuse, and broken OPF wiring.

Core data model: `book_info`
----------------------------

`book_info` is the shared mutable context for one book run. It carries discovered
paths, pipeline settings, and intermediate artifacts between stages.

Typical keys include:

- `folder_name`: working directory for one book
- `m4b_file`: source audiobook path/name
- `epub_file`: working EPUB path/name used for matching
- `out_file`: working `.epub3` file that gets mutated and later moved
- `opf_file`, `opf_dir`: package metadata derived from the working EPUB
- `audio_extension`: output split-audio extension, usually `.m4a`
- `model`: optional semantic model handle (historically used for embeddings)
- `matched_list`: the bridge artifact between semantic matching and timestamp
  alignment

Key invariants
--------------

- Never modify the source EPUB directly. The pipeline works against a copied
  `.epub3` file.
- Matching is done in EPUB spine order, not naive lexical filename order.
- A single HTML file can own many audio chunks, but packaging should still produce
  exactly one SMIL per HTML file.
- OPF overlay wiring points from HTML manifest items to those per-HTML SMIL files.
- Post checks validate artifacts; they do not generate or repair them.

Frequent operator mistakes
--------------------------

1. `matched_list.pkl` can become stale.
   If a new transcript JSON appears after matching was already run, the saved match
   list no longer reflects the current folder contents.

2. The "real packaged EPUB" may not be the EPUB sitting in the book folder.
   After OPF post-processing, the working `.epub3` is moved to a final `.epub` path.
   Validation must distinguish a processed package from the original source EPUB.

3. Matching and alignment are intentionally separate.
   `link_html_with_audio()` chooses the most likely HTML window. It does not create
   timestamps. `create_smil_files()` then performs the token-level alignment inside
   the chosen HTML file.

Section map
-----------

- EPUB/ZIP helpers: path, XML, and archive manipulation
- audio split/transcription: loose artifact generation
- HTML/audio matching: choose the right HTML region per transcript
- segment marking + token alignment: turn text correspondence into timestamps
- packaging/OPF updates: make the EPUB declare and reference the generated assets
- post checks: audit the output and intermediate artifacts

High-risk areas
---------------

- choosing the wrong packaged EPUB candidate for validation
- assuming one audio chunk maps to one HTML file from the beginning
- stale `matched_list.pkl` data after new transcripts appear
- OPF path normalization and relative href calculations
- alignment cases where an audiobook chunk begins in the middle of a chapter file
"""

import difflib
import glob
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import warnings
import xml.dom.minidom as minidom
import zipfile
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm

from mark_sentence import mark_sentences
from transcription_backend import (
    default_model_for_backend,
    detect_transcription_backend,
    transcribe_file,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="zipfile")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["MPLBACKEND"] = "Agg"


# === EPUB and ZIP helpers ===


def preprocess(book_info):
    # Working model:
    # - source `.epub` remains untouched
    # - all mutations happen against a copied `.epub3`
    # - final packaging later moves that working copy to a final `.epub`
    m4b_file = glob.glob("*.m4b")[0]
    epub_file = glob.glob("*.epub")[0]
    out_file = epub_file.replace(".epub", ".epub3")
    shutil.copy(epub_file, out_file)

    with zipfile.ZipFile(out_file, "a") as f:
        for file_name in f.namelist():
            if ".opf" in file_name:
                break
        assert ".opf" in file_name

    book_info["opf_file"] = file_name
    book_info["opf_dir"] = posixpath.dirname(file_name) or "."
    root_level = len(list(Path(file_name).parents))

    return m4b_file, epub_file, out_file, root_level


def delete_file_from_zip(zip_path, file_to_delete):
    # Create a temporary ZIP file
    temp_zip = zip_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as original_zip:
        with zipfile.ZipFile(
            temp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as new_zip:
            # Iterate over all files in the original ZIP
            for item in original_zip.infolist():
                # Only write files that are NOT the one to delete
                if item.filename != file_to_delete:
                    # Read and write the file content
                    data = original_zip.read(item.filename)
                    new_zip.writestr(item, data)

    # Replace the original ZIP with the modified one
    os.replace(temp_zip, zip_path)


def replace_files_in_zip(zip_path, replacements):
    # ZIP update strategy:
    # rewrite the archive into a temporary file while replacing selected entries.
    # This is more predictable than trying to mutate EPUB contents in place.
    temp_zip = zip_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as original_zip:
        with zipfile.ZipFile(
            temp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as new_zip:
            for item in original_zip.infolist():
                if item.filename not in replacements:
                    new_zip.writestr(item, original_zip.read(item.filename))

            for file_name, content in replacements.items():
                new_zip.writestr(file_name, content)

    os.replace(temp_zip, zip_path)


def convert_soup_to_html(soup):
    # BeautifulSoup/minidom give us a stable enough serialization for generated XML,
    # but OPF namespace prefixes like `opf:` can leak into output in ways that are
    # syntactically noisy and not useful for our packaged artifacts.
    xml_str = str(soup).replace("opf:", "")

    # Reparse through minidom to get readable indentation. This is primarily for human
    # inspection/debuggability; the EPUB readers do not care about pretty printing.
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", newl="\n")

    # CORRECTED: Remove ONLY blank lines (whitespace-only lines)
    lines = [line for line in pretty_xml.splitlines() if line.strip() != ""]
    cleaned_xml = "\n".join(lines) + "\n"  # Ensure single trailing newline

    return cleaned_xml


def get_relative_zip_href(source_path, target_path):
    source_dir = posixpath.dirname(source_path) or "."
    return posixpath.relpath(target_path, start=source_dir)


def seconds_to_media_duration(seconds):
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def get_smil_duration(smil_file):
    # A grouped SMIL may reference multiple different audio files. The correct SMIL
    # duration is therefore the sum of its clip durations, not the maximum clip end.
    with open(smil_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")

    total_duration = 0.0
    for audio_el in soup.find_all("audio"):
        clip_begin = audio_el.get("clipBegin", "0s").rstrip("s")
        clip_end = audio_el.get("clipEnd", "0s").rstrip("s")
        try:
            total_duration += max(0.0, float(clip_end) - float(clip_begin))
        except ValueError:
            continue

    return total_duration


def parse_ncx_nav_points(container):
    if container is None:
        return []

    nav_points = []
    for nav_point in container.find_all("navPoint", recursive=False):
        label_el = nav_point.find("text")
        content_el = nav_point.find("content")
        if not label_el or not content_el or not content_el.get("src"):
            continue

        nav_points.append(
            {
                "label": label_el.get_text(strip=True),
                "href": content_el["src"],
                "children": parse_ncx_nav_points(nav_point),
            }
        )

    return nav_points


def render_nav_points(nav_points, level=3):
    indent = "  " * level
    lines = [f"{indent}<ol>"]
    for point in nav_points:
        lines.append(
            f'{indent}  <li><a href="{escape(point["href"], quote=True)}">{escape(point["label"])}</a>'
        )
        if point["children"]:
            lines.append(render_nav_points(point["children"], level + 2))
            lines.append(f"{indent}  </li>")
        else:
            lines[-1] += "</li>"
    lines.append(f"{indent}</ol>")
    return "\n".join(lines)


def build_nav_document(title, nav_points):
    nav_list = render_nav_points(nav_points)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
  <head>
    <title>{escape(title or "Contents")}</title>
  </head>
  <body>
    <nav epub:type="toc" id="toc">
      <h1>Contents</h1>
{nav_list}
    </nav>
  </body>
</html>
"""


def get_spine_ordered_html_files(zip_file):
    # Filename sorting is not a reliable reading order. Front matter, appendices, and
    # generated nav files often sort in surprising ways. The OPF spine is the actual
    # "book order" we want for matching and monotonic progression.
    html_files = [
        file_name
        for file_name in zip_file.namelist()
        if file_name.endswith((".htm", ".html", ".xhtml"))
    ]
    if not html_files:
        return []

    opf_file = next(
        (file_name for file_name in zip_file.namelist() if file_name.endswith(".opf")),
        None,
    )
    if not opf_file:
        return sorted(html_files)

    with zip_file.open(opf_file) as f:
        soup = BeautifulSoup(f.read(), "xml")

    opf_dir = posixpath.dirname(opf_file) or "."
    manifest_map = {}
    for item in soup.find_all("item"):
        item_id = item.get("id")
        href = item.get("href")
        if item_id and href:
            manifest_map[item_id] = posixpath.normpath(posixpath.join(opf_dir, href))

    spine_files = []
    for itemref in soup.find_all("itemref"):
        href = manifest_map.get(itemref.get("idref"))
        if href in html_files and href not in spine_files:
            spine_files.append(href)

    remaining_files = [file_name for file_name in html_files if file_name not in spine_files]
    return spine_files + sorted(remaining_files)


# === Audio splitting and transcription ===


def split_m4b(book_info):
    # The chapter markers embedded in the audiobook are useful for creating chunks, but
    # they are not trusted as final semantic matches to the book text. They simply give
    # us manageable audio segments for transcription and later matching.
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-i",
            book_info["m4b_file"],
            "-show_chapters",
            "-print_format",
            "json",
        ],
        text=True,
    )
    chapters_json = json.loads(output)["chapters"]
    valid_chapters = [
        chapter_info
        for chapter_info in chapters_json
        if float(chapter_info["end_time"]) > float(chapter_info["start_time"])
    ]

    for chapter_info in tqdm(valid_chapters, desc="Splitting audio", unit="chunk"):
        start_time = chapter_info["start_time"]
        end_time = chapter_info["end_time"]
        out_name = f"{str(chapter_info['id']).zfill(3)}" + book_info["audio_extension"]
        if os.path.exists(out_name):
            continue

        # Keep the console focused on overall progress. If ffmpeg fails, surface only
        # its error output for the chunk that failed.
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-nostats",
                "-i",
                book_info["m4b_file"],
                "-ss",
                start_time,
                "-to",
                end_time,
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                "-ar",
                "24000",
                "-ac",
                "1",
                os.path.join(book_info["folder_name"], out_name),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            error_output = (result.stderr or result.stdout or "Unknown ffmpeg error").strip()
            raise RuntimeError(f"ffmpeg failed while creating {out_name}: {error_output}")


def print_nonzero_summary(label, counters):
    parts = [f"{value} {name}" for name, value in counters if value]
    if parts:
        print(f"{label}: " + ", ".join(parts))


def transcribe_audio(book_info):
    # Transcription is per split audio chunk, not against the original `.m4b`. The
    # sibling JSON files become the canonical transcript inputs for both matching and
    # fine-grained timestamp alignment.
    backend = book_info.get("backend") or detect_transcription_backend()
    model = book_info.get("model") or default_model_for_backend(backend)
    language = book_info.get("language") or "en"

    for file in tqdm(
        sorted(glob.glob(f"*{book_info['audio_extension']}")),
        desc="Transcribing audio",
        unit="chunk",
    ):
        json_file = file.replace(book_info["audio_extension"], ".json")
        if os.path.exists(json_file):
            continue

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                result = transcribe_file(
                    file,
                    model,
                    language,
                    backend,
                )
        except Exception as exc:
            error_output = (stderr_buffer.getvalue() or stdout_buffer.getvalue()).strip()
            if error_output:
                raise RuntimeError(
                    f"transcription failed for {file}: {error_output}"
                ) from exc
            raise RuntimeError(f"transcription failed for {file}: {exc}") from exc

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            



# === Semantic / lexical matching between transcripts and HTML ===


def link_html_with_audio(book_info):
    epub_path = book_info.get("epub_file")

    if not epub_path:
        print("ERROR: 'epub_file' not found in book_info.")
        return []
    if not os.path.exists(epub_path):
        print(f"ERROR: EPUB file not found at path: {epub_path}")
        return []

    # Matching model:
    # build overlapping token windows inside each HTML file, then score transcript
    # openings against those windows. This supports mid-file starts and multi-chunk
    # chapters.
    epub_candidates = []
    window_size = 160
    window_step = 80
    overlap_token_min_length = 4
    overlap_threshold = 6
    combined_score_threshold = 0.35
    top_k_windows = 24
    preprocess_stats = {
        "decode_fallbacks": 0,
        "html_skipped_short": 0,
        "html_errors": 0,
    }

    # Step 1: build a candidate window index over EPUB text.
    # Candidate records carry file identity, token offsets, lexical counters, and a
    # global reading-order index used for a soft forward-bias.
    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            html_files = get_spine_ordered_html_files(zf)

            if not html_files:
                print(
                    f"❌ No HTML/XHTML files found in EPUB: {epub_path}. Matched list will be empty."
                )
                return []

            for html_order, file_name in enumerate(
                tqdm(html_files, desc="Indexing EPUB", unit="file")
            ):
                try:
                    with zf.open(file_name) as f:
                        try:
                            raw_html = f.read()
                            html_content = raw_html.decode("utf-8")
                        except UnicodeDecodeError:
                            preprocess_stats["decode_fallbacks"] += 1
                            html_content = raw_html.decode("latin-1", errors="replace")

                        soup = BeautifulSoup(html_content, "lxml")

                        for script_or_style in soup(["script", "style"]):
                            script_or_style.decompose()

                        # Tokenization happens once per HTML file; windows are slices of
                        # that token stream.
                        token_items = extract_text_token_items(
                            soup.get_text(separator=" ", strip=True)
                        )

                        if len(token_items) < 5:
                            preprocess_stats["html_skipped_short"] += 1
                            continue

                        max_start = max(0, len(token_items) - window_size)
                        window_starts = list(range(0, max_start + 1, window_step)) or [0]
                        if window_starts[-1] != max_start:
                            window_starts.append(max_start)

                        # Overlap is required so a transcript opening can land near the
                        # middle of a long chapter file and still score well.
                        window_ranges = []
                        for window_start in window_starts:
                            window_end = min(len(token_items), window_start + window_size)
                            if window_end - window_start < 20:
                                continue
                            window_ranges.append((window_start, window_end))

                        window_texts = [
                            " ".join(
                                token["display"]
                                for token in token_items[window_start:window_end]
                            )
                            for window_start, window_end in window_ranges
                        ]

                        for window_index, ((window_start, window_end), window_text) in enumerate(
                            zip(window_ranges, window_texts)
                        ):
                            window_tokens = [
                                token["token"]
                                for token in token_items[window_start:window_end]
                            ]
                            match_tokens = [
                                token
                                for token in window_tokens
                                if len(token) >= overlap_token_min_length
                            ]
                            epub_candidates.append(
                                {
                                    "file_name": file_name,
                                    "html_order": html_order,
                                    "window_index": window_index,
                                    "window_start": window_start,
                                    "window_end": window_end,
                                    "text_preview": window_text[:200],
                                    "window_tokens": window_tokens,
                                    "match_counter": Counter(match_tokens),
                                    "global_index": len(epub_candidates),
                                }
                            )

                except Exception:
                    preprocess_stats["html_errors"] += 1
                    continue  # Continue to next HTML file

    except zipfile.BadZipFile:
        print(f"❌ EPUB file is a bad zip file: {epub_path}")
        return []
    except Exception as e:
        print(f"ERROR during EPUB processing: {e}")
        return []

    if not epub_candidates:
        print(
            "❌ No valid EPUB HTML candidates could be processed for comparison. Matched list will be empty."
        )
        return []

    print_nonzero_summary(
        "Indexing summary",
        [
            ("decode fallback(s)", preprocess_stats["decode_fallbacks"]),
            ("short/empty HTML file(s) skipped", preprocess_stats["html_skipped_short"]),
            ("HTML file(s) failed", preprocess_stats["html_errors"]),
        ],
    )

    # Step 2: score each transcript against the candidate windows.
    # Current policy prefers lexical overlap plus local token-order agreement over
    # embedding-only matching because short openings and chapter headers are noisy.
    matched_list = []
    json_files = sorted(glob.glob("*.json"))
    last_global_index = 0
    match_stats = {
        "json_skipped_short": 0,
        "json_skipped_probe": 0,
        "no_overlap": 0,
        "no_candidate": 0,
        "below_threshold": 0,
        "backward_warnings": 0,
        "decode_errors": 0,
        "processing_errors": 0,
    }

    if not json_files:
        print(
            "❌ No JSON files found in current directory. Matched list will be empty."
        )
        return []

    for json_file in tqdm(json_files, desc="Matching transcripts", unit="file"):
        try:
            audio_tokens = load_audio_tokens(json_file)
            content_words = [item["word"] for item in audio_tokens]

            if not content_words or len(content_words) < 5:
                match_stats["json_skipped_short"] += 1
                continue

            # Only the transcript opening is used as the probe text. That is usually
            # sufficient to identify the correct chapter/window.
            probe_tokens = [item["token"] for item in audio_tokens[:120] if item["token"]]
            probe_match_tokens = [
                token for token in probe_tokens if len(token) >= overlap_token_min_length
            ]
            if not probe_match_tokens:
                match_stats["json_skipped_probe"] += 1
                continue
            probe_counter = Counter(probe_match_tokens)

            min_allowed_index = max(0, last_global_index - 1)
            # First pass: cheap lexical overlap filter.
            overlap_scored_candidates = []
            for candidate in epub_candidates:
                overlap_count = sum(
                    min(probe_counter[token], candidate["match_counter"].get(token, 0))
                    for token in probe_counter
                )
                if overlap_count > 0:
                    overlap_scored_candidates.append((overlap_count, candidate))

            if not overlap_scored_candidates:
                match_stats["no_overlap"] += 1
                continue

            overlap_scored_candidates.sort(
                key=lambda item: (item[0], -item[1]["global_index"]), reverse=True
            )

            best_match = None
            best_score = 0.0
            best_overlap = 0
            # Second pass: score local token order and add a small forward-order bias.
            for overlap_count, candidate in overlap_scored_candidates[:top_k_windows]:
                sequence_score = difflib.SequenceMatcher(
                    None,
                    candidate["window_tokens"][: len(probe_tokens)],
                    probe_tokens,
                    autojunk=False,
                ).ratio()
                overlap_score = overlap_count / max(1, len(probe_match_tokens))
                order_bonus = 0.02 if candidate["global_index"] >= min_allowed_index else 0.0
                combined_score = (overlap_score * 0.7) + (sequence_score * 0.3) + order_bonus
                if combined_score > best_score:
                    best_score = combined_score
                    best_overlap = overlap_count
                    best_match = candidate

            if best_match is None:
                match_stats["no_candidate"] += 1
                continue

            if best_score >= combined_score_threshold and best_overlap >= overlap_threshold:
                matched_list.append(
                    {
                        "json_file": json_file,
                        "html_file": best_match["file_name"],
                        "score": float(best_score),
                        "html_order": best_match["html_order"],
                        "window_index": best_match["window_index"],
                        "window_start": best_match["window_start"],
                        "window_end": best_match["window_end"],
                        "candidate_index": best_match["global_index"],
                    }
                )
                last_global_index = best_match["global_index"]
            else:
                match_stats["below_threshold"] += 1
                continue

            # Diagnostic only: log suspicious backward jumps after accepting a match.
            if len(matched_list) >= 2:
                prev_html_file = matched_list[-2]["html_file"]
                curr_html_file = matched_list[-1]["html_file"]
                if curr_html_file < prev_html_file:
                    if (
                        "introduction" not in prev_html_file.lower()
                        and "preface" not in prev_html_file.lower()
                        and "intro" not in prev_html_file.lower()
                    ):
                        match_stats["backward_warnings"] += 1

        except json.JSONDecodeError:
            match_stats["decode_errors"] += 1
        except Exception:
            match_stats["processing_errors"] += 1

    print_nonzero_summary(
        "Matching summary",
        [
            ("matched transcript(s)", len(matched_list)),
            ("low-content transcript(s) skipped", match_stats["json_skipped_short"]),
            ("transcript(s) skipped for missing probe tokens", match_stats["json_skipped_probe"]),
            ("transcript(s) with no lexical overlap", match_stats["no_overlap"]),
            ("transcript(s) with no surviving candidate", match_stats["no_candidate"]),
            ("transcript(s) below threshold", match_stats["below_threshold"]),
            ("backward-order warning(s)", match_stats["backward_warnings"]),
            ("JSON decode error(s)", match_stats["decode_errors"]),
            ("transcript processing error(s)", match_stats["processing_errors"]),
        ],
    )

    return matched_list


# === Segment marking and alignment helpers ===


def mark_segments(book_info):
    # Segment ids belong to HTML files, not to audio files. If multiple audio chunks
    # map to the same HTML chapter, we still want one consistent set of segment ids in
    # that HTML file, so we process each HTML file at most once here.
    matched_items = normalize_matched_list(book_info["matched_list"])
    replacements = {}
    language = book_info.get("language") or "en"
    css_zip_path = (
        f"{book_info['opf_dir']}/readaloud.css"
        if book_info["opf_dir"] != "."
        else "readaloud.css"
    )

    with zipfile.ZipFile(book_info["out_file"], "r") as f:
        processed_html_files = set()
        for item in matched_items:
            html_file_name = item["html_file"]
            if html_file_name in processed_html_files:
                continue

            processed_html_files.add(html_file_name)
            with f.open(html_file_name) as html_file:
                html_content = html_file.read()

            processed_html = mark_sentences(
                html_content,
                make_segment_prefix(html_file_name),
                language=language,
            )
            soup = BeautifulSoup(processed_html, "lxml")
            head = soup.head
            css_href = get_relative_zip_href(html_file_name, css_zip_path)
            if head and not head.find(
                "link", attrs={"href": css_href, "rel": "stylesheet"}
            ):
                link = soup.new_tag(
                    "link",
                    rel="stylesheet",
                    href=css_href,
                    type="text/css",
                )
                head.append(link)
            replacements[html_file_name] = str(soup)

            with open(os.path.basename(html_file_name), "w") as html_out:
                html_out.write(convert_soup_to_html(soup))

    if replacements:
        replace_files_in_zip(book_info["out_file"], replacements)


def get_clean_string(s):
    return "".join(char for char in s if char.isalpha() or char == " ")


def clean_token(text):
    """Normalize text for alignment: lowercase and alpha-numeric only."""
    if not text:
        return ""
    return re.sub(r"[^\w]", "", text.lower())


def sanitize_identifier(value):
    value = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if not value:
        return "item"
    if value[0].isdigit():
        value = "item_" + value
    return value


def make_segment_prefix(html_file):
    return f"seg_{sanitize_identifier(posixpath.splitext(html_file)[0])}"


def make_overlay_basename(html_file):
    return f"{sanitize_identifier(posixpath.splitext(html_file)[0])}.smil"


def make_overlay_id(html_file):
    return f"html_overlay.{sanitize_identifier(posixpath.splitext(html_file)[0])}"


def normalize_match_item(item):
    # The codebase historically used tuple-style match records and now uses dict-style
    # records with richer window metadata. This normalizer keeps downstream alignment
    # and packaging code compatible with both formats.
    if isinstance(item, dict):
        normalized = dict(item)
        normalized["json_file"] = normalized.get("json_file") or normalized.get(
            "audio_file", ""
        )
        normalized["html_file"] = normalized.get("html_file") or normalized.get(
            "file_name", ""
        )
        normalized["score"] = float(normalized.get("score", 0.0))
        normalized["window_start"] = int(normalized.get("window_start", 0))
        normalized["window_end"] = int(
            normalized.get("window_end", normalized["window_start"])
        )
        normalized["html_order"] = int(normalized.get("html_order", 0))
        normalized["candidate_index"] = int(
            normalized.get("candidate_index", normalized.get("window_index", 0))
        )
        return normalized

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return {
            "json_file": item[0],
            "html_file": item[1],
            "score": float(item[2]) if len(item) > 2 else 0.0,
            "window_start": 0,
            "window_end": 0,
            "html_order": 0,
            "candidate_index": 0,
        }

    raise ValueError(f"Unsupported matched item: {item}")


def normalize_matched_list(matched_list):
    return [normalize_match_item(item) for item in matched_list]


def extract_text_token_items(text):
    token_items = []
    for word in text.split():
        token = clean_token(word)
        if token:
            token_items.append({"display": word, "token": token})
    return token_items


def load_audio_tokens(json_file):
    # WhisperX outputs can appear either as top-level `word_segments` or nested under
    # per-segment `words`. This loader normalizes both layouts into one token stream.
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_data = data.get("word_segments")
    if audio_data is None:
        audio_data = []
        for segment in data.get("segments", []):
            audio_data.extend(segment.get("words", []))

    audio_tokens = []
    for word_info in audio_data:
        token = clean_token(word_info.get("word", ""))
        if token and "start" in word_info and "end" in word_info:
            audio_tokens.append(
                {
                    "token": token,
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                }
            )

    return audio_tokens


def load_html_segments_and_tokens(zip_path, html_file):
    # After `mark_segments()`, each HTML file contains stable `*-segment` ids. This
    # helper loads the marked HTML and flattens it into segment-aware token records so
    # transcript alignment can map from token matches back to segment ids.
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(html_file) as f:
            soup = BeautifulSoup(f, "lxml")

    segments = soup.select('[id*="-segment"]')
    html_tokens = []
    for seg_index, seg in enumerate(segments):
        seg_text = seg.get_text()
        seg_id = seg.get("id")
        for word in seg_text.split():
            token = clean_token(word)
            if token:
                html_tokens.append(
                    {"token": token, "seg_id": seg_id, "seg_index": seg_index}
                )

    return segments, html_tokens


def build_raw_matches(opcodes, html_tokens, audio_tokens, html_offset=0):
    # `SequenceMatcher` yields equal runs in token space. This helper turns those runs
    # into per-segment timestamp envelopes by collecting every transcript token that
    # aligned to a given HTML segment.
    raw_matches = {}
    matched_token_count = 0
    max_html_idx = html_offset - 1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue

        for relative_html_idx in range(i1, i2):
            offset = relative_html_idx - i1
            audio_idx = j1 + offset
            if audio_idx >= len(audio_tokens):
                break

            html_idx = html_offset + relative_html_idx
            seg_id = html_tokens[html_idx]["seg_id"]
            audio_info = audio_tokens[audio_idx]

            if seg_id not in raw_matches:
                raw_matches[seg_id] = {
                    "start": audio_info["start"],
                    "end": audio_info["end"],
                    "audio_text_parts": [],
                    "segment_index": html_tokens[html_idx]["seg_index"],
                }

            raw_matches[seg_id]["start"] = min(raw_matches[seg_id]["start"], audio_info["start"])
            raw_matches[seg_id]["end"] = max(raw_matches[seg_id]["end"], audio_info["end"])
            raw_matches[seg_id]["audio_text_parts"].append((audio_idx, audio_info["word"]))
            matched_token_count += 1
            max_html_idx = max(max_html_idx, html_idx)

    return raw_matches, matched_token_count, max_html_idx


def finalize_segment_timestamps(raw_matches, segments, total_duration):
    # Raw equal-token matches are usually sparse and jagged. This function turns them
    # into monotonically ordered segment timings by:
    # - ordering them by segment appearance
    # - gap-closing neighboring segments
    # - anchoring the first start to 0 and the last end to the full audio duration
    # - applying a small negative shift so highlights feel less late to the reader
    matched_ordered_list = []
    for seg_index, seg in enumerate(segments):
        seg_id = seg.get("id")
        if seg_id not in raw_matches:
            continue

        data = raw_matches[seg_id]
        data["audio_text_parts"].sort(key=lambda x: x[0])
        matched_ordered_list.append(
            {
                "id": seg_id,
                "segment_index": seg_index,
                "start": float(data["start"]),
                "end": float(data["end"]),
                "audio_text": " ".join(word for _, word in data["audio_text_parts"]),
            }
        )

    if matched_ordered_list:
        for i in range(len(matched_ordered_list) - 1):
            matched_ordered_list[i]["end"] = matched_ordered_list[i + 1]["start"]

        matched_ordered_list[0]["start"] = 0.0
        matched_ordered_list[-1]["end"] = total_duration

        for i, obj in enumerate(matched_ordered_list):
            if i == 0:
                obj["end"] = max(0.0, obj["end"] - 0.2)
            elif i == len(matched_ordered_list) - 1:
                obj["start"] = max(0.0, obj["start"] - 0.2)
            else:
                obj["start"] = max(0.0, obj["start"] - 0.2)
                obj["end"] = max(0.0, obj["end"] - 0.2)

    return matched_ordered_list


def get_audio_duration(audio_path, fallback_matches):
    try:
        return float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                text=True,
            ).strip()
        )
    except Exception:
        if fallback_matches:
            return fallback_matches[-1]["end"] + 1.0
        return 0.0


# === Post-generation validation helpers ===


def build_check_result(
    name, ok, summary, findings=None, metrics=None, skipped=False
):
    return {
        "name": name,
        "ok": ok,
        "skipped": skipped,
        "summary": summary,
        "findings": findings or [],
        "metrics": metrics or {},
    }


def get_book_folder(book_info):
    return book_info.get("folder_name") or os.getcwd()


def resolve_book_path(book_info, file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(get_book_folder(book_info), file_name)


def iter_audio_files(book_info):
    audio_extension = book_info.get("audio_extension")
    if not audio_extension:
        return []

    pattern = os.path.join(get_book_folder(book_info), f"*{audio_extension}")
    return sorted(os.path.basename(path) for path in glob.glob(pattern))


def iter_smil_files(book_info):
    pattern = os.path.join(get_book_folder(book_info), "*.smil")
    return sorted(os.path.basename(path) for path in glob.glob(pattern))


def parse_smil_clock_value(value):
    if value is None:
        return None

    value = value.strip()
    if value.endswith("s"):
        value = value[:-1]

    try:
        return float(value)
    except ValueError:
        return None


def get_audio_inventory(book_info):
    # This inventory is the shared base for several post checks. It joins three pieces
    # of information about every split audio file: file name, transcript word count,
    # and measured duration.
    inventory = []

    for audio_file in iter_audio_files(book_info):
        json_file = os.path.splitext(audio_file)[0] + ".json"
        json_path = resolve_book_path(book_info, json_file)
        word_count = 0
        if os.path.exists(json_path):
            word_count = len(load_audio_tokens(json_path))

        inventory.append(
            {
                "audio_file": audio_file,
                "json_file": json_file,
                "duration": get_audio_duration(
                    resolve_book_path(book_info, audio_file), fallback_matches=[]
                ),
                "word_count": word_count,
            }
        )

    return inventory


def get_substantial_audio_files(book_info, min_duration=180.0, min_words=300):
    return [
        item
        for item in get_audio_inventory(book_info)
        if item["duration"] >= min_duration or item["word_count"] >= min_words
    ]


def parse_smil_audio_refs(book_info):
    # This is the central SMIL audit parser. It walks every generated SMIL file and
    # records both per-reference detail and per-audio aggregates used by multiple post
    # checks such as coverage, duplicate reuse, and invalid clip detection.
    refs = []
    clip_totals = defaultdict(float)
    clip_counts = defaultdict(int)
    smil_audio_counts = {}

    for smil_file in iter_smil_files(book_info):
        smil_path = resolve_book_path(book_info, smil_file)
        with open(smil_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "xml")

        valid_audio_count = 0
        for audio_el in soup.find_all("audio"):
            par = audio_el.find_parent("par")
            src = os.path.basename(audio_el.get("src", ""))
            clip_begin_raw = audio_el.get("clipBegin")
            clip_end_raw = audio_el.get("clipEnd")
            clip_begin = parse_smil_clock_value(clip_begin_raw)
            clip_end = parse_smil_clock_value(clip_end_raw)
            is_valid = (
                bool(src)
                and clip_begin is not None
                and clip_end is not None
                and clip_end > clip_begin
            )

            refs.append(
                {
                    "smil_file": smil_file,
                    "par_id": par.get("id") if par else "",
                    "audio_file": src,
                    "clip_begin": clip_begin,
                    "clip_end": clip_end,
                    "clip_begin_raw": clip_begin_raw,
                    "clip_end_raw": clip_end_raw,
                    "valid": is_valid,
                }
            )

            if is_valid:
                valid_audio_count += 1
                clip_totals[src] += clip_end - clip_begin
                clip_counts[src] += 1

        smil_audio_counts[smil_file] = valid_audio_count

    return {
        "refs": refs,
        "clip_totals": dict(clip_totals),
        "clip_counts": dict(clip_counts),
        "smil_audio_counts": smil_audio_counts,
    }


def get_packaged_epub_candidates(book_info):
    # Validation often needs to inspect "the real packaged EPUB", but that is not
    # always the same file path as `book_info['out_file']`. This helper enumerates the
    # likely candidates in preference order without yet deciding which one is truly the
    # processed package.
    candidates = []

    def add_candidate(path):
        if path:
            candidates.append(os.path.normpath(path))

    def add_epub_family(path):
        path = os.path.normpath(path)
        if path.endswith(".epub3"):
            add_candidate(
                os.path.normpath(
                    os.path.join(
                        os.path.dirname(path),
                        "..",
                        os.path.basename(path).replace(".epub3", ".epub"),
                    )
                )
            )
            add_candidate(path)
            add_candidate(path.replace(".epub3", ".epub"))
        elif path.endswith(".epub"):
            add_candidate(path)

    out_file = book_info.get("out_file")
    if out_file:
        add_epub_family(resolve_book_path(book_info, out_file))

    epub_file = book_info.get("epub_file")
    if epub_file:
        add_epub_family(resolve_book_path(book_info, epub_file))

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    return ordered_candidates


def inspect_epub_package(epub_path):
    # We need a way to distinguish a source/original EPUB from a processed output EPUB.
    # Overlay artifacts such as `smil/`, `readaloud.css`, and OPF `media-overlay`
    # attributes are strong signals that a package has passed through this pipeline.
    if not epub_path or not os.path.exists(epub_path):
        return None

    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            opf_file = next(
                (file_name for file_name in zf.namelist() if file_name.endswith(".opf")),
                None,
            )
            if not opf_file:
                return None

            with zf.open(opf_file) as f:
                soup = BeautifulSoup(f.read(), "xml")

            zip_names = set(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return None

    has_smil_files = any(file_name.startswith("smil/") for file_name in zip_names)
    has_readaloud_css = any(file_name.endswith("readaloud.css") for file_name in zip_names)
    has_media_overlays = any(
        item.get("media-overlay") for item in soup.find_all("item")
    )

    return {
        "epub_path": epub_path,
        "opf_file": opf_file,
        "opf_dir": posixpath.dirname(opf_file) or ".",
        "soup": soup,
        "zip_names": zip_names,
        "has_smil_files": has_smil_files,
        "has_readaloud_css": has_readaloud_css,
        "has_media_overlays": has_media_overlays,
        "processed": has_smil_files or has_readaloud_css or has_media_overlays,
    }


def get_packaged_epub_path(book_info):
    first_existing = None
    for candidate in get_packaged_epub_candidates(book_info):
        epub_data = inspect_epub_package(candidate)
        if not epub_data:
            continue
        if first_existing is None:
            first_existing = candidate
        if epub_data["processed"]:
            return candidate

    return first_existing


def load_epub_opf(book_info, require_processed=False):
    # Prefer a candidate that looks processed. If none exist and the caller explicitly
    # requires processed output, return a sentinel payload instead of a false-positive
    # failure against the original source EPUB.
    candidate_paths = get_packaged_epub_candidates(book_info)
    first_existing = None

    for candidate in candidate_paths:
        epub_data = inspect_epub_package(candidate)
        if not epub_data:
            continue

        epub_data["candidate_paths"] = candidate_paths
        if first_existing is None:
            first_existing = epub_data
        if epub_data["processed"]:
            return epub_data

    if require_processed:
        return {
            "candidate_paths": candidate_paths,
            "processed": False,
        }

    return first_existing


def get_manifest_href_map(opf_soup, opf_dir):
    by_id = {}
    by_path = {}

    for item in opf_soup.find_all("item"):
        item_id = item.get("id")
        href = item.get("href")
        resolved_path = None
        if href:
            resolved_path = posixpath.normpath(posixpath.join(opf_dir, href))

        if item_id:
            by_id[item_id] = item
        if resolved_path:
            by_path[resolved_path] = item

    return {"by_id": by_id, "by_path": by_path}


def test_missing_long_audio(book_info, min_duration=180.0, min_words=300):
    # This check is aimed at "big obvious misses": long or word-heavy audio files that
    # never appear in any SMIL. The first and last split audio files are exempt because
    # audiobook intros/outros often have no sensible overlay target.
    smil_files = iter_smil_files(book_info)
    if not smil_files:
        return build_check_result(
            "missing_long_audio",
            True,
            "No SMIL files were found to validate.",
            metrics={"smil_files": 0},
            skipped=True,
        )

    substantial_audio = get_substantial_audio_files(book_info, min_duration, min_words)
    audio_files = iter_audio_files(book_info)
    if not substantial_audio:
        return build_check_result(
            "missing_long_audio",
            True,
            "No substantial audio files were found for this book.",
            metrics={"substantial_audio_files": 0},
            skipped=True,
        )

    smil_data = parse_smil_audio_refs(book_info)
    referenced_audio = {
        ref["audio_file"] for ref in smil_data["refs"] if ref["valid"] and ref["audio_file"]
    }
    allowed_missing = set()
    if audio_files:
        allowed_missing.add(audio_files[0])
        allowed_missing.add(audio_files[-1])

    missing_substantial = [
        item for item in substantial_audio if item["audio_file"] not in referenced_audio
    ]
    allowed_boundary_missing = [
        item for item in missing_substantial if item["audio_file"] in allowed_missing
    ]
    findings = [
        item for item in missing_substantial if item["audio_file"] not in allowed_missing
    ]

    return build_check_result(
        "missing_long_audio",
        ok=not findings,
        summary=(
            "No substantial interior audio files are missing from SMIL references."
            if not findings
            else f"{len(findings)} substantial interior audio file(s) are not referenced by any SMIL."
        ),
        findings=findings,
        metrics={
            "substantial_audio_files": len(substantial_audio),
            "referenced_audio_files": len(referenced_audio),
            "allowed_missing_boundary_audio_files": sorted(allowed_missing),
            "missing_but_allowed_count": len(allowed_boundary_missing),
        },
    )


def test_missing_transcripts(book_info):
    # The pass condition is intentionally simple: every `.m4a` should have a readable
    # sibling `.json` with at least one usable word. This catches missing, broken, or
    # empty transcription artifacts before later stages fail more opaquely.
    audio_pattern = os.path.join(get_book_folder(book_info), "*.m4a")
    audio_files = sorted(os.path.basename(path) for path in glob.glob(audio_pattern))
    if not audio_files:
        return build_check_result(
            "missing_transcripts",
            True,
            "No .m4a files were found to validate.",
            metrics={"audio_files": 0},
            skipped=True,
        )

    findings = []
    for audio_file in audio_files:
        json_file = os.path.splitext(audio_file)[0] + ".json"
        json_path = resolve_book_path(book_info, json_file)
        if not os.path.exists(json_path):
            findings.append(
                {
                    "audio_file": audio_file,
                    "json_file": json_file,
                    "issue": "missing_transcript",
                }
            )
            continue

        try:
            word_count = len(load_audio_tokens(json_path))
        except Exception:
            findings.append(
                {
                    "audio_file": audio_file,
                    "json_file": json_file,
                    "issue": "unreadable_transcript",
                }
            )
            continue

        if word_count < 1:
            findings.append(
                {
                    "audio_file": audio_file,
                    "json_file": json_file,
                    "issue": "empty_transcript",
                }
            )

    return build_check_result(
        "missing_transcripts",
        ok=not findings,
        summary=(
            "Every .m4a file has a readable transcript with at least one word."
            if not findings
            else f"{len(findings)} .m4a file(s) are missing a usable transcript."
        ),
        findings=findings,
        metrics={"audio_files": len(audio_files)},
    )


def test_low_audio_coverage(
    book_info,
    min_duration=180.0,
    min_words=300,
    coverage_warn_ratio=0.60,
):
    # A file can appear in SMILs and still be suspiciously under-covered. This check
    # compares summed clip durations to the real audio duration to flag transcripts or
    # alignments that only captured a small fraction of a substantial audio file.
    smil_files = iter_smil_files(book_info)
    if not smil_files:
        return build_check_result(
            "low_audio_coverage",
            True,
            "No SMIL files were found to validate.",
            metrics={"smil_files": 0},
            skipped=True,
        )

    substantial_audio = get_substantial_audio_files(book_info, min_duration, min_words)
    if not substantial_audio:
        return build_check_result(
            "low_audio_coverage",
            True,
            "No substantial audio files were found for this book.",
            metrics={"substantial_audio_files": 0},
            skipped=True,
        )

    smil_data = parse_smil_audio_refs(book_info)
    findings = []
    for item in substantial_audio:
        duration = item["duration"]
        covered_seconds = smil_data["clip_totals"].get(item["audio_file"], 0.0)
        clip_count = smil_data["clip_counts"].get(item["audio_file"], 0)
        coverage_ratio = covered_seconds / duration if duration > 0 else 0.0

        if clip_count > 0 and coverage_ratio < coverage_warn_ratio:
            findings.append(
                {
                    **item,
                    "covered_seconds": covered_seconds,
                    "coverage_ratio": coverage_ratio,
                    "clip_count": clip_count,
                }
            )

    return build_check_result(
        "low_audio_coverage",
        ok=not findings,
        summary=(
            "All referenced substantial audio files have acceptable SMIL coverage."
            if not findings
            else f"{len(findings)} substantial audio file(s) have low SMIL coverage."
        ),
        findings=findings,
        metrics={
            "substantial_audio_files": len(substantial_audio),
            "coverage_warn_ratio": coverage_warn_ratio,
        },
    )


def test_duplicate_audio_clips(book_info):
    # Exact duplicate reuse means the same source file and the same clip range appear
    # multiple times in SMILs. That is often a sign of alignment duplication rather
    # than legitimate reuse.
    smil_files = iter_smil_files(book_info)
    if not smil_files:
        return build_check_result(
            "duplicate_audio_clips",
            True,
            "No SMIL files were found to validate.",
            metrics={"smil_files": 0},
            skipped=True,
        )

    smil_data = parse_smil_audio_refs(book_info)
    grouped_refs = defaultdict(list)
    for ref in smil_data["refs"]:
        if not ref["valid"]:
            continue

        clip_key = (
            ref["audio_file"],
            round(ref["clip_begin"], 3),
            round(ref["clip_end"], 3),
        )
        grouped_refs[clip_key].append(
            {"smil_file": ref["smil_file"], "par_id": ref["par_id"]}
        )

    findings = []
    for (audio_file, clip_begin, clip_end), refs in grouped_refs.items():
        if len(refs) > 1:
            findings.append(
                {
                    "audio_file": audio_file,
                    "clip_begin": clip_begin,
                    "clip_end": clip_end,
                    "usage_count": len(refs),
                    "references": refs,
                }
            )

    findings.sort(
        key=lambda item: (
            item["audio_file"],
            item["clip_begin"],
            item["clip_end"],
        )
    )

    return build_check_result(
        "duplicate_audio_clips",
        ok=not findings,
        summary=(
            "No exact duplicate audio clip ranges were reused in SMIL files."
            if not findings
            else f"{len(findings)} exact duplicate audio clip range(s) were reused."
        ),
        findings=findings,
        metrics={
            "smil_files": len(smil_files),
            "valid_audio_refs": sum(1 for ref in smil_data["refs"] if ref["valid"]),
        },
    )


def test_overlapping_audio_clips(book_info, min_overlap_seconds=0.5):
    # Overlap is weaker evidence than an exact duplicate, but substantial overlap on
    # the same source audio often still indicates that two overlay segments are trying
    # to claim the same narrated region.
    smil_files = iter_smil_files(book_info)
    if not smil_files:
        return build_check_result(
            "overlapping_audio_clips",
            True,
            "No SMIL files were found to validate.",
            metrics={"smil_files": 0},
            skipped=True,
        )

    smil_data = parse_smil_audio_refs(book_info)
    refs_by_audio = defaultdict(list)
    for ref in smil_data["refs"]:
        if ref["valid"]:
            refs_by_audio[ref["audio_file"]].append(ref)

    findings = []
    for audio_file, refs in refs_by_audio.items():
        refs = sorted(refs, key=lambda ref: (ref["clip_begin"], ref["clip_end"]))

        for i, current_ref in enumerate(refs):
            for next_ref in refs[i + 1 :]:
                if next_ref["clip_begin"] >= current_ref["clip_end"]:
                    break

                if (
                    round(current_ref["clip_begin"], 3) == round(next_ref["clip_begin"], 3)
                    and round(current_ref["clip_end"], 3) == round(next_ref["clip_end"], 3)
                ):
                    continue

                overlap_seconds = min(
                    current_ref["clip_end"], next_ref["clip_end"]
                ) - max(current_ref["clip_begin"], next_ref["clip_begin"])

                if overlap_seconds >= min_overlap_seconds:
                    findings.append(
                        {
                            "audio_file": audio_file,
                            "overlap_seconds": overlap_seconds,
                            "first": {
                                "smil_file": current_ref["smil_file"],
                                "par_id": current_ref["par_id"],
                                "clip_begin": current_ref["clip_begin"],
                                "clip_end": current_ref["clip_end"],
                            },
                            "second": {
                                "smil_file": next_ref["smil_file"],
                                "par_id": next_ref["par_id"],
                                "clip_begin": next_ref["clip_begin"],
                                "clip_end": next_ref["clip_end"],
                            },
                        }
                    )

    findings.sort(
        key=lambda item: (
            item["audio_file"],
            item["first"]["clip_begin"],
            item["second"]["clip_begin"],
        )
    )

    return build_check_result(
        "overlapping_audio_clips",
        ok=not findings,
        summary=(
            "No overlapping audio clip ranges were reused in SMIL files."
            if not findings
            else f"{len(findings)} overlapping audio clip pair(s) were found."
        ),
        findings=findings,
        metrics={
            "smil_files": len(smil_files),
            "audio_files_checked": len(refs_by_audio),
            "min_overlap_seconds": min_overlap_seconds,
        },
    )


def test_invalid_smil_clips(book_info):
    # This is a structural sanity check on the generated SMIL XML itself. A clip with a
    # missing src or non-increasing begin/end times is invalid regardless of how it was
    # produced.
    smil_files = iter_smil_files(book_info)
    if not smil_files:
        return build_check_result(
            "invalid_smil_clips",
            True,
            "No SMIL files were found to validate.",
            metrics={"smil_files": 0},
            skipped=True,
        )

    smil_data = parse_smil_audio_refs(book_info)
    findings = []
    for ref in smil_data["refs"]:
        issue = None
        if not ref["audio_file"]:
            issue = "missing_audio_src"
        elif ref["clip_begin"] is None:
            issue = "invalid_clip_begin"
        elif ref["clip_end"] is None:
            issue = "invalid_clip_end"
        elif ref["clip_end"] <= ref["clip_begin"]:
            issue = "clip_end_not_after_clip_begin"

        if issue:
            findings.append(
                {
                    "smil_file": ref["smil_file"],
                    "par_id": ref["par_id"],
                    "audio_file": ref["audio_file"],
                    "clip_begin": ref["clip_begin_raw"],
                    "clip_end": ref["clip_end_raw"],
                    "issue": issue,
                }
            )

    return build_check_result(
        "invalid_smil_clips",
        ok=not findings,
        summary=(
            "All SMIL audio clips have valid timing and src values."
            if not findings
            else f"{len(findings)} invalid SMIL audio clip(s) were found."
        ),
        findings=findings,
        metrics={
            "smil_files": len(smil_files),
            "audio_refs": len(smil_data["refs"]),
        },
    )


def test_overlay_without_audio(book_info):
    # An HTML overlay SMIL that contains no valid `<audio>` entries is usually a sign
    # that matching or alignment failed but packaging still created the overlay shell.
    smil_files = iter_smil_files(book_info)
    if not smil_files:
        return build_check_result(
            "overlay_without_audio",
            True,
            "No SMIL files were found to validate.",
            metrics={"smil_files": 0},
            skipped=True,
        )

    smil_data = parse_smil_audio_refs(book_info)
    findings = [
        {"smil_file": smil_file, "audio_count": smil_data["smil_audio_counts"][smil_file]}
        for smil_file in smil_files
        if smil_data["smil_audio_counts"].get(smil_file, 0) == 0
    ]

    return build_check_result(
        "overlay_without_audio",
        ok=not findings,
        summary=(
            "All SMIL files contain at least one valid audio clip."
            if not findings
            else f"{len(findings)} SMIL file(s) contain no valid audio clips."
        ),
        findings=findings,
        metrics={"smil_files": len(smil_files)},
    )


def test_missing_media_overlays(book_info):
    # This is the packaged-EPUB-side counterpart to the loose SMIL checks above. It
    # verifies that each matched HTML file is actually wired in the OPF to its overlay
    # SMIL, and that the referenced overlay file exists inside the package.
    matched_items = normalize_matched_list(book_info.get("matched_list", []))
    if not matched_items:
        return build_check_result(
            "missing_media_overlays",
            True,
            "No matched HTML files were provided in book_info.",
            metrics={"matched_html_files": 0},
            skipped=True,
        )

    opf_data = load_epub_opf(book_info, require_processed=True)
    if not opf_data or not opf_data.get("processed"):
        return build_check_result(
            "missing_media_overlays",
            True,
            "No post-processed EPUB with overlay artifacts was found for validation.",
            metrics={
                "matched_html_files": len({item['html_file'] for item in matched_items}),
                "candidate_paths": opf_data.get("candidate_paths", []) if opf_data else [],
                "processed_epub_found": False,
            },
            skipped=True,
        )

    manifest_map = get_manifest_href_map(opf_data["soup"], opf_data["opf_dir"])
    html_files = list(dict.fromkeys(item["html_file"] for item in matched_items))
    findings = []

    for html_file in html_files:
        html_item = manifest_map["by_path"].get(html_file)
        if not html_item:
            findings.append({"html_file": html_file, "issue": "html_item_missing_from_manifest"})
            continue

        overlay_id = html_item.get("media-overlay")
        if not overlay_id:
            findings.append({"html_file": html_file, "issue": "missing_media_overlay"})
            continue

        overlay_item = manifest_map["by_id"].get(overlay_id)
        if not overlay_item:
            findings.append(
                {
                    "html_file": html_file,
                    "media_overlay": overlay_id,
                    "issue": "overlay_id_missing_from_manifest",
                }
            )
            continue

        overlay_href = overlay_item.get("href")
        if not overlay_href:
            findings.append(
                {
                    "html_file": html_file,
                    "media_overlay": overlay_id,
                    "issue": "overlay_item_missing_href",
                }
            )
            continue

        overlay_path = posixpath.normpath(
            posixpath.join(opf_data["opf_dir"], overlay_href)
        )
        if overlay_path not in opf_data["zip_names"]:
            findings.append(
                {
                    "html_file": html_file,
                    "media_overlay": overlay_id,
                    "overlay_path": overlay_path,
                    "issue": "overlay_file_missing_from_epub",
                }
            )

    return build_check_result(
        "missing_media_overlays",
        ok=not findings,
        summary=(
            "All matched HTML files have valid media-overlay wiring in the OPF."
            if not findings
            else f"{len(findings)} HTML file(s) have overlay wiring issues in the OPF."
        ),
        findings=findings,
        metrics={
            "matched_html_files": len(html_files),
            "epub_path": opf_data["epub_path"],
            "candidate_paths": opf_data.get("candidate_paths", []),
            "processed_epub_found": True,
        },
    )


def run_post_checks(
    book_info,
    min_duration=180.0,
    min_words=300,
    coverage_warn_ratio=0.60,
    min_overlap_seconds=0.5,
):
    # The runner intentionally mixes pre-package and post-package checks. Individual
    # checks decide whether to pass, fail, or skip depending on which artifacts exist.
    # This lets the same function be called at different points in the pipeline.
    results = [
        test_missing_transcripts(book_info),
        test_missing_long_audio(book_info, min_duration=min_duration, min_words=min_words),
        test_low_audio_coverage(
            book_info,
            min_duration=min_duration,
            min_words=min_words,
            coverage_warn_ratio=coverage_warn_ratio,
        ),
        test_duplicate_audio_clips(book_info),
        test_overlapping_audio_clips(
            book_info, min_overlap_seconds=min_overlap_seconds
        ),
        test_invalid_smil_clips(book_info),
        test_overlay_without_audio(book_info),
        test_missing_media_overlays(book_info),
    ]
    ok_results = sum(1 for result in results if result["ok"] and not result["skipped"])
    skipped_results = sum(1 for result in results if result["skipped"])

    return {
        "ok": all(result["ok"] or result["skipped"] for result in results),
        "summary": (
            f"{ok_results} check(s) passed, {skipped_results} skipped, "
            f"{len(results) - ok_results - skipped_results} failed."
        ),
        "results": results,
    }


# === Segment-level alignment and SMIL generation ===


def create_smil_files(book_info, skip=True):
    # Output model:
    # one HTML file -> one SMIL file, even when several audio chunks contribute to it.
    matched_items = normalize_matched_list(book_info["matched_list"])
    if not skip:
        for file in glob.glob("*.smil"):
            os.remove(file)

    html_groups = {}
    for item in matched_items:
        html_groups.setdefault(item["html_file"], []).append(item)

    alignment_stats = {
        "no_audio_tokens": 0,
        "no_alignment": 0,
        "no_segments": 0,
    }

    for html_file, items in tqdm(html_groups.items(), desc="Generating SMIL", unit="file"):
        smil_filename = make_overlay_basename(html_file)

        if os.path.exists(smil_filename) and skip:
            continue

        # Alignment target is the marked HTML already stored in the working EPUB.
        segments, html_tokens = load_html_segments_and_tokens(book_info["out_file"], html_file)
        aggregated_matches = {}
        last_html_idx = 0

        for item in items:
            json_file = item["json_file"]
            audio_tokens = load_audio_tokens(json_file)
            if not audio_tokens:
                alignment_stats["no_audio_tokens"] += 1
                continue

            audio_filename = json_file.replace(".json", book_info["audio_extension"])
            # Search locally around the matched window first; widen only if needed.
            search_start = min(
                len(html_tokens),
                max(last_html_idx, max(0, item["window_start"] - 250)),
            )
            search_end = min(
                len(html_tokens),
                max(
                    item["window_end"] + len(audio_tokens) + 400,
                    search_start + len(audio_tokens) + 1200,
                ),
            )

            # Try progressively wider search windows before falling back to large scans.
            search_ranges = [(search_start, max(search_start, search_end))]
            if search_end < len(html_tokens):
                search_ranges.append((search_start, len(html_tokens)))
            if search_start > last_html_idx:
                search_ranges.append((last_html_idx, len(html_tokens)))

            best_match = None
            for range_start, range_end in search_ranges:
                if range_end <= range_start:
                    continue

                html_slice = html_tokens[range_start:range_end]
                if not html_slice:
                    continue

                # This is token-level alignment, not semantic matching.
                matcher = difflib.SequenceMatcher(
                    None,
                    [token["token"] for token in html_slice],
                    [token["token"] for token in audio_tokens],
                    autojunk=False,
                )
                raw_matches, matched_token_count, max_html_idx = build_raw_matches(
                    matcher.get_opcodes(), html_tokens, audio_tokens, html_offset=range_start
                )

                if best_match is None or matched_token_count > best_match["matched_token_count"]:
                    best_match = {
                        "raw_matches": raw_matches,
                        "matched_token_count": matched_token_count,
                        "max_html_idx": max_html_idx,
                    }

                if matched_token_count >= max(25, int(len(audio_tokens) * 0.2)):
                    break

            if not best_match or best_match["matched_token_count"] == 0:
                alignment_stats["no_alignment"] += 1
                continue

            matched_ordered_list = finalize_segment_timestamps(
                best_match["raw_matches"],
                segments,
                get_audio_duration(
                    os.path.join(book_info["folder_name"], audio_filename),
                    [],
                ),
            )

            if not matched_ordered_list:
                alignment_stats["no_segments"] += 1
                continue

            for segment_match in matched_ordered_list:
                seg_id = segment_match["id"]
                if seg_id in aggregated_matches:
                    continue

                aggregated_matches[seg_id] = {
                    **segment_match,
                    "audio_file": audio_filename,
                    "json_file": json_file,
                }

            last_html_idx = max(last_html_idx, best_match["max_html_idx"] + 1)

        # Build the grouped SMIL shell for this one HTML file.
        soup_smil = BeautifulSoup("<smil/>", "xml")
        smil = soup_smil.smil
        smil["xmlns"] = "http://www.w3.org/ns/SMIL"
        smil["xmlns:epub"] = "http://www.idpf.org/2007/ops"
        smil["version"] = "3.0"

        body = soup_smil.new_tag("body")
        smil.append(body)

        seq = soup_smil.new_tag(
            "seq",
            attrs={
                "id": make_overlay_id(html_file),
                "epub:textref": "../" + html_file,
                "epub:type": "chapter",
            },
        )
        body.append(seq)

        final_timestamps = {
            item["id"]: item
            for item in sorted(
                aggregated_matches.values(), key=lambda match: match["segment_index"]
            )
        }

        for seg in segments:
            seg_id = seg.get("id")

            if seg_id in final_timestamps:
                info = final_timestamps[seg_id]
                start_t = info["start"]
                end_t = info["end"]

                if end_t > start_t:
                    par = soup_smil.new_tag("par", id=seg_id)

                    text_src = f"../{html_file}#{seg_id}"
                    text_el = soup_smil.new_tag("text", src=text_src)
                    par.append(text_el)

                    audio_el = soup_smil.new_tag(
                        "audio",
                        src=f"../audio/{info['audio_file']}",
                        clipBegin=f"{start_t:.3f}s",
                        clipEnd=f"{end_t:.3f}s",
                    )
                    par.append(audio_el)

                    seq.append(par)

        # Write SMIL
        with open(smil_filename, "w", encoding="utf-8") as f:
            f.write(convert_soup_to_html(soup_smil))

    print_nonzero_summary(
        "SMIL summary",
        [
            ("transcript(s) with no alignable words", alignment_stats["no_audio_tokens"]),
            ("transcript(s) with no alignment", alignment_stats["no_alignment"]),
            ("transcript(s) with no matched segments", alignment_stats["no_segments"]),
        ],
    )


# === Packaging and OPF updates ===


def merge_files(book_info):
    # At this stage all generated assets still exist as loose files in the working
    # folder. This function copies them into the working `.epub3` archive under the
    # conventional `audio/`, `smil/`, and CSS paths expected by the OPF.
    css_zip_path = (
        f"{book_info['opf_dir']}/readaloud.css"
        if book_info["opf_dir"] != "."
        else "readaloud.css"
    )

    with zipfile.ZipFile(book_info["out_file"], "a") as f:
        # Get the list of file names in the ZIP
        for file in sorted(glob.glob(f"*{book_info['audio_extension']}")):
            f.write(file, f"audio/{file}")

        for file in sorted(glob.glob("*.smil")):
            f.write(file, f"smil/{file}")

        css_content = """\
            .-epub-media-overlay-active {
              background-color: #ffb;
            }
            """
        f.writestr(css_zip_path, css_content)


def post_processing_opf(book_info):
    # OPF rewrite responsibilities:
    # - declare packaged audio assets
    # - declare one overlay SMIL per matched HTML file
    # - write per-overlay and total media durations
    # - attach `media-overlay` references to HTML manifest items
    # - synthesize a nav document if the source package lacks one
    matched_items = normalize_matched_list(book_info["matched_list"])
    replacements = {}

    with zipfile.ZipFile(book_info["out_file"], "r") as f:
        for file_name in f.namelist():
            if ".opf" in file_name:
                break
        assert ".opf" in file_name

        with f.open(file_name) as opf_file:
            soup = BeautifulSoup(opf_file.read(), "lxml-xml")

        opf_file = file_name
        opf_dir = posixpath.dirname(opf_file) or "."
        package = soup.find("package")
        package["version"] = "3.0"
        prefix_value = package.get("prefix", "")
        media_prefix = "media: http://www.idpf.org/epub/vocab/overlays/#"
        if media_prefix not in prefix_value:
            package["prefix"] = (
                f"{prefix_value} {media_prefix}".strip()
                if prefix_value
                else media_prefix
            )

        metadata = soup.find("metadata")

        manifest = soup.find("manifest")

        # Remove previously generated overlay artifacts before re-adding them.
        for item in manifest.find_all("item"):
            if item.has_attr("media-overlay"):
                del item["media-overlay"]
            item_id = item.get("id", "")
            if (
                item_id.startswith("audio_")
                or item_id.startswith("html_overlay.")
                or item_id == "readaloud_style"
                or item_id == "generated_nav"
            ):
                item.decompose()

        for meta in metadata.find_all("meta", attrs={"property": "media:duration"}):
            meta.decompose()

        nav_item = manifest.find(
            "item",
            attrs={"properties": lambda value: value and "nav" in value.split()},
        )

        # If the source package lacks an XHTML nav, synthesize one from NCX.
        if not nav_item:
            ncx_item = manifest.find(
                "item", attrs={"media-type": "application/x-dtbncx+xml"}
            )
            if ncx_item:
                ncx_zip_path = posixpath.normpath(
                    posixpath.join(opf_dir, ncx_item["href"])
                )
                with f.open(ncx_zip_path) as ncx_file:
                    ncx_soup = BeautifulSoup(ncx_file.read(), "xml")

                nav_points = parse_ncx_nav_points(ncx_soup.find("navMap"))
                title_el = metadata.find(lambda tag: tag.name in {"dc:title", "title"})
                nav_zip_path = f"{opf_dir}/nav.xhtml" if opf_dir != "." else "nav.xhtml"
                nav_item = soup.new_tag(
                    "item",
                    attrs={
                        "id": "generated_nav",
                        "href": posixpath.basename(nav_zip_path),
                        "media-type": "application/xhtml+xml",
                        "properties": "nav",
                    },
                )
                manifest.insert(0, nav_item)
                replacements[nav_zip_path] = build_nav_document(
                    title_el.get_text(strip=True) if title_el else "Contents",
                    nav_points,
                )

        # Declare all packaged split-audio files.
        for file_name in sorted(glob.glob(f"*{book_info['audio_extension']}")):
            item = soup.new_tag(
                "item",
                attrs={
                    "id": f"audio_{file_name.replace('book_info.get(audio_extension)', '')}",
                    "href": get_relative_zip_href(opf_file, f"audio/{file_name}"),
                    "media-type": "audio/mp4",
                },
            )
            manifest.append(item)

        # Exactly one overlay manifest item is created per matched HTML file.
        html_overlay_files = list(dict.fromkeys(item["html_file"] for item in matched_items))
        total_duration = 0.0
        for html_file in html_overlay_files:
            file_name = make_overlay_basename(html_file)
            item = soup.new_tag(
                "item",
                attrs={
                    "id": make_overlay_id(html_file),
                    "href": get_relative_zip_href(opf_file, f"smil/{file_name}"),
                    "media-type": "application/smil+xml",
                },
            )
            manifest.append(item)

            smil_duration = get_smil_duration(file_name)
            total_duration += smil_duration

            duration_meta = soup.new_tag(
                "meta",
                attrs={
                    "property": "media:duration",
                    "refines": f"#{item['id']}",
                },
            )
            duration_meta.string = seconds_to_media_duration(smil_duration)
            metadata.append(duration_meta)

        item = soup.new_tag(
            "item",
            attrs={
                "id": "readaloud_style",
                "href": get_relative_zip_href(
                    opf_file,
                    f"{opf_dir}/readaloud.css" if opf_dir != "." else "readaloud.css",
                ),
                "media-type": "text/css",
            },
        )
        manifest.append(item)

        total_duration_meta = soup.new_tag("meta", attrs={"property": "media:duration"})
        total_duration_meta.string = seconds_to_media_duration(total_duration)
        metadata.append(total_duration_meta)

        # Attach overlay ids to the corresponding HTML manifest items.
        for html_file in html_overlay_files:
            el = soup.find(
                "item",
                attrs={
                    "href": lambda href, html_file=html_file: href
                    and posixpath.normpath(posixpath.join(opf_dir, href)) == html_file
                },
            )
            if el:
                el["media-overlay"] = make_overlay_id(html_file)

    replacements[opf_file] = convert_soup_to_html(soup)
    replace_files_in_zip(book_info["out_file"], replacements)
    # Move the working `.epub3` to the final `.epub` output path.
    shutil.move(
        book_info["out_file"], "../" + book_info["out_file"].replace(".epub3", ".epub")
    )
