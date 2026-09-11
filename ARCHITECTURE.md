# Project Structure

Developer-facing map of the codebase: what each module does, how the pipeline
fits together, and the non-obvious logic worth knowing before you change things.
This documents *functionality*, not line numbers. For installation and CLI usage
see [`README.md`](README.md). For the segmentation algorithm in depth see
[`docs/sentence_segmentation.md`](docs/sentence_segmentation.md).

## Overview

The project turns one audiobook file + one source `.epub` into a Media Overlay
EPUB (text that highlights in sync with narration). Work flows through eight
ordered stages: **prepare → split → transcribe → match → segment → smil →
package → validate**.

`generate_epub_overlay.py` is the CLI entry point and orchestrator. It imports
`pipeline_core.py` (the core engine) and drives its functions as discrete stages.
The source `.epub` is never modified in place.

Two pieces of shared context recur everywhere:

- **`book_info`** — a mutable dict threaded through every engine call. It carries
  discovered paths, the working-EPUB location, OPF metadata, and the
  backend/model/audio settings for the run. It is built in one place
  (`book_info_from_config`) and every file name in it is a basename that the engine
  resolves against `folder_name` (`resolve_book_path`); nothing depends on the
  process working directory.
- **`state.json`** — per-stage status plus a signature of the inputs/config,
  persisted after every stage so an interrupted run resumes automatically.

## Module map

| File | Role |
| --- | --- |
| `generate_epub_overlay.py` | CLI entry, config, `state.json` + resume, stage orchestration |
| `pipeline_core.py` | Core engine: split, transcribe, match, segment, SMIL, package, validate |
| `mark_sentence.py` | XHTML linearize → segment → reconstruct; NLTK tokenizer setup |
| `epub_reference_index.py` | Whole-EPUB `class`/`id` reference index that guides safe tag cleanup |
| `transcription_backend.py` | Platform-aware backend selection (mlx / whisperx) + adapters |
| `tests/` | Pytest suite (see [Tests](#tests)) |
| `requirements.txt` | Dependencies (platform-gated `whisperx` vs `mlx-whisperx`) |
| `docs/sentence_segmentation.md` | Deep dive on sentence/segment boundary detection |
| `run.py` | Local batch runner over a folder tree. **Gitignored**, not shipped |

## The pipeline stages

Each stage is a function in the orchestrator that wraps one or more engine calls.

| Stage | What it produces |
| --- | --- |
| `prepare` | Copies the audio + EPUB into the work dir, locates the OPF, seeds `book_info` |
| `split` | Cuts the audiobook into numbered audio chunks (`000.m4a`, `001.m4a`, …) |
| `transcribe` | A transcript `.json` next to each audio chunk |
| `match` | `matched_list.json`: which transcript chunk(s) map to which HTML file |
| `segment` | Segment spans + read-aloud CSS injected into the working EPUB (`segmented.epub3`) |
| `smil` | One SMIL overlay per HTML file, with real timestamps |
| `package` | Bundles audio/SMIL/CSS, rewrites the OPF, emits the final `*.media-overlay.epub` |
| `validate` | `validation.json`: read-only audit of the result |

## State & resume model

This is the backbone of the orchestrator and the part most worth understanding.

- **Signature-based change detection.** A signature is built from a fingerprint
  of the audio + EPUB (path, size, mtime) plus key config fields. On startup the
  saved signature is compared to the current one. If they match, the run resumes;
  if they differ, the inputs/config changed.

- **Reset preserves expensive work.** When the signature no longer matches, the
  *derived downstream* artifacts are deleted: `matched_list.json`,
  `segmented.epub3`, the working `run/<stem>.epub3` (it carries segment ids),
  `run/*.smil`, `validation.json`, the packaged `<work>/<stem>.epub` **and the
  final output EPUB**. Only these known paths are removed, never a wildcard over
  the work-dir root (which may hold unrelated EPUBs). The audio chunks and
  transcripts in `run/` are **never** thrown away by a reset; their compatibility
  stamps decide reuse per artifact. Transcription is slow and non-deterministic,
  so it is only redone when the inputs actually change in a way that invalidates it.

- **Downstream invalidation.** Whenever a stage actually *executes* (as opposed to
  being skipped), the derived artifacts of every later stage are deleted and those
  stages reset to pending. A re-run match therefore always leads to a re-run
  segment/smil/package/validate; a packaged EPUB can never be reused against a
  newer SMIL set. `prepare` is exempt (it re-runs after every completed run because
  packaging moves the working EPUB away, without changing any input).

- **Per-stage reconciliation is the real resume engine.** On every run, before
  running a stage, the orchestrator re-validates that stage's on-disk artifacts
  against the *current* config. If they check out, the stage is skipped and its
  prior result reused; otherwise it re-runs. `state.json` is treated as a hint,
  not the source of truth — reconciliation can recover correct behavior even
  when the recorded status is stale or the file was edited. Some non-obvious
  reconciliation behavior:
  - `split` is considered done only if every planned chunk file exists, carries a
    matching compatibility stamp **and** passes the duration-completeness check
    (see below). Reconcile (like the stage itself) also prunes `NNN.*` chunk,
    transcript and stamp files whose ordinal is outside the current plan.
  - `match` records the list of transcripts it ran over
    (`artifacts.match_transcript_files`); it is reusable only while that list
    equals the current transcript set and every matched transcript is in it. Some
    transcripts legitimately match nothing (intro/outro chunks), so "every
    transcript appears in the matched list" is deliberately *not* the test.
  - `segment` keeps the working EPUB and the `segmented.epub3` snapshot in sync:
    whichever one already has segment ids is copied to the other.
  - `package` accepts the final output EPUB or the working packaged EPUB only if
    it is byte-identical (size + SHA-256) to the package this state recorded and
    contains every expected `audio/` and `smil/` entry; the missing copy is
    backfilled from the present one. A processed EPUB with the right entry names
    but no record (e.g. after a reset) is not enough.
  - `validate` is bound the same way to the packaged EPUB it validated.

- **Atomic writes.** `state.json` and other JSON/text artifacts are written via a
  temp-file-then-rename so an interrupted write can't corrupt them.

## Work-dir layout

The work dir defaults to `<output-dir>/.<epub-stem>.epubmo`.

```
<work_dir>/                       # root
  run/                            # copied source, audio chunks (NNN.ext),
                                  #   transcripts (*.json), SMIL (*.smil),
                                  #   working <stem>.epub3
  logs/pipeline.log               # detailed stage-by-stage log
  state.json                      # persistent state + signature
  matched_list.json               # match-stage output
  segmented.epub3                 # snapshot after segment marking
  validation.json                 # validate-stage results
  <stem>.epub                     # packaged working EPUB

<output-dir>/
  <stem>.media-overlay.epub       # FINAL output (lives OUTSIDE the work dir)
```

## Engine functionality and tricky logic

Everything below lives in `pipeline_core.py` unless noted. These are the
non-obvious decisions — the things that look wrong until you know why.

### Audio chunk planning & splitting

- **Chapters are preferred but treated as approximate.** Planning reads embedded
  chapter markers (`ffprobe -show_chapters`) and only falls back to fixed-size
  chunks (`--chunk-seconds`, default 600s) when there are no usable chapters.
  Chapter boundaries are used only to get manageable audio segments — they are
  **not** trusted as reading order. Final text correspondence is decided later in
  the match stage. The first chunk is snapped to 0 and the last to the audio
  stream's end when the markers leave more than half a second out, so no audio
  outside the chapter markers is silently dropped.
- **The plan is the only source of chunk names.** Every later stage (transcribe,
  match, package, OPF rewrite) takes its chunk/transcript list from
  `planned_chunk_basenames`, never from a directory glob, and split prunes
  `NNN.*` files whose ordinal is not in the plan. Otherwise chunks left by an
  older plan (new audio, different `--chunk-seconds`) were re-transcribed,
  matched, packaged and declared in the OPF.
- **Too-short chunks are merged forward.** A chapter shorter than ~10s is folded
  into a neighbor. The reason is concrete: a few-second header chunk can fall
  entirely inside one AAC keyframe interval, so `ffmpeg` stream-copy has no clean
  packet boundary to cut on and fails. A too-short *final* chunk is absorbed into
  its predecessor.
- **Chunk ids are merged-list indices**, not original chapter numbers — names are
  zero-padded ordinals over the post-merge list.
- **Completeness uses codec-aware tolerances.** A chunk is "complete" if its
  actual audio-stream duration matches the planned duration within tolerance:
  **1.0s for `copy`, 5.0s for `aac`**. The wider AAC tolerance exists because
  encoder priming delay and frame padding make the reported duration drift more
  than a second; without it, valid chunks would read as "incomplete" and trigger
  a needless full re-split.
- **Reuse + bounded parallelism.** Existing complete chunks are skipped on
  re-runs. Splitting is serial for a single chunk or `--split-jobs 1`; otherwise
  a thread pool (clamped to `[1, cpu_count]`) cuts chunks in parallel and cancels
  all work if any chunk fails.

### Transcription

- Transcribes each **split chunk**, not the original audiobook, writing a sibling
  `.json` per chunk; those JSONs are the canonical transcript artifacts for both
  matching and alignment. A transcript is reused only when its `.meta` stamp
  matches the current backend/model/language and its parent chunk's stamp
  (idempotent/resumable).
- Models are loaded once per run and released after the stage
  (`transcription_backend.release_models`); whisperx used to reload its ASR and
  alignment models for every chunk.
- Backend output is suppressed (stdout/stderr redirected) because WhisperX is
  noisy, but on failure the captured output is re-raised in the error so the real
  cause survives.
- The ASR engine is indirected through `transcription_backend.py`, so the
  pipeline isn't hard-wired to one engine (see that module below).

### Matching: transcript chunks → HTML files (the hard part)

This is the most subtle code in the project. It maps each transcript chunk onto
the right HTML file(s), tolerating that chapter audio and chapter text rarely
line up one-to-one.

- **Spine order, not filename order.** HTML is walked in OPF spine order. Front
  matter, appendices, and generated nav files sort unpredictably by filename, so
  spine order is the authoritative reading order and gives a monotonic
  progression the matcher can bias toward.
- **Overlapping token windows.** Each file's normalized token stream is cut into
  overlapping windows (size 160, step 80). The overlap lets a transcript that
  starts *mid-chapter* anchor against an interior window instead of only the
  file's beginning.
- **Two-stage lexical scoring (deliberately not embeddings).** Short openings and
  headers are noisy, so matching stays lexical: a cheap multiset token-overlap
  count prefilters and ranks candidates, then the top handful are rescored with a
  `difflib` sequence ratio over just the probe-length prefix. The final score
  blends overlap, sequence ratio, and a small forward-order bonus
  (`score_probe_candidates`; thresholds are the `MATCH_*` constants).
- **One alignment routine.** Estimating how far into a transcript a file reaches
  (matching) and timing segments (SMIL) both call `align_tokens`, which runs
  `difflib` over progressively widening HTML ranges (`alignment_search_ranges`)
  and stops once enough tokens match.
- **Failures are loud.** A transcript that cannot be parsed or matched raises with
  the file name; silently counting it would drop that chapter from the overlay.
- **Forward bias is soft in the main path.** Being in reading order only earns a
  small bonus; a slightly out-of-order but strongly matching window can still
  win, and backward jumps are merely logged (intro/preface files are exempt).
- **Multi-chunk chapters / mid-file starts.** After accepting a primary match,
  the matcher estimates how far into the transcript that file consumed (another
  `difflib` alignment), then walks an audio cursor forward and tries to attach
  the *next consecutive spine files* as additional spans of the same transcript.
  A deliberate backtrack (~24 tokens) rewinds the cursor so the next span
  re-checks an overlap region instead of dropping boundary words; each span's
  audio-end is back-filled from the next span's start so spans don't overlap.
- **Short-page / divider-page fallback** (`select_short_page_match`). Tiny
  divider/heading pages (e.g. "PART 6 / THE GREEN CARD MAN") can't clear the
  normal thresholds, so they get a dedicated path. A short page is accepted only if it (a) is genuinely a
  short-page candidate, (b) does **not** move backward in spine order (here the
  forward gate is *hard*, unlike the soft main path), and (c) its tokens appear
  as a contiguous run at the probe start. That prefix check tolerates 1–2 stray
  leading ASR tokens and skips up to 2 printed-but-unspoken running-header tokens
  (e.g. a per-page date stamp), while still requiring a minimum contiguous run to
  reject trivial coincidences.

### Token normalization (shared by match + SMIL)

A small but load-bearing detail: hyphen/dash-joined compounds (`dry-swallowed`)
are split into separate tokens because ASR emits them as two words, but
apostrophes are **not** split (`don't` → `dont` already matches the single ASR
word). Both the matcher and the aligner use this same normalization, so they
agree on what a "token" is.

### Segment marking (`mark_segments` + `mark_sentence.py`)

- `mark_segments` injects stable segment-span ids into each matched HTML file
  **exactly once** (even if several chunks map to that chapter) and attaches the
  read-aloud CSS, writing into the working EPUB.
- **Redundant-markup cleanup is evidence-based and fail-closed.** Before
  segmenting, unreferenced `class`/`id` values, now-bare `<span>`s and empty
  inline leftovers are removed using the whole-EPUB reference index (see
  `epub_reference_index.py`). The index sets are treated as *complete*, so an
  empty set means "strip everything"; when the index cannot be built the cleanup
  is skipped entirely (`None` sets) rather than run with empty ones. Whitespace-only
  inline tags are unwrapped, not deleted, so `foo<i> </i>bar` keeps its space;
  empty block tags (visible spacers) and positional tags (`<td>`, `<li>`) are never
  removed.
- **Why not just `get_text()`?** Naive "flatten text → segment → reinject"
  flattens inline markup, drops empty anchors, drifts/loses void tags, and
  changes whitespace across parse/serialize cycles. `mark_sentence.py` instead
  uses an explicit **linearize → segment → reconstruct** pipeline:
  - *Linearize* builds three positionally-aligned views of a block: the
    `linear_text` the segmenter sees, a per-character `char_map` back to the
    original DOM node + active inline-formatting stack, and a list of zero-width
    nodes. The hard invariant is `len(char_map) == len(linear_text)`.
  - Void tags (`<br>`, `<img>`) become single atomic map entries with placeholder
    characters so they can't silently vanish; empty structural anchors are
    recorded as "insert at offset N" so footnote/nav targets survive despite
    contributing no visible text.
  - *Reconstruct* rebuilds spans from the boundaries + `char_map`, re-inserts the
    zero-width nodes at their exact offsets, groups contiguous same-format runs
    into single text nodes (no per-character "span soup"), and avoids
    whitespace-only wrapper tags that serialize inconsistently. Every container
    found inside a leaf block is treated as a formatting wrapper and re-emitted
    (an allowlist used to drop unlisted tags silently); `INLINE_TAGS` now only
    tells the cleanup which empty leftovers may be removed.
  - Leaf blocks include table cells and captions (`td`, `th`, `caption`) so table
    text is highlightable; a block whose direct children include another block
    is skipped in favour of its children.
  - The document is parsed once; its visible text is captured *before* cleanup and
    compared with the final DOM, so a cleanup pass that altered text is caught,
    not only a segmentation error.
- NLTK tokenizer data is bootstrapped on first run and cached under
  `~/.cache/epub-media-overlay/nltk_data`. See
  [`docs/sentence_segmentation.md`](docs/sentence_segmentation.md) for the
  boundary-detection rules and edge cases.

### SMIL alignment (`create_smil_files`)

- **One SMIL per HTML file**, even when several audio chunks contribute; matches
  are aggregated per file and de-duplicated by segment id. Every SMIL is
  regenerated on each run of the stage (seam resolution works across files, so a
  leftover from an interrupted run would be inconsistent with its neighbours);
  skipping the stage when its outputs are complete is the orchestrator's job.
- Each chunk's transcript words are aligned to that file's segment-aware HTML
  tokens with `difflib` over a hinted, progressively-widening search range,
  stopping early once enough tokens match.
- **Timestamp envelope = min(start) … max(end)** of a segment's aligned words.
  This is robust to the jagged equal-runs `difflib` produces and to hyphen-split
  sub-tokens reusing a parent word's times.
- **Interior gap interpolation.** An interior segment that matched nothing still
  gets an evenly interpolated time span so its on-screen text highlights instead
  of being absorbed by a neighbor — but **only for strictly positive gaps**, and
  never for leading/trailing gaps. Non-positive gaps (common right before an
  ellipsis, where punctuation-only spans produce non-monotonic envelopes) are
  skipped to avoid overlapping clips.
- **Anchoring.** Segment starts are first clamped to be non-decreasing (a
  non-monotonic envelope would otherwise yield a dropped `<par>` and overlapping
  neighbours), then the first segment's start is pinned to 0.0 and the last
  segment's end to the audio's measured duration, and inter-segment gaps are
  closed by snapping each end to the next start, so the overlay fully covers the
  chunk's audio with no holes. Clip times are emitted as 3-decimal SMIL clock
  values, and a `<par>` is written only when `end > start` (`write_smil_file`).

### Packaging & OPF rewrite (`post_processing_opf`)

- Bumps the package to EPUB 3.0 and adds the `media:` overlays prefix
  (idempotently).
- **Re-runs don't accumulate cruft.** It first strips previously generated
  `media-overlay` attributes, generated manifest items, and prior
  `media:duration` metas before re-adding them, so reprocessing stays clean.
- Declares one manifest item per audio file and exactly one overlay SMIL item per
  matched HTML file. Per-overlay `media:duration` is **summed from the SMIL's clip
  durations** (not max-clip-end), because one grouped SMIL can reference several
  audio files; a package-level total is also written.
- If the source has no XHTML `nav`, one is **synthesized from the NCX** and added
  to the manifest.
- **Gotcha:** the final step moves the working `.epub3` to the destination the
  orchestrator passes in (`<work_dir>/<stem>.epub`), which is then copied to the
  final output path. The "real" packaged output is therefore *not* a file inside
  `run/` — validation has to go find the processed package.
- The OPF is located through `find_opf_path` (META-INF/container.xml first, then
  the first `*.opf` entry); every other OPF lookup in the codebase uses the same
  helper.

### Validation (`run_post_checks`)

- **Read-only / audit-only.** Every `check_*` function inspects existing artifacts
  and reports pass / skip / findings; nothing is generated or repaired. The audio
  inventory (one `ffprobe` per chunk, transcript status, word count) and the SMIL
  clip parse are computed once and shared by all checks.
- **Mixes pre- and post-package checks** so the same runner works at different
  points: loose-artifact checks and packaged-EPUB checks coexist, and each one
  *skips* (counts as pass) when its inputs aren't present yet. The packaged-side
  checks must locate the *processed* package (not the source EPUB) first — see
  the packaging gotcha above.
- The checks catch distinct failure classes, e.g.: transcripts missing or empty;
  substantial audio referenced by no SMIL (first/last chunk exempt as
  intro/outro); referenced audio whose clips cover too little of its real
  duration; duplicate or overlapping clips; structurally invalid SMIL; SMIL
  shells with no audio; substantive HTML *between* matched chapters that was never
  matched (the clearest "we skipped part of the book" signal); and HTML not wired
  to an existing overlay in the packaged OPF.

## `transcription_backend.py`

Isolates ASR engine choice so the rest of the pipeline is engine-agnostic.

- Auto-detects the backend: **mlx** on Apple Silicon, **whisperx** elsewhere
  (overridable via `--backend`). `BACKENDS`, `DEFAULT_MODEL_BY_BACKEND` and
  `REQUIRED_MODULE_BY_BACKEND` are the only tables.
- Exposes a single `transcribe_file` that dispatches to the mlx or whisperx
  adapter. whisperx ASR and alignment models are cached per (model, device,
  compute type) / (language, device) for the run and dropped by `release_models`.
- `apply_mlx_cache_limit` optionally caps the mlx Metal buffer cache during
  transcription, so freed GPU/unified memory isn't retained unbounded across
  chunks (controlled by `--mlx-cache-gb`; a no-op on non-mlx backends).

## Module-level environment & config

Set at import time in `transcription_backend.py` because they must take effect
before the ASR/ML libraries load (which happens lazily in that module):

- `TOKENIZERS_PARALLELISM=false` — avoid HuggingFace fork-parallelism
  warnings/deadlocks (the pipeline does its own threading).
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` — keep the legacy torch checkpoint-load
  path working under newer torch "weights-only" defaults.
- `MPLBACKEND=Agg` — force headless matplotlib so a dependency can't try to open
  a GUI window in a non-interactive run.

Tuning constants (all in `pipeline_core.py`): split duration tolerances and the
min-chunk threshold (see splitting above); the match-window geometry
(size/step/min-tokens); and the short-page fallback thresholds (leading-token
tolerance, header-skip, minimum coverage, minimum contiguous run).

## Tests

Tests live in `tests/` and run under pytest (configured via `pyproject.toml`,
which puts the repo root on `sys.path` so the flat source modules import by bare
name).

| File | Covers |
| --- | --- |
| `tests/test_alignment.py` | Token splitting, `align_tokens`, segment alignment and interior gap interpolation, overlap avoidance, monotonic clamp, short-page matching, seam resolution, coverage-gap and unmatched-spine checks |
| `tests/test_segmentation.py` | Sentence/phrase boundary detection (`mark_sentence`), as parametrized `(text, expected_segments)` cases |
| `tests/test_mark_sentences_inline.py` | `mark_sentences` round-trips: inline tags survive, reference-index semantics (None skips cleanup), text integrity, anchor placement, CSS link |
| `tests/test_html_cleaner.py` | `preprocess_remove_redundant_tags`: unreferenced class/id stripping, bare-span unwrap, empty-tag removal, safety controls |
| `tests/test_epub_reference_index.py` | `epub_reference_index`: CSS (incl. `:not()`/`:is()` arguments, `@media`), link/NCX/OPF-guide fragments, JS tokens; optional integration test against a real EPUB |
| `tests/test_compat_stamps.py` | Chunk/transcript compatibility stamps and atomic JSON writes |
| `tests/test_split_plan.py` | Chunk planning (edge snapping) and stale-chunk pruning |
| `tests/test_packaging_serialization.py` | OPF serialization, media types, `find_opf_path`, folder-addressed `preprocess`/`merge_files`/`post_processing_opf`, validation inventory |
| `tests/test_orchestrator_state.py` | Reset scope, downstream invalidation, package/match/split reconcile rules |
| `tests/test_transcription_backend.py` | whisperx model caching and release |

Run the whole suite:

```bash
pytest
```

## See also

- [`README.md`](README.md) — installation, CLI options, outputs, troubleshooting.
- [`docs/sentence_segmentation.md`](docs/sentence_segmentation.md) — segmentation
  boundary detection and edge cases.
