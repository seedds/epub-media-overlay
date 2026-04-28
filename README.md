# epub-media-overlay

Generate a Media Overlay EPUB from an audiobook `.m4b` file and a source `.epub`.

This project provides a resumable command-line pipeline that:

- prepares a working EPUB copy
- splits audiobook chapters into audio chunks
- transcribes each chunk with `mlx-whisperx`
- matches transcript chunks to EPUB HTML files
- injects stable segment ids into HTML
- generates SMIL overlays
- packages the final Media Overlay EPUB
- validates the generated result

## Requirements

- Python 3
- `ffmpeg`
- `ffprobe`
- Apple Silicon environment suitable for `mlx-whisperx`

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install NLTK tokenizer data once:

```bash
python -m nltk.downloader punkt punkt_tab
```

## Files

- `generate_epub_overlay.py`: resumable CLI wrapper and stage/state manager
- `pipeline_core.py`: EPUB/audio matching, SMIL generation, packaging, and validation logic
- `mark_sentence.py`: HTML segmentation logic used for overlay targets

## Usage

Basic run:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output
```

Resume a previous run:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output \
  --resume
```

Restart from scratch:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output \
  --fresh
```

Rerun from a specific stage:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output \
  --resume \
  --from-stage segment
```

Force a stage and everything after it to rerun:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output \
  --resume \
  --force-stage smil
```

Validate only:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output \
  --resume \
  --validate-only
```

## Stages

- `prepare`
- `split`
- `transcribe`
- `match`
- `segment`
- `smil`
- `package`
- `validate`

## Outputs

Final output:

- `<book-stem>.media-overlay.epub`

Working directory default:

- `<output-dir>/.<book-stem>.epubmo`

Important working artifacts:

- `state.json`
- `matched_list.json`
- `segmented.epub3`
- `validation.json`
- `logs/pipeline.log`

## Notes

- The source `.epub` is not modified directly.
- The pipeline expects chapter metadata in the input `.m4b` for chunk generation.
- Validation checks packaging, SMIL clip quality, transcript coverage, and OPF media-overlay wiring.

## Troubleshooting

If `ffmpeg` or `ffprobe` are missing, install them and ensure they are on `PATH`.

If NLTK data is missing, run:

```bash
python -m nltk.downloader punkt punkt_tab
```

If a run is interrupted, restart with `--resume`.
