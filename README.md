# epub-media-overlay

Generate a Media Overlay EPUB from an audiobook `.m4b` file and a source `.epub`.

This project provides a command-line pipeline that automatically resumes prior work when possible and:

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

The pipeline downloads the required NLTK tokenizer data automatically on first run and caches it under `~/.cache/epub-media-overlay/nltk_data`.

## Usage

Basic run:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub
```

Transcription defaults:

- model: `mlx-community/whisper-turbo`
- language: `en`

The `--language` setting is used for both transcription and HTML sentence segmentation.

Override them when needed:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --model mlx-community/whisper-large-v3-turbo \
  --language en
```

Run behavior:

- if compatible work already exists, the pipeline resumes automatically
- if no work exists yet, the pipeline starts from the beginning
- use `--fresh` only when you want to discard previous work and restart from scratch
- if `--output-dir` is omitted, the final EPUB is written next to the source EPUB

Restart from scratch:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --fresh
```

## Outputs

Final output:

- `<book-stem>.media-overlay.epub`
- default location: the source EPUB folder

Working directory default:

- `<resolved-output-dir>/.<book-stem>.epubmo`

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
- The console output and `logs/pipeline.log` include detailed stage-by-stage progress and timings.

## Files

- `generate_epub_overlay.py`: CLI wrapper and state manager with automatic resume
- `pipeline_core.py`: EPUB/audio matching, SMIL generation, packaging, and validation logic
- `mark_sentence.py`: HTML segmentation logic used for overlay targets

## Troubleshooting

If `ffmpeg` or `ffprobe` are missing, install them and ensure they are on `PATH`.

If the first run cannot download NLTK data automatically, make sure the machine has network access and write permission for `~/.cache/epub-media-overlay/nltk_data`.

If a run is interrupted, rerun the same command and the pipeline resumes automatically.
