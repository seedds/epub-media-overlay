# epub-media-overlay

Generate a Media Overlay EPUB from an audiobook `.m4b` file and a source `.epub`.

This project provides a command-line pipeline that automatically resumes prior work when possible and:

- prepares a working EPUB copy
- splits audiobook chapters into audio chunks
- transcribes each chunk with a platform-appropriate backend
- matches transcript chunks to EPUB HTML files
- injects stable segment ids into HTML
- generates SMIL overlays
- packages the final Media Overlay EPUB
- validates the generated result

## Requirements

- Python 3
- `ffmpeg`
- `ffprobe`

Transcription backend by platform:

- Apple Silicon macOS: `mlx-whisperx`
- other platforms: `whisperx`

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

Split-audio defaults:

- `--audio-codec copy` preserves source audio quality in the packaged Media Overlay EPUB
- this usually makes splitting much faster than re-encoding
- output EPUB size can increase because packaged audio stays near source quality

Transcription defaults:

- Apple Silicon macOS model: `mlx-community/whisper-turbo`
- other platforms model: `small`
- language: `en`

The `--language` setting is used for both transcription and HTML sentence segmentation.

Override them when needed:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --model large-v2 \
  --language en
```

Model names depend on the selected backend. For example, non-MLX platforms should use standard Whisper or WhisperX model names such as `small`, `medium`, `large`, or `large-v2`.

Optional split-audio re-encode controls:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --audio-codec aac \
  --audio-bitrate 96k \
  --audio-sample-rate 44100 \
  --audio-channels 2
```

When `--audio-codec aac` is selected, you can optionally set bitrate, sample rate, and channel count. These flags are rejected when `--audio-codec copy` is in use.

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
- Split audio defaults to stream copy so packaged playback preserves source quality.
- Validation checks packaging, SMIL clip quality, transcript coverage, and OPF media-overlay wiring.
- The console output and `logs/pipeline.log` include detailed stage-by-stage progress and timings.

## Files

- `generate_epub_overlay.py`: CLI wrapper and state manager with automatic resume
- `pipeline_core.py`: EPUB/audio matching, SMIL generation, packaging, and validation logic
- `mark_sentence.py`: HTML segmentation logic used for overlay targets
- `transcription_backend.py`: platform-aware transcription backend selection and adapters

## Troubleshooting

If `ffmpeg` or `ffprobe` are missing, install them and ensure they are on `PATH`.

If the first run cannot download NLTK data automatically, make sure the machine has network access and write permission for `~/.cache/epub-media-overlay/nltk_data`.

If a run is interrupted, rerun the same command and the pipeline resumes automatically.
