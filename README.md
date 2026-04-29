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

### Examples

Basic run:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub
```

Run with all optional parameters:

```bash
python generate_epub_overlay.py \
  --m4b /path/to/book.m4b \
  --epub /path/to/book.epub \
  --output-dir /path/to/output \
  --work-dir /path/to/work \
  --model large-v2 \
  --language en \
  --audio-extension .m4a \
  --audio-codec aac \
  --audio-bitrate 96k \
  --audio-sample-rate 44100 \
  --audio-channels 2 \
  --fresh
```

### Defaults

- transcription backend:
  - Apple Silicon macOS: `mlx-whisperx`
  - other platforms: `whisperx`
- transcription model:
  - Apple Silicon macOS: `mlx-community/whisper-turbo`
  - other platforms: `small`
- language: `en`
- audio extension: `.m4a`
- split audio codec: `copy`

With `--audio-codec copy`:

- split audio preserves the source audio stream without re-encoding
- packaged Media Overlay playback keeps source-quality audio
- splitting is usually faster than AAC re-encoding
- final EPUB size may increase because the source-quality audio is packaged

The `--language` setting is used for both transcription and HTML sentence segmentation.

Model names depend on the selected backend. On non-MLX platforms, common values include `small`, `medium`, `large`, and `large-v2`.

### Parameters

`--m4b`

- Required.
- Path to the source audiobook file.
- Must point to an `.m4b` file.

`--epub`

- Required.
- Path to the source ebook file.
- Must point to an `.epub` file.

`--output-dir`

- Optional.
- Directory where the final `<book-stem>.media-overlay.epub` file is written.
- Default: the source EPUB directory.

`--work-dir`

- Optional.
- Directory used for persistent state, logs, transcripts, split audio, and intermediate EPUB artifacts.
- Default: `<output-dir>/.<book-stem>.epubmo`

`--model`

- Optional.
- Transcription model identifier.
- Default:
  - Apple Silicon macOS: `mlx-community/whisper-turbo`
  - other platforms: `small`
- Valid model names depend on the active backend.

`--language`

- Optional.
- Language code used for transcription and HTML sentence segmentation.
- Default: `en`

`--audio-extension`

- Optional.
- Filename extension used for split audio chunks.
- Default: `.m4a`

`--audio-codec`

- Optional.
- Split audio codec mode.
- Supported values: `copy`, `aac`
- Default: `copy`
- `copy` preserves the source audio stream.
- `aac` re-encodes split audio and enables the quality controls below.

`--audio-bitrate`

- Optional.
- AAC bitrate for split audio chunks, such as `64k`, `96k`, or `128k`.
- Only valid when `--audio-codec aac` is used.

`--audio-sample-rate`

- Optional.
- AAC sample rate in Hz, such as `24000` or `44100`.
- Only valid when `--audio-codec aac` is used.

`--audio-channels`

- Optional.
- AAC channel count, such as `1` for mono or `2` for stereo.
- Only valid when `--audio-codec aac` is used.

`--fresh`

- Optional.
- Discards any existing compatible work state and restarts the pipeline from scratch.

### Run behavior

- if compatible work already exists, the pipeline resumes automatically
- if no work exists yet, the pipeline starts from the beginning
- use `--fresh` only when you want to discard previous work and restart from scratch
- if `--output-dir` is omitted, the final EPUB is written next to the source EPUB

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
