"""ASR backend selection and adapters (mlx-whisperx on Apple Silicon, whisperx elsewhere).

Everything the pipeline knows about the speech engines lives here so
`pipeline_core.transcribe_audio` only calls `transcribe_file`. Models are cached for
the life of the process and released with `release_models()` after the transcribe
stage; whisperx used to reload its ASR and alignment models for every audio chunk.
"""

from __future__ import annotations

import gc
import importlib
import os
import platform
from typing import Any

# Environment for the ML libraries. These must be set before torch / transformers /
# matplotlib are imported by the backends, which happens lazily inside this module.
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # no HuggingFace fork-parallelism warnings/deadlocks
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # legacy checkpoint loads under newer torch defaults
os.environ["MPLBACKEND"] = "Agg"  # never open a GUI window from a dependency

BACKEND_MLX = "mlx"
BACKEND_WHISPERX = "whisperx"
BACKENDS = (BACKEND_MLX, BACKEND_WHISPERX)

DEFAULT_MODEL_BY_BACKEND = {
    BACKEND_MLX: "mlx-community/whisper-large-v3-mlx",
    BACKEND_WHISPERX: "small",
}
REQUIRED_MODULE_BY_BACKEND = {
    BACKEND_MLX: "mlx_whisperx",
    BACKEND_WHISPERX: "whisperx",
}


def detect_transcription_backend() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return BACKEND_MLX
    return BACKEND_WHISPERX


def default_model_for_backend(backend: str) -> str:
    return DEFAULT_MODEL_BY_BACKEND[backend]


def transcribe_file(
    file_path: str,
    model: str,
    language: str,
    backend: str,
    batch_size: int,
) -> dict[str, Any]:
    if backend == BACKEND_MLX:
        return _transcribe_with_mlx(file_path, model, language, batch_size)
    return _transcribe_with_whisperx(file_path, model, language, batch_size)


def apply_mlx_cache_limit(backend: str, cache_gb: float | None) -> float | None:
    """Cap the mlx Metal buffer cache so freed GPU memory is not retained unbounded.

    Returns the applied limit in GB, or None when nothing was applied (non-mlx
    backend, no limit requested, or the installed mlx build lacks the setter).
    """
    if backend != BACKEND_MLX or cache_gb is None:
        return None
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError:
        return None
    limit_bytes = int(cache_gb * 1024**3)
    setter = getattr(mx, "set_cache_limit", None) or getattr(
        getattr(mx, "metal", None), "set_cache_limit", None
    )
    if setter is None:
        return None
    setter(limit_bytes)
    return cache_gb


def _transcribe_with_mlx(
    file_path: str,
    model: str,
    language: str,
    batch_size: int,
) -> dict[str, Any]:
    mlx_whisperx = importlib.import_module("mlx_whisperx")
    return mlx_whisperx.transcribe(
        file_path,
        model=model,
        language=language,
        beam_size=1,
        batch_size=batch_size,
    )


# whisperx model caches, keyed so a different model/device/language loads afresh.
_WHISPERX_MODELS: dict[tuple[str, str, str], Any] = {}
_WHISPERX_ALIGN_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}


def _import_torch():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _transcribe_with_whisperx(
    file_path: str,
    model: str,
    language: str,
    batch_size: int,
) -> dict[str, Any]:
    whisperx = importlib.import_module("whisperx")
    torch = _import_torch()

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model_key = (model, device, compute_type)
    model_obj = _WHISPERX_MODELS.get(model_key)
    if model_obj is None:
        model_obj = whisperx.load_model(model, device, compute_type=compute_type)
        _WHISPERX_MODELS[model_key] = model_obj

    audio = whisperx.load_audio(file_path)
    result = model_obj.transcribe(audio, batch_size=batch_size, language=language)

    align_language = result.get("language") or language
    align_key = (align_language, device)
    if align_key not in _WHISPERX_ALIGN_MODELS:
        _WHISPERX_ALIGN_MODELS[align_key] = whisperx.load_align_model(
            language_code=align_language, device=device
        )
    model_a, metadata = _WHISPERX_ALIGN_MODELS[align_key]

    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    aligned.setdefault("language", align_language)

    if "word_segments" not in aligned:
        aligned["word_segments"] = [
            word
            for segment in aligned.get("segments", [])
            for word in segment.get("words", [])
        ]
    return aligned


def release_models() -> None:
    """Drop cached models and return their memory. Call once after transcription."""
    _WHISPERX_MODELS.clear()
    _WHISPERX_ALIGN_MODELS.clear()
    gc.collect()
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
