from __future__ import annotations

import gc
import importlib
import platform
from typing import Any


BACKEND_MLX = "mlx"
BACKEND_WHISPERX = "whisperx"

DEFAULT_MODEL_BY_BACKEND = {
    BACKEND_MLX: "mlx-community/whisper-large-v3-mlx",
    BACKEND_WHISPERX: "small",
}


def detect_transcription_backend() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return BACKEND_MLX
    return BACKEND_WHISPERX


def default_model_for_backend(backend: str) -> str:
    return DEFAULT_MODEL_BY_BACKEND[backend]


def required_module_for_backend(backend: str) -> str:
    if backend == BACKEND_MLX:
        return "mlx_whisperx"
    return "whisperx"


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


def available_backends() -> tuple[str, ...]:
    return (BACKEND_MLX, BACKEND_WHISPERX)


def describe_backend_params(backend: str, batch_size: int) -> dict[str, Any]:
    if backend == BACKEND_MLX:
        # mirrors _transcribe_with_mlx
        return {"beam_size": 1, "batch_size": batch_size}
    # mirrors _transcribe_with_whisperx
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        torch = None
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return {"device": device, "compute_type": compute_type, "batch_size": batch_size}


def is_mlx_backend(backend: str) -> bool:
    return backend == BACKEND_MLX


def apply_mlx_cache_limit(backend: str, cache_gb: float | None) -> float | None:
    """Cap the mlx Metal buffer cache so freed GPU memory is not retained unbounded.

    Returns the applied limit in GB, or None when nothing was applied (non-mlx
    backend, no limit requested, or the installed mlx build lacks the setter).
    """
    if backend != BACKEND_MLX or cache_gb is None:
        return None
    try:
        mx = importlib.import_module("mlx.core")
    except ModuleNotFoundError:
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


def _transcribe_with_whisperx(
    file_path: str,
    model: str,
    language: str,
    batch_size: int,
) -> dict[str, Any]:
    whisperx = importlib.import_module("whisperx")
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        torch = None

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model_obj = whisperx.load_model(model, device, compute_type=compute_type)
    audio = whisperx.load_audio(file_path)
    result = model_obj.transcribe(audio, batch_size=batch_size, language=language)

    align_language = result.get("language") or language
    model_a, metadata = whisperx.load_align_model(language_code=align_language, device=device)
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

    del model_obj
    del model_a
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aligned
