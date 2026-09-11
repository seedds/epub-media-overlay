"""Tests for transcription_backend: whisperx models are loaded once per run.

Run:
  pytest tests/test_transcription_backend.py -q
"""

import sys
import types

import transcription_backend as tb


class _FakeWhisperx:
    def __init__(self):
        self.calls = {"load_model": 0, "load_align_model": 0, "align": 0}

    def load_model(self, model, device, compute_type):
        self.calls["load_model"] += 1
        fake = self

        class Model:
            def transcribe(self, audio, batch_size, language):
                return {"language": language, "segments": [{"text": "hi"}]}

        return Model()

    def load_audio(self, path):
        return f"audio:{path}"

    def load_align_model(self, language_code, device):
        self.calls["load_align_model"] += 1
        return ("align-model", {"lang": language_code})

    def align(self, segments, model_a, metadata, audio, device, return_char_alignments):
        self.calls["align"] += 1
        return {"segments": [{"words": [{"word": "hi", "start": 0.0, "end": 0.4}]}]}


def test_whisperx_models_are_loaded_once_and_released(monkeypatch):
    fake = _FakeWhisperx()
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "torch", None)  # import fails -> cpu/int8 path
    tb.release_models()

    outputs = [tb.transcribe_file(f"{i:03d}.m4a", "small", "en", "whisperx", 1) for i in range(3)]

    assert fake.calls == {"load_model": 1, "load_align_model": 1, "align": 3}
    assert outputs[0]["word_segments"] == [{"word": "hi", "start": 0.0, "end": 0.4}]
    assert outputs[0]["language"] == "en"
    assert tb._WHISPERX_MODELS and tb._WHISPERX_ALIGN_MODELS

    tb.release_models()
    assert not tb._WHISPERX_MODELS and not tb._WHISPERX_ALIGN_MODELS


def test_align_model_cached_per_language(monkeypatch):
    fake = _FakeWhisperx()
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "torch", None)
    tb.release_models()

    tb.transcribe_file("000.m4a", "small", "en", "whisperx", 1)
    tb.transcribe_file("001.m4a", "small", "fr", "whisperx", 1)
    tb.transcribe_file("002.m4a", "small", "en", "whisperx", 1)

    assert fake.calls["load_model"] == 1
    assert fake.calls["load_align_model"] == 2
    tb.release_models()


def test_backend_tables_are_consistent():
    assert set(tb.BACKENDS) == set(tb.DEFAULT_MODEL_BY_BACKEND) == set(tb.REQUIRED_MODULE_BY_BACKEND)
    assert tb.detect_transcription_backend() in tb.BACKENDS
