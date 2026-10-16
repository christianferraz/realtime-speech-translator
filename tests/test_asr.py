"""ASR post-processing: hallucination gating and language reporting.

These test the logic *around* the model, using stub segment objects, so no
weights are downloaded.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from speech_translate.asr import (
    FasterWhisperASR,
    MLXWhisperASR,
    Transcript,
    create_recognizer,
    _looks_like_hallucination,
)
from speech_translate.config import ASRSettings


@dataclass
class StubSegment:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class StubInfo:
    language: str | None = "en"
    language_probability: float = 0.99


class StubModel:
    """Mimics faster-whisper's ``(segments, info)`` return shape."""

    def __init__(self, segments: list[StubSegment], info: StubInfo | None = None) -> None:
        self.segments = segments
        self.info = info or StubInfo()
        self.kwargs: dict = {}

    def transcribe(self, audio, **kwargs):  # noqa: ANN001
        self.kwargs = kwargs
        return iter(self.segments), self.info


def make_asr(segments: list[StubSegment], info: StubInfo | None = None) -> FasterWhisperASR:
    asr = FasterWhisperASR(ASRSettings())
    asr._model = StubModel(segments, info)
    return asr


AUDIO = np.zeros(16_000, dtype=np.float32)


class TestSilenceGating:
    def test_high_no_speech_probability_is_dropped(self) -> None:
        """Whisper's classic failure: confident text over pure silence."""
        asr = make_asr([StubSegment("Thank you for watching!", no_speech_prob=0.95)])
        assert asr.transcribe(AUDIO).text == ""

    def test_low_confidence_segment_is_dropped(self) -> None:
        asr = make_asr([StubSegment("mumble", no_speech_prob=0.1, avg_logprob=-3.0)])
        assert asr.transcribe(AUDIO).text == ""

    def test_confident_speech_is_kept(self) -> None:
        asr = make_asr([StubSegment("Hello world.", no_speech_prob=0.01, avg_logprob=-0.2)])
        assert asr.transcribe(AUDIO).text == "Hello world."

    def test_segments_are_joined(self) -> None:
        asr = make_asr(
            [StubSegment("Hello there."), StubSegment("How are you?")]
        )
        result = asr.transcribe(AUDIO)
        assert result.text == "Hello there. How are you?"
        assert result.segments == ["Hello there.", "How are you?"]

    def test_mixed_segments_keep_only_the_good_ones(self) -> None:
        asr = make_asr(
            [
                StubSegment("Real speech.", no_speech_prob=0.01),
                StubSegment("Thanks for watching!", no_speech_prob=0.99),
            ]
        )
        assert asr.transcribe(AUDIO).text == "Real speech."


class TestHallucinationHeuristics:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "Thank you.",
            "Thanks for watching!",
            "you",
            "...",
            "♪",
            "Subtitles by the Amara.org community",
        ],
    )
    def test_known_boilerplate_is_rejected(self, text: str) -> None:
        assert _looks_like_hallucination(text)

    def test_repeated_phrase_loop_is_rejected(self) -> None:
        assert _looks_like_hallucination("Thank you. Thank you. Thank you.")

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, how are you today?",
            "The meeting is at three o'clock.",
            "Thank you for your help with the project.",
        ],
    )
    def test_real_speech_is_kept(self, text: str) -> None:
        assert not _looks_like_hallucination(text)


class TestLanguageReporting:
    def test_detected_language_is_converted_to_flores(self) -> None:
        """The core of the --src auto fix."""
        asr = make_asr([StubSegment("Bonjour.")], StubInfo(language="fr"))
        assert asr.transcribe(AUDIO, "auto").language == "fra_Latn"

    def test_language_probability_is_reported(self) -> None:
        asr = make_asr([StubSegment("Hi.")], StubInfo(language="en", language_probability=0.87))
        assert asr.transcribe(AUDIO, "auto").language_probability == pytest.approx(0.87)

    def test_unknown_detected_language_falls_back_to_the_request(self) -> None:
        asr = make_asr([StubSegment("...")], StubInfo(language="xx"))
        assert asr.transcribe(AUDIO, "deu_Latn").language == "deu_Latn"

    def test_unknown_detected_language_with_auto_defaults_to_english(self) -> None:
        asr = make_asr([StubSegment("hi")], StubInfo(language=None))
        assert asr.transcribe(AUDIO, "auto").language == "eng_Latn"

    def test_explicit_language_is_translated_to_a_whisper_code(self) -> None:
        asr = make_asr([StubSegment("Hola.")], StubInfo(language="es"))
        asr.transcribe(AUDIO, "spa_Latn")
        assert asr._model.kwargs["language"] == "es"

    def test_auto_passes_no_language_to_the_model(self) -> None:
        asr = make_asr([StubSegment("Hola.")])
        asr.transcribe(AUDIO, "auto")
        assert asr._model.kwargs["language"] is None


class TestDecodingOptions:
    def test_conditioning_on_previous_text_is_off_by_default(self) -> None:
        """It is the main driver of repetition loops in a chunked pipeline."""
        asr = make_asr([StubSegment("Hi.")])
        asr.transcribe(AUDIO)
        assert asr._model.kwargs["condition_on_previous_text"] is False

    def test_vad_filter_is_enabled_by_default(self) -> None:
        asr = make_asr([StubSegment("Hi.")])
        asr.transcribe(AUDIO)
        assert asr._model.kwargs["vad_filter"] is True

    def test_duration_is_reported(self) -> None:
        asr = make_asr([StubSegment("Hi.")])
        assert asr.transcribe(np.zeros(32_000, dtype=np.float32)).duration == pytest.approx(2.0)


def test_transcript_is_falsy_when_empty() -> None:
    assert not Transcript(text="", language="eng_Latn")


class TestBackendSelection:
    """MLX on an Apple Silicon GPU, faster-whisper everywhere else."""

    def test_auto_uses_mlx_when_available(self, monkeypatch) -> None:
        monkeypatch.setattr("speech_translate.config._mlx_available", lambda: True)
        settings = ASRSettings()
        assert settings.resolved_backend() == "mlx"
        assert settings.resolved_model() == "small"

    def test_auto_falls_back_to_cpu_base(self, monkeypatch) -> None:
        monkeypatch.setattr("speech_translate.config._mlx_available", lambda: False)
        settings = ASRSettings()
        assert settings.resolved_backend() == "faster-whisper"
        assert settings.resolved_model() == "base"

    def test_explicit_device_means_faster_whisper(self, monkeypatch) -> None:
        monkeypatch.setattr("speech_translate.config._mlx_available", lambda: True)
        assert ASRSettings(device="cpu").resolved_backend() == "faster-whisper"

    def test_explicit_model_is_kept(self, monkeypatch) -> None:
        monkeypatch.setattr("speech_translate.config._mlx_available", lambda: True)
        assert ASRSettings(model="small").resolved_model() == "small"

    def test_model_without_mlx_conversion_falls_back(self, monkeypatch) -> None:
        monkeypatch.setattr("speech_translate.config._mlx_available", lambda: True)
        recognizer = create_recognizer(ASRSettings(model="distil-small.en"))
        assert isinstance(recognizer, FasterWhisperASR)

    def test_forcing_mlx_with_unknown_model_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="No MLX conversion"):
            create_recognizer(ASRSettings(model="distil-small.en", backend="mlx"))


class TestMLXWhisperASR:
    @pytest.fixture
    def fake_mlx(self, monkeypatch):
        calls: list[dict] = []
        result: dict = {"language": "pt", "segments": []}
        module = types.ModuleType("mlx_whisper")

        def transcribe(audio, **kwargs):  # noqa: ANN001, ANN202
            calls.append(kwargs)
            return result

        module.transcribe = transcribe
        monkeypatch.setitem(sys.modules, "mlx_whisper", module)
        return calls, result

    def segment(self, text: str, no_speech: float = 0.1, logprob: float = -0.2) -> dict:
        return {"text": text, "no_speech_prob": no_speech, "avg_logprob": logprob}

    def test_model_name_maps_to_mlx_repo(self) -> None:
        assert MLXWhisperASR(ASRSettings(model="base")).repo == "mlx-community/whisper-base-mlx"
        assert MLXWhisperASR(ASRSettings(model="org/custom")).repo == "org/custom"

    def test_detected_language_reaches_the_transcript(self, fake_mlx) -> None:
        _, result = fake_mlx
        result["segments"] = [self.segment(" Olá, tudo bem? ")]
        transcript = MLXWhisperASR(ASRSettings(model="base")).transcribe(np.zeros(16_000))
        assert transcript.text == "Olá, tudo bem?"
        assert transcript.language == "por_Latn"

    def test_pinned_language_is_passed_as_whisper_code(self, fake_mlx) -> None:
        calls, _ = fake_mlx
        MLXWhisperASR(ASRSettings(model="base")).transcribe(np.zeros(16_000), "por_Latn")
        assert calls[0]["language"] == "pt"
        assert calls[0]["condition_on_previous_text"] is False

    def test_silence_and_low_confidence_segments_are_dropped(self, fake_mlx) -> None:
        _, result = fake_mlx
        result["segments"] = [
            self.segment("kept"),
            self.segment("silence", no_speech=0.9),
            self.segment("mumble", logprob=-2.0),
        ]
        transcript = MLXWhisperASR(ASRSettings(model="base")).transcribe(np.zeros(16_000))
        assert transcript.text == "kept"

    def test_hallucinations_are_discarded(self, fake_mlx) -> None:
        _, result = fake_mlx
        result["segments"] = [self.segment("Thank you for watching.")]
        transcript = MLXWhisperASR(ASRSettings(model="base")).transcribe(np.zeros(16_000))
        assert transcript.text == ""
