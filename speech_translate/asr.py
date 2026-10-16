"""Automatic speech recognition: faster-whisper (CTranslate2) or MLX.

:class:`MLXWhisperASR` runs the same Whisper checkpoints on the GPU of an
Apple Silicon Mac; :func:`create_recognizer` picks the runtime per
``ASRSettings.backend``.

Replaces ``openai-whisper``. Same weights, same accuracy, but the CTranslate2
runtime is markedly faster and uses less memory, and int8 quantisation makes a
CPU-only laptop viable.

Two bugs from the original implementation are fixed here:

1. The detected language was thrown away, so ``--src auto`` handed the literal
   string ``"auto"`` to the translator. :class:`Transcript` now carries the
   detected language back out.
2. Silence was transcribed as confident nonsense. Segments are now gated on
   ``no_speech_prob`` and average log-probability.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import ASRSettings, default_cpu_threads
from .languages import AUTO, flores_to_whisper, whisper_to_flores

if TYPE_CHECKING:  # pragma: no cover
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

__all__ = [
    "FasterWhisperASR",
    "MLXWhisperASR",
    "SpeechRecognizer",
    "Transcript",
    "create_recognizer",
]


@dataclass
class Transcript:
    """Result of transcribing one utterance."""

    text: str
    #: FLORES-200 code of the language actually detected/used.
    language: str
    language_probability: float = 1.0
    #: Mean ``no_speech_prob`` over the kept segments.
    no_speech_prob: float = 0.0
    duration: float = 0.0
    segments: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text)


class SpeechRecognizer(Protocol):
    """The contract the pipeline depends on, so tests can substitute a fake."""

    def transcribe(self, audio: np.ndarray, language: str = AUTO) -> Transcript: ...


# Whisper emits these when fed silence or music. They are not transcriptions,
# they are the training set's most common subtitle boilerplate leaking out.
_HALLUCINATION_PHRASES = frozenset(
    {
        "thank you.",
        "thanks for watching!",
        "thank you for watching.",
        "thank you for watching!",
        "you",
        "bye.",
        "please subscribe!",
        "subtitles by the amara.org community",
        "www.mooji.org",
        ".",
        "...",
        "♪",
    }
)


def _build_transcript(
    segments: Iterable[object],
    settings: ASRSettings,
    *,
    language: str,
    language_probability: float,
    duration: float,
) -> Transcript:
    """Keep the segments that are plausibly speech and join them.

    Each segment needs ``text``, ``no_speech_prob`` and ``avg_logprob``
    attributes, which both faster-whisper and (wrapped) MLX provide.
    """
    kept: list[str] = []
    no_speech: list[float] = []
    for segment in segments:
        prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        no_speech.append(prob)
        if prob > settings.no_speech_threshold:
            logger.debug("Dropped segment (no_speech_prob=%.2f): %r", prob, segment.text)
            continue
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        if avg_logprob < settings.log_prob_threshold:
            logger.debug("Dropped low-confidence segment (%.2f): %r", avg_logprob, segment.text)
            continue
        text = segment.text.strip()
        if text:
            kept.append(text)

    text = " ".join(kept).strip()
    if _looks_like_hallucination(text):
        logger.debug("Discarded probable hallucination: %r", text)
        text = ""

    return Transcript(
        text=text,
        language=language,
        language_probability=language_probability,
        no_speech_prob=(sum(no_speech) / len(no_speech)) if no_speech else 0.0,
        duration=duration,
        segments=kept,
    )


def _looks_like_hallucination(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    if stripped in _HALLUCINATION_PHRASES:
        return True
    # "Thank you. Thank you. Thank you." -- one phrase looped.
    parts = [p.strip() for p in stripped.split(".") if p.strip()]
    return len(parts) >= 3 and len(set(parts)) == 1


class FasterWhisperASR:
    """faster-whisper backed recogniser.

    The model is loaded lazily on first use so that constructing the object
    (and therefore importing the package) stays cheap.
    """

    def __init__(self, settings: ASRSettings | None = None) -> None:
        self.settings = settings or ASRSettings()
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> WhisperModel:
        from faster_whisper import WhisperModel

        device = self.settings.resolved_device()
        compute_type = self.settings.resolved_compute_type()
        model = self.settings.resolved_model()
        logger.info(
            "Loading ASR model %s (device=%s, compute_type=%s)",
            model,
            device,
            compute_type,
        )
        kwargs: dict[str, object] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": self.settings.cpu_threads or default_cpu_threads(),
        }
        try:
            return WhisperModel(model, **kwargs)
        except ValueError:
            # e.g. float16 requested on a device that cannot do it.
            logger.warning(
                "compute_type=%s unsupported on %s; falling back to float32",
                compute_type,
                device,
            )
            kwargs["compute_type"] = "float32"
            return WhisperModel(model, **kwargs)

    def load(self) -> FasterWhisperASR:
        """Force the model to load now (used to keep warm-up out of timings)."""
        _ = self.model
        return self

    def transcribe(self, audio: np.ndarray, language: str = AUTO) -> Transcript:
        """Transcribe mono float32 audio at 16 kHz.

        Args:
            audio: 1-D float32 waveform in [-1, 1].
            language: FLORES-200 code, or ``"auto"`` to let Whisper detect it.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration = len(audio) / 16_000

        whisper_lang = None if language == AUTO else flores_to_whisper(language)
        if language != AUTO and whisper_lang is None:
            logger.warning(
                "Whisper cannot be pinned to %s; falling back to auto-detect.", language
            )

        segments, info = self.model.transcribe(
            audio,
            language=whisper_lang,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            no_speech_threshold=self.settings.no_speech_threshold,
            log_prob_threshold=self.settings.log_prob_threshold,
            condition_on_previous_text=self.settings.condition_on_previous_text,
        )

        # segments is a generator: this is where decoding actually happens.
        return _build_transcript(
            segments,
            self.settings,
            language=self._detected_language(info, language),
            language_probability=float(getattr(info, "language_probability", 1.0) or 1.0),
            duration=duration,
        )

    @staticmethod
    def _detected_language(info: object, requested: str) -> str:
        """Turn whatever Whisper reports into a FLORES-200 code.

        This is the fix for the ``--src auto`` bug: previously the detected
        language never left this function, so the translator was handed the
        string ``"auto"``.
        """
        raw = getattr(info, "language", None)
        if raw:
            try:
                return whisper_to_flores(raw)
            except Exception:
                logger.warning("Whisper detected %r which has no NLLB equivalent.", raw)
        if requested != AUTO:
            return requested
        return "eng_Latn"


# Whisper names -> MLX conversions published by mlx-community. Anything with a
# "/" is taken to be a Hugging Face repo already.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class MLXWhisperASR:
    """Whisper on the Apple Silicon GPU, through MLX.

    Measured on an M4 Max (9.6 s of speech): ``base`` 59 ms versus 241 ms for
    faster-whisper on 8 CPU threads, and ``large-v3-turbo`` 296 ms versus
    2.4 s. Same weights, same filtering of silence and hallucinations as
    :class:`FasterWhisperASR`.
    """

    def __init__(self, settings: ASRSettings | None = None) -> None:
        self.settings = settings or ASRSettings()
        model = self.settings.resolved_model()
        self.repo = _MLX_REPOS.get(model, model)
        if "/" not in self.repo:
            raise ValueError(
                f"No MLX conversion known for Whisper model {model!r}; pass a Hugging "
                f"Face repo (e.g. mlx-community/whisper-small-mlx) or one of "
                f"{', '.join(sorted(_MLX_REPOS))}."
            )
        self._loaded = False

    def load(self) -> MLXWhisperASR:
        """Download and load the weights now, and compile the first graph."""
        if not self._loaded:
            logger.info("Loading ASR model %s (MLX, Apple GPU)", self.repo)
            self.transcribe(np.zeros(16_000, dtype=np.float32), "eng_Latn")
            self._loaded = True
        return self

    def transcribe(self, audio: np.ndarray, language: str = AUTO) -> Transcript:
        import mlx_whisper

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        whisper_lang = None if language == AUTO else flores_to_whisper(language)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=whisper_lang,
            no_speech_threshold=self.settings.no_speech_threshold,
            logprob_threshold=self.settings.log_prob_threshold,
            condition_on_previous_text=self.settings.condition_on_previous_text,
            verbose=None,
        )
        segments = [SimpleNamespace(**segment) for segment in result.get("segments", [])]
        return _build_transcript(
            segments,
            self.settings,
            language=FasterWhisperASR._detected_language(
                SimpleNamespace(language=result.get("language")), language
            ),
            language_probability=1.0,
            duration=len(audio) / 16_000,
        )


def create_recognizer(settings: ASRSettings | None = None) -> SpeechRecognizer:
    """The recogniser for ``ASRSettings.backend``: MLX on a Mac GPU, else faster-whisper."""
    settings = settings or ASRSettings()
    if settings.resolved_backend() == "mlx":
        try:
            return MLXWhisperASR(settings)
        except ValueError:
            if settings.backend == "mlx":
                raise
            logger.warning(
                "Whisper model %r has no MLX conversion; using faster-whisper on the CPU.",
                settings.resolved_model(),
            )
    return FasterWhisperASR(settings)


def load_asr_model(
    size: str = "small", settings: ASRSettings | None = None
) -> FasterWhisperASR:
    """Backwards-compatible helper mirroring the original API."""
    settings = settings or ASRSettings(model=size)
    return FasterWhisperASR(settings)


def transcribe(
    audio: np.ndarray | str, model: FasterWhisperASR, language: str = AUTO
) -> str:
    """Backwards-compatible helper returning only the text."""
    if isinstance(audio, str):
        from .audio.wav import read_wav

        audio, _ = read_wav(audio)
    return model.transcribe(audio, language).text


def available_models() -> Sequence[str]:
    """Model names worth offering in a UI, fastest first."""
    return (
        "tiny",
        "base",
        "distil-small.en",
        "small",
        "distil-medium.en",
        "medium",
        "distil-large-v3",
        "large-v3",
    )
