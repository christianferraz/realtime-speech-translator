"""Configuration for the speech-translation cascade.

All tunables live here as plain dataclasses. Nothing in this module reads
``sys.argv``, opens a device, or touches the network -- importing
``speech_translate.config`` is free and side-effect-free, which is what makes
the package testable and safe to import from a web worker.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field, replace
from typing import Literal

from .languages import AUTO, resolve_language

Device = Literal["auto", "cpu", "cuda"]
TTSBackendName = Literal["auto", "piper", "system", "none"]
MTBackendName = Literal["auto", "ctranslate2", "transformers"]
ASRBackendName = Literal["auto", "faster-whisper", "mlx"]


def _default_device() -> Device:
    return "auto"


@dataclass
class AudioSettings:
    """Capture/playback format. 16 kHz mono is what Whisper wants natively."""

    sample_rate: int = 16_000
    channels: int = 1
    #: Size of one capture block. 30 ms is the granularity VAD operates on.
    block_ms: int = 30
    input_device: int | None = None
    output_device: int | None = None

    @property
    def block_frames(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass
class VADSettings:
    """Voice-activity detection: when does an utterance start and stop?

    Replaces the old fixed 3-second chunking, which cut words in half and paid
    full ASR cost on silence.
    """

    enabled: bool = True
    #: RMS energy threshold, adaptively raised to the observed noise floor.
    threshold: float = 0.015
    #: Ignore blips shorter than this; they are keyboard clicks, not speech.
    min_speech_ms: int = 250
    #: Silence this long ends the utterance. Every millisecond here is added
    #: to the latency of every sentence; 300 ms also ends utterances at most
    #: comma pauses, which live translation then voices clause by clause.
    min_silence_ms: int = 300
    #: Hard cap so a monologue still produces output. The cut is moved to the
    #: quietest nearby frame so it does not land in the middle of a word.
    #: On a 27 s monologue, 4 s took the voice from 7.8 s to 6.8 s behind;
    #: at 3 s Whisper started mishearing the short pieces.
    max_utterance_ms: int = 4_000
    #: Audio kept from *before* the trigger so we don't clip the first phoneme.
    pre_roll_ms: int = 300


@dataclass
class ASRSettings:
    """Speech recognition settings.

    Two runtimes: faster-whisper (CTranslate2, CPU or CUDA) and MLX, which runs
    Whisper on the GPU of an Apple Silicon Mac. ``backend='auto'`` picks MLX
    when it is installed on such a Mac and no device was forced.

    ``model='auto'`` follows the runtime. Measured live on an M4 Max (9.6 s of
    Portuguese, real-time pace): MLX ``small`` costs ~135 ms per call and
    delivers the last sentence 0.34 s after speech ends, with captions as good
    as ``large-v3-turbo`` (~500 ms per call under live load, 0.75 s) and
    clearly better than CPU ``base`` (~290 ms, 0.60 s), which misheard words
    that ``small`` got right. ``large-v3-turbo`` is one flag away for noisy or
    technical audio. On a CPU, ``base`` is the largest model that stays
    comfortably ahead of live speech.
    """

    model: str = "auto"
    backend: ASRBackendName = "auto"
    device: Device = field(default_factory=_default_device)
    #: ``auto`` picks int8 on CPU and float16 on GPU.
    compute_type: str = "auto"
    #: Greedy decoding: beams buy little for short utterances and cost latency.
    beam_size: int = 1
    #: faster-whisper's built-in Silero VAD, applied inside the model.
    vad_filter: bool = True
    #: Drop segments the model itself flags as probably-not-speech.
    no_speech_threshold: float = 0.6
    #: Drop low-confidence segments (a classic hallucination signature).
    log_prob_threshold: float = -1.0
    #: Whisper loops ("Thank you. Thank you. Thank you.") when it can see its
    #: own previous output. In a chunked real-time loop it is pure downside.
    condition_on_previous_text: bool = False
    #: 0 means :func:`default_cpu_threads`.
    cpu_threads: int = 0

    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        # An explicit --device cpu/cuda is a request for faster-whisper.
        if self.device == "auto" and _mlx_available():
            return "mlx"
        return "faster-whisper"

    def resolved_model(self) -> str:
        if self.model != "auto":
            return self.model
        return "small" if self.resolved_backend() == "mlx" else "base"

    def resolved_device(self) -> str:
        """Device for faster-whisper; MLX always runs on the Apple GPU."""
        if self.device != "auto":
            return self.device
        return "cuda" if _cuda_available() else "cpu"

    def resolved_compute_type(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        return "float16" if self.resolved_device() == "cuda" else "int8"


@dataclass
class MTSettings:
    """NLLB-200 settings.

    The 1.3B distilled checkpoint is the default: on an Apple M4 Max it costs
    ~390 ms per sentence against ~210 ms for 600M, and in exchange it stops
    mistranslating short stock phrases ("Thank you for watching" -> "Obrigado
    por me vigiar") and produces Brazilian rather than European Portuguese
    forms. Small CPUs (a free Space) should pass the 600M model explicitly.

    ``backend='auto'`` runs the model on CTranslate2 (int8 on CPU), which is
    several times faster than PyTorch for the same checkpoint. The model is
    converted once on first use and cached; plain ``transformers`` is the
    fallback when the conversion is not possible.
    """

    model: str = "facebook/nllb-200-distilled-1.3B"
    backend: MTBackendName = "auto"
    #: ``auto`` picks int8 on CPU and float16 on GPU (CTranslate2 only).
    compute_type: str = "auto"
    #: Where converted CTranslate2 models are kept.
    cache_dir: str | None = None
    device: Device = field(default_factory=_default_device)
    num_beams: int = 1
    max_new_tokens: int = 256
    #: 0 means :func:`default_cpu_threads` (CTranslate2 only).
    cpu_threads: int = 0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if _cuda_available() else "cpu"


@dataclass
class TTSSettings:
    """Speech synthesis settings.

    ``backend='auto'`` prefers Piper and degrades to the OS voice, then to
    subtitles-only, so the pipeline never hard-fails just because a voice is
    missing.
    """

    backend: TTSBackendName = "auto"
    #: Explicit Piper voice name (e.g. ``en_US-lessac-medium``).
    voice: str | None = None
    #: >1.0 is slower speech. Piper calls this ``length_scale``.
    length_scale: float = 1.0
    use_cuda: bool = False
    download_dir: str | None = None


@dataclass
class Settings:
    """Everything one run of the pipeline needs."""

    src: str = AUTO
    tgt: str = "spa_Latn"
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VADSettings = field(default_factory=VADSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    mt: MTSettings = field(default_factory=MTSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    #: Drop the oldest pending utterance when the queue is this deep. Staying
    #: near-live matters more than translating every backlogged utterance.
    max_pending_utterances: int = 4
    #: Hold partial text until sentence-final punctuation before synthesising.
    buffer_to_sentences: bool = True

    def __post_init__(self) -> None:
        self.src = resolve_language(self.src, allow_auto=True)
        self.tgt = resolve_language(self.tgt)

    def with_languages(self, src: str | None = None, tgt: str | None = None) -> Settings:
        """Copy with different languages, validating them."""
        return replace(
            self,
            src=resolve_language(src, allow_auto=True) if src else self.src,
            tgt=resolve_language(tgt) if tgt else self.tgt,
        )


def default_cpu_threads() -> int:
    """Threads for a CTranslate2 model when none were configured.

    CTranslate2 defaults to 4, which leaves most of a modern laptop idle;
    8 measured ~30% faster for Whisper and NLLB on an Apple M4 Max. Past 8 the
    gain is in the noise, and two models share the machine.
    """
    return max(1, min(8, os.cpu_count() or 4))


def _mlx_available() -> bool:
    """Apple Silicon with mlx-whisper installed, checked without importing MLX."""
    if os.environ.get("SPEECH_TRANSLATE_FORCE_CPU"):
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    import importlib.util

    return importlib.util.find_spec("mlx_whisper") is not None


def _cuda_available() -> bool:
    """Check for CUDA without importing torch at module import time."""
    if os.environ.get("SPEECH_TRANSLATE_FORCE_CPU"):
        return False
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
