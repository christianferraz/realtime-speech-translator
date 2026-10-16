"""Machine translation on NLLB-200 (distilled 1.3B by default, 600M on small CPUs).

Replaces M2M100-418M. NLLB is better maintained, covers 200 languages, and the
distilled 600M checkpoint is stronger than M2M100-418M at comparable cost.

The API difference that matters: M2M100 used ``tokenizer.get_lang_id(tgt)`` with
ISO codes, NLLB uses FLORES-200 codes (``spa_Latn``) resolved through the
tokenizer's vocabulary. Getting this wrong yields fluent output in the *wrong*
language, which is why :meth:`NLLBTranslator.translate` validates up front.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .config import MTSettings, default_cpu_threads
from .languages import AUTO, UnsupportedLanguageError, language_name, resolve_language

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

__all__ = [
    "CT2NLLBTranslator",
    "NLLBTranslator",
    "Translation",
    "Translator",
    "create_translator",
]


@dataclass
class Translation:
    text: str
    src: str
    tgt: str

    def __bool__(self) -> bool:
        return bool(self.text)


class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str) -> Translation: ...


class NLLBTranslator:
    """NLLB-200 translator with lazy model loading."""

    def __init__(self, settings: MTSettings | None = None) -> None:
        self.settings = settings or MTSettings()
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # -- loading ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.settings.resolved_device()
        logger.info("Loading MT model %s (device=%s)", self.settings.model, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.model)
        self._model = model.to(device).eval()

    def load(self) -> NLLBTranslator:
        """Force the model to load now (keeps warm-up out of latency numbers)."""
        self._ensure_loaded()
        return self

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._ensure_loaded()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        self._ensure_loaded()
        assert self._model is not None
        return self._model

    # -- language plumbing ----------------------------------------------
    def _lang_token_id(self, flores_code: str) -> int:
        """Vocabulary id of a FLORES language token.

        ``convert_tokens_to_ids`` returns the *unknown* id for a code the
        checkpoint does not know, which would silently translate into a random
        language -- so an unknown code is turned into a loud error instead.
        """
        tokenizer = self.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(flores_code)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise UnsupportedLanguageError(
                f"{flores_code!r} is not a language known to {self.settings.model!r}."
            )
        return int(token_id)

    def supported_codes(self) -> set[str]:
        """FLORES codes this checkpoint actually knows."""
        extra = getattr(self.tokenizer, "additional_special_tokens", None) or []
        return {t for t in extra if "_" in t}

    # -- translation -----------------------------------------------------
    def translate(self, text: str, src: str, tgt: str) -> Translation:
        """Translate ``text`` from ``src`` to ``tgt`` (FLORES-200 codes)."""
        text = (text or "").strip()
        if not text:
            return Translation(text="", src=src, tgt=tgt)

        src_code = resolve_language(src, allow_auto=True)
        tgt_code = resolve_language(tgt)

        if src_code == AUTO:
            # The pipeline should have substituted the detected language by
            # now. Failing loudly beats guessing and mistranslating.
            raise UnsupportedLanguageError(
                "Source language is still 'auto'; pass the language detected by "
                "the recogniser (Transcript.language) instead."
            )

        if src_code == tgt_code:
            return Translation(text=text, src=src_code, tgt=tgt_code)

        import torch

        tokenizer = self.tokenizer
        tokenizer.src_lang = src_code
        device = self.settings.resolved_device()
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self._lang_token_id(tgt_code),
                max_new_tokens=self.settings.max_new_tokens,
                num_beams=self.settings.num_beams,
            )
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        logger.debug(
            "Translated %s -> %s: %r -> %r",
            language_name(src_code),
            language_name(tgt_code),
            text,
            out,
        )
        return Translation(text=out, src=src_code, tgt=tgt_code)


class CT2NLLBTranslator(NLLBTranslator):
    """NLLB-200 on CTranslate2, the runtime faster-whisper already uses.

    Same checkpoint and tokenizer as :class:`NLLBTranslator`, but decoding runs
    in CTranslate2 with int8 weights on CPU, which is several times faster
    than PyTorch eager mode -- and translation is the slowest stage of the
    cascade, so this is where live latency is won or lost.

    CTranslate2 needs the model in its own format. It is converted once, on
    first use, into ``MTSettings.cache_dir`` and reused afterwards.
    """

    def __init__(self, settings: MTSettings | None = None) -> None:
        super().__init__(settings)
        self._ct2: object | None = None

    def _ensure_loaded(self) -> None:
        if self._ct2 is not None:
            return
        import ctranslate2
        from transformers import AutoTokenizer

        device = self.settings.resolved_device()
        compute_type = self.settings.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        model_dir = self.converted_model_dir()
        if not (model_dir / "model.bin").exists():
            self._convert(model_dir)

        logger.info(
            "Loading MT model %s on CTranslate2 (device=%s, compute_type=%s)",
            self.settings.model,
            device,
            compute_type,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model)
        self._ct2 = ctranslate2.Translator(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            intra_threads=self.settings.cpu_threads or default_cpu_threads(),
        )

    def converted_model_dir(self) -> Path:
        root = Path(
            self.settings.cache_dir
            or os.environ.get("SPEECH_TRANSLATE_CACHE")
            or Path.home() / ".cache" / "speech_translate"
        )
        return root / "ct2" / self.settings.model.replace("/", "--")

    def _convert(self, model_dir: Path) -> None:
        from ctranslate2.converters import TransformersConverter

        logger.info(
            "Converting %s to CTranslate2 (one-off, takes a minute) ...", self.settings.model
        )
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        # Convert into a scratch directory and rename, so an interrupted
        # conversion never leaves a half-written model that looks complete.
        scratch = Path(tempfile.mkdtemp(prefix="convert-", dir=model_dir.parent))
        try:
            # float16 storage halves the download-to-disk size; the weights are
            # re-quantised to the requested compute type at load time.
            TransformersConverter(self.settings.model).convert(
                str(scratch), quantization="float16", force=True
            )
            if model_dir.exists():
                shutil.rmtree(model_dir)
            scratch.rename(model_dir)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @property
    def model(self) -> object:
        self._ensure_loaded()
        return self._ct2

    def translate(self, text: str, src: str, tgt: str) -> Translation:
        text = (text or "").strip()
        if not text:
            return Translation(text="", src=src, tgt=tgt)

        src_code = resolve_language(src, allow_auto=True)
        tgt_code = resolve_language(tgt)
        if src_code == AUTO:
            raise UnsupportedLanguageError(
                "Source language is still 'auto'; pass the language detected by "
                "the recogniser (Transcript.language) instead."
            )
        if src_code == tgt_code:
            return Translation(text=text, src=src_code, tgt=tgt_code)

        tokenizer = self.tokenizer
        self._lang_token_id(tgt_code)  # loud error for an unknown target
        tokenizer.src_lang = src_code
        source = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(text, truncation=True, max_length=512)
        )
        results = self.model.translate_batch(
            [source],
            target_prefix=[[tgt_code]],
            beam_size=self.settings.num_beams,
            max_decoding_length=self.settings.max_new_tokens,
        )
        # The first token is the forced target-language tag.
        tokens = results[0].hypotheses[0][1:]
        out = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
        ).strip()
        logger.debug(
            "Translated %s -> %s: %r -> %r",
            language_name(src_code),
            language_name(tgt_code),
            text,
            out,
        )
        return Translation(text=out, src=src_code, tgt=tgt_code)


def create_translator(settings: MTSettings | None = None) -> Translator:
    """The fastest translator that works here, per ``MTSettings.backend``."""
    settings = settings or MTSettings()
    if settings.backend == "transformers":
        return NLLBTranslator(settings)
    if settings.backend == "ctranslate2":
        return CT2NLLBTranslator(settings)
    try:
        import ctranslate2  # noqa: F401
        from ctranslate2.converters import TransformersConverter  # noqa: F401
    except Exception:
        logger.info("CTranslate2 unavailable; translating with PyTorch instead.")
        return NLLBTranslator(settings)
    return _FallbackTranslator(settings)


class _FallbackTranslator:
    """CTranslate2, falling back to PyTorch if the model cannot be converted."""

    def __init__(self, settings: MTSettings) -> None:
        self.settings = settings
        self._inner: NLLBTranslator = CT2NLLBTranslator(settings)
        self._loaded = False

    def load(self) -> _FallbackTranslator:
        if self._loaded:
            return self
        try:
            self._inner.load()
        except Exception:
            logger.warning(
                "Could not run %s on CTranslate2; using PyTorch instead.",
                self.settings.model,
                exc_info=True,
            )
            self._inner = NLLBTranslator(self.settings)
            self._inner.load()
        self._loaded = True
        return self

    def translate(self, text: str, src: str, tgt: str) -> Translation:
        return self.load()._inner.translate(text, src, tgt)

    def supported_codes(self) -> set[str]:
        return self.load()._inner.supported_codes()


def load_translation_model(settings: MTSettings | None = None) -> NLLBTranslator:
    """Backwards-compatible helper mirroring the original API."""
    return NLLBTranslator(settings)


def translate_text(text: str, src: str, tgt: str, translator: NLLBTranslator) -> str:
    """Backwards-compatible helper returning only the text."""
    return translator.translate(text, src, tgt).text
