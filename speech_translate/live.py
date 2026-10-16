"""Live translation with provisional captions, for push-style audio sources.

:class:`RealtimeSession` owns the microphone. A browser is the opposite: audio
arrives in chunks pushed by the client, and the request that delivers a chunk
must return quickly or the next chunks queue up behind it. The previous web
demo ran ASR -> MT -> TTS inside that request, so every translated sentence
stalled capture for its full processing time and the session drifted further
behind the longer it ran.

:class:`LiveTranslator` splits the two:

* :meth:`feed` only runs the (cheap) VAD segmenter and returns at once;
* a worker thread does the model work;
* :meth:`poll` hands back whatever is ready.

Between finished utterances the worker also re-transcribes the utterance that
is *still being spoken* and translates it, so captions follow the speaker
instead of appearing only after they pause. Those partial results are
provisional -- they are shown, then replaced -- and are never spoken: speech is
synthesised only for finished sentences, which is what keeps the voice from
stuttering through every revision.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import numpy as np

from .audio.vad import Utterance, UtteranceSegmenter
from .config import AudioSettings, VADSettings
from .languages import AUTO
from .pipeline import Pipeline, PipelineResult, StageTimings, timed
from .segmentation import SentenceBuffer, split_clauses
from .tts import SpeechAudio

logger = logging.getLogger(__name__)

__all__ = ["LiveTranslator", "LiveUpdate"]


@dataclass
class LiveUpdate:
    """What changed since the previous :meth:`LiveTranslator.poll`."""

    #: Sentences that are done: transcript, translation and timings.
    finals: list[PipelineResult] = field(default_factory=list)
    #: Provisional caption for speech still in progress ("" when none).
    partial_transcript: str = ""
    partial_translation: str = ""
    #: Speech to start playing now, or None.
    speech: SpeechAudio | None = None
    #: False when nothing at all changed, so a UI can skip re-rendering.
    changed: bool = False


class LiveTranslator:
    """Push audio in, poll captions and speech out.

    Args:
        pipeline_for: returns the pipeline to use for a target language and
            whether speech is wanted. Called on the worker thread, so model
            loading never blocks :meth:`feed`.
        lock: held around every model call, for callers that share one
            pipeline between sessions.
        partials: whether to produce provisional captions while speaking.
        partial_interval: minimum seconds between two provisional transcripts.
            Whisper on the GPU is cheap, so the source text can refresh often.
        partial_translate_interval: minimum seconds between two provisional
            *translations*. NLLB on the CPU is the expensive stage (0.4-2 s per
            call for the 1.3B model); re-translating every caption kept ~7
            cores busy and made finished sentences -- the ones that get
            spoken -- queue behind captions nobody would hear.
        min_partial_audio: speech needed before the first provisional caption;
            Whisper on less than this is mostly guesswork.
        max_speech_backlog: seconds of unplayed speech kept. Older clips are
            dropped: in a live translator, being current beats being complete.
        base_speed: speaking rate when speech is not behind. Piper's voices
            are unhurried and translations run long, so the default is a
            little above 1.
        max_speed: fastest speaking rate used to catch up. Translated speech
            is usually longer than the original (a 26.7 s English clip came
            out as 31 s of Portuguese), so at a fixed rate the voice falls
            further behind with every sentence.
        catch_up_seconds: how far behind the voice must be to reach
            ``max_speed``; the rate rises linearly up to it.
        clause_min_words: translate and speak clause by clause (split at
            commas) once a clause has this many words; 0 waits for whole
            sentences. On a monologue of long sentences this took the voice
            from ~8 s behind to about one clause behind.
        stall_timeout: seconds without any audio, mid-utterance, after which
            the utterance is finalised. A paused video or a browser that stops
            sending silent chunks never delivers the silence that would end
            it, and the last sentence would hang as a caption forever.
    """

    def __init__(
        self,
        pipeline_for: Callable[[str, bool], Pipeline],
        *,
        src: str = AUTO,
        tgt: str = "spa_Latn",
        speak: bool = True,
        audio: AudioSettings | None = None,
        vad: VADSettings | None = None,
        lock: AbstractContextManager | None = None,
        partials: bool = True,
        partial_interval: float = 0.6,
        partial_translate_interval: float = 1.2,
        min_partial_audio: float = 0.8,
        max_speech_backlog: float = 12.0,
        base_speed: float = 1.15,
        max_speed: float = 1.6,
        catch_up_seconds: float = 3.0,
        clause_min_words: int = 5,
        stall_timeout: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pipeline_for = pipeline_for
        self.src = src
        self.tgt = tgt
        self.speak = speak
        self.audio = audio or AudioSettings()
        self.partials = partials
        self.partial_interval = partial_interval
        self.partial_translate_interval = partial_translate_interval
        self.min_partial_audio = min_partial_audio
        self.max_speech_backlog = max_speech_backlog
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.catch_up_seconds = catch_up_seconds
        self.stall_timeout = stall_timeout
        self.clause_min_words = clause_min_words
        self._clock = clock
        self._last_feed = clock()
        self._model_lock = lock or nullcontext()

        self._segmenter = UtteranceSegmenter(self.audio, vad)
        self._segmenter_lock = threading.Lock()
        #: Bumped whenever an utterance completes, so a provisional caption
        #: computed from audio that has since been finalised can be discarded.
        self._generation = 0
        self._utterances: queue.Queue[Utterance | None] = queue.Queue()
        self._buffer = SentenceBuffer()
        self._last_language = src

        self._out_lock = threading.Lock()
        self._finals: list[PipelineResult] = []
        self._partial = ("", "")
        self._partial_dirty = False
        self._clips: list[SpeechAudio] = []
        self._playing_until = 0.0
        self._last_partial_at = 0.0
        self._last_partial_translation_at = 0.0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> LiveTranslator:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="live", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Finish the utterance in progress, then stop the worker."""
        self.end_of_stream()
        self._utterances.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def end_of_stream(self) -> None:
        """The source stopped: translate whatever was said last."""
        with self._segmenter_lock:
            pending = list(self._segmenter.flush())
            self._generation += 1
        for utterance in pending:
            self._utterances.put(utterance)

    def configure(self, *, src: str, tgt: str, speak: bool) -> None:
        """Change languages or speech output; takes effect on the next sentence."""
        self.src, self.tgt, self.speak = src, tgt, speak

    # -- producer side ---------------------------------------------------
    def feed(self, audio: np.ndarray) -> None:
        """Push mono float32 audio at ``AudioSettings.sample_rate``."""
        self._last_feed = self._clock()
        block = self.audio.block_frames
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._segmenter_lock:
            for start in range(0, len(audio), block):
                frame = audio[start : start + block]
                if frame.size < block:
                    frame = np.pad(frame, (0, block - frame.size))
                for utterance in self._segmenter.push(frame):
                    self._generation += 1
                    self._utterances.put(utterance)

    # -- consumer side ---------------------------------------------------
    def poll(self) -> LiveUpdate:
        """Collect finished sentences, the current caption and due speech."""
        now = self._clock()
        with self._out_lock:
            update = LiveUpdate(finals=self._finals, changed=bool(self._finals))
            self._finals = []
            update.partial_transcript, update.partial_translation = self._partial
            if self._partial_dirty:
                update.changed = True
                self._partial_dirty = False
            # Only hand out the next clip once the previous one has finished,
            # so a new clip never cuts off the one still playing.
            if self._clips and now >= self._playing_until:
                update.speech = _concatenate(self._clips)
                self._clips = []
                # A little slack for the browser to start playback.
                self._playing_until = now + update.speech.duration + 0.15
                update.changed = True
        return update

    @property
    def is_speaking(self) -> bool:
        return self._segmenter.is_speaking

    # -- worker ----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._utterances.get(timeout=0.05)
            except queue.Empty:
                self._idle()
                continue
            if utterance is None:
                break
            try:
                self._finalise(utterance)
            except Exception:
                logger.exception("Failed to translate an utterance")
        try:
            self._release(self._buffer.flush(), 0.0, StageTimings())
        except Exception:  # pragma: no cover - shutdown best effort
            logger.debug("Failed to flush the sentence buffer", exc_info=True)

    def _idle(self) -> None:
        """No finished utterance waiting: flush stale text or update the caption."""
        if (
            self._segmenter.is_speaking
            and self._clock() - self._last_feed > self.stall_timeout
        ):
            # Audio stopped arriving mid-utterance; finish it rather than wait
            # for silence that will never be sent.
            self.end_of_stream()
            return
        if not self._segmenter.is_speaking and self._buffer.pending:
            # The speaker stopped after a cut-off fragment; do not strand it.
            self._release(self._buffer.flush(), 0.0, StageTimings())
            return
        if self._buffer.expired():
            self._release(self._buffer.flush(), 0.0, StageTimings())
        if self.partials:
            self._update_partial()

    def _finalise(self, utterance: Utterance) -> None:
        pipeline = self._pipeline_for(self.tgt, self.speak)
        timings = StageTimings()
        with self._model_lock, timed(timings, "asr"):
            transcript = pipeline.recognizer.transcribe(utterance.audio, self.src)
        if transcript.text:
            self._last_language = transcript.language

        # Text from an utterance cut off by the length cap is held until its
        # clause ends; a real pause releases everything.
        ready = self._buffer.add(transcript.text) if transcript.text else []
        if not utterance.truncated:
            ready += self._buffer.flush()
        elif self.clause_min_words:
            ready += self._buffer.take_clauses(self.clause_min_words)
        if self.clause_min_words:
            ready = [c for sentence in ready for c in split_clauses(sentence, self.clause_min_words)]
        self._release(ready, utterance.duration, timings)
        self._set_partial("", "")

    def _release(self, sentences: list[str], duration: float, timings: StageTimings) -> None:
        """Translate, voice and publish finished sentences, one at a time.

        Translating sentence by sentence rather than the whole utterance at
        once gets the first one out sooner, and NLLB tends to drop a sentence
        when given several.
        """
        for index, sentence in enumerate(sentences):
            first = index == 0
            self._emit(
                sentence,
                duration if first else 0.0,
                timings if first else StageTimings(),
            )

    def _emit(self, text: str, duration: float, timings: StageTimings) -> None:
        pipeline = self._pipeline_for(self.tgt, self.speak)
        result = PipelineResult(
            transcript=text,
            source_language=self._last_language,
            target_language=self.tgt,
            audio_duration=duration,
            timings=timings,
        )
        result.translation = self._translate(pipeline, text, timings)
        speed = 1.0
        if self.speak and result.translation:
            speed = self._catch_up_speed()
            with self._model_lock, timed(timings, "tts"):
                result.speech = pipeline.synthesize(result.translation, speed=speed)

        logger.info(
            "Final %s -> %s (voice x%.2f): %r -> %r",
            result.source_language,
            self.tgt,
            speed,
            text,
            result.translation,
        )
        with self._out_lock:
            self._finals.append(result)
            if result.speech is not None and len(result.speech.audio):
                self._clips.append(result.speech)
                self._trim_backlog()

    def _translate(self, pipeline: Pipeline, text: str, timings: StageTimings) -> str:
        language = self._last_language
        if language in (self.tgt, AUTO):
            return text
        with self._model_lock, timed(timings, "mt"):
            return pipeline.translator.translate(text, language, self.tgt).text

    def _update_partial(self) -> None:
        now = self._clock()
        if now - self._last_partial_at < self.partial_interval:
            return
        if not self._utterances.empty():
            return  # a finished sentence is waiting; it goes first
        with self._segmenter_lock:
            audio = self._segmenter.current_audio()
            generation = self._generation
        if audio is None or len(audio) < self.min_partial_audio * self.audio.sample_rate:
            return
        self._last_partial_at = now

        pipeline = self._pipeline_for(self.tgt, self.speak)
        # Once a language has been detected, pin it: detection on a second of
        # audio is unreliable, and skipping it makes the caption cheaper.
        language = self.src if self.src != AUTO else self._last_language
        with self._model_lock:
            transcript = pipeline.recognizer.transcribe(audio, language)
        if not transcript.text or generation != self._generation:
            return  # nothing heard, or the utterance was finalised meanwhile
        if self._last_language == AUTO:
            self._last_language = transcript.language

        text = f"{self._buffer.pending} {transcript.text}".strip()
        previous_text, previous_translation = self._partial
        if text == previous_text:
            return  # nothing new was heard; the caption is already current
        if (
            not self._utterances.empty()
            or self._clock() - self._last_partial_translation_at < self.partial_translate_interval
        ):
            # Refresh the source text now, translate it on a later pass.
            if generation == self._generation:
                self._set_partial(text, previous_translation)
            return
        self._last_partial_translation_at = self._clock()
        translation = self._translate(pipeline, text, StageTimings())
        if generation == self._generation:
            self._set_partial(text, translation)

    def _set_partial(self, transcript: str, translation: str) -> None:
        with self._out_lock:
            if self._partial != (transcript, translation):
                self._partial = (transcript, translation)
                self._partial_dirty = True

    def _catch_up_speed(self) -> float:
        """Speaking rate for the next sentence, from how far behind speech is."""
        with self._out_lock:
            behind = max(0.0, self._playing_until - self._clock()) + sum(
                clip.duration for clip in self._clips
            )
        if self.catch_up_seconds <= 0:
            return self.max_speed if behind > 0 else self.base_speed
        boost = (self.max_speed - self.base_speed) * min(1.0, behind / self.catch_up_seconds)
        return self.base_speed + boost

    def _trim_backlog(self) -> None:
        while (
            len(self._clips) > 1
            and sum(clip.duration for clip in self._clips) > self.max_speech_backlog
        ):
            self._clips.pop(0)
            logger.warning("Speech is behind real time; dropped an older sentence.")

    def __enter__(self) -> LiveTranslator:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def _concatenate(clips: list[SpeechAudio]) -> SpeechAudio:
    """Join queued clips, with a short pause between sentences."""
    rate = clips[0].sample_rate
    gap = np.zeros(int(rate * 0.12), dtype=np.float32)
    parts: list[np.ndarray] = []
    for clip in clips:
        if clip.sample_rate != rate:  # the voice changed mid-queue
            continue
        if parts:
            parts.append(gap)
        parts.append(np.asarray(clip.audio, dtype=np.float32))
    return SpeechAudio(audio=np.concatenate(parts), sample_rate=rate, voice=clips[0].voice)
