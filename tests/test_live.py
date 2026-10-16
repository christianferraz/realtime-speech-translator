"""The browser-side live translator, driven with fake models."""

from __future__ import annotations

import time

import numpy as np
import pytest

from speech_translate.config import AudioSettings, Settings, VADSettings
from speech_translate.live import LiveTranslator, LiveUpdate
from speech_translate.pipeline import Pipeline
from speech_translate.tts import SpeechAudio

from .conftest import FakeRecognizer, FakeTranslator, FakeTTS, quiet, tone

VAD = VADSettings(
    threshold=0.01, min_speech_ms=90, min_silence_ms=300, max_utterance_ms=3_000, pre_roll_ms=90
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_live(text: str = "Hello there.", **kwargs):
    recognizer = FakeRecognizer(text=text, language="eng_Latn")
    translator = FakeTranslator()
    tts = FakeTTS()
    pipeline = Pipeline(
        Settings(src="auto", tgt="spa_Latn"), recognizer=recognizer, translator=translator, tts=tts
    )
    kwargs.setdefault("clock", FakeClock())
    live = LiveTranslator(
        lambda tgt, speak: pipeline,
        tgt="spa_Latn",
        audio=AudioSettings(),
        vad=VAD,
        **kwargs,
    )
    return live, recognizer, translator, tts


def poll_until(live: LiveTranslator, predicate, timeout: float = 3.0) -> LiveUpdate:
    """Poll, accumulating finals, until ``predicate(update)`` holds."""
    deadline = time.monotonic() + timeout
    finals = []
    while time.monotonic() < deadline:
        update = live.poll()
        finals += update.finals
        update.finals = finals
        if predicate(update):
            return update
        time.sleep(0.01)
    raise AssertionError("condition not reached")


class TestFinals:
    def test_a_pause_produces_a_translated_sentence(self) -> None:
        with make_live(partials=False)[0] as live:
            live.feed(np.concatenate([quiet(0.2), tone(1.0), quiet(0.6)]))
            update = poll_until(live, lambda u: u.finals)
        assert update.finals[0].transcript == "Hello there."
        assert update.finals[0].translation == "Hola."

    def test_feed_does_not_wait_for_the_models(self) -> None:
        """The bug this module exists for: capture stalled behind translation."""
        live, recognizer, _, _ = make_live(partials=False)
        slow = recognizer.transcribe

        def transcribe(audio, language="auto"):  # noqa: ANN001, ANN202
            time.sleep(0.5)
            return slow(audio, language)

        recognizer.transcribe = transcribe
        with live:
            started = time.perf_counter()
            for _ in range(3):
                live.feed(np.concatenate([tone(0.5), quiet(0.5)]))
            assert time.perf_counter() - started < 0.2

    def test_each_sentence_is_translated_separately(self) -> None:
        live, _, translator, _ = make_live(text="One thing. Another thing.", partials=False)
        with live:
            live.feed(np.concatenate([tone(1.0), quiet(0.6)]))
            poll_until(live, lambda u: len(u.finals) == 2)
        assert [call[0] for call in translator.calls] == ["One thing.", "Another thing."]

    def test_end_of_stream_translates_speech_still_in_progress(self) -> None:
        live = make_live(partials=False)[0]
        with live:
            live.feed(tone(1.0))  # no trailing silence
            live.end_of_stream()
            update = poll_until(live, lambda u: u.finals)
        assert update.finals[0].translation == "Hola."


class TestPartials:
    def test_caption_appears_while_still_speaking(self) -> None:
        live, _, _, tts = make_live(partial_interval=0.0, min_partial_audio=0.5)
        with live:
            live.feed(tone(1.0))  # still talking: no utterance has ended
            update = poll_until(live, lambda u: u.partial_translation)
            assert tts.calls == [], "provisional captions must never be spoken"
        assert update.partial_transcript == "Hello there."
        assert update.partial_translation == "Hola."
        assert not update.finals

    def test_caption_clears_once_the_sentence_is_final(self) -> None:
        live = make_live(partial_interval=0.0, min_partial_audio=0.5)[0]
        with live:
            live.feed(tone(1.0))
            poll_until(live, lambda u: u.partial_translation)
            live.feed(quiet(0.6))
            update = poll_until(live, lambda u: u.finals and not u.partial_translation)
        assert update.finals[0].translation == "Hola."

    def test_no_caption_for_too_little_audio(self) -> None:
        live, recognizer, _, _ = make_live(partial_interval=0.0, min_partial_audio=2.0)
        with live:
            live.feed(tone(1.0))
            time.sleep(0.2)
            assert live.poll().partial_translation == ""
        # Only the end-of-stream final ran ASR, never a partial.
        assert len(recognizer.calls) == 1

    def test_partials_can_be_disabled(self) -> None:
        live, recognizer, _, _ = make_live(partials=False)
        with live:
            live.feed(tone(1.0))
            time.sleep(0.2)
            assert recognizer.calls == []


class TestSpeech:
    def clip(self, seconds: float) -> SpeechAudio:
        return SpeechAudio(audio=np.zeros(int(22_050 * seconds), dtype=np.float32), sample_rate=22_050)

    def test_next_clip_waits_for_the_previous_one_to_finish(self) -> None:
        clock = FakeClock()
        live = make_live(clock=clock)[0]
        live._clips = [self.clip(1.0)]
        first = live.poll()
        assert first.speech is not None

        live._clips = [self.clip(1.0)]
        assert live.poll().speech is None, "would cut off the clip still playing"
        clock.now += 1.2
        assert live.poll().speech is not None

    def test_queued_clips_are_joined(self) -> None:
        live = make_live()[0]
        live._clips = [self.clip(0.5), self.clip(0.5)]
        speech = live.poll().speech
        assert speech is not None
        assert speech.duration == pytest.approx(1.12, abs=0.01)

    def test_backlog_drops_the_oldest_speech(self) -> None:
        live = make_live(max_speech_backlog=2.0)[0]
        live._clips = [self.clip(1.0), self.clip(1.0), self.clip(1.0)]
        live._trim_backlog()
        assert len(live._clips) == 2

    def test_spoken_finals_queue_speech(self) -> None:
        live, _, _, tts = make_live(partials=False)
        with live:
            live.feed(np.concatenate([tone(1.0), quiet(0.6)]))
            update = poll_until(live, lambda u: u.speech is not None)
        assert tts.calls == ["Hola."]
        assert update.speech.duration > 0

    def test_subtitles_only_never_synthesises(self) -> None:
        live, _, _, tts = make_live(partials=False, speak=False)
        with live:
            live.feed(np.concatenate([tone(1.0), quiet(0.6)]))
            poll_until(live, lambda u: u.finals)
        assert tts.calls == []


def test_same_language_is_not_translated() -> None:
    live, _, translator, _ = make_live(partials=False)
    live.configure(src="auto", tgt="eng_Latn", speak=False)
    with live:
        live.feed(np.concatenate([tone(1.0), quiet(0.6)]))
        update = poll_until(live, lambda u: u.finals)
    assert update.finals[0].translation == "Hello there."
    assert translator.calls == []


class TestCatchUp:
    """Translated speech is longer than the original; the voice must not drift."""

    def clip(self, seconds: float) -> SpeechAudio:
        return SpeechAudio(audio=np.zeros(int(22_050 * seconds), dtype=np.float32), sample_rate=22_050)

    def test_base_speed_when_nothing_is_queued(self) -> None:
        live = make_live(base_speed=1.15)[0]
        assert live._catch_up_speed() == pytest.approx(1.15)

    def test_speeds_up_with_the_backlog(self) -> None:
        live = make_live(base_speed=1.0, max_speed=1.5, catch_up_seconds=4.0)[0]
        live._clips = [self.clip(2.0)]
        assert live._catch_up_speed() == pytest.approx(1.25)

    def test_speed_is_capped(self) -> None:
        live = make_live(base_speed=1.0, max_speed=1.5, catch_up_seconds=4.0)[0]
        live._clips = [self.clip(10.0)]
        assert live._catch_up_speed() == pytest.approx(1.5)

    def test_speech_still_playing_counts_as_backlog(self) -> None:
        clock = FakeClock()
        live = make_live(clock=clock, base_speed=1.0, max_speed=1.5, catch_up_seconds=4.0)[0]
        live._playing_until = clock.now + 4.0
        assert live._catch_up_speed() == pytest.approx(1.5)

    def test_synthesis_receives_the_speed(self) -> None:
        live, _, _, tts = make_live(partials=False)
        with live:
            live.feed(np.concatenate([tone(1.0), quiet(0.6)]))
            poll_until(live, lambda u: u.finals)
        assert tts.speeds == [pytest.approx(live.base_speed)]


def test_utterance_is_finalised_when_audio_stops_arriving() -> None:
    """A paused video sends no silence, so the last sentence would hang."""
    clock = FakeClock()
    live = make_live(clock=clock, partials=False, stall_timeout=1.0)[0]
    with live:
        live.feed(tone(1.0))  # mid-utterance, then nothing more arrives
        time.sleep(0.2)
        assert live.poll().finals == []
        clock.now += 1.5
        update = poll_until(live, lambda u: u.finals)
    assert update.finals[0].translation == "Hola."


class TestPartialCost:
    """Captions must not starve the sentences that get spoken."""

    def test_unchanged_caption_is_not_retranslated(self) -> None:
        live, _, translator, _ = make_live(
            partial_interval=0.0, partial_translate_interval=0.0, min_partial_audio=0.5
        )
        with live:
            live.feed(tone(1.0))
            poll_until(live, lambda u: u.partial_translation)
            time.sleep(0.3)  # many idle passes over the same audio
            calls = len(translator.calls)
        assert calls == 1

    def test_translation_is_throttled_but_source_text_updates(self) -> None:
        live, recognizer, translator, _ = make_live(
            partial_interval=0.0, partial_translate_interval=60.0, min_partial_audio=0.5
        )
        with live:
            live.feed(tone(1.0))
            poll_until(live, lambda u: u.partial_translation == "Hola.")
            recognizer.text = "Hello there, friend"
            update = poll_until(live, lambda u: u.partial_transcript == "Hello there, friend")
            assert update.partial_translation == "Hola."  # kept, not re-translated yet
            assert len(translator.calls) == 1


def test_long_sentences_are_translated_clause_by_clause() -> None:
    text = "If you have any questions, leave them in the comments, and I will answer them all."
    live, _, translator, _ = make_live(text=text, partials=False, clause_min_words=5)
    with live:
        live.feed(np.concatenate([tone(1.0), quiet(0.6)]))
        poll_until(live, lambda u: len(u.finals) == 3)
    assert [call[0] for call in translator.calls] == [
        "If you have any questions,",
        "leave them in the comments,",
        "and I will answer them all.",
    ]
