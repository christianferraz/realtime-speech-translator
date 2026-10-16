"""Sentence buffering before synthesis."""

from __future__ import annotations

import time

from speech_translate.segmentation import SentenceBuffer


class TestSentenceExtraction:
    def test_incomplete_text_is_held_back(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("I went to the") == []
        assert buffer.pending == "I went to the"

    def test_complete_sentence_is_released(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("Hello there.") == ["Hello there."]
        assert buffer.pending == ""

    def test_fragments_are_joined_into_one_sentence(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("I went to the") == []
        assert buffer.add("store yesterday.") == ["I went to the store yesterday."]

    def test_multiple_sentences_released_in_order(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_trailing_fragment_is_kept(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("Done. And then") == ["Done."]
        assert buffer.pending == "And then"

    def test_question_and_exclamation_end_sentences(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("Really?") == ["Really?"]
        assert buffer.add("Wow!") == ["Wow!"]

    def test_cjk_punctuation_ends_sentences(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("你好。") == ["你好。"]

    def test_devanagari_danda_ends_sentences(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("नमस्ते।") == ["नमस्ते।"]

    def test_closing_quote_stays_with_the_sentence(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add('He said "hello."') == ['He said "hello."']


class TestAbbreviations:
    def test_title_abbreviation_does_not_split(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("Dr. Smith is here.") == ["Dr. Smith is here."]

    def test_eg_does_not_split(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("Use fruit, e.g. apples and pears.") == [
            "Use fruit, e.g. apples and pears."
        ]

    def test_initials_do_not_split(self) -> None:
        buffer = SentenceBuffer()
        assert buffer.add("J. Smith arrived.") == ["J. Smith arrived."]


class TestFlushing:
    def test_length_cap_forces_a_flush(self) -> None:
        buffer = SentenceBuffer(max_chars=20)
        released = buffer.add("a" * 25)
        assert released == ["a" * 25]
        assert buffer.pending == ""

    def test_flush_releases_incomplete_text(self) -> None:
        buffer = SentenceBuffer()
        buffer.add("no punctuation here")
        assert buffer.flush() == ["no punctuation here"]
        assert buffer.pending == ""

    def test_flush_when_empty_returns_nothing(self) -> None:
        assert SentenceBuffer().flush() == []

    def test_expiry_releases_a_stranded_fragment(self) -> None:
        buffer = SentenceBuffer(max_wait=0.01)
        buffer.add("waiting")
        assert not buffer.expired(time.monotonic() - 1)
        time.sleep(0.02)
        assert buffer.expired()
        assert buffer.flush_if_expired() == ["waiting"]

    def test_no_expiry_when_buffer_is_empty(self) -> None:
        buffer = SentenceBuffer(max_wait=0.0)
        assert not buffer.expired()
        assert buffer.flush_if_expired() == []

    def test_reset_clears_everything(self) -> None:
        buffer = SentenceBuffer()
        buffer.add("something")
        buffer.reset()
        assert buffer.pending == ""
        assert not buffer.expired()


def test_empty_and_whitespace_input_is_ignored() -> None:
    buffer = SentenceBuffer()
    assert buffer.add("") == []
    assert buffer.add("   ") == []
    assert buffer.add(None) == []  # type: ignore[arg-type]
    assert buffer.pending == ""


class TestClauses:
    """Long sentences are voiced clause by clause, like an interpreter would."""

    def test_long_sentence_splits_at_commas(self) -> None:
        from speech_translate.segmentation import split_clauses

        text = "If you have any questions, leave them in the comments, and I will try to answer."
        assert split_clauses(text, min_words=5) == [
            "If you have any questions,",
            "leave them in the comments,",
            "and I will try to answer.",
        ]

    def test_short_pieces_are_merged(self) -> None:
        from speech_translate.segmentation import split_clauses

        assert split_clauses("Yes, I know.", min_words=5) == ["Yes, I know."]
        assert split_clauses("So, today we will look at the borrow checker.", min_words=5) == [
            "So, today we will look at the borrow checker."
        ]

    def test_numbers_and_times_are_not_split(self) -> None:
        from speech_translate.segmentation import split_clauses

        text = "It costs 1,5 euros and the talk starts at 3:30 in the main hall."
        assert split_clauses(text, min_words=3) == [text]

    def test_zero_disables_splitting(self) -> None:
        from speech_translate.segmentation import split_clauses

        text = "If you have any questions, leave them in the comments."
        assert split_clauses(text, min_words=0) == [text]

    def test_take_clauses_keeps_the_unfinished_tail(self) -> None:
        buffer = SentenceBuffer()
        buffer.add("which a lot of people find confusing at first, is actually your")
        assert buffer.take_clauses(min_words=5) == ["which a lot of people find confusing at first,"]
        assert buffer.pending == "is actually your"
