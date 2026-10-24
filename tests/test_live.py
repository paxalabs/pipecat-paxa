import pytest

from pipecat_paxa._live import LegWords, LiveSession, take_frame, utf16_len


def ready_session(grace: float = 0.5) -> LiveSession:
    session = LiveSession(grace=grace)
    session.started()
    return session


def charge(session: LiveSession, seq: int, *texts: str):
    return session.on_charged(seq, sum(utf16_len(t) for t in texts))


def test_nothing_goes_out_before_the_session_starts():
    session = LiveSession()
    session.add_text("a", "Hello.")
    assert session.next_frame(0.0) is None
    session.started()
    assert session.next_frame(0.0) == {"type": "text", "text": "Hello.\n"}


def test_sentences_that_arrive_during_a_leg_share_the_next_frame():
    session = ready_session()
    session.add_text("a", "One.")
    assert session.next_frame(0.0) == {"type": "text", "text": "One.\n"}
    session.add_text("a", "Two.")
    assert session.next_frame(0.1) is None  # sent, no leg yet
    charge(session, 1, "One.\n")
    session.add_text("a", "Three.")
    assert session.next_frame(0.2) is None  # leg 1 is speaking
    assert session.on_flushed(1).leg is not None
    assert session.next_frame(0.3) == {"type": "text", "text": "Two. Three.\n"}


def test_a_frame_that_starts_no_leg_holds_the_next_for_the_grace_time():
    session = ready_session(grace=0.5)
    session.add_text("a", "[whispering]")
    assert session.next_frame(0.0) == {"type": "text", "text": "[whispering]\n"}
    session.add_text("a", "Secret.")
    assert session.next_frame(0.4) is None
    assert session.wait_time(0.4) == pytest.approx(0.1)
    assert session.next_frame(0.5) == {"type": "text", "text": "Secret.\n"}
    # Paxa's leg takes both frames: the tag waited in its buffer.
    leg = charge(session, 1, "[whispering]\n", "Secret.\n")
    assert leg is not None and leg.text == "[whispering]\nSecret.\n"


def test_context_closes_on_the_answer_to_its_flush():
    session = ready_session()
    session.add_text("a", "One.")
    session.request_flush("a")
    assert session.next_frame(0.0)["type"] == "text"
    assert session.next_frame(0.0) is None  # the flush waits for the leg
    charge(session, 1, "One.\n")
    outcome = session.on_flushed(1)
    assert outcome.leg is not None and outcome.finished is None
    assert session.next_frame(0.1) == {"type": "flush"}
    # Nothing was left to speak, so Paxa repeats the last leg's seq.
    outcome = session.on_flushed(1)
    assert outcome.leg is None and outcome.finished is not None
    assert outcome.finished.context_id == "a"


def test_a_leg_started_by_the_flush_closes_the_context():
    session = ready_session(grace=0.5)
    session.add_text("a", "Hi <emph")
    session.request_flush("a")
    session.next_frame(0.0)
    # Paxa held back the unfinished mark, so the auto leg is shorter than the frame.
    leg = session.on_charged(1, 3)
    assert leg is not None and leg.text is None
    session.on_flushed(1)
    assert session.next_frame(0.1) == {"type": "flush"}
    leg = session.on_charged(2, 6)
    assert leg is not None and leg.answers_flush
    assert session.on_flushed(2).finished is not None


def test_contexts_take_the_socket_in_turn():
    session = ready_session()
    session.add_text("a", "First.")
    session.add_text("b", "Second.")
    session.request_flush("a")
    session.request_flush("b")
    assert session.next_frame(0.0) == {"type": "text", "text": "First.\n"}
    charge(session, 1, "First.\n")
    session.on_flushed(1)
    assert session.next_frame(0.1) == {"type": "flush"}
    assert session.next_frame(0.1) is None  # b waits for a's answer
    assert session.on_flushed(1).finished.context_id == "a"
    assert session.next_frame(0.2) == {"type": "text", "text": "Second.\n"}
    leg = charge(session, 2, "Second.\n")
    assert leg is not None and leg.context.context_id == "b"


def test_interrupting_sent_speech_cancels_and_drops_until_cancelled():
    session = ready_session()
    session.add_text("a", "One.")
    session.next_frame(0.0)
    charge(session, 1, "One.\n")
    assert session.interrupt("a") is True
    assert session.leg() is None
    session.add_text("b", "Next.")
    assert session.next_frame(0.1) is None
    # Events for the abandoned speech are ignored.
    assert session.on_charged(2, 5) is None
    assert session.on_flushed(1).finished is None
    # A second interruption has nothing new on the socket to cancel.
    assert session.interrupt("b") is False
    session.add_text("c", "Again.")
    session.on_cancelled()
    assert session.next_frame(0.2) == {"type": "text", "text": "Again.\n"}


def test_interrupting_unsent_speech_needs_no_cancel():
    session = ready_session()
    session.add_text("a", "One.")
    assert session.interrupt("a") is False
    assert session.next_frame(0.0) is None


def test_interrupting_a_queued_context_leaves_the_one_speaking():
    session = ready_session()
    session.add_text("a", "Speaking.")
    session.add_text("b", "Queued.")
    session.next_frame(0.0)
    assert session.interrupt("b") is False
    assert not session.cancelling
    leg = charge(session, 1, "Speaking.\n")
    assert leg is not None and leg.context.context_id == "a"


def test_keepalive_only_when_idle_and_its_answer_is_ignored():
    session = ready_session()
    assert session.keepalive_ready()
    session.keepalive_sent()
    assert not session.keepalive_ready()
    session.add_text("a", "Hi.")
    session.request_flush("a")
    session.next_frame(0.0)
    # The keepalive's answer arrives first and closes nothing.
    assert session.on_flushed(0).finished is None
    charge(session, 1, "Hi.\n")
    session.on_flushed(1)
    session.next_frame(0.1)
    assert session.on_flushed(1).finished.context_id == "a"
    assert session.keepalive_ready()


def test_tagged_text_is_marked():
    session = ready_session()
    session.add_text("a", "[laughs] That is funny.")
    session.next_frame(0.0)
    session.add_text("b", "Plain.")
    assert session._contexts["a"].tagged
    assert not session._contexts["b"].tagged


def test_rejected_frame_starts_no_leg():
    session = ready_session(grace=0.5)
    session.add_text("a", "Bad [british accent] tag.")
    session.next_frame(0.0)
    session.on_frame_rejected()
    session.add_text("a", "Good.")
    assert session.next_frame(0.0) == {"type": "text", "text": "Good.\n"}
    leg = charge(session, 1, "Good.\n")
    assert leg is not None and leg.text == "Good.\n"


def test_failed_leg_lets_the_next_frame_out():
    session = ready_session()
    session.add_text("a", "One.")
    session.next_frame(0.0)
    charge(session, 1, "One.\n")
    session.add_text("a", "Two.")
    assert session.on_leg_failed(1) is None
    assert session.next_frame(0.1) == {"type": "text", "text": "Two.\n"}


def test_abandon_drops_the_context_on_the_socket():
    session = ready_session()
    session.add_text("a", "One.")
    session.next_frame(0.0)
    abandoned = session.abandon_active()
    assert abandoned is not None and abandoned.context_id == "a"
    assert session.cancelling
    session.on_cancelled()
    # The flush that re-arms Paxa's auto-flush is answered and ignored.
    assert session.on_flushed(0).finished is None
    assert session.keepalive_ready()


def test_leg_offsets_follow_the_context_audio():
    session = ready_session()
    session.add_text("a", "One.")
    session.next_frame(0.0)
    leg = charge(session, 1, "One.\n")
    assert leg is not None and leg.offset == 0.0
    leg.context.audio_bytes += 48000  # one second
    session.on_flushed(1)
    session.add_text("a", "Two.")
    session.next_frame(0.1)
    leg = charge(session, 2, "Two.\n")
    assert leg is not None and leg.offset == 1.0


def test_characters_are_counted_as_paxa_counts_them():
    session = ready_session()
    session.add_text("a", "Hi 👍")
    session.next_frame(0.0)
    # The emoji is two UTF-16 code units, so the leg reports 6 characters.
    leg = session.on_charged(1, 6)
    assert leg is not None and leg.text == "Hi 👍\n"


def test_take_frame_batches_and_cuts_long_sentences():
    pending = ["a" * 5, "b" * 5, "c" * 5]
    assert take_frame(pending, 11) == "aaaaa bbbbb"
    assert pending == ["ccccc"]
    pending = ["word " * 10]
    head = take_frame(pending, 12)
    assert head == "word word"
    assert pending == ["word " * 8]


def test_take_frame_never_cuts_before_a_thai_mark():
    text = "ที่" * 10
    pending = [text]
    head = take_frame(pending, 2)
    assert head == "ท"
    assert pending == [text[1:]]


def test_words_keep_english_punctuation():
    words = LegWords("Hello, world. How are you?\n", 0.0)
    out = words.add([{"text": "Hello", "start": 0.0}, {"text": "world", "start": 0.4}])
    assert out == [("Hello,", 0.0)]
    out = words.add(
        [
            {"text": "How", "start": 1.0},
            {"text": "are", "start": 1.2},
            {"text": "you", "start": 1.4},
        ]
    )
    assert out == [("world.", 0.4), ("How", 1.0), ("are", 1.2)]
    assert words.finish() == [("you?", 1.4)]


def test_thai_words_of_a_phrase_join():
    words = LegWords("สวัสดีครับ ยินดีต้อนรับ\n", 2.0)
    out = words.add(
        [
            {"text": "สวัสดี", "start": 0.0},
            {"text": "ครับ", "start": 0.3},
            {"text": "ยินดี", "start": 0.6},
        ]
    )
    assert out == [("สวัสดีครับ", 2.0)]
    out = words.add([{"text": "ต้อนรับ", "start": 0.9}])
    assert out == []
    assert words.finish() == [("ยินดีต้อนรับ", 2.6)]


def test_mixed_thai_and_latin_keep_their_spaces():
    words = LegWords("ราคา 500 บาท ใช้ Wi-Fi ฟรี\n", 0.0)
    spans = ["ราคา", "500", "บาท", "ใช้", "Wi", "Fi", "ฟรี"]
    out = words.add([{"text": t, "start": float(i)} for i, t in enumerate(spans)])
    out += words.finish()
    assert [w for w, _ in out] == ["ราคา", "500", "บาท", "ใช้", "Wi-Fi", "ฟรี"]
    assert [t for _, t in out] == [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


def test_words_without_the_leg_text_fall_back_to_the_spans():
    words = LegWords(None, 0.0)
    out = words.add([{"text": "Hello", "start": 0.0}, {"text": "ครับ", "start": 0.5}])
    assert out == [("Hello", 0.0)]
    assert words.finish() == [("ครับ", 0.5)]


def test_a_span_missing_from_the_text_stands_alone():
    words = LegWords("Hello there\n", 0.0)
    out = words.add([{"text": "Hello", "start": 0.0}, {"text": "hullo", "start": 0.3}])
    out += words.add([{"text": "there", "start": 0.6}])
    out += words.finish()
    assert out == [("Hello", 0.0), ("hullo", 0.3), ("there", 0.6)]


def test_malformed_spans_are_skipped():
    words = LegWords("Hi there\n", 0.0)
    out = words.add([{"text": "Hi", "start": "0"}, "x", {"text": "", "start": 0.1}])
    assert out == []
    assert words.finish() == []
