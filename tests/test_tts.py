import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

from pipecat_paxa import PaxaHttpTTSService, PaxaTTSService, __version__
from pipecat_paxa import tts as tts_module

from .fake_paxa import API_KEY, Behavior, FakePaxa, leg_pcm, utf16_len, wav_header

CLIENT = f"pipecat-paxa/{__version__}"


def audio_of(frames) -> bytes:
    return b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame))


def errors(frames) -> list[ErrorFrame]:
    """The service's errors, without Pipecat's own report of a context that stayed silent."""
    return [f for f in frames if isinstance(f, ErrorFrame) and not silent_context(f)]


def silent_context(frame) -> bool:
    return isinstance(frame, ErrorFrame) and "completed with no audio" in frame.error


def texts_sent(session) -> list[str]:
    return [f["text"] for f in session.frames if f.get("type") == "text"]


def spoken_audio(session) -> bytes:
    return b"".join(leg_pcm(leg.seq, utf16_len(leg.text)) for leg in session.legs)


def live(paxa: FakePaxa, **kwargs) -> PaxaTTSService:
    return PaxaTTSService(api_key=API_KEY, base_url=paxa.base_url, **kwargs)


def http(paxa: FakePaxa, **kwargs) -> PaxaHttpTTSService:
    return PaxaHttpTTSService(api_key=API_KEY, base_url=paxa.base_url, **kwargs)


def llm_turn(*tokens: str, gap: float = 0.0) -> list:
    frames: list = [LLMFullResponseStartFrame()]
    for token in tokens:
        frames.append(LLMTextFrame(token))
        if gap:
            frames.append(SleepFrame(sleep=gap))
    frames.append(LLMFullResponseEndFrame())
    return frames


# Live service


async def test_live_speaks_a_sentence():
    async with FakePaxa() as paxa:
        down, up = await run_test(
            live(paxa), frames_to_send=[TTSSpeakFrame(text="Hello there."), SleepFrame(sleep=0.2)]
        )
    kinds = [type(f) for f in down]
    assert kinds.count(TTSStartedFrame) == 1 and kinds.count(TTSStoppedFrame) == 1
    assert kinds.index(TTSStartedFrame) < kinds.index(TTSAudioRawFrame)
    assert audio_of(down) == leg_pcm(1, utf16_len("Hello there.\n"))
    assert all(f.sample_rate == 24000 for f in down if isinstance(f, TTSAudioRawFrame))
    assert [f.text for f in down if isinstance(f, TTSTextFrame)] == ["Hello there."]
    assert not errors(up)

    (session,) = paxa.sessions
    assert session.headers["x-paxa-client"] == CLIENT
    assert session.headers["User-Agent"] == CLIENT
    assert session.headers["Authorization"] == f"Bearer {API_KEY}"
    assert session.frames == [
        {"type": "start", "model": "paxa-tts-flash-v2", "voice": "foithong", "format": "wav"},
        {"type": "text", "text": "Hello there.\n"},
        {"type": "flush"},
    ]


async def test_live_speaks_a_turn_in_order():
    async with FakePaxa() as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[
                *llm_turn("Hello there. ", "How are you ", "today? ", "I am fine."),
                SleepFrame(sleep=0.3),
            ],
        )
    (session,) = paxa.sessions
    assert " ".join(t.strip() for t in texts_sent(session)) == (
        "Hello there. How are you today? I am fine."
    )
    assert audio_of(down) == spoken_audio(session)
    assert [type(f) for f in down].count(TTSStoppedFrame) == 1
    assert [f for f in session.frames if f["type"] == "flush"] == [{"type": "flush"}]
    assert not errors(up)


async def test_live_batches_a_burst_of_sentences():
    sentences = [f"Sentence number {i}." for i in range(100)]
    async with FakePaxa() as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[*llm_turn(*(s + " " for s in sentences)), SleepFrame(sleep=1.0)],
        )
    (session,) = paxa.sessions
    assert session.close_code != 4413
    assert len(texts_sent(session)) < len(sentences)
    assert " ".join(t.strip() for t in texts_sent(session)) == " ".join(sentences)
    assert audio_of(down) == spoken_audio(session)
    assert not errors(up)


async def test_live_header_split_across_frames():
    async with FakePaxa(Behavior(split_header=True)) as paxa:
        down, _ = await run_test(
            live(paxa), frames_to_send=[TTSSpeakFrame(text="Split."), SleepFrame(sleep=0.2)]
        )
    assert audio_of(down) == leg_pcm(1, utf16_len("Split.\n"))


async def test_live_interruption_cancels_without_reconnecting():
    long_text = "This sentence takes a while to say. " * 8
    async with FakePaxa(Behavior(chunk_delay=0.02)) as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[
                TTSSpeakFrame(text=long_text.strip()),
                SleepFrame(sleep=0.15),
                InterruptionFrame(),
                TTSSpeakFrame(text="After."),
                SleepFrame(sleep=0.3),
            ],
        )
    (session,) = paxa.sessions
    assert {"type": "cancel"} in session.frames
    assert session.legs[0].cancelled
    after = leg_pcm(2, utf16_len("After.\n"))
    audio = audio_of(down)
    assert audio.endswith(after)
    # Nothing of the interrupted leg plays after the next one starts.
    assert len(audio) < len(leg_pcm(1, utf16_len(long_text))) + len(after)
    assert not errors(up)


async def test_live_rate_limit_drops_the_turn_and_recovers():
    async with FakePaxa(Behavior(rate_limited_attempts={1})) as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[
                TTSSpeakFrame(text="One."),
                SleepFrame(sleep=0.2),
                TTSSpeakFrame(text="Two."),
                SleepFrame(sleep=0.2),
            ],
        )
    (error,) = errors(up)
    assert error.category == ErrorCategory.RATE_LIMIT
    assert "rate_limited" in error.error
    # Pipecat also reports the turn that produced no audio.
    assert sum(silent_context(f) for f in up) == 1
    (session,) = paxa.sessions
    assert [leg.text for leg in session.legs] == ["Two.\n"]
    assert audio_of(down) == leg_pcm(1, utf16_len("Two.\n"))


async def test_live_failed_leg_reports_and_continues():
    async with FakePaxa(Behavior(failing_legs={1})) as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[
                TTSSpeakFrame(text="Lost."),
                SleepFrame(sleep=0.2),
                TTSSpeakFrame(text="Kept."),
                SleepFrame(sleep=0.2),
            ],
        )
    (error,) = errors(up)
    assert error.category == ErrorCategory.SERVER
    assert sum(silent_context(f) for f in up) == 1
    assert audio_of(down) == leg_pcm(2, utf16_len("Kept.\n"))


async def test_live_refused_tag_reports_and_continues():
    async with FakePaxa() as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[
                TTSSpeakFrame(text="Cheerio [british accent] mate."),
                SleepFrame(sleep=0.7),
                TTSSpeakFrame(text="Plain."),
                SleepFrame(sleep=0.2),
            ],
        )
    (error,) = errors(up)
    assert error.category == ErrorCategory.APPLICATION
    assert "tag_invalid" in error.error
    assert sum(silent_context(f) for f in up) == 1
    assert audio_of(down) == leg_pcm(1, utf16_len("Plain.\n"))


async def test_live_unknown_voice_makes_the_service_unusable():
    async with FakePaxa() as paxa:
        tts = live(paxa, settings=PaxaTTSService.Settings(voice="nobody"))
        _, up = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)]
        )
    assert errors(up)[0].category == ErrorCategory.INVALID_REQUEST
    assert "unknown_voice" in errors(up)[0].error
    assert not tts.is_usable
    assert len(paxa.sessions) == 1


async def test_live_bad_key_is_an_authentication_error():
    async with FakePaxa() as paxa:
        tts = PaxaTTSService(api_key="pxa_wrong", base_url=paxa.base_url)
        _, up = await run_test(tts, frames_to_send=[SleepFrame(sleep=0.1)])
    (error,) = errors(up)
    assert error.category == ErrorCategory.AUTHENTICATION
    assert "unauthorized" in error.error and "pxa_wrong" not in error.error
    assert not tts.is_usable
    assert paxa.sessions == []


async def test_live_tagged_turn_does_not_carry_into_the_next():
    async with FakePaxa() as paxa:
        await run_test(
            live(paxa),
            frames_to_send=[
                TTSSpeakFrame(text="[whispering] I have a secret."),
                SleepFrame(sleep=0.2),
                TTSSpeakFrame(text="Back to normal."),
                SleepFrame(sleep=0.2),
            ],
        )
    (session,) = paxa.sessions
    kinds = [f["type"] for f in session.frames]
    assert kinds == ["start", "text", "flush", "cancel", "text", "flush"]
    assert session.legs[0].carry is None
    assert session.legs[1].carry is None


async def test_live_word_timestamps_keep_spacing():
    async with FakePaxa() as paxa:
        down, up = await run_test(
            live(paxa, word_timestamps=True),
            frames_to_send=[
                TTSSpeakFrame(text="Hello, world. How are you?"),
                SleepFrame(sleep=0.2),
                TTSSpeakFrame(text="สวัสดีครับ ยินดีต้อนรับ"),
                SleepFrame(sleep=0.2),
            ],
        )
    (session,) = paxa.sessions
    assert session.frames[0]["timestamps"] == "word"
    words = [f for f in down if isinstance(f, TTSTextFrame)]
    assert [f.text for f in words] == [
        "Hello,",
        "world.",
        "How",
        "are",
        "you?",
        "สวัสดีครับ",
        "ยินดีต้อนรับ",
    ]
    assert all(f.pts for f in words)
    joined = concatenate_aggregated_text(
        [TextPartForConcatenation(f.text, f.includes_inter_frame_spaces) for f in words]
    )
    assert joined == "Hello, world. How are you? สวัสดีครับ ยินดีต้อนรับ"
    assert not errors(up)


async def test_live_resamples_to_the_pipeline_rate():
    async with FakePaxa() as paxa:
        down, _ = await run_test(
            live(paxa),
            frames_to_send=[TTSSpeakFrame(text="Resampled."), SleepFrame(sleep=0.2)],
            pipeline_params=PipelineParams(audio_out_sample_rate=16000),
        )
    frames = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert frames and all(f.sample_rate == 16000 for f in frames)
    samples = len(audio_of(down)) // 2
    expected = utf16_len("Resampled.\n") * 240 * 2 // 3
    # Pipecat's stream resampler holds back up to about 40 ms until more audio arrives.
    assert expected - 700 <= samples <= expected


async def test_live_settings_update_opens_a_new_session():
    async with FakePaxa() as paxa:
        tts = live(paxa, settings=PaxaTTSService.Settings(language=Language.TH, speed=1.2))
        down, up = await run_test(
            tts,
            frames_to_send=[
                TTSUpdateSettingsFrame(delta=PaxaTTSService.Settings(voice="massaman")),
                SleepFrame(sleep=0.1),
                TTSSpeakFrame(text="New voice."),
                SleepFrame(sleep=0.2),
            ],
        )
    assert [s.frames[0]["voice"] for s in paxa.sessions] == ["foithong", "massaman"]
    assert paxa.sessions[1].frames[0]["language"] == "th"
    assert paxa.sessions[1].frames[0]["speed"] == 1.2
    assert audio_of(down) == leg_pcm(1, utf16_len("New voice.\n"))
    assert not errors(up)


async def test_live_unusable_settings_update_is_refused():
    async with FakePaxa() as paxa:
        tts = live(paxa)
        _, up = await run_test(
            tts,
            frames_to_send=[
                TTSUpdateSettingsFrame(
                    delta=PaxaTTSService.Settings(language=Language.FR, speed=3.0)
                ),
                SleepFrame(sleep=0.1),
            ],
        )
    (error,) = errors(up)
    assert error.category == ErrorCategory.APPLICATION
    assert tts.settings.language is None and tts.settings.speed is None
    assert len(paxa.sessions) == 1


async def test_live_unexpected_audio_format_is_not_played():
    async with FakePaxa(Behavior(header=wav_header(rate=16000))) as paxa:
        tts = live(paxa)
        down, up = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Odd."), SleepFrame(sleep=0.3)]
        )
    assert audio_of(down) == b""
    assert errors(up)[0].category == ErrorCategory.SERVER
    assert not tts.is_usable


async def test_live_keepalive_holds_an_idle_session_open(monkeypatch):
    monkeypatch.setattr(tts_module, "_KEEPALIVE_INTERVAL", 0.1)
    async with FakePaxa(Behavior(idle_timeout=0.4)) as paxa:
        down, up = await run_test(
            live(paxa),
            frames_to_send=[
                SleepFrame(sleep=1.0),
                TTSSpeakFrame(text="Still here."),
                SleepFrame(sleep=0.2),
            ],
        )
    (session,) = paxa.sessions
    assert session.close_code != 4408
    assert sum(f["type"] == "flush" for f in session.frames) >= 3
    assert audio_of(down) == leg_pcm(1, utf16_len("Still here.\n"))
    assert not errors(up)


async def test_live_reconnects_when_a_cancel_goes_unanswered(monkeypatch):
    monkeypatch.setattr(tts_module, "_RESPONSE_TIMEOUT", 0.3)
    long_text = "This sentence takes a while to say. " * 8
    async with FakePaxa(Behavior(ignore_cancel=True, chunk_delay=0.02)) as paxa:
        down, _ = await run_test(
            live(paxa),
            frames_to_send=[
                TTSSpeakFrame(text=long_text.strip()),
                SleepFrame(sleep=0.15),
                InterruptionFrame(),
                SleepFrame(sleep=1.5),
                TTSSpeakFrame(text="Recovered."),
                SleepFrame(sleep=0.3),
            ],
        )
    first, second = paxa.sessions
    assert {"type": "cancel"} in first.frames
    assert [leg.text for leg in second.legs] == ["Recovered.\n"]
    assert audio_of(down).endswith(leg_pcm(1, utf16_len("Recovered.\n")))


async def test_live_unpunctuated_thai_goes_out_phrase_by_phrase():
    tokens = ["สวัสดี", "ค่ะ", " ยินดี", "ต้อนรับ", "สู่ร้าน", "อาหาร", "ค่ะ", " วันนี้", "มีโต๊ะ", "ว่าง", "ค่ะ"]
    async with FakePaxa() as paxa:
        down, up = await run_test(
            live(paxa), frames_to_send=[*llm_turn(*tokens, gap=0.01), SleepFrame(sleep=0.3)]
        )
    (session,) = paxa.sessions
    assert texts_sent(session) == ["สวัสดีค่ะ ยินดีต้อนรับสู่ร้านอาหารค่ะ\n", "วันนี้มีโต๊ะว่างค่ะ\n"]
    assert audio_of(down) == spoken_audio(session)
    assert not errors(up)


def test_live_refuses_token_streaming():
    with pytest.raises(ValueError, match="TOKEN"):
        PaxaTTSService(api_key=API_KEY, text_aggregation_mode=TextAggregationMode.TOKEN)


def test_api_key_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("PAXA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PAXA_API_KEY"):
        PaxaTTSService()
    monkeypatch.setenv("PAXA_API_KEY", API_KEY)
    assert PaxaTTSService()._api_key == API_KEY


def test_settings_are_checked():
    with pytest.raises(ValueError, match="speed"):
        PaxaTTSService(api_key=API_KEY, settings=PaxaTTSService.Settings(speed=2.0))
    with pytest.raises(ValueError, match="reads auto, th, en or zh"):
        PaxaTTSService(api_key=API_KEY, settings=PaxaTTSService.Settings(language=Language.FR))
    with pytest.raises(ValueError, match="voice"):
        PaxaHttpTTSService(api_key=API_KEY, settings=PaxaHttpTTSService.Settings(voice=""))


def test_languages_map_to_paxa_codes():
    for language, code in [
        (Language.TH, "th"),
        (Language.TH_TH, "th"),
        (Language.EN_GB, "en"),
        (Language.ZH_CN, "zh"),
        (Language.CMN, "zh"),
        ("auto", "auto"),
    ]:
        tts = PaxaTTSService(api_key=API_KEY, settings=PaxaTTSService.Settings(language=language))
        assert tts.settings.language == code


def test_base_url_takes_an_openai_style_v1_suffix():
    tts = PaxaTTSService(api_key=API_KEY, base_url="https://api.paxalabs.com/v1/")
    assert tts._base_url == "https://api.paxalabs.com"


# HTTP service


async def test_http_speaks_a_sentence():
    async with FakePaxa() as paxa:
        down, up = await run_test(
            http(paxa, settings=PaxaHttpTTSService.Settings(language=Language.TH, speed=0.9)),
            frames_to_send=[TTSSpeakFrame(text="สวัสดีครับ")],
        )
    assert audio_of(down) == leg_pcm(1, utf16_len("สวัสดีครับ"))
    kinds = [type(f) for f in down]
    assert TTSStartedFrame in kinds and TTSStoppedFrame in kinds
    assert not errors(up)
    (request,) = paxa.http_requests
    assert request["json"] == {
        "text": "สวัสดีครับ",
        "voice": "foithong",
        "model": "paxa-tts-flash-v2",
        "format": "wav",
        "stream": True,
        "speed": 0.9,
        "language": "th",
    }
    assert request["headers"]["x-paxa-client"] == CLIENT
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"


async def test_http_header_split_across_chunks():
    async with FakePaxa(Behavior(split_header=True)) as paxa:
        down, _ = await run_test(http(paxa), frames_to_send=[TTSSpeakFrame(text="Split.")])
    assert audio_of(down) == leg_pcm(1, utf16_len("Split."))


async def test_http_error_is_reported_by_code():
    async with FakePaxa() as paxa:
        down, up = await run_test(
            http(paxa, settings=PaxaHttpTTSService.Settings(voice="nobody")),
            frames_to_send=[TTSSpeakFrame(text="Hi.")],
        )
    (error,) = errors(up)
    assert error.category == ErrorCategory.INVALID_REQUEST
    assert error.error == (
        "Paxa TTS error (HTTP 400): unknown_voice (GET /v1/voices lists the voice ids)"
    )
    assert audio_of(down) == b""


async def test_http_unexpected_audio_format_is_not_played():
    async with FakePaxa(Behavior(header=wav_header(channels=2))) as paxa:
        down, up = await run_test(http(paxa), frames_to_send=[TTSSpeakFrame(text="Odd.")])
    assert audio_of(down) == b""
    assert errors(up)[0].category == ErrorCategory.SERVER


async def test_http_unreachable_api_is_a_connectivity_error():
    tts = PaxaHttpTTSService(api_key=API_KEY, base_url="http://127.0.0.1:9")
    _, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])
    assert errors(up)[0].category == ErrorCategory.CONNECTIVITY


async def test_http_speaks_thai_phrase_by_phrase():
    tokens = ["สวัสดี", "ค่ะ", " ยินดี", "ต้อนรับ", "สู่ร้าน", "อาหาร", "ค่ะ", " วันนี้", "มีโต๊ะ", "ว่าง", "ค่ะ"]
    async with FakePaxa() as paxa:
        await run_test(http(paxa), frames_to_send=llm_turn(*tokens))
    assert [r["json"]["text"] for r in paxa.http_requests] == [
        "สวัสดีค่ะ ยินดีต้อนรับสู่ร้านอาหารค่ะ",
        "วันนี้มีโต๊ะว่างค่ะ",
    ]
