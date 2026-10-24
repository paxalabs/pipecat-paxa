import pytest
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator

from pipecat_paxa import PaxaHttpTTSService, PaxaTTSService, _thai
from pipecat_paxa._thai import ThaiTextAggregator, match_endofsentence

FIRST = "สวัสดีค่ะ ยินดีต้อนรับสู่ร้านอาหารค่ะ"


async def _feed(aggregator, tokens):
    out = []
    for token in tokens:
        out.extend([agg.text async for agg in aggregator.aggregate(token)])
    return out


def test_space_between_thai_words_ends_a_sentence():
    assert match_endofsentence(FIRST + " ต้อง") == len(FIRST)
    assert not match_endofsentence(FIRST + " ")


def test_short_phrase_stays_with_the_next():
    assert not match_endofsentence("สวัสดีค่ะ ยินดี")


def test_no_end_after_abbreviation_or_repetition_marks():
    assert not match_endofsentence("วันพรุ่งนี้ที่กรุงเทพฯ มีฝนตก")
    assert not match_endofsentence("คุณครูประจำชั้นพูดช้าๆ กับนักเรียน")


def test_no_end_next_to_latin_words_or_digits():
    assert not match_endofsentence("กรุณาตรวจสอบอีเมลของคุณใน Inbox ภายในห้านาที")
    assert not match_endofsentence("ยอดเงินคงเหลือในบัญชีของคุณคือ 500 บาท")


def test_whichever_ends_first_wins():
    assert match_endofsentence("ได้ค่ะ! " + FIRST + " ต้อง") == len("ได้ค่ะ!")
    assert match_endofsentence(FIRST + " ต้องการจองไหมคะ? ได้") == len(FIRST)


def test_other_text_is_left_to_pipecat():
    assert match_endofsentence("Hello. World") == len("Hello.")
    assert not match_endofsentence("Dr. Smith")
    assert match_endofsentence("你好。我们") == len("你好。")


async def test_streamed_thai_tokens():
    aggregator = ThaiTextAggregator()
    tokens = ["สวัสดี", "ค่ะ", " ยินดี", "ต้อนรับ", "สู่ร้าน", "ค่ะ", " วันนี้", "มีโต๊ะ", "ว่าง", "ค่ะ"]
    assert await _feed(aggregator, tokens) == ["สวัสดีค่ะ ยินดีต้อนรับสู่ร้านค่ะ"]
    rest = await aggregator.flush()
    assert rest is not None and rest.text == "วันนี้มีโต๊ะว่างค่ะ"


async def test_pipecat_holds_the_same_reply():
    # The behavior the aggregator exists for, on the Pipecat this package is tested with.
    aggregator = SimpleTextAggregator()
    tokens = ["สวัสดี", "ค่ะ", " ยินดี", "ต้อนรับ", "สู่ร้าน", "ค่ะ", " วันนี้", "มีโต๊ะ", "ว่าง", "ค่ะ"]
    assert await _feed(aggregator, tokens) == []


async def test_english_is_aggregated_as_pipecat_does():
    text = 'Hi there. Dr. Smith paid $29.99! "Did you see it?" Yes I did. Next'
    tokens = [text[i : i + 3] for i in range(0, len(text), 3)]
    assert await _feed(ThaiTextAggregator(), tokens) == await _feed(SimpleTextAggregator(), tokens)


class _ThaiSkipTags(SkipTagsAggregator, ThaiTextAggregator):
    pass


async def test_space_inside_a_closed_tag_is_not_an_end():
    aggregator = _ThaiSkipTags([("<spell>", "</spell>")])
    text = "รหัสยืนยันของคุณคือ <spell>ก ข ค ง</spell> กรุณาจดไว้ด้วยนะคะ ขอบคุณค่ะ"
    assert await _feed(aggregator, [text]) == [
        "รหัสยืนยันของคุณคือ <spell>ก ข ค ง</spell> กรุณาจดไว้ด้วยนะคะ"
    ]
    rest = await aggregator.flush()
    assert rest is not None and rest.text == "ขอบคุณค่ะ"


@pytest.fixture
def fresh_decision(monkeypatch):
    monkeypatch.setattr(_thai, "_decision", {})


async def test_services_install_thai_sentence_ends(fresh_decision, monkeypatch):
    monkeypatch.setenv("PAXA_API_KEY", "pxa_test_key")
    for service in (PaxaHttpTTSService(), PaxaTTSService()):
        await _thai.use_thai_sentence_ends(service)
        assert type(service._text_aggregator) is ThaiTextAggregator
        assert type(service._skipped_text_aggregator) is ThaiTextAggregator


async def test_pipecat_with_thai_sentence_ends_is_left_alone(fresh_decision, monkeypatch):
    # A Pipecat whose own aggregator ends Thai sentences keeps it.
    monkeypatch.setattr(_thai, "SimpleTextAggregator", ThaiTextAggregator)
    monkeypatch.setenv("PAXA_API_KEY", "pxa_test_key")
    service = PaxaHttpTTSService()
    original = service._text_aggregator
    await _thai.use_thai_sentence_ends(service)
    assert service._text_aggregator is original


async def test_aggregator_that_fails_its_check_is_not_installed(fresh_decision, monkeypatch):
    async def broken(self, char, *, protected_end=0):
        raise AttributeError("_lookahead_state")

    monkeypatch.setattr(ThaiTextAggregator, "_check_sentence_with_lookahead", broken)
    monkeypatch.setenv("PAXA_API_KEY", "pxa_test_key")
    service = PaxaHttpTTSService()
    original = service._text_aggregator
    await _thai.use_thai_sentence_ends(service)
    assert service._text_aggregator is original
