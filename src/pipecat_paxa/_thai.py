#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Thai sentence ends for Pipecat's sentence aggregation.

Thai is written without sentence punctuation: a space between two Thai words
ends a sentence or clause. ``SimpleTextAggregator`` waits for punctuation, so on
its own it holds an unpunctuated Thai reply until the LLM finishes it.

:class:`ThaiTextAggregator` adds one sentence end to ``SimpleTextAggregator``: a
space after a Thai letter, vowel or tone mark and before a Thai consonant or
leading vowel, once the sentence before it is 20 characters long. ``ฯ``
(abbreviation) and ``ๆ`` (repetition) don't end a sentence, since a space follows
them inside one too. The services install it only when the installed Pipecat
doesn't end Thai sentences itself.
"""

import re

from loguru import logger
from pipecat.services.tts_service import TTSService
from pipecat.utils.string import SENTENCE_ENDING_PUNCTUATION
from pipecat.utils.string import match_endofsentence as _pipecat_match_endofsentence
from pipecat.utils.text.base_text_aggregator import Aggregation, BaseTextAggregator, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

_THAI = re.compile(r"[\u0e00-\u0e7f]")
# A Thai letter, vowel or tone mark, followed by whitespace and a consonant or
# leading vowel.
_THAI_SENTENCE_END = re.compile(
    r"[\u0e01-\u0e2e\u0e30-\u0e3a\u0e40-\u0e45\u0e47-\u0e4e](?=\s+[\u0e01-\u0e2e\u0e40-\u0e44])"
)
# A shorter Thai phrase, such as an item in a list, stays with the next one.
_THAI_MIN_SENTENCE_LENGTH = 20


def is_thai(char: str) -> bool:
    """Check whether a character belongs to the Thai script.

    Args:
        char: A single character.

    Returns:
        True if the character is in the Thai Unicode block.
    """
    return "\u0e00" <= char <= "\u0e7f"


def _match_thai_sentence_end(text: str) -> int:
    """Return the end of the first Thai sentence long enough to speak on its own, or 0."""
    for match in _THAI_SENTENCE_END.finditer(text):
        if len(text[: match.end()].lstrip()) >= _THAI_MIN_SENTENCE_LENGTH:
            return match.end()
    return 0


def match_endofsentence(text: str, *, language: str | None = None) -> int:
    """Find the end of the first sentence, also ending one at a space between Thai words.

    Pipecat's ``match_endofsentence`` decides everything else, and punctuation
    before the first Thai sentence end still ends the sentence there.

    Args:
        text: The text to search.
        language: Language code for Pipecat's sentence tokenizer.

    Returns:
        The position of the end of the first sentence, or 0 if none is complete.
    """
    stripped = text.rstrip()
    thai_end = _match_thai_sentence_end(stripped)
    if not thai_end:
        return _pipecat_match_endofsentence(text, language=language)
    return _pipecat_match_endofsentence(stripped[:thai_end], language=language) or thai_end


class ThaiTextAggregator(SimpleTextAggregator):
    """``SimpleTextAggregator`` that also ends a sentence at a space between Thai words.

    While the buffer holds no Thai, ``SimpleTextAggregator`` decides alone.
    """

    async def _check_sentence_with_lookahead(
        self, char: str, *, protected_end: int = 0
    ) -> Aggregation | None:
        if not _THAI.search(self._text):
            return await super()._check_sentence_with_lookahead(char, protected_end=protected_end)

        states = type(self._lookahead_state)
        is_punctuation = char in SENTENCE_ENDING_PUNCTUATION
        is_thai_space = char.isspace() and len(self._text) > 1 and is_thai(self._text[-2])
        result = None

        if self._advance_lookahead(char, is_punctuation=is_punctuation):
            candidate = self._text[:-1] if is_punctuation else self._text
            if protected_end:
                # Blank out punctuation, and Thai letters (a space between them
                # ends a sentence), up to protected_end with equal-width spaces,
                # so boundary offsets still refer to self._text.
                protected = "".join(
                    " " if c in SENTENCE_ENDING_PUNCTUATION or is_thai(c) else c
                    for c in candidate[:protected_end]
                )
                candidate = protected + candidate[protected_end:]
            eos_marker = match_endofsentence(candidate, language=self.language)
            if eos_marker:
                result = Aggregation(
                    text=self._text[:eos_marker].strip(" "), type=TextType.SENTENCE
                )
                self._text = self._text[eos_marker:]
                self._lookahead_state = states.IDLE

        # A space after a Thai word needs a lookahead, as punctuation does.
        if is_punctuation or is_thai_space:
            self._lookahead_state = states.AWAITING_CHARACTER

        return result


_PROBE_THAI = "สวัสดีครับ ยินดีต้อนรับสู่บริการของเรา วันนี้มีอะไรให้ช่วยไหมครับ"
_PROBE_THAI_FIRST = "สวัสดีครับ ยินดีต้อนรับสู่บริการของเรา"
_PROBE_OTHER = 'Hello there. Dr. Smith paid $29.99 today! "Did you see it?" 好的。我们走吧。'

_decision: dict[str, bool] = {}


async def _sentences(aggregator: BaseTextAggregator, text: str) -> list[str]:
    sentences = [aggregation.text async for aggregation in aggregator.aggregate(text)]
    if (rest := await aggregator.flush()) is not None:
        sentences.append(rest.text)
    return sentences


async def thai_aggregation_needed() -> bool:
    """Whether the installed Pipecat needs :class:`ThaiTextAggregator` to end Thai sentences.

    False when Pipecat's own aggregator already ends them. Also False, with a
    warning, when this Pipecat's aggregator changed in a way the subclass doesn't
    follow. The answer is worked out once per process.
    """
    if "needed" not in _decision:
        _decision["needed"] = await _decide()
    return _decision["needed"]


async def _decide() -> bool:
    if len(await _sentences(SimpleTextAggregator(), _PROBE_THAI)) > 1:
        return False
    try:
        thai = await _sentences(ThaiTextAggregator(), _PROBE_THAI)
        other = await _sentences(ThaiTextAggregator(), _PROBE_OTHER)
        expected_other = await _sentences(SimpleTextAggregator(), _PROBE_OTHER)
    except Exception as e:
        logger.warning(f"pipecat-paxa: Thai sentence ends are off for this Pipecat version: {e}")
        return False
    if thai[:1] != [_PROBE_THAI_FIRST] or other != expected_other:
        logger.warning(
            "pipecat-paxa: Thai sentence ends are off for this Pipecat version, "
            "so an unpunctuated Thai reply is spoken once the LLM finishes it"
        )
        return False
    return True


async def use_thai_sentence_ends(service: TTSService) -> None:
    """Give a sentence-aggregating service Thai sentence ends, when Pipecat lacks them.

    Args:
        service: A TTS service that aggregates LLM text into sentences.
    """
    if not await thai_aggregation_needed():
        return
    current = getattr(service, "_text_aggregator", None)
    skipped = getattr(service, "_skipped_text_aggregator", None)
    if type(current) is not SimpleTextAggregator or type(skipped) is not SimpleTextAggregator:
        logger.warning(f"{service}: Thai sentence ends are off: unexpected text aggregator")
        return
    language = service.text_aggregation_language
    service._text_aggregator = ThaiTextAggregator(
        aggregation_type=TextType.SENTENCE, language=language
    )
    service._skipped_text_aggregator = ThaiTextAggregator(
        aggregation_type=TextType.SENTENCE, language=language
    )
