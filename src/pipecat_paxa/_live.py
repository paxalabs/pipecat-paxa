#
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Client-side bookkeeping for one ``/v1/tts/live`` session.

Paxa synthesizes one leg at a time and answers in order: a ``charged`` receipt,
the leg's audio and ``timing`` spans, then ``flushed``. The socket carries no
context ids, so this module keeps the order of what was sent and works out which
Pipecat audio context each leg belongs to:

- Text goes out one frame per batch of sentences, each ending in a newline,
  which Paxa flushes at once, so every frame becomes a leg of its own.
- A frame goes out once the previous leg has flushed: Paxa closes a session that
  queues 32 frames behind a leg. Sentences that arrive meanwhile share the next
  frame.
- Contexts use the socket one at a time. A context ends with an explicit
  ``flush``, sent once legs have taken its text, and the ``flushed`` that
  answers it marks the context complete.
- An interruption sends ``cancel``. Paxa drops every frame sent before it, so
  everything until ``cancelled`` is discarded and nothing is sent meanwhile.
"""

import re
import unicodedata
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ._api import BYTES_PER_SECOND

# Stays well inside Paxa's per-leg ceiling (max_chars, 5000 today) and its frame limit.
MAX_FRAME_CHARS = 2000
# How long a sent frame may go without starting a leg (a frame of tags alone
# starts none) before the next frame goes out anyway.
LEG_START_GRACE = 0.5

# Text holding a tag: Paxa carries a tag's delivery into the following legs and
# keeps a tag with no words after it in its buffer.
_TAG_START = re.compile(r"[\[［<]")


def utf16_len(text: str) -> int:
    """Length in UTF-16 code units, the unit Paxa counts characters in."""
    return len(text.encode("utf-16-le")) // 2


@dataclass(eq=False)
class LiveContext:
    """A Pipecat audio context's text on its way through the socket.

    Parameters:
        context_id: The Pipecat audio context id.
        pending: Sentences not sent yet.
        flush_requested: Pipecat ended the turn, so the context closes once its text is spoken.
        flush_sent: The closing ``flush`` went out.
        sent: Some frame of this context reached the socket.
        tagged: Its text holds a tag, so the session's carried delivery resets after it.
        audio_bytes: PCM bytes delivered for this context so far.
    """

    context_id: str
    pending: list[str] = field(default_factory=list)
    flush_requested: bool = False
    flush_sent: bool = False
    sent: bool = False
    tagged: bool = False
    audio_bytes: int = 0


@dataclass(eq=False)
class LiveLeg:
    """One synthesis leg, from its ``charged`` receipt to its ``flushed``.

    Parameters:
        seq: Paxa's leg number on this connection.
        context: The context the leg speaks for.
        text: The text the leg speaks, when the frames it took are known.
        offset: Seconds of the context's audio before this leg.
        answers_flush: The context's closing ``flush`` started this leg, so its
            ``flushed`` also completes the context.
    """

    seq: int
    context: LiveContext
    text: str | None
    offset: float
    answers_flush: bool = False


@dataclass
class FlushedOutcome:
    """What a ``flushed`` event completed.

    Parameters:
        leg: The leg that finished, if the event closed one.
        finished: The context whose closing flush the event answered.
    """

    leg: LiveLeg | None = None
    finished: LiveContext | None = None


@dataclass
class _SentFrame:
    context: LiveContext
    size: int
    text: str


class LiveSession:
    """Order and flow control for one live connection, without any I/O."""

    def __init__(self, *, max_frame_chars: int = MAX_FRAME_CHARS, grace: float = LEG_START_GRACE):
        """Initialize an empty session.

        Args:
            max_frame_chars: Largest text frame to send, in characters.
            grace: Seconds a frame may go without starting a leg before the next goes out.
        """
        self._max_frame_chars = max_frame_chars
        self._grace = grace
        self.ready = False
        self.cancelling = False
        self._contexts: OrderedDict[str, LiveContext] = OrderedDict()
        self._unclaimed: deque[_SentFrame] = deque()
        self._leg: LiveLeg | None = None
        # Contexts whose closing flush awaits its answer, in send order; None
        # marks a keepalive flush.
        self._answers: deque[LiveContext | None] = deque()
        self._last_text_sent = 0.0

    def reset(self) -> None:
        """Forget everything, for a new connection."""
        self.ready = False
        self.cancelling = False
        self._contexts.clear()
        self._unclaimed.clear()
        self._leg = None
        self._answers.clear()
        self._last_text_sent = 0.0

    # What the service reports.

    def started(self) -> None:
        """Paxa confirmed the session with ``started``."""
        self.ready = True

    def add_text(self, context_id: str, text: str) -> None:
        """Queue a sentence for a context."""
        context = self._contexts.get(context_id)
        if context is None:
            context = self._contexts[context_id] = LiveContext(context_id)
        context.pending.append(text)

    def request_flush(self, context_id: str) -> None:
        """Close a context once its queued text is spoken."""
        context = self._contexts.get(context_id)
        if context is not None:
            context.flush_requested = True

    def interrupt(self, context_id: str) -> bool:
        """Drop a context Pipecat interrupted.

        Returns:
            True when the context had reached the socket, so a ``cancel`` must go out.
        """
        head = self._head()
        context = self._contexts.pop(context_id, None)
        if context is None or context is not head or not context.sent or self.cancelling:
            return False
        self.begin_cancel()
        return True

    def abandon_active(self) -> LiveContext | None:
        """Give up the context on the socket, whose text Paxa kept unspoken.

        The caller sends ``cancel``, which clears Paxa's buffer, then an empty
        ``flush``: a rate-limited leg stops Paxa from flushing text by itself
        until the next explicit flush. That flush's answer is ignored.

        Returns:
            The abandoned context, if one was on the socket.
        """
        head = self._head()
        if head is not None:
            del self._contexts[head.context_id]
        self.begin_cancel()
        self._answers.append(None)
        return head

    def begin_cancel(self) -> None:
        """Record that a ``cancel`` is going out: Paxa drops every frame sent before it."""
        self.cancelling = True
        self._leg = None
        self._unclaimed.clear()
        self._answers.clear()

    def on_cancelled(self) -> None:
        """Paxa answered the ``cancel``."""
        self.cancelling = False

    # What the service sends.

    def next_frame(self, now: float) -> dict[str, Any] | None:
        """The next frame to send, or None until an event or the grace time allows one."""
        if not self.ready or self.cancelling:
            return None
        context = self._head()
        if context is None or not self._gate_open(now):
            return None
        if context.pending:
            text = take_frame(context.pending, self._max_frame_chars) + "\n"
            self._unclaimed.append(_SentFrame(context, utf16_len(text), text))
            self._last_text_sent = now
            context.sent = True
            context.tagged = context.tagged or bool(_TAG_START.search(text))
            return {"type": "text", "text": text}
        if context.flush_requested and not context.flush_sent:
            context.flush_sent = True
            context.sent = True
            self._answers.append(context)
            return {"type": "flush"}
        return None

    def wait_time(self, now: float) -> float | None:
        """Seconds until :meth:`next_frame` returns a frame without any event, if it will."""
        if not self.ready or self.cancelling or self._leg is not None or not self._unclaimed:
            return None
        context = self._head()
        if context is None:
            return None
        if not context.pending and not (context.flush_requested and not context.flush_sent):
            return None
        return max(0.0, self._last_text_sent + self._grace - now)

    def keepalive_ready(self) -> bool:
        """Whether the session is idle, so a keepalive flush can't be taken for anything else."""
        return (
            self.ready
            and not self.cancelling
            and not self._contexts
            and self._leg is None
            and not self._unclaimed
            and not self._answers
        )

    def keepalive_sent(self) -> None:
        """Record a keepalive flush, whose answer is then ignored."""
        self._answers.append(None)

    @property
    def awaiting_server(self) -> bool:
        """Whether Paxa owes the session an event."""
        return self.cancelling or self._leg is not None or bool(self._answers)

    # What Paxa sends.

    def on_charged(self, seq: int, chars: int) -> LiveLeg | None:
        """A leg started. Returns it, or None while a cancel is pending."""
        if self.cancelling:
            return None
        claimed: list[_SentFrame] = []
        total = 0
        while self._unclaimed and total < chars:
            frame = self._unclaimed.popleft()
            claimed.append(frame)
            total += frame.size
        exact = total == chars
        if not exact:
            # The leg took part of a frame (a tag Paxa held back) or text this
            # session never matched, so its text is unknown from here on.
            self._unclaimed.clear()
        context = claimed[0].context if claimed else self._head()
        if context is None:
            return None
        self._leg = LiveLeg(
            seq=seq,
            context=context,
            text="".join(frame.text for frame in claimed) if exact and claimed else None,
            offset=context.audio_bytes / BYTES_PER_SECOND,
            answers_flush=not claimed and bool(self._answers) and self._answers[0] is context,
        )
        return self._leg

    def leg(self, seq: int | None = None) -> LiveLeg | None:
        """The leg in flight, if any, and if ``seq`` is given, only when it matches."""
        if self.cancelling or self._leg is None:
            return None
        if seq is not None and self._leg.seq != seq:
            return None
        return self._leg

    def on_flushed(self, seq: int) -> FlushedOutcome:
        """A leg finished, or an explicit flush was answered."""
        if self.cancelling:
            return FlushedOutcome()
        leg = self._leg
        if leg is not None:
            if leg.seq != seq:
                return FlushedOutcome()
            self._leg = None
            return FlushedOutcome(leg=leg, finished=self._answer() if leg.answers_flush else None)
        # Nothing was left to speak, so Paxa answered the flush at once.
        return FlushedOutcome(finished=self._answer())

    def on_leg_failed(self, seq: int) -> LiveContext | None:
        """A leg failed after its receipt. Returns the context it completes, if any."""
        leg = self.leg(seq)
        if leg is None:
            return None
        self._leg = None
        return self._answer() if leg.answers_flush else None

    def on_frame_rejected(self) -> None:
        """Paxa refused a text frame, which therefore starts no leg."""
        if not self.cancelling and self._unclaimed:
            self._unclaimed.popleft()

    def _answer(self) -> LiveContext | None:
        if not self._answers:
            return None
        context = self._answers.popleft()
        if context is not None:
            self._contexts.pop(context.context_id, None)
            # Frames no leg took (tags with no words) stay in Paxa's buffer; a
            # tagged context is followed by a cancel that clears it.
            self._unclaimed.clear()
        return context

    def _head(self) -> LiveContext | None:
        return next(iter(self._contexts.values()), None)

    def _gate_open(self, now: float) -> bool:
        return self._leg is None and (
            not self._unclaimed or now - self._last_text_sent >= self._grace
        )


def take_frame(pending: list[str], max_chars: int) -> str:
    """Pop queued sentences into one text frame of at most ``max_chars`` characters.

    Sentences are joined with a space. A sentence longer than ``max_chars`` is cut
    at its last whitespace inside the limit, and its rest stays first in the queue.
    """
    parts: list[str] = []
    size = 0
    while pending:
        sentence = pending[0]
        added = len(sentence) + (1 if parts else 0)
        if size + added <= max_chars:
            parts.append(pending.pop(0))
            size += added
            continue
        if not parts:
            head, rest = _split_sentence(sentence, max_chars)
            parts.append(head)
            if rest:
                pending[0] = rest
            else:
                pending.pop(0)
        break
    return " ".join(parts)


def _split_sentence(text: str, max_chars: int) -> tuple[str, str]:
    cut = max_chars
    while cut > 0 and not text[cut].isspace():
        cut -= 1
    if cut == 0:
        # No whitespace to cut at: cut at the limit, but never between a character
        # and the combining marks after it, such as Thai vowel and tone marks.
        cut = max_chars
        while cut > 1 and unicodedata.category(text[cut]).startswith("M"):
            cut -= 1
    return text[:cut].rstrip(), text[cut:].lstrip()


class LegWords:
    """Turns one leg's timing spans into the words Pipecat times, spaced as the leg's text is.

    Pipecat joins timed words with single spaces, and Thai writes the words of a
    phrase without them. Spans with no whitespace between them in the leg's text
    therefore join into one word, so the joined words read as the text was
    written: ``สวัสดีครับ ยินดีต้อนรับ``, ``Hello, world.``. A word is held until the
    next span arrives, or the leg ends, since only then is it known whether the
    next span continues it.
    """

    def __init__(self, text: str | None, offset: float):
        """Initialize for a leg.

        Args:
            text: The text the leg speaks, or None if unknown.
            offset: Seconds of the context's audio before this leg.
        """
        self._text = text.rstrip() if text is not None else None
        self._offset = offset
        self._cursor = 0
        # The word being built: where it starts and its last span ends in the
        # text (-1 if not found), its last span's text, and its start time.
        self._word: tuple[int, int, str, float] | None = None

    def add(self, spans: Sequence[object]) -> list[tuple[str, float]]:
        """Take a ``timing`` event's spans and return the words now complete."""
        words: list[tuple[str, float]] = []
        for span in spans:
            parsed = _parse_span(span)
            if parsed is None:
                continue
            text, start = parsed
            position = self._locate(text)
            if self._word is not None:
                if self._continues(position):
                    first, _, _, word_start = self._word
                    self._word = (first, position + len(text), text, word_start)
                    continue
                words.append(self._release(position))
            end = position + len(text) if position >= 0 else -1
            self._word = (position, end, text, self._offset + start)
        return words

    def finish(self) -> list[tuple[str, float]]:
        """Return the leg's last word."""
        if self._word is None:
            return []
        word = self._release(None)
        self._word = None
        return [word]

    def _locate(self, word: str) -> int:
        if self._text is None:
            return -1
        index = self._text.find(word, self._cursor)
        if index >= 0:
            self._cursor = index + len(word)
        return index

    def _continues(self, position: int) -> bool:
        """Whether a span found at ``position`` touches the word being built."""
        if self._word is None or self._text is None or position < 0:
            return False
        first, end, _, _ = self._word
        if first < 0:
            return False
        return not any(c.isspace() for c in self._text[end:position])

    def _release(self, next_position: int | None) -> tuple[str, float]:
        assert self._word is not None
        first, end, text, start = self._word
        if first < 0 or self._text is None:
            return text, start
        # Punctuation after the last span belongs to the word, up to the next space.
        stop = len(self._text) if next_position is None or next_position < 0 else next_position
        while end < stop and not self._text[end].isspace():
            end += 1
        return self._text[first:end], start


def _parse_span(span: object) -> tuple[str, float] | None:
    if not isinstance(span, dict):
        return None
    text = span.get("text")
    start = span.get("start")
    if not isinstance(text, str) or not text or isinstance(start, bool):
        return None
    if not isinstance(start, int | float):
        return None
    return text, float(start)
