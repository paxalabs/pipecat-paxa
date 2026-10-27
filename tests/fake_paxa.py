#
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A local stand-in for the Paxa API: ``POST /v1/tts`` and the ``/v1/tts/live`` socket.

It follows the documented contract: Bearer or ``x-api-key`` auth, RFC 9457 error
bodies, streamed pcm (16-bit mono, no header) at the requested sample rate with
``audio/pcm;rate=<R>;channels=1``, and the live socket's frames and events. A
rate the model doesn't serve answers ``unsupported_format``, as Paxa does. Only
pcm is served: the stand-in answers ``unsupported_format`` for every other
format, which Paxa serves. The live session works like Paxa's: frames are
handled one at a time, a ``cancel`` jumps the queue and drops every frame sent
before it, text auto-flushes on a newline, end punctuation or 400 buffered
characters, a flush with nothing to speak answers ``flushed`` with the last
leg's seq, and a session closes with 4413 once 32 frames wait.

Audio is deterministic: a leg of N characters is 10 ms per character at the
requested rate (N * rate // 100 samples), and every sample is the leg's seq.
"""

import asyncio
import json
import re
import struct
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

API_KEY = "pxa_test_key"
VOICES = {"foithong", "massaman"}
MAX_CHARS = 5000
AUTO_FLUSH_CHARS = 400
MAX_PENDING_FRAMES = 32
FORMATS = ("mp3", "opus", "wav", "pcm", "mulaw", "alaw")
# The rates the schema takes, and the ones each model serves.
RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
DEFAULT_RATE = 24000
MODEL_RATES = {"paxa-tts-flash-v2": set(RATES), "paxa-tts-flash-v1": {DEFAULT_RATE}}
MODELS = set(MODEL_RATES)
SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’」』)）]*\s*$")
TAG = re.compile(r"\[[^\]\n]*\]")
REFUSED_TAG = "[british accent]"


def leg_pcm(seq: int, chars: int, rate: int = DEFAULT_RATE) -> bytes:
    """The pcm the fake speaks at ``rate`` for leg ``seq`` of ``chars`` characters."""
    return struct.pack("<h", seq % 32000) * (chars * rate // 100)


def problem(code: str, status: int) -> web.Response:
    return web.Response(
        status=status,
        body=json.dumps({"title": code, "status": status}).encode(),
        content_type="application/problem+json",
    )


def serves(behavior: "Behavior", model: str, audio_format: str, rate: int) -> bool:
    """Whether the fake serves ``audio_format`` at ``rate`` on ``model``."""
    return (
        audio_format == "pcm" and rate in MODEL_RATES[model] and rate not in behavior.unserved_rates
    )


def spoken(text: str) -> bool:
    return bool(TAG.sub(" ", text).strip())


def held_back(text: str) -> int:
    """Characters of a tag still being typed at the end of ``text``."""
    bracket = text.rfind("[")
    if bracket != -1 and not re.search(r"[\]\n]", text[bracket + 1 :]):
        return len(text) - bracket
    return 0


def word_spans(text: str) -> list[dict[str, Any]]:
    """Word spans as Paxa sends them: the word without punctuation, timed by character."""
    spans = []
    for match in re.finditer(r"[^\s\[\]]+", TAG.sub(lambda m: " " * len(m.group()), text)):
        word = match.group().strip(".,!?;:\"'")
        if not word:
            continue
        spans.append({"text": word, "start": match.start() / 100, "end": match.end() / 100})
    return spans


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


@dataclass
class Behavior:
    """Knobs a test turns to make the fake misbehave or slow down."""

    chunk_bytes: int = 4800
    chunk_delay: float = 0.002
    charge_delay: float = 0.0
    # Leg attempts (1-based, counting refused ones) to answer with rate_limited.
    rate_limited_attempts: set[int] = field(default_factory=set)
    # Legs (by seq) whose synthesis fails after the receipt.
    failing_legs: set[int] = field(default_factory=set)
    # Rates refused with unsupported_format even where the model serves them.
    unserved_rates: set[int] = field(default_factory=set)
    # The content type of POST /v1/tts audio, to test that the wrong one isn't played.
    content_type: str | None = None
    # The live `started` event echoes the format and rate it will send, as Paxa does.
    # started_echo=False is a server from before the echo; the overrides make a session
    # that differs from the request, to test that its audio isn't played.
    started_echo: bool = True
    started_format: str | None = None
    started_rate: int | None = None
    idle_timeout: float | None = None
    # Never answer cancel, to test the client's recovery.
    ignore_cancel: bool = False
    # Accept the upgrade, then send nothing at all.
    mute: bool = False
    # Seconds to wait before answering the WebSocket upgrade.
    upgrade_delay: float = 0.0
    # Seconds each server message waits on the way to the client. With chunk_delay=0,
    # Paxa finishes a leg long before the client has heard it, as the real API does:
    # it streams about 60 times faster than real time.
    delivery_delay: float = 0.0


@dataclass
class Leg:
    seq: int
    text: str
    carry: str | None
    cancelled: bool = False


@dataclass
class SessionLog:
    headers: dict[str, str]
    # The start frame, and the pcm rate the session was served at once it started.
    start: dict[str, Any] | None = None
    rate: int | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    legs: list[Leg] = field(default_factory=list)
    # What the server sent, in order: event types with their seq ("flushed:1"),
    # and "audio" for binary frames.
    sent: list[str] = field(default_factory=list)
    close_code: int | None = None


class FakePaxa:
    """Serves the fake API on 127.0.0.1 and records what clients sent."""

    def __init__(self, behavior: Behavior | None = None):
        self.behavior = behavior or Behavior()
        self.sessions: list[SessionLog] = []
        self.http_requests: list[dict[str, Any]] = []
        self._runner: web.AppRunner | None = None
        self.base_url = ""

    async def __aenter__(self) -> "FakePaxa":
        app = web.Application()
        app.router.add_post("/v1/tts", self._tts)
        app.router.add_get("/v1/tts/live", self._live)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.base_url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._runner:
            await self._runner.cleanup()

    @staticmethod
    def _authorized(request: web.Request) -> bool:
        bearer = request.headers.get("Authorization", "")
        return bearer == f"Bearer {API_KEY}" or request.headers.get("x-api-key") == API_KEY

    # POST /v1/tts

    async def _tts(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return problem("unauthorized", 401)
        try:
            body = await request.json()
        except ValueError:
            return problem("validation", 400)
        self.http_requests.append({"headers": dict(request.headers), "json": body})
        text = body.get("text")
        speed = body.get("speed", 1.0)
        if (
            not isinstance(text, str)
            or not text
            or not isinstance(body.get("voice"), str)
            or not isinstance(body.get("model"), str)
            or body.get("format", "mp3") not in FORMATS
            or body.get("sample_rate", DEFAULT_RATE) not in RATES
            or body.get("language", "auto") not in ("auto", "th", "en", "zh")
            or not isinstance(speed, int | float)
            or not 0.5 <= speed <= 1.5
        ):
            return problem("validation", 400)
        if body["model"] not in MODELS:
            return problem("unknown_model", 400)
        rate = body.get("sample_rate", DEFAULT_RATE)
        if not serves(self.behavior, body["model"], body.get("format", "mp3"), rate):
            return problem("unsupported_format", 400)
        if body["voice"] not in VOICES:
            return problem("unknown_voice", 400)
        if len(text) > MAX_CHARS:
            return problem("text_too_long", 400)
        if REFUSED_TAG in text:
            return problem("tag_invalid", 400)

        response = web.StreamResponse(
            headers={
                "content-type": self.behavior.content_type or f"audio/pcm;rate={rate};channels=1",
                "x-request-id": "req_test",
                "x-credits-charged": "0.5",
            }
        )
        await response.prepare(request)
        data = leg_pcm(1, utf16_len(text), rate)
        size = self.behavior.chunk_bytes
        await response.write(data[:size])
        for at in range(size, len(data), size):
            await asyncio.sleep(self.behavior.chunk_delay)
            await response.write(data[at : at + size])
        await response.write_eof()
        return response

    # WS /v1/tts/live

    async def _live(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return problem("unauthorized", 401)
        if self.behavior.upgrade_delay:
            await asyncio.sleep(self.behavior.upgrade_delay)
        ws = web.WebSocketResponse(autoping=True)
        await ws.prepare(request)
        log = SessionLog(headers=dict(request.headers))
        self.sessions.append(log)
        await _LiveSession(ws, log, self.behavior).run()
        return ws


class _LiveSession:
    def __init__(self, ws: web.WebSocketResponse, log: SessionLog, behavior: Behavior):
        self.ws = ws
        self.log = log
        self.behavior = behavior
        self.phase = "awaiting_start"
        self.timestamps: str | None = None
        self.rate = DEFAULT_RATE
        self.buffer = ""
        self.carry: str | None = None
        self.seq = 0
        self.attempts = 0
        self.last_flushed = 0
        self.armed = True
        self.cancel_requested = False
        self.frame_seq = 0
        self.cancel_through = 0
        self.pending = 0
        self.queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        self.outbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.closed = False
        self.last_activity = asyncio.get_running_loop().time()

    async def run(self) -> None:
        worker = asyncio.create_task(self._work())
        idle = asyncio.create_task(self._idle_watch())
        writer = asyncio.create_task(self._write())
        try:
            async for message in self.ws:
                if message.type != WSMsgType.TEXT:
                    continue
                self.last_activity = asyncio.get_running_loop().time()
                try:
                    frame = json.loads(message.data)
                except ValueError:
                    frame = None
                if isinstance(frame, dict):
                    self.log.frames.append(frame)
                if self.behavior.mute:
                    continue
                if self.pending >= MAX_PENDING_FRAMES:
                    await self.close(4413, "client read too slowly")
                    break
                if (
                    self.phase == "active"
                    and isinstance(frame, dict)
                    and frame.get("type") == "cancel"
                ):
                    if not self.behavior.ignore_cancel:
                        await self.apply_cancel()
                    continue
                self.pending += 1
                self.frame_seq += 1
                await self.queue.put((self.frame_seq, frame))
        finally:
            worker.cancel()
            idle.cancel()
            writer.cancel()
            self.closed = True
            self.log.close_code = self.log.close_code or self.ws.close_code

    async def _idle_watch(self) -> None:
        timeout = self.behavior.idle_timeout
        if timeout is None:
            return
        while not self.closed:
            await asyncio.sleep(timeout / 10)
            if asyncio.get_running_loop().time() - self.last_activity > timeout:
                await self.close(4408, "idle timeout")
                return

    async def _work(self) -> None:
        while True:
            frame_no, frame = await self.queue.get()
            try:
                await self.handle(frame, frame_no)
            finally:
                self.pending -= 1
                if frame_no >= self.cancel_through:
                    await self.settle_cancel(0)
                if frame_no == self.cancel_through:
                    self.buffer = ""

    async def send(self, event: dict[str, Any]) -> bool:
        if self.closed or self.ws.closed:
            return False
        seq = event.get("seq")
        self.log.sent.append(event["type"] if seq is None else f"{event['type']}:{seq}")
        await self._deliver(json.dumps(event, ensure_ascii=False))
        return True

    async def send_audio(self, chunk: bytes) -> bool:
        if self.closed or self.ws.closed:
            return False
        self.log.sent.append("audio")
        await self._deliver(chunk)
        return True

    async def _deliver(self, message: str | bytes) -> None:
        self.last_activity = asyncio.get_running_loop().time()
        if self.behavior.delivery_delay:
            await self.outbox.put(message)
        elif isinstance(message, bytes):
            await self.ws.send_bytes(message)
        else:
            await self.ws.send_str(message)

    async def _write(self) -> None:
        while True:
            message = await self.outbox.get()
            await asyncio.sleep(self.behavior.delivery_delay)
            if self.ws.closed:
                return
            if isinstance(message, bytes):
                await self.ws.send_bytes(message)
            else:
                await self.ws.send_str(message)

    async def close(self, code: int, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        self.log.close_code = code
        await self.ws.close(code=code, message=reason.encode())

    async def apply_cancel(self) -> None:
        self.cancel_requested = True
        self.cancel_through = self.frame_seq
        self.carry = None
        if self.pending == 0:
            self.buffer = ""
            await self.settle_cancel(0)

    async def settle_cancel(self, seq: int) -> None:
        if not self.cancel_requested:
            return
        self.cancel_requested = False
        await self.send({"type": "cancelled", "seq": seq})

    async def handle(self, frame: Any, frame_no: int) -> None:
        if self.closed:
            return
        if not isinstance(frame, dict) or not _valid(frame):
            await self.send({"type": "error", "code": "validation"})
            return
        if frame_no <= self.cancel_through and frame["type"] != "end":
            self.buffer = ""
            return
        if self.phase == "awaiting_start":
            if frame["type"] != "start":
                await self.send({"type": "error", "code": "live_protocol"})
                await self.close(4400, "protocol violation")
                return
            await self.start(frame)
            return
        if self.phase == "ending" or frame["type"] == "start":
            await self.send({"type": "error", "code": "live_protocol"})
            await self.close(4400, "protocol violation")
            return
        if frame["type"] == "text":
            await self.text(frame["text"], frame_no)
        elif frame["type"] == "flush":
            self.armed = True
            await self.run_leg()
        elif frame["type"] == "end":
            self.phase = "ending"
            delivered = await self.run_leg() if self.buffer else True
            if delivered:
                await self.send({"type": "done", "total_chars": 0, "total_credits": 0})
            await self.close(1000, "session complete")

    async def start(self, frame: dict[str, Any]) -> None:
        self.log.start = frame
        if frame["model"] not in MODELS:
            await self.send({"type": "error", "code": "unknown_model"})
            await self.close(4400, "unknown model")
            return
        rate = frame.get("sample_rate", DEFAULT_RATE)
        if not serves(self.behavior, frame["model"], frame.get("format", "mp3"), rate):
            await self.send({"type": "error", "code": "unsupported_format"})
            await self.close(4400, "unsupported format")
            return
        if frame["voice"] not in VOICES:
            await self.send({"type": "error", "code": "unknown_voice"})
            await self.close(4400, "unknown voice")
            return
        self.phase = "active"
        self.rate = self.log.rate = rate
        self.timestamps = frame.get("timestamps")
        started: dict[str, Any] = {
            "type": "started",
            "connection": "conn_test",
            "model": frame["model"],
            "voice": frame["voice"],
        }
        if self.behavior.started_echo:
            started["format"] = self.behavior.started_format or frame.get("format", "mp3")
            started["sample_rate"] = self.behavior.started_rate or rate
        await self.send(started)

    async def text(self, text: str, frame_no: int) -> None:
        if len(text) > MAX_CHARS:
            await self.send({"type": "error", "code": "text_too_long"})
            return
        if REFUSED_TAG in self.buffer + text:
            await self.send({"type": "error", "code": "tag_invalid"})
            return
        self.buffer += text
        if self.armed and (
            "\n" in text or SENTENCE_END.search(self.buffer) or len(self.buffer) >= AUTO_FLUSH_CHARS
        ):
            await self.run_leg(auto=True)

    async def run_leg(self, auto: bool = False) -> bool:
        leg_end = len(self.buffer) - (held_back(self.buffer) if auto else 0)
        if leg_end == 0 or not spoken(self.buffer[:leg_end]):
            if auto or self.phase == "ending":
                return True
            self.armed = True
            await self.send({"type": "flushed", "seq": self.last_flushed})
            return True

        self.attempts += 1
        if self.attempts in self.behavior.rate_limited_attempts:
            self.armed = False
            await self.send({"type": "error", "code": "rate_limited", "retry_after": 1})
            return False

        if self.cancel_requested:
            self.buffer = ""
            await self.settle_cancel(0)
            return False

        self.seq += 1
        seq = self.seq
        text = self.buffer[:leg_end]
        self.buffer = self.buffer[leg_end:]
        leg = Leg(seq=seq, text=text, carry=self.carry)
        self.log.legs.append(leg)
        tags = TAG.findall(text)
        if tags and not self.cancel_requested:
            self.carry = tags[-1]

        if self.behavior.charge_delay:
            await asyncio.sleep(self.behavior.charge_delay)
        chars = utf16_len(text)
        if not await self.send(
            {"type": "charged", "seq": seq, "chars": chars, "credits": 0.01, "total_credits": 0.01}
        ):
            return False
        if self.cancel_requested:
            leg.cancelled = True
            self.last_flushed = seq
            await self.settle_cancel(seq)
            return False

        if seq in self.behavior.failing_legs:
            await self.send({"type": "error", "code": "provider_error", "seq": seq})
            return False

        data = leg_pcm(seq, chars, self.rate)
        size = self.behavior.chunk_bytes
        chunks = [data[at : at + size] for at in range(0, len(data), size)]
        spans = word_spans(text) if self.timestamps == "word" else []
        halves = [spans[: len(spans) // 2], spans[len(spans) // 2 :]]
        for index, chunk in enumerate(chunks):
            if self.cancel_requested:
                leg.cancelled = True
                break
            await asyncio.sleep(self.behavior.chunk_delay)
            if not await self.send_audio(chunk):
                return False
            if spans and index == 0 and halves[0]:
                await self.send({"type": "timing", "seq": seq, "spans": halves[0]})
        else:
            if spans and halves[1]:
                await self.send({"type": "timing", "seq": seq, "spans": halves[1]})

        self.last_flushed = seq
        if self.cancel_requested:
            leg.cancelled = True
            await self.settle_cancel(seq)
            return False
        await self.send({"type": "flushed", "seq": seq})
        return True


def _valid(frame: dict[str, Any]) -> bool:
    kind = frame.get("type")
    if kind == "start":
        speed = frame.get("speed", 1.0)
        return (
            isinstance(frame.get("model"), str)
            and isinstance(frame.get("voice"), str)
            and frame.get("format", "mp3") in FORMATS
            and frame.get("sample_rate", DEFAULT_RATE) in RATES
            and frame.get("timestamps", "word") in ("word", "sentence", "utterance")
            and isinstance(speed, int | float)
            and 0.5 <= speed <= 1.5
            and frame.get("language", "auto") in ("auto", "th", "en", "zh")
        )
    if kind == "text":
        return isinstance(frame.get("text"), str) and len(frame["text"]) >= 1
    return kind in ("flush", "cancel", "end")
