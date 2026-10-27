#
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Paxa text-to-speech services for Pipecat.

:class:`PaxaTTSService` streams speech over Paxa's live WebSocket
(``/v1/tts/live``), and :class:`PaxaHttpTTSService` streams each sentence from
``POST /v1/tts``. Both speak Thai, English and Mandarin, mixed freely, and
default to ``paxa-tts-flash-v2`` with the voice ``foithong``.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory, classify_http_status_code
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given
from websockets.exceptions import InvalidStatus
from websockets.protocol import State

from ._api import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE,
    check_language,
    check_speed,
    error_category,
    error_code,
    error_message,
    is_pcm_at,
    is_started_pcm_at,
    live_url,
    normalize_base_url,
    paxa_language,
    problem_code,
    request_headers,
    request_sample_rate,
)
from ._live import LegWords, LiveSession
from ._thai import use_thai_sentence_ends

# How long Paxa may take to confirm a new session with ``started``.
_SESSION_START_TIMEOUT = 10.0
# How long Paxa may stay silent while it owes the session an event before the
# connection is treated as lost.
_RESPONSE_TIMEOUT = 20.0
# Paxa closes a session after 300 seconds without a client frame. An idle session
# sends an empty flush, which costs nothing, well inside that.
_KEEPALIVE_INTERVAL = 120.0
# Settings the live session fixes at its start.
_SESSION_SETTINGS = frozenset({"model", "voice", "language", "speed"})
# Errors after which Paxa closes the connection; they are reported even for
# speech that was cancelled.
_CLOSING_ERRORS = frozenset(
    {
        "unauthorized",
        "insufficient_credits",
        "key_limit",
        "unknown_model",
        "unknown_voice",
        "unsupported_format",
        "live_protocol",
        "internal",
    }
)
# Errors with which Paxa refuses one text frame, leaving the session as it was.
_REJECTED_FRAME_ERRORS = frozenset({"text_too_long", "tag_invalid", "validation"})
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=30)


@dataclass
class PaxaTTSSettings(TTSSettings):
    """Settings for :class:`PaxaTTSService` and :class:`PaxaHttpTTSService`.

    Parameters:
        speed: Speaking rate, from 0.5 to 1.5. ``None`` speaks at the voice's own
            rate (1.0).
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("PAXA_API_KEY")
    if not key:
        raise ValueError("A Paxa API key is required: pass api_key or set PAXA_API_KEY")
    return key


def _resolve_base_url(base_url: str | None) -> str:
    return normalize_base_url(base_url or os.environ.get("PAXA_BASE_URL") or DEFAULT_BASE_URL)


def _initial_settings(settings: PaxaTTSSettings | None) -> PaxaTTSSettings:
    initial = PaxaTTSSettings(model=DEFAULT_MODEL, voice=DEFAULT_VOICE, language=None, speed=None)
    if settings is not None:
        initial.apply_update(settings)
    return initial


def _take_sentence_mode(kwargs: dict[str, Any]) -> None:
    """Refuse token streaming: Paxa synthesizes each piece of text it is sent as one leg."""
    mode = kwargs.pop("text_aggregation_mode", None)
    aggregate_sentences = kwargs.pop("aggregate_sentences", None)
    if mode == TextAggregationMode.TOKEN or aggregate_sentences is False:
        raise ValueError(
            "Paxa services speak whole sentences, so TextAggregationMode.TOKEN is not supported"
        )


def _check_settings(settings: PaxaTTSSettings) -> None:
    for name in ("model", "voice"):
        value = getattr(settings, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    check_language(settings.language)
    speed = settings.speed
    if speed is not None and (isinstance(speed, bool) or not isinstance(speed, int | float)):
        raise ValueError(f"speed must be a number, got {speed!r}")
    check_speed(speed)


def _drop_unusable(delta: TTSSettings) -> list[str]:
    """Remove the fields of a settings update Paxa can't take, and name them."""
    dropped: list[str] = []
    language = delta.language
    if is_given(language) and language is not None:
        resolved = "auto" if str(language).strip().lower() == "auto" else paxa_language(language)
        if resolved is None:
            dropped.append(f"language {str(language)!r}")
            delta.language = NOT_GIVEN
        else:
            delta.language = resolved
    for name in ("model", "voice"):
        value = getattr(delta, name)
        if is_given(value) and (not isinstance(value, str) or not value.strip()):
            dropped.append(name)
            setattr(delta, name, NOT_GIVEN)
    if isinstance(delta, PaxaTTSSettings) and is_given(delta.speed) and delta.speed is not None:
        try:
            check_speed(delta.speed)
        except (TypeError, ValueError):
            dropped.append(f"speed {delta.speed!r}")
            delta.speed = NOT_GIVEN
    return dropped


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


class PaxaTTSService(WebsocketTTSService):
    """Paxa text-to-speech over the live WebSocket.

    One live session serves the whole conversation. Each sentence goes out as
    soon as it is complete, and an interruption cancels the speech in flight
    without reconnecting. Paxa sends 16-bit mono pcm at the pipeline's output
    rate when the model serves that rate, so it plays as it arrives. Otherwise
    it sends 24 kHz, which is resampled to the pipeline's rate.

    Paxa bills speech with word timestamps at 1.25 times the standard rate, so
    they are off unless ``word_timestamps`` is set. Without them, each sentence's
    text is pushed with its audio.

    Event handlers available:

    - on_connected: Called when the live session is ready for text.
    - on_disconnected: Called when the connection closes.
    - on_connection_error: Called when the connection fails.

    Example::

        tts = PaxaTTSService(
            settings=PaxaTTSService.Settings(voice="massaman", language=Language.TH),
        )
    """

    Settings = PaxaTTSSettings
    _settings: PaxaTTSSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        word_timestamps: bool = False,
        settings: PaxaTTSSettings | None = None,
        **kwargs,
    ):
        """Initialize the Paxa live TTS service.

        Args:
            api_key: Paxa API key. Defaults to the ``PAXA_API_KEY`` environment variable.
            base_url: API origin. Defaults to the ``PAXA_BASE_URL`` environment
                variable, then ``https://api.paxalabs.com``.
            word_timestamps: Whether to time every spoken word, for word-level
                text frames. Paxa bills timed speech at 1.25 times the standard rate.
            settings: Runtime-updatable settings. Unset fields keep their defaults:
                model ``paxa-tts-flash-v2``, voice ``foithong``.
            **kwargs: Additional arguments passed to the parent service.

        Raises:
            ValueError: If no API key is set, a setting is outside what Paxa
                takes, or ``TextAggregationMode.TOKEN`` is requested.
        """
        api_key = _resolve_api_key(api_key)
        _take_sentence_mode(kwargs)
        super().__init__(
            push_text_frames=not word_timestamps,
            push_stop_frames=True,
            push_start_frame=True,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            settings=_initial_settings(settings),
            **kwargs,
        )
        _check_settings(self._settings)

        self._api_key = api_key
        self._base_url = _resolve_base_url(base_url)
        self._word_timestamps = word_timestamps

        self._live = LiveSession()
        # The pcm rate the open session asked for in its start frame.
        self._session_rate = DEFAULT_SAMPLE_RATE
        self._odd_byte = b""
        self._leg_words: LegWords | None = None
        self._connection_id: str | None = None
        self._session_started = asyncio.Event()
        self._wake = asyncio.Event()
        self._last_activity = 0.0
        self._receive_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the service reports TTFB and usage metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a language to Paxa's ``language`` field.

        Args:
            language: The language to convert.

        Returns:
            ``th``, ``en`` or ``zh``, or None for a language Paxa doesn't read.
        """
        return paxa_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and open the live session.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await use_thai_sentence_ends(self)
        await self._connect()

    async def flush_audio(self, context_id: str | None = None):
        """Close a context once its text is spoken, as the LLM turn ended.

        Args:
            context_id: The context to close. Defaults to the active context.
        """
        context_id = context_id or self.get_active_audio_context_id()
        if context_id:
            self._live.request_flush(context_id)
            self._wake.set()

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the speech of an interrupted context.

        Paxa stops the leg in flight and drops the text it holds; the session
        stays open for the next turn.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        if self._live.interrupt(context_id):
            await self._send({"type": "cancel"})
        await super().on_audio_context_interrupted(context_id)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Queue a sentence for the live session; its audio arrives on the receive task.

        Args:
            text: The text to speak.
            context_id: The audio context the text belongs to.

        Yields:
            None, as the audio is delivered by the receive task.
        """
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()
        self._live.add_text(context_id, text)
        self._wake.set()
        await self.start_tts_usage_metrics(text)
        yield None

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, opening a new session for session-wide settings.

        Args:
            delta: A :class:`TTSSettings` (or ``PaxaTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        dropped = _drop_unusable(delta)
        if dropped:
            await self.push_error(
                error_msg=f"Paxa TTS ignored settings it can't use: {', '.join(dropped)}",
                category=ErrorCategory.APPLICATION,
            )
        changed = await super()._update_settings(delta)
        if changed.keys() & _SESSION_SETTINGS:
            # The live session fixes model, voice, speed and language at its start.
            await self._disconnect()
            await self._connect()
        self._warn_unhandled_updated_settings(changed.keys() - _SESSION_SETTINGS)
        return changed

    # Connection.

    async def _connect(self):
        await super()._connect()
        # A receive or send task ends when its connection is given up on, and a
        # new connection needs new ones.
        if self._receive_task and self._receive_task.done():
            self._receive_task = None
        if self._send_task and self._send_task.done():
            self._send_task = None
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
        if self._websocket and not self._send_task:
            self._send_task = self.create_task(self._send_task_handler())
        self._create_keepalive_task(_KEEPALIVE_INTERVAL)

    async def _disconnect(self):
        await super()._disconnect()
        await self._cancel_keepalive_task()
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug(f"{self}: connecting to Paxa")
            self._reset_session()
            # Pipecat sets the output rate in setup(), before the service connects.
            start = self._start_frame()
            websocket = await self._websocket_connect(
                live_url(self._base_url), additional_headers=request_headers(self._api_key)
            )
            self._websocket = websocket
            self._session_rate = start["sample_rate"]
            await websocket.send(json.dumps(start))
            self._last_activity = time.monotonic()
        except Exception as e:
            message, category, exception = _connection_failure(e)
            await self.push_error(error_msg=message, exception=exception, category=category)
            self._websocket = None
            await self._call_event_handler("on_connection_error", message)

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Paxa")
                await self._websocket.close()
        except Exception as e:
            logger.debug(f"{self}: error closing the Paxa connection: {e}")
        finally:
            self._websocket = None
            self._reset_session()
            await self._call_event_handler("on_disconnected")

    def _reset_session(self):
        self._live.reset()
        self._session_started.clear()
        self._odd_byte = b""
        self._leg_words = None
        self._connection_id = None

    def _start_frame(self) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "type": "start",
            "model": self._settings.model,
            "voice": self._settings.voice,
            "format": "pcm",
            "sample_rate": request_sample_rate(self._settings.model, self.sample_rate),
        }
        if self._word_timestamps:
            frame["timestamps"] = "word"
        if self._settings.speed is not None:
            frame["speed"] = self._settings.speed
        if self._settings.language is not None:
            frame["language"] = self._settings.language
        return frame

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise ConnectionError(f"{self}: not connected to Paxa")

    def _is_keepalive_ready(self) -> bool:
        return super()._is_keepalive_ready() and self._live.keepalive_ready()

    async def _send_keepalive(self):
        # An empty flush is answered at once and costs nothing.
        self._live.keepalive_sent()
        await self._send({"type": "flush"})

    # Sending.

    async def _send(self, frame: Mapping[str, Any]) -> bool:
        websocket = self._websocket
        if websocket is None or websocket.state is not State.OPEN:
            return False
        try:
            await websocket.send(json.dumps(frame, ensure_ascii=False))
        except Exception as e:
            # The receive task reports the lost connection and reconnects.
            logger.debug(f"{self}: send to Paxa failed: {e}")
            return False
        self._last_activity = time.monotonic()
        return True

    async def _send_task_handler(self):
        while True:
            now = time.monotonic()
            timeout = self._live.wait_time(now)
            if self._live.awaiting_server:
                silence = now - self._last_activity
                if silence >= _RESPONSE_TIMEOUT:
                    await self._drop_unresponsive_connection(silence)
                    continue
                remaining = _RESPONSE_TIMEOUT - silence
                timeout = remaining if timeout is None else min(timeout, remaining)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                pass
            self._wake.clear()
            while (frame := self._live.next_frame(time.monotonic())) is not None:
                if not await self._send(frame):
                    break

    async def _drop_unresponsive_connection(self, silence: float):
        logger.warning(f"{self}: no response from Paxa for {silence:.0f} s, reconnecting")
        self._last_activity = time.monotonic()
        websocket = self._websocket
        if websocket is not None:
            # A close with an error code makes the receive task reconnect.
            await websocket.close(code=4000, reason="no response")

    # Receiving.

    async def _receive_messages(self):
        messages = aiter(self._get_websocket())
        if not await self._receive_until_confirmed(
            messages, self._session_started, self._handle_message, _SESSION_START_TIMEOUT
        ):
            return
        await self._call_event_handler("on_connected")
        async for message in messages:
            await self._handle_message(message)

    async def _handle_message(self, message: str | bytes):
        self._last_activity = time.monotonic()
        if isinstance(message, bytes):
            await self._handle_audio(message)
            return
        try:
            event = json.loads(message)
        except ValueError:
            event = None
        if not isinstance(event, dict):
            logger.warning(f"{self}: unreadable event from Paxa")
            return
        kind = event.get("type")
        if kind == "started":
            if not is_started_pcm_at(event, self._session_rate):
                # Audio at another rate would play at the wrong speed, so none is played, as
                # on HTTP. Asking again would get the same session.
                await self.push_error(
                    error_msg=f"Paxa started the live session as {event.get('format')} at "
                    f"{event.get('sample_rate')} Hz, not the pcm at {self._session_rate} Hz "
                    "it was asked for",
                    category=ErrorCategory.SERVER,
                    force_treat_as_permanent=True,
                )
                if self._websocket is not None:
                    await self._websocket.close()
                return
            connection = event.get("connection")
            self._connection_id = connection if isinstance(connection, str) else None
            # Paxa's connection id identifies the session in its support logs.
            logger.debug(f"{self}: Paxa live session {self._connection_id} started")
            self._live.started()
            self._session_started.set()
            self._wake.set()
        elif kind == "charged":
            self._on_charged(event)
        elif kind == "timing":
            await self._on_timing(event)
        elif kind == "flushed":
            await self._on_flushed(event)
        elif kind == "cancelled":
            self._live.on_cancelled()
            self._wake.set()
        elif kind == "error":
            await self._on_error(event)

    async def _handle_audio(self, data: bytes):
        leg = self._live.leg()
        if leg is None:
            # Audio of cancelled speech, which arrives until Paxa answers the cancel.
            return
        pcm = self._odd_byte + data
        if len(pcm) % 2:
            self._odd_byte, pcm = pcm[-1:], pcm[:-1]
        else:
            self._odd_byte = b""
        if not pcm:
            return
        leg.context.audio_seconds += len(pcm) / (2 * self._session_rate)
        audio = pcm
        if self._session_rate != self.sample_rate:
            audio = await self._resampler.resample(pcm, self._session_rate, self.sample_rate)
        if audio:
            context_id = leg.context.context_id
            await self.append_to_audio_context(
                context_id, TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
            )

    def _on_charged(self, event: dict[str, Any]):
        seq, chars = event.get("seq"), event.get("chars")
        if not _is_int(seq) or not _is_int(chars):
            return
        leg = self._live.on_charged(seq, chars)
        if leg is None:
            return
        # Every leg's audio is a stream of its own, of whole 16-bit samples.
        self._odd_byte = b""
        self._leg_words = LegWords(leg.text, leg.offset) if self._word_timestamps else None

    async def _on_timing(self, event: dict[str, Any]):
        seq, spans = event.get("seq"), event.get("spans")
        leg = self._live.leg(seq) if _is_int(seq) else None
        if leg is None or self._leg_words is None or not isinstance(spans, list):
            return
        words = self._leg_words.add(spans)
        if words:
            await self.add_word_timestamps(words, leg.context.context_id)

    async def _on_flushed(self, event: dict[str, Any]):
        seq = event.get("seq")
        if not _is_int(seq):
            return
        outcome = self._live.on_flushed(seq)
        if outcome.leg is not None and self._leg_words is not None:
            words = self._leg_words.finish()
            self._leg_words = None
            if words:
                await self.add_word_timestamps(words, outcome.leg.context.context_id)
        if outcome.finished is not None:
            await self._finish_context(outcome.finished.context_id, outcome.finished.tagged)
        self._wake.set()

    async def _on_error(self, event: dict[str, Any]):
        code = error_code(event.get("code"))
        seq = event.get("seq")
        if self._live.cancelling and code not in _CLOSING_ERRORS:
            logger.debug(f"{self}: Paxa error for cancelled speech: {code}")
            return
        if code in ("rate_limited", "provider_unavailable") and not _is_int(seq):
            # Paxa keeps the text unspoken; the cancel clears it, so a later turn
            # doesn't start with it.
            await self._abandon_active()
        elif _is_int(seq):
            finished = self._live.on_leg_failed(seq)
            if finished is not None:
                await self._finish_context(finished.context_id, finished.tagged)
        elif code in _REJECTED_FRAME_ERRORS:
            self._live.on_frame_rejected()
        await self.push_error(error_msg=error_message(code), category=error_category(code))
        self._wake.set()

    async def _abandon_active(self):
        context = self._live.abandon_active()
        await self._send({"type": "cancel"})
        await self._send({"type": "flush"})
        if context is not None:
            await self._close_audio_context(context.context_id)

    async def _finish_context(self, context_id: str, tagged: bool):
        await self._close_audio_context(context_id)
        if tagged:
            # Paxa carries a tag's delivery into the next leg and keeps a tag with
            # no words after it, so a cancel starts the next context afresh.
            self._live.begin_cancel()
            await self._send({"type": "cancel"})

    async def _close_audio_context(self, context_id: str):
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)


class PaxaHttpTTSService(TTSService):
    """Paxa text-to-speech over ``POST /v1/tts``.

    Each sentence is one streamed request. Paxa sends 16-bit mono pcm at the
    pipeline's output rate when the model serves that rate, so it plays as it
    arrives. Otherwise it sends 24 kHz, which is resampled to the pipeline's rate.

    Example::

        tts = PaxaHttpTTSService(settings=PaxaHttpTTSService.Settings(voice="massaman"))
    """

    Settings = PaxaTTSSettings
    _settings: PaxaTTSSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: PaxaTTSSettings | None = None,
        **kwargs,
    ):
        """Initialize the Paxa HTTP TTS service.

        Args:
            api_key: Paxa API key. Defaults to the ``PAXA_API_KEY`` environment variable.
            base_url: API origin. Defaults to the ``PAXA_BASE_URL`` environment
                variable, then ``https://api.paxalabs.com``.
            aiohttp_session: Session for the requests. When None, the service
                creates one and closes it at cleanup; a session you pass is yours
                to close.
            settings: Runtime-updatable settings. Unset fields keep their defaults:
                model ``paxa-tts-flash-v2``, voice ``foithong``.
            **kwargs: Additional arguments passed to the parent service.

        Raises:
            ValueError: If no API key is set, a setting is outside what Paxa
                takes, or ``TextAggregationMode.TOKEN`` is requested.
        """
        api_key = _resolve_api_key(api_key)
        _take_sentence_mode(kwargs)
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            settings=_initial_settings(settings),
            **kwargs,
        )
        _check_settings(self._settings)

        self._api_key = api_key
        self._base_url = _resolve_base_url(base_url)
        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the service reports TTFB, processing and usage metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a language to Paxa's ``language`` field.

        Args:
            language: The language to convert.

        Returns:
            ``th``, ``en`` or ``zh``, or None for a language Paxa doesn't read.
        """
        return paxa_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await use_thai_sentence_ends(self)

    async def cleanup(self):
        """Close the HTTP session the service created."""
        await super().cleanup()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta; the next request uses it.

        Args:
            delta: A :class:`TTSSettings` (or ``PaxaHttpTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        dropped = _drop_unusable(delta)
        if dropped:
            await self.push_error(
                error_msg=f"Paxa TTS ignored settings it can't use: {', '.join(dropped)}",
                category=ErrorCategory.APPLICATION,
            )
        changed = await super()._update_settings(delta)
        self._warn_unhandled_updated_settings(changed.keys() - _SESSION_SETTINGS)
        return changed

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize a sentence with a streamed ``POST /v1/tts`` request.

        Args:
            text: The text to speak.
            context_id: The audio context the text belongs to.

        Yields:
            Audio frames, or an ErrorFrame if the request fails.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
        rate = request_sample_rate(self._settings.model, self.sample_rate)
        try:
            async with self._session.post(
                f"{self._base_url}/v1/tts",
                json=self._request_body(text, rate),
                headers=request_headers(self._api_key),
                timeout=_HTTP_TIMEOUT,
            ) as response:
                if response.status != 200:
                    code = problem_code(await response.read())
                    yield ErrorFrame(
                        error=error_message(code, status=response.status),
                        category=error_category(code, response.status)
                        or classify_http_status_code(response.status),
                    )
                    return
                # Paxa's request id identifies the request in its support logs.
                logger.debug(f"{self}: Paxa request {response.headers.get('x-request-id')}")
                if not is_pcm_at(response.headers.get("Content-Type", ""), rate):
                    yield ErrorFrame(
                        error=f"Paxa TTS returned audio that is not mono pcm at the {rate} Hz "
                        "the request asked for",
                        category=ErrorCategory.SERVER,
                    )
                    return
                await self.start_tts_usage_metrics(text)
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_any(), in_sample_rate=rate, context_id=context_id
                ):
                    yield frame
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError) as e:
            yield ErrorFrame(
                error=f"Paxa TTS request failed: {type(e).__name__}",
                exception=e,
                category=ErrorCategory.CONNECTIVITY,
            )
        except aiohttp.ClientError as e:
            # Other client errors keep the request, headers included, so the
            # exception itself isn't passed on.
            yield ErrorFrame(
                error=f"Paxa TTS request failed: {type(e).__name__}",
                category=ErrorCategory.UNKNOWN,
            )

    def _request_body(self, text: str, sample_rate: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "text": text,
            "voice": self._settings.voice,
            "model": self._settings.model,
            "format": "pcm",
            "sample_rate": sample_rate,
            "stream": True,
        }
        if self._settings.speed is not None:
            body["speed"] = self._settings.speed
        if self._settings.language is not None:
            body["language"] = self._settings.language
        return body


def _connection_failure(e: Exception) -> tuple[str, ErrorCategory, Exception | None]:
    """Describe a failed connection without echoing anything the request carried."""
    if isinstance(e, InvalidStatus):
        status = e.response.status_code
        code = problem_code(e.response.body or b"")
        category = error_category(code, status) or classify_http_status_code(status)
        return error_message(code, status=status), category, None
    return (
        f"Paxa TTS connection failed: {type(e).__name__}",
        ErrorCategory.CONNECTIVITY,
        e,
    )
