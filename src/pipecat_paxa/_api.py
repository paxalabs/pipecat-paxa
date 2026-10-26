#
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What the Paxa API expects from a client: defaults, headers, audio framing and errors."""

import json
import re
import struct

from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory

from ._version import __version__

DEFAULT_BASE_URL = "https://api.paxalabs.com"
DEFAULT_MODEL = "paxa-tts-flash-v2"
DEFAULT_VOICE = "foithong"

CLIENT_NAME = f"pipecat-paxa/{__version__}"

# Paxa serves WAV as 16-bit mono PCM at 24 kHz on every model.
SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2

LANGUAGES = frozenset({"auto", "th", "en", "zh"})
MIN_SPEED = 0.5
MAX_SPEED = 1.5


def request_headers(api_key: str) -> dict[str, str]:
    """Headers for every Paxa request, the WebSocket upgrade included.

    ``x-paxa-client`` tells Paxa which integration sent the request.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": CLIENT_NAME,
        "x-paxa-client": CLIENT_NAME,
    }


def normalize_base_url(base_url: str) -> str:
    """Return the API origin, also accepting an OpenAI-style base URL ending in ``/v1``."""
    origin = base_url.strip().rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]
    if not origin.startswith(("https://", "http://")):
        raise ValueError("base_url must start with https:// or http://")
    return origin


def live_url(base_url: str) -> str:
    """Build the ``/v1/tts/live`` WebSocket URL from a normalized API origin."""
    scheme, rest = base_url.split("://", 1)
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/v1/tts/live"


def paxa_language(language: Language | str) -> str | None:
    """Map a Pipecat language to Paxa's ``language`` field, or None if Paxa has no match.

    Paxa reads Thai, English and Mandarin. Regional variants map to their base
    language: ``th-TH`` to ``th``, ``en-GB`` to ``en``, ``zh-CN`` and ``cmn`` to ``zh``.
    """
    base = str(language).strip().lower().replace("_", "-").split("-")[0]
    if base == "cmn":
        return "zh"
    return base if base in LANGUAGES else None


def check_speed(speed: float | None) -> None:
    """Raise ValueError for a speed outside Paxa's 0.5 to 1.5 range."""
    if speed is not None and not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")


def check_language(language: object) -> None:
    """Raise ValueError for a language Paxa does not read."""
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"Paxa reads auto, th, en or zh, got {language!r}")


class WavHeaderError(Exception):
    """Paxa sent audio in a format the service does not play."""


_WAV_HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")
# RIFF, WAVE, a 16-byte fmt chunk of PCM (1), mono, 24 kHz, 16 bits, then data.
_EXPECTED_WAV = (b"RIFF", b"WAVE", b"fmt ", 16, 1, 1, SAMPLE_RATE, 16, b"data")


class WavHeaderStripper:
    """Strips the 44-byte header from a WAV stream, leaving its 16-bit PCM.

    Paxa also serves headerless pcm. The services ask for WAV so they can check
    the format before any audio plays: 24 kHz 16-bit mono PCM behind a canonical
    44-byte header, with 0xFFFFFFFF sizes when streamed. A header that describes
    any other audio raises WavHeaderError instead of being played.
    """

    def __init__(self) -> None:
        """Initialize the stripper, expecting a header first."""
        self._header = bytearray()
        self._in_header = True

    def reset(self) -> None:
        """Expect a new header, as every leg of a live session is a WAV stream of its own."""
        self._header.clear()
        self._in_header = True

    def feed(self, data: bytes) -> bytes:
        """Return the PCM in ``data``, holding back header bytes until all 44 arrive.

        Args:
            data: The next bytes of the WAV stream.

        Returns:
            The PCM bytes that follow the header, possibly empty.

        Raises:
            WavHeaderError: If the header describes anything but 24 kHz 16-bit mono PCM.
        """
        if not self._in_header:
            return data
        missing = _WAV_HEADER.size - len(self._header)
        self._header += data[:missing]
        if len(self._header) < _WAV_HEADER.size:
            return b""
        riff, _, wave, fmt, fmt_size, encoding, channels, rate, _, _, bits, data_id, _ = (
            _WAV_HEADER.unpack(self._header)
        )
        if (riff, wave, fmt, fmt_size, encoding, channels, rate, bits, data_id) != _EXPECTED_WAV:
            raise WavHeaderError("Paxa TTS returned audio in an unexpected format")
        self._in_header = False
        return data[missing:]


# Paxa error codes are lowercase identifiers. Anything else found where a code
# belongs is dropped, so an error never carries response text into logs.
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

_ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    "validation": ErrorCategory.INVALID_REQUEST,
    "live_protocol": ErrorCategory.INVALID_REQUEST,
    "unknown_model": ErrorCategory.INVALID_REQUEST,
    "unknown_voice": ErrorCategory.INVALID_REQUEST,
    # The text, not the service, was refused, so the next sentence may speak.
    "text_too_long": ErrorCategory.APPLICATION,
    "tag_invalid": ErrorCategory.APPLICATION,
    "unspeakable_text": ErrorCategory.APPLICATION,
    "content_blocked": ErrorCategory.APPLICATION,
    "unauthorized": ErrorCategory.AUTHENTICATION,
    "insufficient_credits": ErrorCategory.QUOTA,
    "key_limit": ErrorCategory.QUOTA,
    "rate_limited": ErrorCategory.RATE_LIMIT,
    "concurrency_limited": ErrorCategory.RATE_LIMIT,
    "internal": ErrorCategory.SERVER,
    "provider_error": ErrorCategory.SERVER,
    "provider_unavailable": ErrorCategory.SERVER,
}

_ERROR_HINTS: dict[str, str] = {
    "unauthorized": "check PAXA_API_KEY",
    "insufficient_credits": "top up in the Paxa dashboard",
    "key_limit": "the API key reached its spending cap",
    "unknown_voice": "GET /v1/voices lists the voice ids",
    "unknown_model": "GET /v1/models lists the model ids",
    "rate_limited": "the plan's requests per minute are used up",
    "concurrency_limited": "the plan's concurrent connections are in use",
    "tag_invalid": "the text holds a tag the voice does not read",
}


def error_code(value: object) -> str | None:
    """Return ``value`` if it looks like a Paxa error code, else None."""
    if isinstance(value, str) and _ERROR_CODE_RE.fullmatch(value):
        return value
    return None


def problem_code(body: bytes | bytearray) -> str | None:
    """Read the error code from an RFC 9457 problem body, ``{"title": "<code>", ...}``."""
    try:
        problem = json.loads(body)
    except ValueError:
        return None
    return error_code(problem.get("title")) if isinstance(problem, dict) else None


def error_category(code: str | None, status: int | None = None) -> ErrorCategory | None:
    """The Pipecat error category for a Paxa error code, falling back to the HTTP status."""
    if code in _ERROR_CATEGORIES:
        return _ERROR_CATEGORIES[code]
    if status == 429:
        return ErrorCategory.RATE_LIMIT
    if status is not None and status >= 500:
        return ErrorCategory.SERVER
    return None


def error_message(code: str | None, *, status: int | None = None) -> str:
    """Describe a Paxa error by its code, never by response text."""
    message = "Paxa TTS error"
    if status is not None:
        message += f" (HTTP {status})"
    if code:
        message += f": {code}"
        if hint := _ERROR_HINTS.get(code):
            message += f" ({hint})"
    return message
