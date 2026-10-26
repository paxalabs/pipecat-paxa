#
# Copyright (c) 2026, Paxa Labs
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What the Paxa API expects from a client: defaults, headers, audio formats and errors."""

import json
import re

from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory

from ._version import __version__

DEFAULT_BASE_URL = "https://api.paxalabs.com"
DEFAULT_MODEL = "paxa-tts-flash-v2"
DEFAULT_VOICE = "foithong"

CLIENT_NAME = f"pipecat-paxa/{__version__}"

# Paxa's pcm is 16-bit little-endian mono with no header, at 24 kHz unless the
# request names another rate its model serves. Every rate costs the same, and a
# rate the model doesn't serve answers unsupported_format.
DEFAULT_SAMPLE_RATE = 24000
MODEL_SAMPLE_RATES: dict[str, frozenset[int]] = {
    "paxa-tts-flash-v2": frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000}),
    "paxa-tts-flash-v1": frozenset({DEFAULT_SAMPLE_RATE}),
}

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


def request_sample_rate(model: object, output_rate: int) -> int:
    """The pcm rate to ask Paxa for: the pipeline's own if the model serves it, else 24 kHz.

    At the pipeline's rate the audio plays as it arrives. At 24 kHz, which every
    model serves, Pipecat resamples it.
    """
    rates = MODEL_SAMPLE_RATES.get(model, frozenset()) if isinstance(model, str) else frozenset()
    return output_rate if output_rate in rates else DEFAULT_SAMPLE_RATE


def is_pcm_at(content_type: str, rate: int) -> bool:
    """Whether a content type is mono pcm at ``rate``, as ``audio/pcm;rate=<R>;channels=1``."""
    mime, *params = content_type.split(";")
    fields: dict[str, str] = {}
    for param in params:
        name, _, value = param.partition("=")
        fields[name.strip().lower()] = value.strip().strip('"')
    return (
        mime.strip().lower() == "audio/pcm"
        and fields.get("rate") == str(rate)
        and fields.get("channels", "1") == "1"
    )


# Paxa error codes are lowercase identifiers. Anything else found where a code
# belongs is dropped, so an error never carries response text into logs.
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

_ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    "validation": ErrorCategory.INVALID_REQUEST,
    "live_protocol": ErrorCategory.INVALID_REQUEST,
    "unknown_model": ErrorCategory.INVALID_REQUEST,
    "unknown_voice": ErrorCategory.INVALID_REQUEST,
    "unsupported_format": ErrorCategory.INVALID_REQUEST,
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
    "unsupported_format": "the model doesn't serve pcm at the sample rate asked for",
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
