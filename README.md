# pipecat-paxa

[Paxa](https://paxalabs.com) text-to-speech for [Pipecat](https://github.com/pipecat-ai/pipecat).
Paxa's voices speak Thai, English and Mandarin, including sentences that mix them.

- `PaxaTTSService` streams speech over Paxa's [live WebSocket](https://paxalabs.com/docs/live-streaming).
  One session serves the whole conversation, and an interruption stops the speech in flight
  without reconnecting.
- `PaxaHttpTTSService` synthesizes each sentence with a streamed `POST /v1/tts` request.

This integration is built and maintained by Paxa Labs, the company behind the Paxa API.

Tested with Pipecat 1.14.0 on Python 3.11, 3.12 and 3.13.

## Installation

```bash
uv add pipecat-paxa
```

## Prerequisites

Create an API key in the [Paxa dashboard](https://paxalabs.com/app/keys) and set it as an
environment variable:

```bash
export PAXA_API_KEY="pxa_..."
```

## Usage

Place the service after your LLM and before the transport output:

```python
from pipecat.pipeline.pipeline import Pipeline

from pipecat_paxa import PaxaTTSService

tts = PaxaTTSService()  # reads PAXA_API_KEY

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

Choose the voice, language and speed with `Settings`:

```python
from pipecat.transcriptions.language import Language

tts = PaxaTTSService(
    settings=PaxaTTSService.Settings(voice="massaman", language=Language.TH, speed=1.1),
)
```

`PaxaHttpTTSService` takes the same settings and goes in the same place.

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `api_key` | `PAXA_API_KEY` | Paxa API key. |
| `base_url` | `PAXA_BASE_URL`, then `https://api.paxalabs.com` | API origin. |
| `word_timestamps` | `False` | `PaxaTTSService` only. Times every spoken word; see [Word timestamps](#word-timestamps). |
| `aiohttp_session` | `None` | `PaxaHttpTTSService` only. A session for the requests; without one, the service creates and closes its own. |
| `settings` | | See [Settings](#settings). |

Both services refuse `TextAggregationMode.TOKEN`: Paxa synthesizes whole sentences.

### Settings

| Setting | Default | Description |
| --- | --- | --- |
| `model` | `"paxa-tts-flash-v2"` | Every voice on Flash v2 speaks Thai, English and Mandarin and reads direction tags. `"paxa-tts-flash-v1"` is the legacy model, with regional Thai accents. |
| `voice` | `"foithong"` | A voice id from [`GET /v1/voices`](https://paxalabs.com/docs/api/voices), such as `foithong` or `massaman`, or a designed voice id (`voice_...`). |
| `language` | `None` | How digits, codes and symbols are read: `Language.TH`, `Language.EN` or `Language.ZH`. `None` lets Paxa decide (`auto`). Words keep the language of their script, so mixed text needs no setting. |
| `speed` | `None` | Speaking rate, from 0.5 to 1.5. `None` speaks at the voice's own rate. |

Change them during a conversation with a `TTSUpdateSettingsFrame`. `PaxaTTSService` opens a
new live session to apply them.

## Thai replies

Thai is written without sentence punctuation; a space between two Thai words ends a phrase
or sentence. Pipecat's sentence aggregation waits for punctuation, so on its own it holds an
unpunctuated Thai reply until the LLM has written all of it. Both services also end a
sentence at a space between Thai words, once the sentence before it is 20 characters long,
so a Thai reply starts speaking after its first phrase. A space after `ฯ` or `ๆ`, or next to
Latin letters or digits, doesn't end a sentence. With a Pipecat version that ends Thai
sentences itself, the services use Pipecat's rule.

## Direction tags

Flash v2 voices read direction tags written in the text, such as `[whispering]` or
`[laughs]`, so you can ask the LLM to use them. A direction lasts until the end of the
reply. See the [text-to-speech docs](https://paxalabs.com/docs/text-to-speech) for the tags.

## Word timestamps

`PaxaTTSService(word_timestamps=True)` times every spoken word, so the conversation context
keeps exactly the words spoken before an interruption, and clients can show words as they
play. Words of a Thai phrase come as one timed unit, so the text reads as it was written.
Paxa bills timed speech at 1.25 times the standard rate.

## Running the example

[`examples/foundational/voice_paxa.py`](examples/foundational/voice_paxa.py) is a voice agent
that uses OpenAI for speech recognition and replies and Paxa for speech. It listens for
Thai and greets in Thai; change `LANGUAGE` and `GREETING` in the file for another language.

```bash
cp .env.example .env  # then add OPENAI_API_KEY and PAXA_API_KEY
uv sync --extra examples
uv run examples/foundational/voice_paxa.py
```

Open http://localhost:7860 in your browser and talk to the bot.

## Notes

- **Audio**: Both services produce 24 kHz 16-bit mono audio, which Pipecat resamples when
  the pipeline's output rate differs, as on 8 kHz phone calls.
- **Plan limits**: An open live session holds one of your plan's concurrent speech
  connections for as long as the service runs, and each synthesized segment counts as a
  request toward the plan's per-minute limit. Speech refused for a limit is reported as an
  error with `ErrorCategory.RATE_LIMIT`, and the next reply is spoken normally.
- **Errors**: Errors carry Paxa's error code, such as `unknown_voice` or
  `insufficient_credits`. The [error reference](https://paxalabs.com/docs/errors) explains
  each one.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check && uv run ruff format --check && uv run pyright
```

The tests run against a local stand-in for the Paxa API, so they need no API key.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
