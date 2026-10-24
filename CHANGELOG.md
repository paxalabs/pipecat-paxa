# Changelog

## 0.1.0

- `PaxaTTSService`: Paxa text-to-speech over the live WebSocket, with one session per
  conversation, interruption by cancel, and optional word timestamps.
- `PaxaHttpTTSService`: Paxa text-to-speech over a streamed `POST /v1/tts` request.
- Thai replies are spoken phrase by phrase: both services end a sentence at a space
  between Thai words when the installed Pipecat doesn't.
- Tested with Pipecat 1.14.0.
