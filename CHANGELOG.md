# Changelog

## 0.1.0

- `PaxaTTSService`: Paxa text-to-speech over the live WebSocket, with one session per
  conversation, interruption by cancel, and optional word timestamps.
- `PaxaHttpTTSService`: Paxa text-to-speech over a streamed `POST /v1/tts` request.
- Audio comes as raw pcm at the pipeline's output rate whenever the model serves it,
  so it isn't resampled: Flash v2 serves 8, 16, 22.05, 24, 32, 44.1 and 48 kHz. Other
  rates, and Flash v1, get 24 kHz, which Pipecat resamples.
- Thai replies are spoken phrase by phrase: both services end a sentence at a space
  between Thai words when the installed Pipecat doesn't.
- Tested with Pipecat 1.14.0.
