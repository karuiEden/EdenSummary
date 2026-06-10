# ASR Server Contract

EdenSummary connects to any OpenAI-compatible transcription endpoint.

## Required endpoint

```
POST {WHISPER_API_BASE}/audio/transcriptions
Authorization: Bearer {WHISPER_API_KEY}
```

## Request

Standard OpenAI `audio/transcriptions` multipart form:

| Field | Value |
|---|---|
| `file` | WAV audio (16 kHz, mono, converted by ffmpeg) |
| `model` | value of `WHISPER_MODEL` |
| `response_format` | always `verbose_json` |
| `timestamp_granularities` | `["segment"]` |
| `language` | BCP-47 code if provided by client, omitted otherwise (auto-detect) |

## Response

Must follow the OpenAI `verbose_json` schema:

```json
{
  "language": "ru",
  "segments": [
    { "text": "segment text" },
    ...
  ]
}
```

The `language` field must be a BCP-47 primary subtag (`ru`, `en`, `de`, etc.) or a
Whisper full name (`russian`, `english`). It is used to localize summary section headers.

## Compatible providers

| Provider | `WHISPER_API_BASE` |
|---|---|
| [Groq](https://console.groq.com) | `https://api.groq.com/openai/v1` |
| [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) | `http://whisper-server:8000/v1` (local via `docker compose --profile local-asr up`) |
| Any OpenAI-compatible ASR | Set accordingly |

## Notes

- ASR routing and model selection belong on the ASR server side, not in the application.
  EdenSummary treats ASR as a black-box LLM-compatible service.
- Language detection is the responsibility of the ASR server. The application passes an
  optional hint and uses whatever language the server returns.