# EdenSummary

Self-hostable pipeline for automatic transcription and summarization of meeting recordings.

Accepts an audio or video file → transcribes via Whisper-compatible ASR → summarizes via LLM (map-reduce) → sends the result by email and exposes it through the API.

## Architecture

```
Client
  │  POST /v1/jobs (audio file)
  ▼
FastAPI ──► Celery/Redis ──► Worker
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 ffmpeg       ASR        MinIO/S3
               (→ WAV)    (Whisper)   (store files)
                                │
                            chunks
                                │
                            LLM (map-reduce)
                                │
                          Summary ──► Email
                                │
                          MinIO/S3 (store result)
```

## Quick Start

**Prerequisites:** Docker, Docker Compose, [MinIO AIStor license](https://min.io) (free tier available)

```bash
git clone https://github.com/karuiEden/EdenSummary.git
cd EdenSummary

# Get a free AIStor license and place it at:
# ./minio/minio.license

cp .env.example .env
# Fill in the required values (see Configuration section)

docker compose up -d
```

The stack will:
1. Run database migrations automatically
2. Create the MinIO bucket
3. Start the API on `http://localhost:8000`
4. Start the Celery worker and Flower dashboard on `http://localhost:5555`

> **Using external S3?** Set `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` to your
> provider (AWS S3, Cloudflare R2, etc.) and remove the `minio`/`createbuckets` services from compose.

## Configuration

All configuration is via environment variables (`.env` file).

| Variable | Required | Default | Description |
|---|---|---|---|
| `X_API_KEY` | ✅ | — | API authentication key |
| **ASR** | | | |
| `WHISPER_API_BASE` | ✅ | — | ASR server base URL (e.g. `https://api.groq.com/openai/v1`) |
| `WHISPER_API_KEY` | ✅ | — | ASR server API key |
| `WHISPER_MODEL` | | `large-v3` | Whisper model name |
| `WHISPER_LANGUAGE` | | _(auto)_ | Force ASR language (BCP-47, e.g. `ru`). Leave empty for auto-detection |
| `MAX_CHARS` | | `4000` | Max characters per chunk sent to LLM |
| **LLM** | | | |
| `LLM_MODEL` | ✅ | — | litellm model string (e.g. `groq/llama-3.3-70b-versatile`, `openai/gpt-4o`) |
| `LLM_API_KEY` | ✅ | — | LLM provider API key |
| `LLM_API_BASE` | | _(provider default)_ | Override API base URL (for self-hosted models) |
| `LLM_MAX_RETRIES` | | — | litellm retry count on provider errors |
| `LLM_TEMPERATURE` | | — | Sampling temperature |
| `LLM_TIMEOUT` | | — | Per-LLM-call timeout in seconds |
| `LLM_PARSE_MAX_ATTEMPTS` | | — | Max JSON parse retries before failing the job |
| `LLM_MAX_WORKERS` | | `5` | Parallel chunk workers (ThreadPool) |
| **SMTP** | | | |
| `SMTP_HOST` | ✅ | — | SMTP server host |
| `SMTP_PORT` | | `587` | SMTP port |
| `SMTP_USERNAME` | ✅ | — | SMTP login |
| `SMTP_PASSWORD` | ✅ | — | SMTP password |
| `SMTP_SENDER` | | _(same as username)_ | From address |
| **Redis** | | | |
| `REDIS_PASSWORD` | ✅ | — | Redis auth password |
| **PostgreSQL** | | | |
| `DB_HOST` | | `postgres` | Database host |
| `DB_PORT` | | `5432` | Database port |
| `DB_USERNAME` | ✅ | — | Database user |
| `DB_PASSWORD` | ✅ | — | Database password |
| `DB_NAME` | ✅ | — | Database name |
| **S3 / MinIO** | | | |
| `S3_ENDPOINT` | ✅ | — | S3-compatible endpoint URL |
| `S3_ACCESS_KEY` | ✅ | — | S3 access key |
| `S3_SECRET_KEY` | ✅ | — | S3 secret key |
| `S3_BUCKET` | ✅ | — | Bucket name |
| `S3_REGION` | ✅ | — | Region (e.g. `us-east-1`) |
| `MINIO_ROOT_USER` | | — | MinIO root user (local MinIO only) |
| `MINIO_ROOT_PASSWORD` | | — | MinIO root password (local MinIO only) |
| **Flower** | | | |
| `FLOWER_USER` | | — | Flower dashboard username |
| `FLOWER_PASSWORD` | | — | Flower dashboard password |
| **Tuning** | | | |
| `JOB_SOFT_TIMEOUT` | | `21600` | Per-job soft timeout in seconds (SIGTERM sent at this point) |
| `JOB_TIMEOUT_GRACE` | | `300` | Grace window after soft timeout before SIGKILL |
| `LOG_LEVEL` | | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MAX_UPLOAD_MB` | | `500` | Maximum upload file size in MB |

## API

All endpoints require the `x-api-key` header.

### Submit a job

```
POST /v1/jobs
Content-Type: multipart/form-data

file      — audio/video file
emails    — comma-separated email list (optional)
language  — BCP-47 language hint for ASR (optional, e.g. "ru", "en")
```

Response `202`:
```json
{ "status": "queued", "job_id": "uuid" }
```

### Check status

```
GET /v1/jobs/{job_id}
```

Statuses: `queued` → `asr_running` → `summary_running` → `done` / `failed` / `email_failed`

### Get result

```
GET /v1/jobs/{job_id}/result
```

Available only when status is `done`. Returns the rendered summary text plus the
structured summary fields:

```json
{
  "job_id": "uuid",
  "status": "done",
  "summary": "Decisions\n- ...",
  "structured": { "title": "...", "tldr": [], "decisions": [], "action_items": [], "risks": [] }
}
```

`structured` is `null` for jobs processed before structured output was stored.

### Submit corrections

```
PATCH /v1/jobs/{job_id}/result
Content-Type: application/json

{ "title": "...", "tldr": [], "decisions": [], "action_items": [], "risks": [] }
```

Send the full corrected summary (the shape returned in `structured`). Each changed
field is recorded as feedback used to calibrate the quality scores. **This does not
change the stored summary** — `GET /result` still returns the original; corrections
are collected as labels only. Available only when status is `done`.

### Health

```
GET /health
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)