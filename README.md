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
| `WHISPER_ASR_CHUNK_SECONDS` | | `300` | Audio longer than this (seconds) is split into pieces before transcription so each request stays under the ASR API's per-file cap. 16 kHz mono WAV ≈ 32 KB/s, so 300 s ≈ 9.6 MB — kept below Groq's 25 MB free-tier limit (uploads near the cap get connection-reset in practice). Raise it for a self-hosted ASR server with no size limit |
| `MAX_CHARS` | | `8000` | Max characters per chunk sent to the LLM (map-reduce path) |
| `CHUNK_OVERLAP_CHARS` | | `1200` | Overlap (characters) between consecutive chunks so a claim on a chunk boundary isn't dropped |
| **LLM** | | | |
| `LLM_MODEL` | ✅ | — | litellm model string (e.g. `groq/llama-3.3-70b-versatile`, `openai/gpt-4o`) |
| `LLM_API_KEY` | ✅ | — | LLM provider API key |
| `LLM_API_BASE` | | _(provider default)_ | Override API base URL (for self-hosted models) |
| `LLM_MAX_RETRIES` | | — | litellm retry count on provider errors |
| `LLM_TEMPERATURE` | | — | Sampling temperature |
| `LLM_TIMEOUT` | | — | Per-LLM-call timeout in seconds |
| `LLM_PARSE_MAX_ATTEMPTS` | | — | Max JSON parse retries before failing the job |
| `LLM_MAX_WORKERS` | | `5` | Parallel chunk workers (ThreadPool) |
| `LLM_SINGLE_PASS_TOKEN_LIMIT` | | `32000` | Transcripts up to this estimated token count are summarized in one pass; longer ones fall back to map-reduce |
| `LLM_JSON_MODE` | | `true` | Request native JSON output from the LLM (`response_format`) |
| **Quality (judge / SummQ)** | | | |
| `LLM_JUDGE_ENABLED` | | `true` | Enable the Tier-2 faithfulness judge (post-terminal) |
| `LLM_JUDGE_MODEL` | | _(same as `LLM_MODEL`)_ | Judge model — kept separate from the summarizer (summarizer ≠ judge). Falls back to the main LLM |
| `LLM_JUDGE_API_BASE` | | _(`LLM_API_BASE`)_ | Judge endpoint override (e.g. a different provider) |
| `LLM_JUDGE_API_KEY` | | _(`LLM_API_KEY`)_ | Judge API key override |
| `LLM_JUDGE_TOKEN_LIMIT` | | `32000` | Skip the judge when the transcript exceeds this estimated token count |
| `LLM_SUMMQ_ENABLED` | | `true` | Enable Q4 SummQ QA-consistency check (post-terminal) |
| `LLM_SUMMQ_THRESHOLD` | | `0.7` | Consistency score below this marks the summary `below_threshold` |
| `LLM_SUMMQ_MAX_QUESTIONS` | | `8` | Max fact-check questions generated per summary |
| **Regeneration (opt-in)** | | | |
| `LLM_SUMMQ_REGEN` | | `false` | Enable keep-if-better regeneration: a summary that fails the consistency checks is repaired in-pipeline (before the email) and the repaired version kept only if no metric regressed. Off by default — see [Regeneration](#regeneration-opt-in--off-by-default) |
| `LLM_SUMMQ_REGEN_TRIGGER` | | `both` | Trigger policy: `summq` (SummQ below threshold) or `both` (SummQ **and** Tier-2 faithfulness below threshold — two independent judges must agree) |
| `LLM_JUDGE_FAITHFULNESS_THRESHOLD` | | `0.8` | Tier-2 score below this counts as a faithfulness flag for the `both` trigger |
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
| `REAPER_STALE_SECONDS` | | `1800` | A job stuck in a non-terminal status with no `updated_at` change for longer than this is reaped → `FAILED` |

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

Available only when status is `done`. Returns the rendered summary text, the
structured summary fields, and three advisory quality signals (see
[Quality monitoring](#quality-monitoring)):

```json
{
  "job_id": "uuid",
  "status": "done",
  "summary": "Decisions\n- ...",
  "structured": { "title": "...", "tldr": [], "decisions": [], "action_items": [], "risks": [] },
  "quality_flags": { "passed": true, "checked": ["numbers"], "flags": [] },
  "quality_eval":  { "overall_score": 1.0, "field_scores": {}, "claims": [] },
  "summq_eval":    { "consistency_score": 0.75, "below_threshold": false, "items": [] }
}
```

`structured` is `null` for jobs processed before structured output was stored.

`quality_flags` (Tier-1) is present as soon as the job is `done`. `quality_eval`
(Tier-2 faithfulness) and `summq_eval` (SummQ) are produced by post-terminal async
tasks, so they read **`null` on the first fetch(es)** and populate seconds-to-minutes
later — re-poll. (When opt-in regeneration is enabled they are computed in-pipeline
and available immediately.)

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

## Quality monitoring

Every summary is checked for faithfulness to the transcript by three **independent,
advisory** signals. They are advisory by design: a signal is recorded on the job and
surfaced via `GET /result`, but it **never blocks or alters delivery**, and a failure
in one never affects another. The guiding rule is **summarizer ≠ judge** — the judge
runs on a separate model so it doesn't repeat the summarizer's mistakes.

| Signal | When | Field | What it checks |
|---|---|---|---|
| **Tier-1 — inline guards** | in-pipeline, <100 ms, 100% of jobs | `quality_flags` | Deterministic grounding of numbers and names from the summary against the transcript (spoken numbers folded to digits for en/ru). Flags ungrounded values |
| **Tier-2 — faithfulness judge** | post-terminal, async | `quality_eval` | One LLM call grades every summary claim `supported` / `partial` / `unsupported` (1.0 / 0.5 / 0.0) against the transcript, with an evidence quote; reports overall and per-field scores |
| **Q4 — SummQ** | post-terminal, async | `summq_eval` | Turns the summary into a fact-check quiz, answers each question **blind** from the transcript (the summary's own answer is hidden → no confirmation bias), and compares deterministically. An independent angle on the judge |

`quality_flags` is ready the moment a job is `done`; `quality_eval` and `summq_eval`
are computed by post-terminal tasks and arrive a little later (poll `GET /result`
again).

**Q3 — edit calibration.** `PATCH /v1/jobs/{id}/result` records which summary fields
a reviewer changed as labels (changed → positive, untouched → genuine negative). These
calibrate the judge scores over time. The stored summary is never modified — see
[Submit corrections](#submit-corrections).

### Regeneration (opt-in — off by default)

Enable with `LLM_SUMMQ_REGEN=true`. When a summary fails the consistency checks,
EdenSummary repairs the unsupported claims (using the summarizer model) and adopts the
repaired version **only if it scores no worse** on the same metrics (*keep-if-better*),
all in-pipeline before the email is sent. The trigger is set by `LLM_SUMMQ_REGEN_TRIGGER`
and is conservative by default (`both`): it requires **both** the faithfulness judge and
SummQ to flag the summary. The metrics computed during regeneration are persisted
directly, so the matching post-terminal eval is skipped for that job (single source of
truth).

> **Honest boundary.** keep-if-better guarantees *not worse on the faithfulness
> metrics*, **not** *not worse in overall quality*: SummQ builds its questions from the
> summary, so a score can be raised by **deleting** a claim (no question is generated
> for it), creating structural pressure toward sparser summaries. The repair prompt
> mitigates this but does not remove it. For that reason the feature is **dark-launch /
> opt-in and off by default**; confident enablement needs a completeness-aware
> evaluation and real-world data.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)