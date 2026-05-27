# EdenSummary

AI-сервис для автоматической транскрипции и суммаризации записей совещаний.

Принимает аудио или видео файл, транскрибирует его через Whisper, суммаризирует через LLM и отправляет результат на email.

## Как это работает

1. Клиент загружает файл через `POST /v1/jobs`
2. Сервис сохраняет файл и запускает обработку в фоне
3. Аудио конвертируется в WAV через ffmpeg
4. Транскрипция выполняется через faster-whisper
5. Текст разбивается на чанки и суммаризируется через LLM (map-reduce)
6. Результат отправляется на email и становится доступен через API

## Требования

- Python 3.14+
- ffmpeg
- CUDA (опционально, для GPU-инференса Whisper)

## Установка

```bash
uv sync
```

## Конфигурация

Создай `.env` файл в корне проекта:

```env
# API
X_API_KEY=your_secret_key

# Whisper
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=ru
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
MAX_CHARS=4000

# LLM (совместим с любым провайдером через litellm)
LLM_MODEL=openai/gpt-4o
LLM_API_KEY=your_llm_api_key
LLM_API_BASE=             # опционально, для self-hosted моделей
LLM_LANGUAGE=ru
LLM_MAX_RETRIES=3
LLM_TEMPERATURE=0.2

# SMTP
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=user@example.com
SMTP_PASSWORD=your_password
SMTP_SENDER=user@example.com

# Прочее
OUTPUT_DIR=output
```

## Запуск

```bash
uv run uvicorn system.api:app --host 0.0.0.0 --port 8000
```

## API

Все запросы требуют заголовок `x-api-key`.

### Загрузить файл

```
POST /v1/jobs
Content-Type: multipart/form-data

file    — аудио/видео файл (.mp3, .wav, .mp4, .mov, .m4a)
emails  — список email через запятую (опционально)
```

Ответ `202`:
```json
{ "status": "queued", "job_id": "uuid" }
```

### Статус задачи

```
GET /v1/jobs/{job_id}
```

Возможные статусы: `queued`, `asr_running`, `summary_running`, `done`, `failed`, `smtp_failed`

### Результат

```
GET /v1/jobs/{job_id}/result
```

Доступен только при статусе `done`. Возвращает транскрипт и суммаризацию.

### Healthcheck

```
GET /health
```
