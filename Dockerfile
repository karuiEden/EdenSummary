FROM python:3.14
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libmagic1  \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m app

COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=app:app . .

USER app

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "eden_summary.api.api:app", "--host", "0.0.0.0", "--port", "8000"]