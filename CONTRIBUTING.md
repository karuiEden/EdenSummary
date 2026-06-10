# Contributing

## Local setup

```bash
git clone https://github.com/karuiEden/EdenSummary.git
cd EdenSummary
uv sync
cp .env.example .env  # fill in values
```

## Running checks

```bash
uv run ruff check eden_summary/   # lint
uv run mypy eden_summary/         # type check
uv run pytest tests/unit -v       # unit tests
```

All three must pass before opening a PR. CI runs the same checks on every push.

## Integration tests

Requires a running stack:

```bash
docker compose up -d db redis minio createbuckets migrate
uv run pytest tests/integration -v
```

## Commit style

```
type: short description
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## Pull requests

1. Fork → branch from `master`
2. Make changes, ensure all checks pass
3. Open PR against `master` with a description of what and why