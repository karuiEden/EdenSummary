import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from eden_summary.core.models import Job


# minimal "audio" payload; the real format check is mocked via check_file
_FAKE_AUDIO = b"\xff\xfb\x90\x00" + b"\x00" * 64


def _patch_create_deps(task_mock: MagicMock):
    """check_file → True, S3-upload → no-op, Celery-task → mock."""
    return (
        patch("eden_summary.api.api.check_file", new=AsyncMock(return_value=True)),
        patch("eden_summary.core.job_store.upload_file"),
        patch("eden_summary.api.api.process_job", task_mock),
    )


async def _create_job(client, auth_headers, **kwargs) -> str:
    task = MagicMock()
    p1, p2, p3 = _patch_create_deps(task)
    with p1, p2, p3:
        r = await client.post(
            "/v1/jobs",
            headers=auth_headers,
            files={"file": ("meeting.mp3", _FAKE_AUDIO, "audio/mpeg")},
            **kwargs,
        )
    assert r.status_code == 202
    return r.json()["job_id"]


# ---------- /health ----------

class TestHealth:
    async def test_no_key_returns_401(self, client):
        assert (await client.get("/health")).status_code == 401

    async def test_ok(self, client, auth_headers):
        r = await client.get("/health", headers=auth_headers)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ---------- POST /v1/jobs ----------

class TestCreateJob:
    async def test_no_key_returns_401(self, client):
        r = await client.post(
            "/v1/jobs",
            files={"file": ("meeting.mp3", _FAKE_AUDIO, "audio/mpeg")},
        )
        assert r.status_code == 401

    async def test_invalid_file_returns_415(self, client, auth_headers):
        with patch("eden_summary.api.api.check_file", new=AsyncMock(return_value=False)):
            r = await client.post(
                "/v1/jobs",
                headers=auth_headers,
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
        assert r.status_code == 415

    async def test_valid_file_returns_202_and_enqueues(self, client, auth_headers):
        task = MagicMock()
        p1, p2, p3 = _patch_create_deps(task)
        with p1, p2, p3:
            r = await client.post(
                "/v1/jobs",
                headers=auth_headers,
                files={"file": ("meeting.mp3", _FAKE_AUDIO, "audio/mpeg")},
            )
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued"
        assert "job_id" in body
        task.delay.assert_called_once_with(body["job_id"])  # task enqueued

    async def test_valid_file_with_emails(self, client, auth_headers):
        job_id = await _create_job(
            client, auth_headers, data={"emails": "alice@example.com, bob@example.com"}
        )
        assert job_id


# ---------- GET /v1/jobs/{id} (status) ----------

class TestStatus:
    async def test_no_key_returns_401(self, client):
        assert (await client.get(f"/v1/jobs/{uuid4()}")).status_code == 401

    async def test_invalid_uuid_returns_400(self, client, auth_headers):
        r = await client.get("/v1/jobs/not-a-uuid", headers=auth_headers)
        assert r.status_code == 400

    async def test_unknown_job_returns_404(self, client, auth_headers):
        r = await client.get(f"/v1/jobs/{uuid4()}", headers=auth_headers)
        assert r.status_code == 404

    async def test_existing_job_returns_queued(self, client, auth_headers):
        job_id = await _create_job(client, auth_headers)
        r = await client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] == "queued"


# ---------- GET /v1/jobs/{id}/result ----------

class TestResult:
    async def test_no_key_returns_401(self, client):
        assert (await client.get(f"/v1/jobs/{uuid4()}/result")).status_code == 401

    async def test_invalid_uuid_returns_400(self, client, auth_headers):
        r = await client.get("/v1/jobs/not-a-uuid/result", headers=auth_headers)
        assert r.status_code == 400

    async def test_unknown_job_returns_404(self, client, auth_headers):
        # no job → NON_EXISTING → 404 (consistent with /status and PATCH /result)
        r = await client.get(f"/v1/jobs/{uuid4()}/result", headers=auth_headers)
        assert r.status_code == 404

    async def test_queued_job_returns_409(self, client, auth_headers):
        job_id = await _create_job(client, auth_headers)
        r = await client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert r.status_code == 409  # not ready yet

    async def test_done_job_returns_summary(self, client, auth_headers, db_session):
        job_id = str(uuid4())
        async with db_session.begin():
            db_session.add(Job(
                id=job_id,
                status="done",
                emails=[],
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                artifacts={"summary": f"{job_id}/summary.txt"},
                error=None,
                warning=None,
            ))

        summary_text = "# Meeting Summary\n\nAgreed to have a call on Friday."

        def _fake_download(s3_path, local_path):
            with open(local_path, "w", encoding="UTF-8") as f:
                f.write(summary_text)

        with patch(
            "eden_summary.core.job_store.download_file", side_effect=_fake_download
        ):
            r = await client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert body["summary"] == summary_text
        assert body["structured"] is None  # no summary_json artifact on this job


# ---------- PATCH /v1/jobs/{id}/result ----------

_STRUCTURED = {"title": "T", "tldr": ["a"], "decisions": ["d1"],
               "action_items": ["x"], "risks": ["r1"]}


async def _insert_done_job(db_session, job_id, artifacts):
    async with db_session.begin():
        db_session.add(Job(
            id=job_id, status="done", emails=[],
            created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
            artifacts=artifacts, error=None, warning=None,
        ))


def _patch_download():
    def _fake(s3_path, local_path):
        with open(local_path, "w", encoding="UTF-8") as f:
            json.dump(_STRUCTURED, f)
    return patch("eden_summary.core.job_store.download_file", side_effect=_fake)


class TestEditResult:
    async def test_no_key_returns_401(self, client):
        r = await client.patch(f"/v1/jobs/{uuid4()}/result", json=_STRUCTURED)
        assert r.status_code == 401

    async def test_invalid_uuid_returns_400(self, client, auth_headers):
        r = await client.patch("/v1/jobs/not-a-uuid/result", headers=auth_headers, json=_STRUCTURED)
        assert r.status_code == 400

    async def test_unknown_job_returns_404(self, client, auth_headers):
        r = await client.patch(f"/v1/jobs/{uuid4()}/result", headers=auth_headers, json=_STRUCTURED)
        assert r.status_code == 404

    async def test_not_done_job_returns_409(self, client, auth_headers):
        job_id = await _create_job(client, auth_headers)  # status queued
        r = await client.patch(f"/v1/jobs/{job_id}/result", headers=auth_headers, json=_STRUCTURED)
        assert r.status_code == 409

    async def test_edit_records_changed_field(self, client, auth_headers, db_session):
        job_id = str(uuid4())
        await _insert_done_job(db_session, job_id, {
            "summary": f"{job_id}/summary.txt",
            "summary_json": f"{job_id}/summary_json.json",
        })
        edited = {**_STRUCTURED, "risks": ["r1", "new risk"]}
        with _patch_download():
            r = await client.patch(f"/v1/jobs/{job_id}/result", headers=auth_headers, json=edited)
        assert r.status_code == 200
        body = r.json()
        assert body["recorded"] == 4
        assert body["edited_fields"] == ["risks"]