"""record_result_edits wiring: DONE-guard, structured-summary load, per-field
upsert. Session and S3 are mocked (no live Postgres), so this runs in CI."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from eden_summary.core import JobStatus
from eden_summary.core.job_store import record_result_edits


class _AsyncCM:
    async def __aenter__(self):
        return None
    async def __aexit__(self, *exc):
        return False


def _make_db(job):
    db = MagicMock()
    db.begin = MagicMock(side_effect=lambda: _AsyncCM())  # fresh CM per `begin()`
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
    db.execute = AsyncMock(return_value=result)
    return db


_STORED = {'title': 'T', 'tldr': ['a'], 'decisions': ['d1'],
           'action_items': ['x'], 'risks': ['r1']}


def _fake_download(key, local_path):
    with open(local_path, 'w', encoding='UTF-8') as f:
        json.dump(_STORED, f)


def _job(status=JobStatus.DONE, artifacts=None):
    return SimpleNamespace(
        id='job-1',
        status=status,
        artifacts=artifacts if artifacts is not None else {
            'summary': 'job-1/summary.txt',
            'summary_json': 'job-1/summary_json.json',
        },
    )


async def _record(job, submitted):
    db = _make_db(job)
    with patch('eden_summary.core.job_store.download_file', side_effect=_fake_download):
        resp = await record_result_edits('job-1', submitted, db)
    return db, resp


async def test_records_edit_for_changed_field():
    db, resp = await _record(_job(), {**_STORED, 'risks': ['r1', 'new']})
    assert resp['recorded'] == 4
    assert resp['edited_fields'] == ['risks']
    assert db.execute.await_count == 5  # 1 load + 4 field upserts


async def test_unchanged_summary_still_writes_negatives():
    db, resp = await _record(_job(), dict(_STORED))
    assert resp['recorded'] == 4
    assert resp['edited_fields'] == []
    assert db.execute.await_count == 5  # negatives are genuine labels, written too


async def test_non_done_job_rejected_without_writes():
    db, resp = await _record(_job(status=JobStatus.SUMMARY_RUNNING), dict(_STORED))
    assert 'error' in resp
    assert db.execute.await_count == 1  # only the load, no upserts


async def test_missing_structured_artifact_rejected():
    db, resp = await _record(_job(artifacts={'summary': 'job-1/summary.txt'}), dict(_STORED))
    assert 'error' in resp
    assert db.execute.await_count == 1


async def test_unknown_job_returns_non_existing():
    _db, resp = await _record(None, dict(_STORED))
    assert resp['status'] == JobStatus.NON_EXISTING
