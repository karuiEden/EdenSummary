"""get_result wiring: surfaces the text summary, the structured summary, and the
three quality signals. Session and S3 are mocked (no live Postgres), so this runs
in CI. Mirrors test_record_result_edits' harness."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from eden_summary.core import JobStatus
from eden_summary.core.job_store import get_result


class _AsyncCM:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def _make_db(job):
    db = MagicMock()
    db.begin = MagicMock(side_effect=lambda: _AsyncCM())
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
    db.execute = AsyncMock(return_value=result)
    return db


_STRUCTURED = {'title': 'T', 'tldr': ['a'], 'decisions': ['d1'],
               'action_items': ['x'], 'risks': ['r1']}


def _fake_download(key, local_path):
    # summary.txt -> text; summary_json.json -> structured dict
    payload = json.dumps(_STRUCTURED) if key.endswith('.json') else 'the summary text'
    with open(local_path, 'w', encoding='UTF-8') as f:
        f.write(payload)


def _job(status=JobStatus.DONE, *, quality_flags=None, quality_eval=None, summq_eval=None):
    return SimpleNamespace(
        id='job-1',
        status=status,
        artifacts={'summary': 'job-1/summary.txt', 'summary_json': 'job-1/summary_json.json'},
        quality_flags=quality_flags,
        quality_eval=quality_eval,
        summq_eval=summq_eval,
    )


async def _get(job):
    db = _make_db(job)
    with patch('eden_summary.core.job_store.download_file', side_effect=_fake_download):
        return await get_result('job-1', db)


async def test_done_job_surfaces_quality_signals():
    resp = await _get(_job(
        quality_flags={'passed': False, 'flags': [{'kind': 'number'}]},
        quality_eval={'overall': 0.6, 'field_scores': {'risks': 0.0}},
        summq_eval={'consistency_score': 0.5, 'below_threshold': True},
    ))
    assert resp['status'] == JobStatus.DONE
    assert resp['summary'] == 'the summary text'
    assert resp['structured'] == _STRUCTURED
    assert resp['quality_flags']['passed'] is False
    assert resp['quality_eval']['overall'] == 0.6
    assert resp['summq_eval']['below_threshold'] is True


async def test_async_evals_null_before_they_land():
    # The post-terminal evals haven't written yet: keys present but null, not missing.
    resp = await _get(_job(quality_flags={'passed': True, 'flags': []}))
    assert resp['quality_flags'] == {'passed': True, 'flags': []}
    assert resp['quality_eval'] is None
    assert resp['summq_eval'] is None


async def test_unknown_job_returns_non_existing():
    resp = await _get(None)
    assert resp['status'] == JobStatus.NON_EXISTING


async def test_non_done_job_has_no_quality_keys():
    # Not done yet -> early return with just status, no quality fields.
    resp = await _get(_job(status=JobStatus.SUMMARY_RUNNING))
    assert resp['status'] == JobStatus.SUMMARY_RUNNING
    assert 'quality_flags' not in resp
