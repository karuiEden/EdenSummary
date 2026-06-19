import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from eden_summary.core import JobStatus
from eden_summary.quality.faithfulness import JudgeResult
from eden_summary.worker import worker


class _AsyncCM:
    async def __aenter__(self):
        return None
    async def __aexit__(self, *exc):
        return False


def _db_with_job(job):
    db = MagicMock()
    db.begin = MagicMock(return_value=_AsyncCM())
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
    db.execute = AsyncMock(return_value=result)
    return db


class _SessionCM:
    def __init__(self, db):
        self._db = db
    async def __aenter__(self):
        return self._db
    async def __aexit__(self, *exc):
        return False


def _touch(key, local_path):  # stand-in for download_file: create the local file
    with open(local_path, "w", encoding="UTF-8") as f:
        f.write("{}")


def _run_evaluate(job, **patches):
    db = _db_with_job(job)
    with patch("eden_summary.worker.worker.AsyncLocalSession", return_value=_SessionCM(db)), \
         patch("eden_summary.worker.worker.update_job", new=AsyncMock()) as upd, \
         patch("eden_summary.worker.worker.download_file", new=MagicMock(side_effect=_touch)), \
         patch("eden_summary.worker.worker.judge_faithfulness",
               return_value=patches.get("judge", JudgeResult(evaluated=True, overall_score=1.0))) as judge, \
         patch("eden_summary.worker.worker.json.load",
               side_effect=patches.get("json_load", [{"segments": ["hi"], "language": "en"},
                                                     {"title": "T", "tldr": [], "decisions": ["d"],
                                                      "action_items": [], "risks": []}])):
        asyncio.run(worker._evaluate("job-1"))
    return upd, judge


def _job(status, artifacts):
    return SimpleNamespace(id="job-1", status=status, artifacts=artifacts)


_FULL_ARTIFACTS = {"transcript": "job-1/transcript.json", "summary_json": "job-1/summary_json.json"}


class TestEvaluateSelfGuards:
    def test_skips_when_not_done(self):
        upd, judge = _run_evaluate(_job(JobStatus.SUMMARY_RUNNING, _FULL_ARTIFACTS))
        judge.assert_not_called()
        upd.assert_not_called()

    def test_skips_when_job_missing(self):
        upd, judge = _run_evaluate(None)
        judge.assert_not_called()
        upd.assert_not_called()

    def test_skips_when_artifacts_missing(self):
        upd, judge = _run_evaluate(_job(JobStatus.DONE, {"summary": "job-1/summary.txt"}))
        judge.assert_not_called()
        upd.assert_not_called()


class TestEvaluateHappyPath:
    def test_runs_judge_and_writes_quality_eval(self):
        result = JudgeResult(evaluated=True, overall_score=0.75, field_scores={"decisions": 0.75})
        upd, judge = _run_evaluate(_job(JobStatus.DONE, _FULL_ARTIFACTS), judge=result)
        judge.assert_called_once()
        # transcript joined from segments, summary reconstructed
        called_summary, called_transcript = judge.call_args.args
        assert called_transcript == "hi"
        assert called_summary.decisions == ["d"]
        # quality_eval persisted, status untouched (no status kwarg)
        upd.assert_awaited_once()
        assert upd.await_args.kwargs["quality_eval"]["overall_score"] == 0.75
        assert "status" not in upd.await_args.kwargs


class TestEvaluateNeverFails:
    def test_task_swallows_errors(self):
        # evaluate_summary wraps _evaluate; an exception must not propagate
        with patch("eden_summary.worker.worker._evaluate",
                   side_effect=RuntimeError("boom")):
            worker.evaluate_summary("job-1")  # must not raise


class TestCalibrationLabels:
    # Q3: the judge score is joined to each edit at fit time (not snapshotted),
    # so edits whose field was never scored must be dropped, not zero-filled.
    def test_joins_score_by_field_and_skips_unscored_job(self):
        edits = [
            SimpleNamespace(job_id="j1", field="risks", edited=True),
            SimpleNamespace(job_id="j1", field="decisions", edited=False),
            SimpleNamespace(job_id="j2", field="tldr", edited=True),  # j2 has no eval
        ]
        scores_by_job = {"j1": {"risks": 0.0, "decisions": 1.0}}
        assert worker._labels_from_edits(edits, scores_by_job) == [(0.0, True), (1.0, False)]

    def test_field_without_score_is_skipped(self):
        edits = [SimpleNamespace(job_id="j1", field="risks", edited=True)]
        scores_by_job = {"j1": {"decisions": 0.5}}  # risks not scored
        assert worker._labels_from_edits(edits, scores_by_job) == []
