"""§6.6 Regeneration (keep-if-better). LLM calls are mocked — these verify the
control logic the feature lives or dies on: the trigger policies, keep-if-better in
both directions, the repair-prompt wiring, and the worker's single-source-of-truth
post-terminal skip matrix."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from eden_summary.quality.faithfulness import JudgeResult
from eden_summary.quality.regen import repair_if_inconsistent
from eden_summary.quality.summq import SummQItem, SummQResult
from eden_summary.summarize.summarize import Summary, repair_summary
from eden_summary.worker import worker


def _summary(**fields) -> Summary:
    base = dict(title="T", tldr=[], decisions=[], action_items=[], risks=[])
    base.update(fields)
    return Summary(**base)


def _summq(score, threshold=0.7, items=()) -> SummQResult:
    items = list(items)
    return SummQResult(
        evaluated=True, items=items, consistency_score=score,
        n_questions=len(items), n_consistent=sum(1 for i in items if i.consistent),
        threshold=threshold,
    )


def _judge(score) -> JudgeResult:
    return JudgeResult(evaluated=True, overall_score=score, field_scores={})


# --------------------------------------------------------------- repair_summary
class TestRepairSummary:
    def test_wires_prompt_and_parses(self):
        canned = json.dumps({"title": "T2", "tldr": ["x"], "decisions": [],
                             "action_items": [], "risks": []})
        with patch("eden_summary.summarize.summarize._complete", return_value=canned) as comp:
            out = repair_summary("the transcript text",
                                 _summary(decisions=["bad claim"]),
                                 ["issue-A", "issue-B"])
        assert isinstance(out, Summary)
        assert out.title == "T2" and out.tldr == ["x"]
        # the repair brief (transcript + flagged issues) reaches the model
        user = comp.call_args.args[0][-1]["content"]
        assert "the transcript text" in user
        assert "issue-A" in user and "issue-B" in user


# ------------------------------------------------------- repair_if_inconsistent
def _run(trigger, summq_list, judge_list=(), repaired=None):
    cfg = SimpleNamespace(summq_regen_trigger=trigger, judge_faithfulness_threshold=0.8)
    rep = repaired if repaired is not None else _summary(decisions=["fixed"])
    with patch("eden_summary.quality.regen.get_llm_cfg", return_value=cfg), \
         patch("eden_summary.quality.regen.verify_summq_consistency",
               side_effect=list(summq_list)), \
         patch("eden_summary.quality.regen.judge_faithfulness",
               side_effect=list(judge_list)), \
         patch("eden_summary.quality.regen.repair_summary",
               return_value=rep) as repair_mock:
        summary, payload = repair_if_inconsistent(_summary(decisions=["orig"]), "transcript")
    return summary, payload, repair_mock


class TestTriggerSkips:
    def test_summq_above_threshold_no_repair(self):
        summary, payload, repair = _run("summq", [_summq(0.9)])
        assert summary.decisions == ["orig"]
        repair.assert_not_called()
        assert "summq_eval" in payload and "quality_eval" not in payload

    def test_both_summq_low_but_judge_high_no_repair(self):
        # 'both' needs BOTH judges to flag: judge 0.9 >= 0.8 → not triggered
        summary, payload, repair = _run("both", [_summq(0.5)], [_judge(0.9)])
        assert summary.decisions == ["orig"]
        repair.assert_not_called()
        assert "summq_eval" in payload and "quality_eval" in payload


class TestKeepIfBetterSummq:
    def test_repaired_better_is_kept(self):
        summary, payload, repair = _run("summq", [_summq(0.5), _summq(0.9)])
        repair.assert_called_once()
        assert summary.decisions == ["fixed"]
        assert payload["summq_eval"]["consistency_score"] == 0.9

    def test_repaired_worse_is_reverted(self):
        summary, payload, repair = _run("summq", [_summq(0.5), _summq(0.4)])
        repair.assert_called_once()
        assert summary.decisions == ["orig"]
        assert payload["summq_eval"]["consistency_score"] == 0.5


class TestKeepIfBetterBoth:
    def test_both_metrics_improve_is_kept(self):
        summary, payload, _ = _run(
            "both", [_summq(0.5), _summq(0.8)], [_judge(0.6), _judge(0.9)])
        assert summary.decisions == ["fixed"]
        assert payload["summq_eval"]["consistency_score"] == 0.8
        assert payload["quality_eval"]["overall_score"] == 0.9

    def test_faithfulness_regression_is_reverted(self):
        # SummQ up (0.5->0.9) but faithfulness down (0.6->0.5): keep-if-better reverts
        summary, payload, _ = _run(
            "both", [_summq(0.5), _summq(0.9)], [_judge(0.6), _judge(0.5)])
        assert summary.decisions == ["orig"]
        assert payload["quality_eval"]["overall_score"] == 0.6


class TestIssuesBrief:
    def test_failing_items_and_unsupported_claims_become_issues(self):
        items = [SummQItem(question="Price?", summary_answer="25", transcript_answer="NOT_STATED",
                           consistent=False)]
        summq = [_summq(0.5, items=items), _summq(0.9, items=items)]
        judge = [JudgeResult(evaluated=True, overall_score=0.6, field_scores={}, claims=[]),
                 _judge(0.9)]
        with patch("eden_summary.quality.regen.get_llm_cfg",
                   return_value=SimpleNamespace(summq_regen_trigger="both",
                                                judge_faithfulness_threshold=0.8)), \
             patch("eden_summary.quality.regen.verify_summq_consistency", side_effect=summq), \
             patch("eden_summary.quality.regen.judge_faithfulness", side_effect=judge), \
             patch("eden_summary.quality.regen.repair_summary",
                   return_value=_summary(decisions=["fixed"])) as repair:
            repair_if_inconsistent(_summary(decisions=["orig"]), "transcript")
        issues = repair.call_args.args[2]
        assert any("Price?" in i for i in issues)


# ------------------------------------------ worker post-terminal dispatch (state-based)
class _BeginCM:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _SessionCM:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *exc):
        return False


def _dispatch(quality_eval, summq_eval, judge_enabled=True, summq_enabled=True):
    """Worker dispatches a post-terminal eval iff it is enabled AND not already
    persisted (the regen path stores it in-pipeline). State-based, so it self-heals
    when regen ran but wrote nothing (e.g. a rate limit)."""
    cfg = SimpleNamespace(judge_enabled=judge_enabled, summq_enabled=summq_enabled)
    job = SimpleNamespace(id="job-1", quality_eval=quality_eval, summq_eval=summq_eval)
    db = MagicMock()
    db.begin = MagicMock(return_value=_BeginCM())
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
    db.execute = AsyncMock(return_value=result)
    with patch.object(worker.pipeline, "process_job", new=AsyncMock()), \
         patch.object(worker, "AsyncLocalSession", return_value=_SessionCM(db)), \
         patch.object(worker, "get_llm_cfg", return_value=cfg), \
         patch.object(worker, "evaluate_summary") as ev, \
         patch.object(worker, "verify_summq") as sq:
        worker.process_job("job-1")
    return ev, sq


class TestWorkerDispatchesMissingEvals:
    def test_dispatches_both_when_none_persisted(self):
        # regen off — or regen enabled but its in-pipeline computation wrote nothing
        ev, sq = _dispatch(quality_eval=None, summq_eval=None)
        ev.delay.assert_called_once_with("job-1")
        sq.delay.assert_called_once_with("job-1")

    def test_skips_summq_when_already_persisted(self):
        # regen 'summq': SummQ persisted in-pipeline, judge still pending
        ev, sq = _dispatch(quality_eval=None, summq_eval={"consistency_score": 0.9})
        ev.delay.assert_called_once_with("job-1")
        sq.delay.assert_not_called()

    def test_skips_both_when_both_persisted(self):
        # regen 'both': both metrics persisted in-pipeline
        ev, sq = _dispatch(quality_eval={"overall_score": 1.0},
                           summq_eval={"consistency_score": 0.9})
        ev.delay.assert_not_called()
        sq.delay.assert_not_called()

    def test_respects_enabled_flags(self):
        ev, sq = _dispatch(quality_eval=None, summq_eval=None,
                           judge_enabled=False, summq_enabled=False)
        ev.delay.assert_not_called()
        sq.delay.assert_not_called()
