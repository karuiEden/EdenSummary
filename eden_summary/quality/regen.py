"""Regeneration (keep-if-better) — opt-in, off by default.

Closes the quality loop: when a finished summary fails the consistency checks,
repair its unsupported claims and adopt the repaired version ONLY if it scores no
worse on the same metrics (keep-if-better). Runs in-pipeline, before the email, so
the served/emailed result is the chosen one.

Returns the final summary plus the metrics to persist. Those metrics are the single
source of truth: the worker skips the corresponding post-terminal task for this job
(re-running it would overwrite the score with a non-deterministic re-evaluation).

Limitation: keep-if-better guarantees "not worse on the faithfulness metrics",
not "not worse in quality" — SummQ builds its questions from the summary, so a score
can be raised by deleting a claim. The repair prompt mitigates but does not remove
this, which is why the feature is off by default.
"""
import logging

from eden_summary.core import LLMConfig, get_llm_cfg
from eden_summary.metrics import summq_regen_total
from eden_summary.quality.faithfulness import JudgeResult, judge_faithfulness
from eden_summary.quality.summq import SummQResult, verify_summq_consistency
from eden_summary.summarize.summarize import Summary, repair_summary

logger = logging.getLogger(__name__)


def _summq_below(result: SummQResult) -> bool:
    return (result.evaluated and result.consistency_score is not None
            and result.consistency_score < result.threshold)


def _judge_low(judge: JudgeResult | None, threshold: float) -> bool:
    return (judge is not None and judge.overall_score is not None
            and judge.overall_score < threshold)


def _collect_issues(summq: SummQResult, judge: JudgeResult | None) -> list[str]:
    """Build the repair brief: failing SummQ questions plus (for the 'both' policy)
    the claims the judge marked unsupported."""
    issues: list[str] = []
    for item in summq.items:
        if not item.consistent:
            transcript_answer = item.transcript_answer or 'not stated'
            issues.append(f'Q: {item.question} — summary says "{item.summary_answer}", '
                          f'transcript says "{transcript_answer}"')
    if judge is not None:
        for claim in judge.claims:
            if claim.verdict == 'unsupported':
                issues.append(f'[{claim.field}] {claim.claim}')
    return issues


def _eval_payload(summq: SummQResult, judge: JudgeResult | None) -> dict:
    """Metrics to persist for the FINAL summary. Only metrics actually computed are
    included (matches the worker post-terminal skip matrix)."""
    payload: dict = {'summq_eval': summq.to_metadata()}
    if judge is not None:
        payload['quality_eval'] = judge.to_metadata()
    return payload


def _score(summq: SummQResult) -> float:
    return summq.consistency_score if summq.consistency_score is not None else 0.0


def _jscore(judge: JudgeResult | None) -> float:
    if judge is None or judge.overall_score is None:
        return 0.0
    return judge.overall_score


def _is_better(summq: SummQResult, judge: JudgeResult | None,
               r_summq: SummQResult, r_judge: JudgeResult | None, both: bool) -> bool:
    """keep-if-better: no scored metric regressed and at least one strictly improved."""
    s0, s1 = _score(summq), _score(r_summq)
    if not both:
        return s1 > s0
    j0, j1 = _jscore(judge), _jscore(r_judge)
    return s1 >= s0 and j1 >= j0 and (s1 > s0 or j1 > j0)


def repair_if_inconsistent(summary: Summary, transcript: str) -> tuple[Summary, dict]:
    """Score the summary; if the configured trigger fires, repair it and keep the
    repaired version only when no metric regressed. Returns (final_summary, payload)
    where payload carries the final version's metrics for in-pipeline persistence."""
    cfg: LLMConfig = get_llm_cfg()
    both = cfg.summq_regen_trigger == 'both'

    summq = verify_summq_consistency(summary, transcript)
    judge = judge_faithfulness(summary, transcript) if both else None

    triggered = _summq_below(summq)
    if both:
        triggered = triggered and _judge_low(judge, cfg.judge_faithfulness_threshold)

    if not triggered:
        summq_regen_total.labels(outcome='skipped').inc()
        return summary, _eval_payload(summq, judge)

    issues = _collect_issues(summq, judge)
    repaired = repair_summary(transcript, summary, issues)
    r_summq = verify_summq_consistency(repaired, transcript)
    r_judge = judge_faithfulness(repaired, transcript) if both else None

    if _is_better(summq, judge, r_summq, r_judge, both):
        summq_regen_total.labels(outcome='applied').inc()
        logger.info("Regeneration applied (summq %s -> %s)",
                    summq.consistency_score, r_summq.consistency_score)
        return repaired, _eval_payload(r_summq, r_judge)

    summq_regen_total.labels(outcome='reverted').inc()
    logger.info("Regeneration reverted: repaired not better (summq %s -> %s)",
                summq.consistency_score, r_summq.consistency_score)
    return summary, _eval_payload(summq, judge)
