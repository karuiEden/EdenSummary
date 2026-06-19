import logging
from dataclasses import asdict, dataclass, field

from litellm import completion

from eden_summary.core import LLMConfig, get_llm_cfg
from eden_summary.summarize.summarize import (
    Summary, _estimate_tokens, _parse_json, _render_item,
)

logger = logging.getLogger(__name__)

# Fields verified in this order. DIAL-SUMMER: hallucinations cluster at the END of
# generation, so 'risks' (produced last) are the least trustworthy and are graded
# first while the judge's attention is freshest.
_FIELD_ORDER = ('risks', 'decisions', 'action_items', 'tldr')

_VERDICTS = {'supported', 'partial', 'unsupported'}
_VERDICT_SCORE = {'supported': 1.0, 'partial': 0.5, 'unsupported': 0.0}

JUDGE_SYSTEM_PROMPT = """You are a meeting-summary faithfulness judge. You are NOT \
a summarizer — you grade an existing summary against the source transcript.

For each claim you are given, decide whether the TRANSCRIPT supports it:
- "supported": the transcript directly states the claim
- "partial": partially supported, or supported but with distorted details (e.g. a wrong number/name)
- "unsupported": the transcript does not state this — a fabrication

Judge ONLY against the transcript. Do not use outside knowledge. Do not reward or
penalize the summary for omitting things — you grade only the claims given.
Return ONLY valid JSON."""

JUDGE_USER_PROMPT = """Transcript:
{transcript}

Claims to grade (each has an id and the summary field it came from):
{claims}

Return JSON of this exact shape:
{{"claims": [{{"id": <int>, "verdict": "supported|partial|unsupported", "evidence": "<short quote from the transcript, or empty if unsupported>"}}]}}
Grade every claim id exactly once."""


@dataclass(frozen=True)
class ClaimVerdict:
    field: str       # summary field the claim came from ('risks', 'decisions', ...)
    claim: str       # the rendered claim text that was graded
    verdict: str     # 'supported' | 'partial' | 'unsupported'
    score: float     # 1.0 | 0.5 | 0.0
    evidence: str    # supporting quote from the transcript ('' when unsupported)


@dataclass(frozen=True)
class JudgeResult:
    evaluated: bool
    reason: str = ''                                  # why skipped, when evaluated=False
    claims: list[ClaimVerdict] = field(default_factory=list)
    overall_score: float | None = None                # mean claim score
    field_scores: dict[str, float] = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            'evaluated': self.evaluated,
            'reason': self.reason,
            'overall_score': self.overall_score,
            'field_scores': self.field_scores,
            'claims': [asdict(c) for c in self.claims],
        }


def _collect_claims(summary: Summary) -> list[tuple[str, str]]:
    """Flatten the summary into (field, claim_text) pairs in verification order,
    skipping empty/placeholder items. The title is not graded (it's a label, not
    a factual claim)."""
    claims: list[tuple[str, str]] = []
    for fld in _FIELD_ORDER:
        for item in getattr(summary, fld):
            text = _render_item(item)
            if text:
                claims.append((fld, text))
    return claims


def _judge_complete(messages: list[dict]) -> str:
    config: LLMConfig = get_llm_cfg()
    kwargs: dict = dict(
        model=config.effective_judge_model,
        api_key=config.effective_judge_api_key,
        api_base=config.effective_judge_api_base,
        max_retries=config.max_retries,
        timeout=config.timeout,
        temperature=config.temperature,
        messages=messages,
    )
    if config.json_mode:
        kwargs['response_format'] = {'type': 'json_object'}
    response = completion(**kwargs)
    return str(response.choices[0].message.content)


def _scores(verdicts: list[ClaimVerdict]) -> tuple[float | None, dict[str, float]]:
    if not verdicts:
        return None, {}
    overall = sum(v.score for v in verdicts) / len(verdicts)
    field_scores: dict[str, float] = {}
    for fld in _FIELD_ORDER:
        fvs = [v.score for v in verdicts if v.field == fld]
        if fvs:
            field_scores[fld] = sum(fvs) / len(fvs)
    return overall, field_scores


def judge_faithfulness(summary: Summary, transcript: str) -> JudgeResult:
    """Grade every summary claim against the transcript in ONE judge LLM call
    (summarizer ≠ judge, no ensembles). Returns per-claim verdicts plus overall
    and per-field scores. Skips (evaluated=False) when there are no claims or the
    transcript exceeds the judge token budget. Language-agnostic — the LLM judge
    handles any language, unlike the Tier-1 regex guard."""
    config: LLMConfig = get_llm_cfg()
    claims = _collect_claims(summary)
    if not claims:
        return JudgeResult(evaluated=False, reason='no claims to grade')
    estimated = _estimate_tokens(transcript)
    if estimated > config.judge_token_limit:
        return JudgeResult(
            evaluated=False,
            reason=f'transcript too long for judge ({estimated} > {config.judge_token_limit} tokens)',
        )

    numbered = '\n'.join(f'{i}. [{fld}] {text}' for i, (fld, text) in enumerate(claims))
    content = _judge_complete([
        {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
        {'role': 'user', 'content': JUDGE_USER_PROMPT.format(transcript=transcript, claims=numbered)},
    ])
    parsed = _parse_json(content)

    # v1: assumes the judge echoes integer ids matching our enumeration. A model
    # that emits string ids ("0") or drops the array → every claim falls through
    # to the unsupported default below (a silent all-flag result). Acceptable for
    # an advisory v1, but harden this when swapping judge models via OpenRouter.
    by_id = {}
    for entry in parsed.get('claims', []):
        if isinstance(entry, dict) and 'id' in entry:
            by_id[entry['id']] = entry

    verdicts: list[ClaimVerdict] = []
    for i, (fld, text) in enumerate(claims):
        entry = by_id.get(i, {})
        verdict = str(entry.get('verdict', 'unsupported')).lower()
        if verdict not in _VERDICTS:
            verdict = 'unsupported'
        verdicts.append(ClaimVerdict(
            field=fld,
            claim=text,
            verdict=verdict,
            score=_VERDICT_SCORE[verdict],
            evidence=str(entry.get('evidence', '')),
        ))

    overall, field_scores = _scores(verdicts)
    return JudgeResult(
        evaluated=True,
        claims=verdicts,
        overall_score=overall,
        field_scores=field_scores,
    )
