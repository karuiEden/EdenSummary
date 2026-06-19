"""Q4 SummQ: QA-based consistency check.

A deliberately INDEPENDENT signal to the Tier-2 claim judge (faithfulness.py).
The claim judge asks "does the transcript support this claim?"; SummQ instead
turns the summary into a short fact-check quiz, answers each question from the
transcript *blind* (never shown the summary's own answer), then compares the two
answers deterministically. Answering blind is the whole point — handing the model
the summary's answer would just rebuild the claim judge with confirmation bias.
Disagreement (or an unanswerable question) means the summary asserts something the
transcript does not, i.e. a faithfulness gap caught from a different angle (QAGS /
QuestEval recipe).

v1 is compute-and-log only: the consistency score and a below_threshold flag are
stored on the job, but no regeneration is applied. The regeneration loop is
deferred — at temperature 0 a naive re-summarization is a near no-op, so it needs
a repair prompt plus real data before it can be validated.
"""
import logging
import re
from dataclasses import asdict, dataclass, field

from litellm import completion

from eden_summary.core import LLMConfig, get_llm_cfg
from eden_summary.quality.numbers import spoken_to_digits
from eden_summary.summarize.summarize import (
    Summary, _estimate_tokens, _parse_json, _render_item,
)

logger = logging.getLogger(__name__)

# Fields quizzed. Title is excluded (a label, not a checkable fact), same as the
# Tier-2 judge. Order is informational only — SummQ scores are pooled, not ranked.
_QUIZ_FIELDS = ('decisions', 'action_items', 'risks', 'tldr')

GEN_SYSTEM_PROMPT = """You turn a meeting summary into a short fact-check quiz. For \
the most important, checkable facts in the summary, write closed questions. Each \
question must have a SHORT, literal answer — a number, name, date, or short noun \
phrase — copied verbatim from the summary (never a full sentence). Ask only about \
facts actually stated in the summary. Return ONLY valid JSON."""

GEN_USER_PROMPT = """Summary to quiz (JSON):
{summary}

Write up to {max_q} questions covering the most important checkable facts
(decisions, action items, risks, key points). Return JSON of this exact shape:
{{"questions": [{{"question": "<question>", "answer": "<short literal answer from the summary>"}}]}}"""

ANSWER_SYSTEM_PROMPT = """You answer questions about a meeting using ONLY the \
provided transcript. Give a SHORT, literal answer — a number, name, date, or short \
phrase, copied from the transcript. If the transcript does not state the answer, \
reply with exactly "NOT_STATED". Never guess and never use outside knowledge. \
Return ONLY valid JSON."""

ANSWER_USER_PROMPT = """Transcript:
{transcript}

Questions:
{questions}

Return JSON of this exact shape:
{{"answers": [{{"id": <int>, "answer": "<short answer from the transcript, or NOT_STATED>"}}]}}
Answer every question id exactly once."""


@dataclass(frozen=True)
class SummQItem:
    question: str
    summary_answer: str       # reference answer extracted from the summary (call 1)
    transcript_answer: str    # answer found in the transcript, blind (call 2)
    consistent: bool          # deterministic agreement of the two answers


@dataclass(frozen=True)
class SummQResult:
    evaluated: bool
    reason: str = ''                                  # why skipped, when evaluated=False
    items: list[SummQItem] = field(default_factory=list)
    consistency_score: float | None = None            # fraction of consistent answers
    n_questions: int = 0
    n_consistent: int = 0
    threshold: float = 0.7

    def to_metadata(self) -> dict:
        # below_threshold is recorded even with no apply-path so the deferred
        # regeneration decision has data once real summaries accumulate.
        below = self.consistency_score is not None and self.consistency_score < self.threshold
        return {
            'evaluated': self.evaluated,
            'reason': self.reason,
            'consistency_score': self.consistency_score,
            'threshold': self.threshold,
            'below_threshold': below,
            'n_questions': self.n_questions,
            'n_consistent': self.n_consistent,
            'items': [asdict(i) for i in self.items],
        }


# ---- deterministic answer comparison (no LLM, no anchoring) -------------------

_WORD_RE = re.compile(r'\w+', re.UNICODE)
_DIGITS_RE = re.compile(r'\d[\d.]*')
_NOT_STATED_MARKERS = {
    '', 'not_stated', 'notstated', 'not stated', 'unknown', 'n/a', 'na', 'none',
    'not mentioned', 'not specified', 'unspecified', 'нет', 'не указано', 'не упоминается',
}


def _is_not_stated(text: str) -> bool:
    return text.strip().lower() in _NOT_STATED_MARKERS


def _norm(text: str) -> str:
    """Lowercase, fold spoken numbers to digits, and drop grouping commas so
    '1,000' and '1000' (and 'fifteen' / '15') compare equal."""
    return spoken_to_digits(text.lower()).replace(',', '')


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(_norm(text)))


def _numbers(text: str) -> set[str]:
    """Numeric tokens after normalization; a trailing '.' is trimmed."""
    return {n.rstrip('.') for n in _DIGITS_RE.findall(_norm(text))}


def _answer_match(summary_answer: str, transcript_answer: str, threshold: float = 0.5) -> bool:
    """Deterministic, anchoring-free comparison of two short extractive answers.

    A 'not stated' transcript answer is always a mismatch — the summary asserted
    something the transcript does not. Numbers are gated hard: every number in the
    summary's answer must appear in the transcript's answer, so a fabricated figure
    (80M where the transcript says 50M, or nothing) is always inconsistent even
    when scale words like 'million' overlap. The remaining (non-numeric) tokens are
    matched by subset or token-set F1."""
    if _is_not_stated(transcript_answer):
        return False
    # Numeric agreement is mandatory: a summary number absent from the transcript
    # answer is a fabrication, regardless of surrounding word overlap.
    s_nums = _numbers(summary_answer)
    if s_nums and not s_nums <= _numbers(transcript_answer):
        return False
    s, t = _tokens(summary_answer), _tokens(transcript_answer)
    if not s or not t:
        return False
    if s <= t or t <= s:
        return True
    inter = len(s & t)
    if inter == 0:
        return False
    precision, recall = inter / len(t), inter / len(s)
    return (2 * precision * recall / (precision + recall)) >= threshold


# ---- LLM plumbing ------------------------------------------------------------

def _summq_complete(messages: list[dict]) -> str:
    """One judge LLM call. Uses the same model/credentials as the Tier-2 judge
    (summarizer ≠ judge); SummQ is just a second, independent use of it."""
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


def _has_claims(summary: Summary) -> bool:
    return any(_render_item(item) for fld in _QUIZ_FIELDS for item in getattr(summary, fld))


def _generate_questions(summary: Summary, max_q: int) -> list[tuple[str, str]]:
    """Call 1: summary -> [(question, reference_answer)]. Keeps only well-formed
    pairs (both non-empty) and caps the count."""
    rendered = {fld: [_render_item(i) for i in getattr(summary, fld)] for fld in _QUIZ_FIELDS}
    content = _summq_complete([
        {'role': 'system', 'content': GEN_SYSTEM_PROMPT},
        {'role': 'user', 'content': GEN_USER_PROMPT.format(summary=rendered, max_q=max_q)},
    ])
    parsed = _parse_json(content)
    pairs: list[tuple[str, str]] = []
    for entry in parsed.get('questions', []):
        if not isinstance(entry, dict):
            continue
        question = str(entry.get('question', '')).strip()
        answer = str(entry.get('answer', '')).strip()
        if question and answer:
            pairs.append((question, answer))
        if len(pairs) >= max_q:
            break
    return pairs


def _answer_from_transcript(transcript: str, pairs: list[tuple[str, str]]) -> dict[int, str]:
    """Call 2: answer each question from the transcript ALONE. The summary's
    reference answer is deliberately never shown — that is what makes SummQ
    independent of the claim judge and free of confirmation bias."""
    numbered = '\n'.join(f'{i}. {q}' for i, (q, _ref) in enumerate(pairs))
    content = _summq_complete([
        {'role': 'system', 'content': ANSWER_SYSTEM_PROMPT},
        {'role': 'user', 'content': ANSWER_USER_PROMPT.format(transcript=transcript, questions=numbered)},
    ])
    parsed = _parse_json(content)
    # v1: assumes the model echoes integer ids matching our enumeration. A dropped
    # or string-keyed entry falls through to '' below (NOT_STATED → inconsistent),
    # a conservative default. Harden when swapping judge models via OpenRouter.
    answers: dict[int, str] = {}
    for entry in parsed.get('answers', []):
        if isinstance(entry, dict) and isinstance(entry.get('id'), int):
            answers[entry['id']] = str(entry.get('answer', ''))
    return answers


def verify_summq_consistency(summary: Summary, transcript: str) -> SummQResult:
    """Score a summary's QA-consistency against the transcript in two judge calls.

    Skips (evaluated=False) when the summary has no checkable claims, the
    transcript exceeds the judge token budget, or no questions could be generated.
    Language-agnostic — the LLM handles any language; the deterministic compare
    folds spoken numbers to digits via the shared ITN helper."""
    config: LLMConfig = get_llm_cfg()
    threshold = config.summq_consistency_threshold
    if not _has_claims(summary):
        return SummQResult(evaluated=False, reason='no claims to quiz', threshold=threshold)
    estimated = _estimate_tokens(transcript)
    if estimated > config.judge_token_limit:
        return SummQResult(
            evaluated=False,
            reason=f'transcript too long for judge ({estimated} > {config.judge_token_limit} tokens)',
            threshold=threshold,
        )

    pairs = _generate_questions(summary, config.summq_max_questions)
    if not pairs:
        return SummQResult(evaluated=False, reason='no questions generated', threshold=threshold)
    answers = _answer_from_transcript(transcript, pairs)

    items: list[SummQItem] = []
    for i, (question, reference) in enumerate(pairs):
        transcript_answer = answers.get(i, '')
        items.append(SummQItem(
            question=question,
            summary_answer=reference,
            transcript_answer=transcript_answer,
            consistent=_answer_match(reference, transcript_answer),
        ))
    n = len(items)
    n_consistent = sum(1 for it in items if it.consistent)
    return SummQResult(
        evaluated=True,
        items=items,
        consistency_score=n_consistent / n if n else None,
        n_questions=n,
        n_consistent=n_consistent,
        threshold=threshold,
    )
