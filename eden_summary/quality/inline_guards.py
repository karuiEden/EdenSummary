import logging
import re
from dataclasses import asdict, dataclass

from eden_summary.quality.numbers import spoken_to_digits
from eden_summary.summarize.summarize import Summary, _extract_numbers, _primary_subtag

logger = logging.getLogger(__name__)

# Languages whose spelled-out numbers spoken_to_digits can normalize. The number
# guard grounds summary digits against the transcript, which only works once both
# sides share a representation — so it must run ONLY for these languages. For any
# other language (the transcript spells numbers out in words we don't convert) the
# check would flag every number as ungrounded (~100% false positives), so unknown
# languages skip the number check rather than emit noise.
_NUMBER_GROUNDING_LANGS = {'en', 'ru'}


@dataclass(frozen=True)
class QualityFlag:
    """A single quality concern detected by an inline guard. Advisory only —
    flags are recorded on the job, they never block or fail delivery."""
    kind: str       # machine-readable category, e.g. 'ungrounded_number'
    severity: str   # 'warning' for now (all Tier 1 checks are advisory)
    detail: str     # human-readable explanation
    value: str      # the offending surface string from the summary


@dataclass(frozen=True)
class InlineGuardResult:
    flags: list[QualityFlag]
    checked: tuple[str, ...]  # names of the checks that actually ran

    @property
    def passed(self) -> bool:
        return not self.flags

    def to_metadata(self) -> dict:
        """Serialize for the job's quality_flags column. Always produced — an
        empty 'flags' with a non-empty 'checked' means 'ran, found nothing',
        which is distinct from a NULL column ('guards did not run')."""
        return {
            'passed': self.passed,
            'checked': list(self.checked),
            'flags': [asdict(flag) for flag in self.flags],
        }


def _normalize_commas(text: str) -> str:
    """Bring commas to one canonical form. A comma followed by 1-2 digits is a
    Russian decimal separator collapsed to its integer part ('12,50'->'12'); a
    comma before 3 digits is English grouping, dropped ('1,000'->'1000'). MUST be
    applied to both the summary number and the transcript so they compare in the
    same form — using it on only one side reintroduces a false positive when the
    transcript renders the decimal in digits."""
    text = re.sub(r'(\d),\d{1,2}(?!\d)', r'\1', text)
    return text.replace(',', '')


def _digit_core(surface: str) -> str:
    """Reduce a number's surface form to its bare digit token for grounding:
    normalize commas (see _normalize_commas) and drop surrounding currency/units,
    keep the numeric core (digits with an optional decimal point or time colon).
    '€25'->'25', '1,000'->'1000', '15 million'->'15', '12.50'->'12.50',
    '12,50'->'12' (split spoken decimals ground on the whole-number part)."""
    match = re.search(r'\d[\d.:]*', _normalize_commas(surface))
    return match.group().rstrip('.:') if match else ''


def _is_grounded(core: str, transcript_no_commas: str) -> bool:
    """True if the digit core appears in the transcript as a standalone number,
    not embedded in a longer one ('50' must not match inside '509'). Boundaries
    are digit-only on purpose: when a match is ambiguous we err toward 'grounded'
    so the guard stays high-precision (a noisy flag is worse than a missed one)."""
    if not core:
        return True
    return re.search(rf'(?<!\d){re.escape(core)}(?!\d)', transcript_no_commas) is not None


def _check_numbers(summary_text: str, transcript: str) -> list[QualityFlag]:
    """Flag numbers present in the summary but absent from the transcript —
    candidate fabrications (the −4 case under I-CALM). The reverse (a number in
    the transcript missing from the summary) is NOT flagged: omission scores 0,
    and recall is the summarizer's job, not this guard's.

    The transcript is first run through spoken_to_digits (lightweight ITN):
    speech transcripts spell numbers out ('twenty five euros') while summaries
    emit digits ('25 euros'), and without this every number would read as
    ungrounded — ~100% false positives on speech. Coverage gaps in that
    converter (e.g. Russian declension) leave
    their numbers unconverted and so remain residual false-positive sources, not
    false negatives — precision here is bounded by ITN coverage, not guaranteed."""
    transcript_no_commas = _normalize_commas(spoken_to_digits(transcript))
    flags: list[QualityFlag] = []
    for surface in _extract_numbers(summary_text):
        core = _digit_core(surface)
        if core and not _is_grounded(core, transcript_no_commas):
            flags.append(QualityFlag(
                kind='ungrounded_number',
                severity='warning',
                detail=f'Number "{surface}" in the summary was not found in the transcript',
                value=surface,
            ))
    return flags


_NUMBERS = 'numbers'


def run_inline_guards(summary: Summary, transcript: str,
                      language: str | None = None) -> InlineGuardResult:
    """Run all Tier 1 inline guards over a finished summary. Fast (<100ms),
    deterministic, advisory. Never raises: on any internal error it logs and
    returns an empty result with no checks recorded, so a guard failure can
    never fail an otherwise-good job. Grounding is checked against the summary
    text actually delivered (Summary.to_text()).

    The number check runs only for languages we can normalize (see
    _NUMBER_GROUNDING_LANGS); for others it is skipped and 'numbers' is left out
    of `checked` (an honest 'not verified', distinct from 'verified, clean')
    rather than emitting false positives."""
    try:
        checks: list[str] = []
        flags: list[QualityFlag] = []
        if _primary_subtag(language) in _NUMBER_GROUNDING_LANGS:
            flags += _check_numbers(summary.to_text(), transcript)
            checks.append(_NUMBERS)
        else:
            logger.info("Number guard skipped: language %r not normalizable", language)
        return InlineGuardResult(flags=flags, checked=tuple(checks))
    except Exception:
        logger.exception("Inline quality guards failed; returning no flags")
        return InlineGuardResult(flags=[], checked=())