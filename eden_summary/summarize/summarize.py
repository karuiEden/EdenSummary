import concurrent.futures
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List

from litellm import completion

from eden_summary.core import LLMConfig, get_llm_cfg
from eden_summary.transcribe import chunk_segments

logger = logging.getLogger(__name__)


_NUM = r'\d[\d,]*(?:\.\d+)?'
_NUMBER_PATTERNS = [
    re.compile(rf'[€$£]\s?{_NUM}'),                                                # €25, $1,000.50
    re.compile(rf'{_NUM}\s*%'),                                                    # 50%
    re.compile(rf'{_NUM}\s+(?:percent|процент\w*)', re.I),                         # 20 percent, 34 процента
    re.compile(rf'{_NUM}\s+(?:hundred|thousand|million|billion|trillion'
               rf'|тысяч\w*|миллион\w*|миллиард\w*|триллион\w*)'                   # 15 million, 50 миллионов
               rf'(?:\s+(?:euros?|dollars?|pounds?|cents?|евро|рубл\w*|доллар\w*))?', re.I),
    re.compile(rf'{_NUM}\s+(?:euros?|dollars?|pounds?|cents?|usd|eur|gbp'
               rf'|евро|рубл\w*|доллар\w*)', re.I),                                # 25 euros, 25 евро
    re.compile(r'\b\d{1,2}:\d{2}\b'),                                              # 10:30
    re.compile(r'\b\d+\.\d+\b'),                                                   # 12.50
    re.compile(r'\b\d{1,3}(?:,\d{3})+\b'),                                         # 1,000
]


def _extract_numbers(text: str) -> list[str]:
    """Extract salient numeric facts (money, magnitudes, percentages, times,
    decimals) as their exact surface strings, de-duplicated and in order of
    appearance.

    Not used for prompt-side "numeric anchoring": injecting these with a hard
    "use ONLY these values" constraint measurably hurt faithfulness — see
    docs/ml-experiments.md (Q1a). Now consumed read-only by the Tier 1 inline
    quality guard (eden_summary/quality) to flag summary numbers that are not
    grounded in the transcript."""
    spans = []
    for pattern in _NUMBER_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), match.group()))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    selected = []
    last_end = -1
    for start, end, raw in spans:
        if start >= last_end:
            selected.append(raw)
            last_end = end
    out: list[str] = []
    seen = set()
    for raw in selected:
        norm = re.sub(r'\s+', ' ', raw.strip())
        if norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
    return out


SYSTEM_PROMPT: str = """ You are a business meeting analyst.                                                                                                                                                                    
  Extract structured information from meeting transcripts.                                                                                                                                                                        
  Follow these rules strictly:                                                                                                                                                                                                    
  - Extract ONLY information explicitly mentioned in the transcript                                                                                                                                                               
  - Do not add recommendations, advice, or any information not present in the text                                                                                                                                                
  - If a field has no data, return an empty list                                                                                                                                                                                  
  - An empty list is always better than a fabricated item                                                                                                                                                                         
  - Respond in the same language as the transcript                                                                                                                                                                                
  - Return ONLY valid JSON, no markdown, no explanations                                                                                                                                                                          
                                                                                                                                                                                                                                  
  Scoring: +1 correct item, −4 fabricated item, 0 omitted item. No explicit evidence → empty list. 
"""

CHUNK_USER_PROMPT: str = """Analyze the following meeting transcript fragment and extract:
- decisions: concrete decisions that were made
- action_items: specific tasks assigned to someone, include who and deadline if mentioned
- risks: problems, blockers, or open questions raised

Transcript fragment:
{chunk}

Omit any item you cannot directly quote from the fragment (omission scores 0; fabrication scores −4).
"""

REDUCE_USER_PROMPT: str = """
Merge the following chunk analyses into a single final meeting summary.
Remove duplicates, keep only what was explicitly mentioned.

Fields to produce:
- title: short meeting title, max 8 words, based on main topic discussed
- tldr: 3-5 bullet points summarizing the entire meeting
- decisions: all decisions made
- action_items: all tasks assigned
- risks: all problems and open questions

Chunk analyses:
{chunks}

Omit any item you cannot directly quote from the chunk analyses (omission scores 0; fabrication scores −4).
"""

SINGLE_PASS_PROMPT: str = """Analyze the following complete meeting transcript and produce a structured summary.

Pay equal attention to the beginning, middle, and end of the meeting — important decisions
and action items are often raised in the middle of a discussion and must not be dropped.

Fields to produce:
- title: short meeting title, max 8 words, based on the main topic discussed
- tldr: 3-5 bullet points summarizing the entire meeting
- decisions: concrete decisions that were made
- action_items: specific tasks assigned to someone, include who and deadline if mentioned
- risks: problems, blockers, or open questions raised

Transcript:
{transcript}

Omit any item you cannot directly quote from the transcript (omission scores 0; fabrication scores −4).
"""

_DEFAULT_LOCALE = 'ru'

_HEADERS: dict[str, dict[str, str]] = {
    'ru': {
        'tldr': 'TL;DR',
        'decisions': 'Решения',
        'action_items': 'Задачи',
        'risks': 'Риски',
    },
    'en': {
        'tldr': 'TL;DR',
        'decisions': 'Decisions',
        'action_items': 'Action Items',
        'risks': 'Risks',
    },
}


def _primary_subtag(bcp47: str | None) -> str | None:
    if not bcp47:
        return None
    return bcp47.split('-')[0].lower()


_PLACEHOLDERS = {'', 'unspecified', 'none', 'n/a', 'na', 'tbd',
                 'not specified', 'not mentioned', 'unknown', 'null'}


def _clean(value: object) -> str | None:
    """Strip a value to text, treating placeholder fillers as absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text.lower() not in _PLACEHOLDERS else None


def _render_item(item: object) -> str:
    """Render a list item that may be a plain string or a structured dict
    (e.g. action items as {task, who, deadline}) into one readable line."""
    if isinstance(item, dict):
        task = _clean(item.get('task') or item.get('action')
                      or item.get('item') or item.get('description'))
        if task is None:
            # unknown dict shape — join whatever non-placeholder values remain
            return ', '.join(v for v in (_clean(x) for x in item.values()) if v)
        who = _clean(item.get('who') or item.get('owner')
                     or item.get('assignee') or item.get('responsible'))
        deadline = _clean(item.get('deadline') or item.get('due')
                          or item.get('by') or item.get('when'))
        meta = []
        if who:
            meta.append(who)
        if deadline:
            meta.append(f'by {deadline}')
        return f'{task} ({"; ".join(meta)})' if meta else task
    return str(item).strip()


@dataclass(frozen=True)
class Summary:
    title: str
    tldr: List[str]
    decisions: List[str]
    action_items: List[str]
    risks: List[str]

    def to_text(self, lang: str | None = None) -> str:
        locale = _primary_subtag(lang) or _DEFAULT_LOCALE
        headers = _HEADERS.get(locale, _HEADERS[_DEFAULT_LOCALE])
        sections = []
        for key, header in headers.items():
            items = getattr(self, key)
            if not items:
                continue
            rendered = [text for text in (_render_item(item) for item in items) if text]
            if not rendered:
                continue
            section = [header] + [f"- {text}" for text in rendered]
            sections.append("\n".join(section))
        return "\n\n".join(sections)

def _ensure_list(value: List[str] | str | None) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return []

def _parse_json(text: str) -> dict:
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError('JSON object not found')
    return json.loads(text[start:end+1])

def _dict_to_summary(summary_dict: dict) -> Summary:
    return Summary(
        title=summary_dict.get('title', 'Meeting Summary'),
        tldr=_ensure_list(summary_dict.get('tldr', [])),
        decisions=_ensure_list(summary_dict.get('decisions', [])),
        action_items=_ensure_list(summary_dict.get('action_items', [])),
        risks=_ensure_list(summary_dict.get('risks', [])),
    )

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English) — enough to choose
    single-pass vs map-reduce without pulling in a tokenizer dependency."""
    return len(text) // 4

def _complete(messages: list[dict]) -> str:
    """Single entry point for LLM calls. Centralizes the shared completion kwargs
    and, when json_mode is on, requests native JSON output (response_format) so
    the model cannot return non-JSON — the _parse_json retry below stays only as
    a safety net. JSON mode requires the word 'json' to appear in the prompt;
    SYSTEM_PROMPT already instructs 'Return ONLY valid JSON'."""
    config: LLMConfig = get_llm_cfg()
    kwargs: dict = dict(
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_retries=config.max_retries,
        timeout=config.timeout,
        temperature=config.temperature,
        messages=messages,
    )
    if config.json_mode:
        kwargs['response_format'] = {'type': 'json_object'}
    response = completion(**kwargs)
    return str(response.choices[0].message.content)

def summarize_chunk(chunk: str, _attempt: int = 0) -> dict:
    config: LLMConfig = get_llm_cfg()
    content = _complete([
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': CHUNK_USER_PROMPT.format(chunk=chunk)},
    ])
    try:
        return _parse_json(content)
    except (json.JSONDecodeError, ValueError) as e:
        if _attempt + 1 >= config.max_parse_attempts:
            logger.error(f"Failed to parse LLM output after {_attempt + 1} attempts: {e}")
            raise
        logger.warning(f'JSON parse error (attempt {_attempt}, retrying LLM call: {e}')
        time.sleep(0.5 * (_attempt + 1))
        return summarize_chunk(chunk, _attempt + 1)



def build_summary(chunks: List[str], _attempt: int = 0) -> Summary:
    config: LLMConfig = get_llm_cfg()
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        summary_chunks = list(executor.map(summarize_chunk, chunks))

    content = _complete([
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': REDUCE_USER_PROMPT.format(chunks=summary_chunks)},
    ])
    try:
        summary_dict = _parse_json(content)
    except (json.JSONDecodeError, ValueError) as e:
        if _attempt + 1 >= config.max_parse_attempts:
            logger.error(f"Failed to parse LLM output after {_attempt + 1} attempts: {e}")
            raise
        logger.warning(f'JSON parse error (attempt {_attempt}, retrying LLM call: {e}')
        time.sleep(0.5 * (_attempt + 1))
        return build_summary(chunks, _attempt + 1)
    logger.debug("Reduce response: %s", summary_dict)
    return _dict_to_summary(summary_dict)


def _summarize_single_pass(transcript: str, _attempt: int = 0) -> Summary:
    """Summarize the whole transcript in one LLM call (no map-reduce). Used when
    the transcript fits the model context — avoids the information loss that the
    map→reduce boundary introduces (see docs/ml-experiments.md, Q1a)."""
    config: LLMConfig = get_llm_cfg()
    content = _complete([
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': SINGLE_PASS_PROMPT.format(transcript=transcript)},
    ])
    try:
        summary_dict = _parse_json(content)
    except (json.JSONDecodeError, ValueError) as e:
        if _attempt + 1 >= config.max_parse_attempts:
            logger.error(f"Failed to parse LLM output after {_attempt + 1} attempts: {e}")
            raise
        logger.warning(f'JSON parse error (attempt {_attempt}, retrying LLM call: {e}')
        time.sleep(0.5 * (_attempt + 1))
        return _summarize_single_pass(transcript, _attempt + 1)
    return _dict_to_summary(summary_dict)


def summarize_transcript(segments: List[str]) -> Summary:
    """Summarize a transcript. Single pass when it fits the model context,
    map-reduce as the fallback for longer transcripts."""
    config: LLMConfig = get_llm_cfg()
    full_text = '\n'.join(segments)
    estimated = _estimate_tokens(full_text)
    if estimated <= config.single_pass_token_limit:
        logger.info("Summarizing in a single pass (~%d tokens)", estimated)
        return _summarize_single_pass(full_text)
    logger.info("Transcript ~%d tokens exceeds single-pass limit; using map-reduce", estimated)
    return build_summary(chunk_segments(segments))
