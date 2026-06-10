import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from typing import List

from litellm import completion

from eden_summary.core import LLMConfig, get_llm_cfg

logger = logging.getLogger(__name__)


SYSTEM_PROMPT: str = """ You are a business meeting analyst. 
Extract structured information from meeting transcripts.
Follow these rules strictly:
- Extract ONLY information explicitly mentioned in the transcript
- Do not add recommendations, advice, or any information not present in the text
- If a field has no data, return an empty list
- Respond in the same language as the transcript
- Return ONLY valid JSON, no markdown, no explanations
"""

CHUNK_USER_PROMPT: str = """Analyze the following meeting transcript fragment and extract:
- decisions: concrete decisions that were made
- action_items: specific tasks assigned to someone, include who and deadline if mentioned
- risks: problems, blockers, or open questions raised

Transcript fragment:
{chunk}
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
"""

@dataclass(frozen=True)
class Summary:
    title: str
    tldr: List[str]
    decisions: List[str]
    action_items: List[str]
    risks: List[str]

    def to_text(self) -> str:
        headers = {
            'tldr': 'TL;DR',
            'decisions': 'Решения',
            'action_items': 'Задачи',
            'risks': 'Риски'
        }
        sections = []
        for key, header in headers.items():
            items = getattr(self, key)
            if not items:
                continue
            section = [header] + [f"- {item}" for item in items]
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

def summarize_chunk(chunk: str, _attempt: int = 0) -> dict:
    config: LLMConfig = get_llm_cfg()
    response = completion(
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_retries=config.max_retries,
        timeout=config.timeout,
        temperature=config.temperature,
        messages=[{
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': CHUNK_USER_PROMPT.format(chunk=chunk)
        }]
    )
    try:
        return _parse_json(str(response.choices[0].message.content))
    except (json.JSONDecodeError, ValueError) as e:
        if _attempt + 1 >= config.max_parse_attempts:
            logger.error(f"Failed to parse LLM output after {_attempt + 1} attempts: {e}")
            raise
        logger.warning(f'JSON parse error (attempt {_attempt}, retrying LLM call: {e}')
        time.sleep(0.5 * (_attempt + 1))
        return summarize_chunk(chunk, _attempt + 1)



def build_summary(chunks: List[str]) -> Summary:
    config: LLMConfig = get_llm_cfg()
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        summary_chunks = list(executor.map(summarize_chunk, chunks))

    response = completion(
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_retries=config.max_retries,
        temperature=config.temperature,
        timeout=config.timeout,
        messages=[{
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': REDUCE_USER_PROMPT.format(chunks=summary_chunks)
        }]
    )
    summary_dict = _parse_json(str(response.choices[0].message.content))
    logger.debug("Reduce response: %s", summary_dict)
    summary = Summary(
        title=summary_dict.get('title', 'Meeting Summary'),
        tldr=_ensure_list(summary_dict.get('tldr', [])),
        decisions=_ensure_list(summary_dict.get('decisions', [])),
        action_items=_ensure_list(summary_dict.get('action_items', [])),
        risks=_ensure_list(summary_dict.get('risks', [])),
    )
    return summary
