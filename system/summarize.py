import concurrent.futures
import functools
import json
from dataclasses import dataclass
from typing import List

from litellm import completion

from system.config import LLMConfig, get_llm_cfg


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


def _parse_json(text: str) -> dict:
    formated_text = text[text.find('{'):text.rfind('}') + 1]
    text_dict = json.loads(formated_text)
    return text_dict

def summarize_chunk(chunk: str, config: LLMConfig) -> dict:
    config: LLMConfig = get_llm_cfg()
    response = completion(
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_retries=config.max_retries,
        temperature=config.temperature,
        messages=[{
            'role': 'user',
            'content': f'Make summary of text. Response lang: {config.lang}. Return response in json format with fields: "decisions", "action_items", "risks". Text: {chunk}'}]
    )
    summary_chunk = _parse_json(response.choices[0].message.content)
    return summary_chunk

def build_summary(chunks: List[str]) -> Summary:
    config: LLMConfig = get_llm_cfg()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summary_chunks = list(executor.map(summarize_chunk, chunks))

    response = completion(
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_retries=config.max_retries,
        temperature=config.temperature,
        messages=[{
            'role': 'user',
            'content': f'Make final summary from chunks. Lang of response:{config.lang}. Return ONLY valid JSON, no markdown, no explanation with fields: "title, "tldr", "decisions", "action_items", "risks". Field "title" must contains upon 10 words. Chunks: {summary_chunks}'
        }]
    )
    summary_dict = _parse_json(response.choices[0].message.content)

    summary = Summary(
        title=summary_dict['title'],
        tldr=summary_dict['tldr'],
        decisions=summary_dict['decisions'],
        action_items=summary_dict['action_items'],
        risks=summary_dict['risks']
    )
    return summary
