from pathlib import Path
from typing import List

import litellm

from eden_summary.core import WhisperConfig, get_whisper_cfg

litellm.drop_params = True
_ASR_PROTOCOL = "openai"

def transcribe(audio_path: Path) -> List[str]:
    config: WhisperConfig = get_whisper_cfg()
    with open(audio_path, 'rb') as f:
        response = litellm.transcription(
            model=config.model,
            file=f,
            api_base=config.api_base,
            api_key=config.api_key,
            custom_llm_provider=_ASR_PROTOCOL,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=config.lang
        )
    segments = getattr(response, 'segments', None) or []
    return [segment.get("text", "") for segment in segments if segment.get("text", "").strip()]


def chunk_segments(segments: List[str]) -> List[str]:
    max_chars = get_whisper_cfg().chunk_max_chars
    chunk = ''
    chunks = []
    for segment in segments:
        chunk += segment
        if len(chunk) >= max_chars:
            chunks.append(chunk)
            chunk = ''
    if chunk != '':
        chunks.append(chunk)
    return chunks