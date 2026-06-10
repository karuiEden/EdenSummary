from dataclasses import dataclass
from pathlib import Path
from typing import List

import litellm

from eden_summary.core import WhisperConfig, get_whisper_cfg

litellm.drop_params = True
_ASR_PROTOCOL = "openai"
_BCP47_MAP = {
    'russian': 'ru', 'ru': 'ru',
    'english': 'en', 'en': 'en',
    'german': 'de', 'de': 'de',
    'french': 'fr', 'fr': 'fr',
    'spanish': 'es', 'es': 'es',
    'chinese': 'zh', 'zh': 'zh',
}
@dataclass
class Transcription:
    segments: List[str]
    language: str | None

def _to_bcp47(provider_lang: str | None) -> str | None:
    if not provider_lang:
        return None
    return _BCP47_MAP.get(provider_lang.lower())

def transcribe(audio_path: Path, language: str | None) -> Transcription:
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
            language=language or config.lang
        )
    raw_segments = getattr(response, 'segments', None) or []
    segments = [segment.get("text", "") for segment in raw_segments if segment.get("text", "").strip()]
    return Transcription(
        segments=segments,
        language=_to_bcp47(getattr(response, 'language', None))
    )



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