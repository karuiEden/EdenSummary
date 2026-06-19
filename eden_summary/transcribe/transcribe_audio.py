import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import litellm

from eden_summary.core import WhisperConfig, get_whisper_cfg
from eden_summary.transcribe.audio import get_duration, split_audio

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

def _transcribe_file(audio_path: Path, language: str | None) -> Transcription:
    """One ASR request for a file that already fits the provider's size cap."""
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


def transcribe(audio_path: Path, language: str | None) -> Transcription:
    """Transcribe audio, splitting long inputs so each ASR request stays under the
    provider's per-file size cap (OpenAI/Groq ~25 MB). Short files take a single
    request — behaviour unchanged. Longer ones are cut into asr_chunk_seconds pieces,
    transcribed sequentially (Groq TPM), and the segment texts concatenated in order.
    Timestamps are not offset: Transcription carries text only. Boundary cuts can
    clip a word at a seam — a known limitation; VAD/silence-aware split is the future
    upgrade."""
    config: WhisperConfig = get_whisper_cfg()
    if get_duration(audio_path) <= config.asr_chunk_seconds:
        return _transcribe_file(audio_path, language)

    segments: List[str] = []
    detected: str | None = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        for chunk in split_audio(audio_path, config.asr_chunk_seconds, Path(tmp_dir)):
            part = _transcribe_file(chunk, language)
            segments.extend(part.segments)
            if detected is None:
                detected = part.language
    return Transcription(segments=segments, language=detected)



def chunk_segments(segments: List[str]) -> List[str]:
    cfg = get_whisper_cfg()
    max_chars = cfg.chunk_max_chars
    overlap_chars = cfg.chunk_overlap_chars
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    new_since_flush = False
    for segment in segments:
        current.append(segment)
        current_len += len(segment)
        new_since_flush = True
        if current_len >= max_chars:
            chunks.append(''.join(current))
            new_since_flush = False
            # seed the next chunk with the trailing ~overlap_chars of segments
            overlap: List[str] = []
            overlap_len = 0
            for seg in reversed(current):
                if overlap_len >= overlap_chars:
                    break
                overlap.insert(0, seg)
                overlap_len += len(seg)
            current = overlap
            current_len = overlap_len
    if current and new_since_flush:
        chunks.append(''.join(current))
    return chunks