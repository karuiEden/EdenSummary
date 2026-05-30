from pathlib import Path
from typing import List

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from eden_summary.core.config import WhisperConfig, get_whisper_cfg


def transcribe(audio_path: Path) -> List[Segment]:
    config: WhisperConfig = get_whisper_cfg()
    model = WhisperModel(config.model, device=config.device, compute_type=config.compute_type)
    segments, _ = model.transcribe(audio=str(audio_path), language=config.lang, vad_filter=True)
    return list(segments)


def chunk_segments(segments: List[Segment]) -> List[str]:
    max_chars = get_whisper_cfg().chunk_max_chars
    chunk = ''
    chunks = []
    for segment in segments:
        chunk += segment.text
        if len(chunk) >= max_chars:
            chunks.append(chunk)
            chunk = ''
    if chunk != '':
        chunks.append(chunk)
    return chunks