from dataclasses import dataclass
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from system.config import WhisperConfig



def transcribe(audio_path: Path, config: WhisperConfig) -> List[Segment]:
    model = WhisperModel(config.model, device=config.device, compute_type=config.compute_type)
    segments, _ = model.transcribe(audio=str(audio_path), language=config.lang, vad_filter=True)
    return list(segments)


def chunk_segments(segments: List[Segment], max_chars: int) -> List[str]:
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