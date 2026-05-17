from dataclasses import dataclass
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from system.config import WhisperConfig

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str



def transcribe(audio_path: Path, config: WhisperConfig) -> List[Segment]:
    model = WhisperModel()