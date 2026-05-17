from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

X_API_KEY = os.getenv("X_API_KEY")

audio_formats = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4a": "audio/mp4",
}

@dataclass(frozen=True)
class WhisperConfig:
    model: str
    lang: str
    device: str
    compute_type: str

    @classmethod
    def from_env(cls):
        return cls(
            model = os.getenv('WHISPER_MODEL', 'large-v3'),
            lang = os.getenv('WHISPER_LANGUAGE', 'ru'),
            device = os.getenv('WHISPER_DEVICE', 'auto'),
            compute_type = os.getenv('WHISPER_COMPUTE_TYPE', 'auto')
        )
