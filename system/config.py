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

