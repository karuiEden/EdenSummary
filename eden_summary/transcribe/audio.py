import subprocess
from pathlib import Path

_FFMPEG_TIMEOUT = 1800
_FFPROBE_TIMEOUT = 30

def convert_to_wav(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-i", str(input_path), "-ar", "16000", "-ac", "1", "-y", str(output_path)], capture_output=True,
        timeout=_FFMPEG_TIMEOUT,
        check=True
    )

def get_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", '-i', str(path), '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0'],
        capture_output=True, check=True,
        timeout=_FFPROBE_TIMEOUT
    )
    duration = float(res.stdout.decode().strip())
    return duration

def is_audio(path: str) -> bool:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True,
        timeout=_FFPROBE_TIMEOUT,
        text=True
    )
    return res.stdout.strip() == 'audio' and res.returncode == 0