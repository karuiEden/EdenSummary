import subprocess
from pathlib import Path
from typing import List

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

def split_audio(input_path: Path, chunk_seconds: int, out_dir: Path) -> List[Path]:
    """Split a (16 kHz mono WAV) file into <=chunk_seconds pieces so each fits under
    the ASR API's per-file size cap. One ffmpeg pass via the segment muxer; we
    re-encode to pcm_s16le (not -c copy) because some ffmpeg builds emit segment WAVs
    with broken headers under stream-copy. Returns chunk paths in playback order."""
    pattern = str(out_dir / "chunk_%03d.wav")
    subprocess.run(
        ["ffmpeg", "-i", str(input_path), "-f", "segment", "-segment_time", str(chunk_seconds),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", pattern],
        capture_output=True,
        timeout=_FFMPEG_TIMEOUT,
        check=True
    )
    return sorted(out_dir.glob("chunk_*.wav"))