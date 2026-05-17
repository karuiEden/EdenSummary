import subprocess
from pathlib import Path


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-i", str(input_path), "-ar", "16000", "-ac", "1", "-y", str(output_path)], capture_output=True,
        check=True
    )

def get_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", '-i', str(path), '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0'], capture_output=True, check=True
    )
    duration = float(res.stdout)
    return duration