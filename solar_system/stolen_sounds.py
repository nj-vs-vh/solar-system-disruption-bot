import subprocess
import signal
import time
from pathlib import Path


WRECKAGE_SYSTEM_STREAM_URL = "https://wreckage-systems.club/radio/8000/stream192.mp3"


def download(output: Path, duration: float):
    print("Downloading stream...")
    cmd = [
        "ffmpeg",
        "-t",
        str(duration + 10),
        "-i",
        str(WRECKAGE_SYSTEM_STREAM_URL),
        "-c",
        "copy",
        "-map",
        "a",
        str(output),
        "-y",
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd)


def add_audio(simulation_vid: Path, audio: Path, output: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(simulation_vid),
            "-i",
            str(audio),
            "-c",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output),
            "-y",
        ]
    )


if __name__ == "__main__":
    test_audio = Path("audio.mp4")
    # download(test_audio, 120)
    add_audio(Path("simulation.mp4"), test_audio, Path("simulation-with-sound.mp4"))
