import subprocess
import signal
import time
from pathlib import Path


WRECKAGE_SYSTEM_STREAM_URL = "https://www.youtube.com/watch?v=FauuCZlN2IA"


def download(output: Path, duration: float):
    print("Listing video formats")
    res = subprocess.run(["youtube-dl", "--list-formats", WRECKAGE_SYSTEM_STREAM_URL], capture_output=True)
    stdout = res.stdout.decode("utf-8")
    format_code = None
    for line in stdout.splitlines():
        parts = line.split()
        if parts[1] == 'mp4':
            format_code = parts[0]
            break
    if format_code is None:
        raise RuntimeError(f"Unable to list formats with youtube-dl, output:\n'{res.stdout}'")

    print("Obtaining manifest")
    res = subprocess.run(["youtube-dl", "-f", format_code, "-g", WRECKAGE_SYSTEM_STREAM_URL], capture_output=True)
    stdout = res.stdout.decode("utf-8")
    manifest_url = stdout.strip()
    print(format_code)

    print("Downloading stream...")
    ffmpeg = subprocess.Popen(["ffmpeg", "-i", manifest_url, "-c", "copy", str(output), "-y"])
    print(f"Sleeping for {duration}")
    time.sleep(duration + 15)
    ffmpeg.send_signal(signal.SIGINT)
    ffmpeg.wait()


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
