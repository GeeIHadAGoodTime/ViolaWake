"""
violawake-collect CLI — Record positive wake word samples for model training.

Entry point: ``violawake-collect`` (declared in pyproject.toml).

Usage::

    violawake-collect --word "jarvis" --output data/jarvis/positives/ --count 50

    Records 50 1.5-second audio clips of the wake word "jarvis" to the given
    output directory. Each clip is saved as positives/sample_001.wav, etc.

    The recording loop displays a countdown timer, so you can pace your recordings:
    each clip will record automatically when the timer reaches 0.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path


def _record_clip(sample_rate: int = 16_000, duration: float = 1.5) -> bytes | None:
    """Record a single audio clip from the default microphone."""
    try:
        import pyaudio
    except ImportError:
        print("ERROR: pyaudio is required. Install with: pip install pyaudio", file=sys.stderr)
        return None

    pa = pyaudio.PyAudio()
    n_samples = int(sample_rate * duration)

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=n_samples,
        )
        data = stream.read(n_samples, exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
        return data
    except Exception as e:
        print(f"ERROR: Could not record from microphone: {e}", file=sys.stderr)
        return None
    finally:
        pa.terminate()


def _save_wav(data: bytes, path: Path, sample_rate: int = 16_000) -> None:
    """Save raw PCM bytes as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-collect",
        description="Record positive wake word samples for model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--word",
        required=True,
        metavar="WORD",
        help="The wake word you are recording (used for display only)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        metavar="N",
        help="Number of samples to record (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        metavar="SEC",
        help="Duration of each recording in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Pause between recordings in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        metavar="HZ",
        help="Sample rate in Hz (default: 16000)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing samples to continue numbering
    existing = sorted(output_dir.glob("sample_*.wav"))
    start_idx = len(existing) + 1

    print(f"Recording '{args.word}' wake word samples")
    print(f"Output: {output_dir}")
    print(f"Count: {args.count} | Duration: {args.duration}s | Delay: {args.delay}s")
    print()
    print("Press Ctrl+C to stop early.")
    print()

    recorded = 0
    try:
        for i in range(start_idx, start_idx + args.count):
            filename = f"sample_{i:04d}.wav"
            path = output_dir / filename

            # Countdown
            print(f"[{i}/{start_idx + args.count - 1}] Ready in ", end="", flush=True)
            for t in range(int(args.delay), 0, -1):
                print(f"{t}... ", end="", flush=True)
                time.sleep(1.0)

            print(f"SAY '{args.word}' NOW!", end=" ", flush=True)
            data = _record_clip(args.sample_rate, args.duration)

            if data is not None:
                _save_wav(data, path, args.sample_rate)
                print(f"✓ {filename}")
                recorded += 1
            else:
                print("✗ (recording failed)")

    except KeyboardInterrupt:
        print()
        print("Recording stopped early.")

    print()
    print(f"Recorded {recorded} samples to {output_dir}")

    if recorded < 20:
        print()
        print("TIP: For good model accuracy, collect at least 50 samples.")
        print("     Use different speaking styles, distances, and room positions.")


if __name__ == "__main__":
    main()
