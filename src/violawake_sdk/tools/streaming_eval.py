"""Streaming FAPH evaluation -- process continuous audio chunk-by-chunk.

Simulates real-time wake word detection on continuous audio to measure
false accepts per hour (FAPH). Unlike batch evaluation which processes
clips independently, this feeds audio through the detector frame-by-frame
as it would happen in production.

Usage::

    # Evaluate FAPH on a long audio file
    python -m violawake_sdk.tools.streaming_eval \\
        --audio data/streaming/stream_0000.wav \\
        --model temporal_cnn --threshold 0.80

    # Evaluate FAPH on an entire directory of streaming files
    python -m violawake_sdk.tools.streaming_eval \\
        --audio-dir data/streaming/ \\
        --model temporal_cnn --threshold 0.80 --confirm 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def streaming_faph(
    detector,
    audio_path: str,
    frame_ms: int = 20,
    sample_rate: int = 16000,
) -> dict:
    """Measure FAPH by processing audio file in streaming fashion.

    Simulates real-time wake word detection on continuous audio.
    The detector's ``detect()`` method is called for each frame, which
    applies the full 4-gate decision policy (RMS floor, threshold,
    cooldown, playback suppression) plus optional multi-window confirmation.

    Args:
        detector: A ``WakeDetector`` instance (already initialized).
        audio_path: Path to a WAV audio file (must be 16kHz mono).
        frame_ms: Frame duration in milliseconds (default 20, matching production).
        sample_rate: Expected sample rate (default 16000).

    Returns:
        Dict with:
            - ``total_hours``: Duration of audio processed in hours.
            - ``n_false_accepts``: Number of detection triggers.
            - ``faph``: False accepts per hour.
            - ``trigger_timestamps``: List of timestamps (seconds) where triggers occurred.

    Raises:
        ValueError: If audio sample rate doesn't match expected rate.
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for streaming evaluation. Install with: pip install soundfile"
        ) from None

    import numpy as np

    audio, sr = sf.read(audio_path, dtype="int16")
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate}Hz, got {sr}Hz")

    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)

    frame_samples = int(sample_rate * frame_ms / 1000)
    triggers: list[float] = []

    # Reset detector state before processing
    detector.reset()

    for start in range(0, len(audio) - frame_samples + 1, frame_samples):
        frame = audio[start : start + frame_samples]
        if detector.detect(frame.tobytes()):
            timestamp = start / sample_rate
            triggers.append(timestamp)

    total_hours = len(audio) / sample_rate / 3600
    return {
        "total_hours": round(total_hours, 4),
        "n_false_accepts": len(triggers),
        "faph": round(len(triggers) / total_hours, 2) if total_hours > 0 else 0,
        "trigger_timestamps": [round(t, 3) for t in triggers],
    }


def streaming_faph_directory(
    detector,
    audio_dir: str,
    frame_ms: int = 20,
    sample_rate: int = 16000,
    verbose: bool = True,
) -> dict:
    """Measure FAPH across all WAV files in a directory.

    Processes each file independently (resetting detector between files)
    and aggregates results.

    Args:
        detector: A ``WakeDetector`` instance.
        audio_dir: Directory containing WAV files.
        frame_ms: Frame duration in milliseconds.
        sample_rate: Expected sample rate.
        verbose: Print per-file progress.

    Returns:
        Dict with aggregate FAPH, per-file results, and total hours.
    """
    audio_files = sorted(list(Path(audio_dir).rglob("*.wav")))

    if not audio_files:
        return {
            "error": f"No WAV files found in {audio_dir}",
            "total_hours": 0,
            "n_false_accepts": 0,
            "faph": 0,
        }

    total_hours = 0.0
    total_triggers = 0
    per_file_results: dict[str, dict] = {}
    errors: list[str] = []

    for i, f in enumerate(audio_files):
        if verbose:
            print(f"  [{i + 1}/{len(audio_files)}] Processing {f.name}...", end="", flush=True)

        try:
            result = streaming_faph(detector, str(f), frame_ms, sample_rate)
            per_file_results[str(f)] = result
            total_hours += result["total_hours"]
            total_triggers += result["n_false_accepts"]

            if verbose:
                n_fa = result["n_false_accepts"]
                dur = result["total_hours"] * 3600
                suffix = f" ({n_fa} triggers)" if n_fa > 0 else ""
                print(f" {dur:.1f}s{suffix}")

        except Exception as e:
            errors.append(f"{f.name}: {e}")
            if verbose:
                print(f" ERROR: {e}")

    aggregate_faph = total_triggers / total_hours if total_hours > 0 else 0

    return {
        "total_hours": round(total_hours, 4),
        "n_false_accepts": total_triggers,
        "faph": round(aggregate_faph, 2),
        "n_files": len(audio_files),
        "n_errors": len(errors),
        "errors": errors,
        "per_file": per_file_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-streaming-eval",
        description="Streaming FAPH evaluation — process continuous audio chunk-by-chunk.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", help="Path to a single WAV audio file")
    group.add_argument("--audio-dir", help="Directory of WAV files to process")

    parser.add_argument(
        "--model",
        default="temporal_cnn",
        help="Model name or path (default: temporal_cnn)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Detection threshold (default: 0.80)",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        default=3,
        help="Multi-window confirmation count (default: 3)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown between detections in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=20,
        help="Frame duration in milliseconds (default: 20)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    from violawake_sdk.wake_detector import WakeDetector

    print(
        f"Loading detector: model={args.model}, threshold={args.threshold}, confirm={args.confirm}"
    )
    detector = WakeDetector(
        model=args.model,
        threshold=args.threshold,
        confirm_count=args.confirm,
        cooldown_s=args.cooldown,
    )

    if args.audio:
        if not Path(args.audio).exists():
            print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)

        print(f"Processing: {args.audio}")
        result = streaming_faph(detector, args.audio, args.frame_ms)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\nResults:")
            print(
                f"  Duration:        {result['total_hours'] * 3600:.1f}s ({result['total_hours']:.4f}h)"
            )
            print(f"  False accepts:   {result['n_false_accepts']}")
            print(f"  FAPH:            {result['faph']}")
            if result["trigger_timestamps"]:
                print(f"  Trigger times:   {result['trigger_timestamps']}")

    elif args.audio_dir:
        if not Path(args.audio_dir).is_dir():
            print(f"ERROR: Directory not found: {args.audio_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Processing directory: {args.audio_dir}")
        result = streaming_faph_directory(
            detector,
            args.audio_dir,
            args.frame_ms,
            verbose=not args.json,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\nAggregate Results:")
            print(f"  Files processed: {result['n_files']}")
            print(
                f"  Total duration:  {result['total_hours'] * 3600:.1f}s ({result['total_hours']:.4f}h)"
            )
            print(f"  False accepts:   {result['n_false_accepts']}")
            print(f"  FAPH:            {result['faph']}")
            if result["n_errors"] > 0:
                print(f"  Errors:          {result['n_errors']}")


if __name__ == "__main__":
    main()
