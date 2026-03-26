"""CLI tool for wake word confusable false-accept testing.

Usage::

    python -m violawake_sdk.tools.test_confusables \
        --model models/viola_mlp_oww.onnx \
        --wake-word viola \
        --threshold 0.80
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from violawake_sdk.audio import load_audio
from violawake_sdk.tools.confusables import generate_confusable_tts_audio, generate_confusables
from violawake_sdk.wake_detector import FRAME_SAMPLES, WakeDetector

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3")


@dataclass(frozen=True)
class ConfusableTestResult:
    """Per-confusable score and pass/fail outcome."""

    word: str
    score: float | None
    passed: bool
    false_accept: bool
    audio_source: str
    error: str | None = None


@dataclass(frozen=True)
class ConfusableTestSummary:
    """Aggregate false-accept metrics for a confusable test run."""

    total_candidates: int
    total_tested: int
    false_accepts: int
    false_accept_rate: float
    worst_offenders: tuple[ConfusableTestResult, ...]


@dataclass(frozen=True)
class ConfusableTestReport:
    """Serializable report for a full confusables test run."""

    model_path: str
    wake_word: str
    threshold: float
    audio_backend: str
    results: tuple[ConfusableTestResult, ...]
    summary: ConfusableTestSummary


def _slugify(text: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in text.lower())
    return slug.strip("_") or "sample"


def _iter_candidate_audio_paths(audio_dir: Path, confusable: str) -> tuple[Path, ...]:
    normalized = " ".join(confusable.lower().split())
    stems = {
        confusable,
        normalized,
        normalized.replace(" ", "_"),
        normalized.replace(" ", "-"),
        _slugify(confusable),
    }

    paths: list[Path] = []
    for stem in stems:
        for extension in _AUDIO_EXTENSIONS:
            paths.append(audio_dir / f"{stem}{extension}")
    return tuple(paths)


def _load_audio_from_dir(confusable: str, audio_dir: Path, sample_rate: int) -> tuple[np.ndarray, str]:
    for candidate_path in _iter_candidate_audio_paths(audio_dir, confusable):
        if not candidate_path.exists():
            continue

        audio = load_audio(candidate_path, target_sr=sample_rate)
        if audio is None:
            raise RuntimeError(f"Failed to load audio file: {candidate_path}")
        return np.asarray(audio, dtype=np.float32), str(candidate_path)

    raise FileNotFoundError(f"No audio found for confusable '{confusable}' in {audio_dir}")


def _iter_audio_frames(audio: np.ndarray) -> list[np.ndarray]:
    flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flattened.size == 0:
        raise ValueError("Audio clip is empty")

    frame_count = max(1, int(np.ceil(flattened.size / FRAME_SAMPLES)))
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        start = index * FRAME_SAMPLES
        frame = flattened[start : start + FRAME_SAMPLES]
        if frame.size < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - frame.size))
        frames.append(frame.astype(np.float32, copy=False))
    return frames


def score_confusable_audio(detector: WakeDetector, audio: np.ndarray) -> float:
    """Score an audio clip by taking the maximum per-frame wake score."""
    best_score = 0.0
    for frame in _iter_audio_frames(audio):
        score = float(detector.process(frame))
        best_score = max(best_score, score)
    return best_score


def _format_score(score: float | None) -> str:
    if score is None:
        return "ERR"
    return f"{score:.3f}"


def _result_label(result: ConfusableTestResult) -> str:
    if result.error is not None:
        return "ERROR"
    if result.false_accept:
        return "FAIL"
    return "PASS"


def _format_results_table(results: tuple[ConfusableTestResult, ...]) -> str:
    headers = ("Confusable", "Score", "Pass/Fail", "Audio Source")
    rows = [
        (
            result.word,
            _format_score(result.score),
            _result_label(result),
            result.audio_source,
        )
        for result in results
    ]

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def _format_row(row: tuple[str, str, str, str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    divider = "  ".join("-" * width for width in widths)
    return "\n".join([_format_row(headers), divider, *(_format_row(row) for row in rows)])


def _build_summary(
    results: tuple[ConfusableTestResult, ...],
    top_k: int = 5,
) -> ConfusableTestSummary:
    tested_results = tuple(result for result in results if result.error is None)
    false_accepts = tuple(result for result in tested_results if result.false_accept)
    worst_offenders = tuple(
        sorted(
            tested_results,
            key=lambda result: result.score if result.score is not None else -1.0,
            reverse=True,
        )[:top_k]
    )

    total_tested = len(tested_results)
    false_accept_rate = (len(false_accepts) / total_tested) if total_tested else 0.0
    return ConfusableTestSummary(
        total_candidates=len(results),
        total_tested=total_tested,
        false_accepts=len(false_accepts),
        false_accept_rate=false_accept_rate,
        worst_offenders=worst_offenders,
    )


def report_to_dict(report: ConfusableTestReport) -> dict[str, object]:
    """Convert a confusable report into a JSON-friendly dictionary."""
    return asdict(report)


def run_confusable_tests(
    model_path: str | Path,
    wake_word: str,
    threshold: float = 0.80,
    count: int = 50,
    audio_dir: str | Path | None = None,
    voice: str = "af_heart",
    sample_rate: int = 16_000,
) -> ConfusableTestReport:
    """Run the confusable false-accept test suite for a wake word model."""
    confusables = generate_confusables(wake_word, count=count)
    if not confusables:
        raise ValueError(f"No confusables generated for wake word '{wake_word}'")

    detector = WakeDetector(model=str(model_path), threshold=threshold)
    resolved_audio_dir = Path(audio_dir) if audio_dir is not None else None

    kokoro_audio: dict[str, np.ndarray] = {}
    audio_backend = "audio_dir"
    try:
        kokoro_audio = generate_confusable_tts_audio(
            confusables,
            voice=voice,
            sample_rate=sample_rate,
        )
        audio_backend = "kokoro"
    except Exception as exc:
        logger.info("Kokoro TTS unavailable for confusable test: %s", exc)
        if resolved_audio_dir is None:
            raise RuntimeError(
                "kokoro-onnx is unavailable and no --audio-dir was provided"
            ) from exc
        audio_backend = "audio_dir"

    results: list[ConfusableTestResult] = []
    for confusable in confusables:
        try:
            if resolved_audio_dir is not None:
                try:
                    audio, source = _load_audio_from_dir(
                        confusable,
                        resolved_audio_dir,
                        sample_rate=sample_rate,
                    )
                except FileNotFoundError:
                    if confusable not in kokoro_audio:
                        raise
                    audio = kokoro_audio[confusable]
                    source = "kokoro"
                    audio_backend = "audio_dir+kokoro"
                else:
                    if audio_backend == "kokoro":
                        audio_backend = "audio_dir+kokoro"
            else:
                audio = kokoro_audio[confusable]
                source = "kokoro"

            score = score_confusable_audio(detector, audio)
            false_accept = score >= threshold
            results.append(
                ConfusableTestResult(
                    word=confusable,
                    score=score,
                    passed=not false_accept,
                    false_accept=false_accept,
                    audio_source=source,
                )
            )
        except Exception as exc:
            logger.warning("Failed to score confusable %s: %s", confusable, exc)
            results.append(
                ConfusableTestResult(
                    word=confusable,
                    score=None,
                    passed=False,
                    false_accept=False,
                    audio_source="unavailable",
                    error=str(exc),
                )
            )

    ordered_results = tuple(
        sorted(
            results,
            key=lambda result: (
                result.score if result.score is not None else -1.0,
                result.word,
            ),
            reverse=True,
        )
    )
    summary = _build_summary(ordered_results)
    return ConfusableTestReport(
        model_path=str(model_path),
        wake_word=wake_word,
        threshold=threshold,
        audio_backend=audio_backend,
        results=ordered_results,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-confusables",
        description="Generate phonetic confusables and test false accepts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to the wake word ONNX model",
    )
    parser.add_argument(
        "--wake-word",
        required=True,
        metavar="WORD",
        help="Wake word to generate confusables for",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        metavar="FLOAT",
        help="False-accept threshold (default: 0.80)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of confusables to test (default: 50)",
    )
    parser.add_argument(
        "--audio-dir",
        metavar="DIR",
        help="Directory containing prerecorded confusable clips",
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        metavar="VOICE",
        help="Kokoro voice to use when TTS is available (default: af_heart)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        metavar="HZ",
        help="Target audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON report after the human-readable summary",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    if audio_dir is not None and not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        report = run_confusable_tests(
            model_path=model_path,
            wake_word=args.wake_word,
            threshold=args.threshold,
            count=args.count,
            audio_dir=audio_dir,
            voice=args.voice,
            sample_rate=args.sample_rate,
        )
    except Exception as exc:
        print(f"ERROR: Confusable test failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Model:             {Path(report.model_path).name}")
    print(f"Wake word:         {report.wake_word}")
    print(f"Threshold:         {report.threshold:.2f}")
    print(f"Audio backend:     {report.audio_backend}")
    print(f"Total candidates:  {report.summary.total_candidates}")
    print()
    print(_format_results_table(report.results))
    print()
    print(f"Total tested:      {report.summary.total_tested}")
    print(f"False accepts:     {report.summary.false_accepts}")
    print(f"False accept rate: {report.summary.false_accept_rate:.1%}")

    if report.summary.worst_offenders:
        worst_line = ", ".join(
            f"{result.word} ({_format_score(result.score)})"
            for result in report.summary.worst_offenders
        )
        print(f"Worst offenders:   {worst_line}")

    if args.json:
        print()
        print(json.dumps(report_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
