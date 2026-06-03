#!/usr/bin/env python
"""Run Lane 3 WASM-vs-Python parity and negative probes."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
SAMPLE_SECONDS = 1.2
SAMPLE_FRAMES = int(SAMPLE_RATE * SAMPLE_SECONDS)
DEFAULT_TOLERANCE = 1e-3

RECORDING_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-US-AriaNeural",
    "en-US-AndrewNeural",
    "en-US-BrandonNeural",
    "en-US-CoraNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
    "en-GB-RyanNeural",
]

PHRASES = [
    "hey viola",
    "ok viola",
    "viola wake up",
    "viola start listening",
    "please wake up viola",
    "viola can you hear me",
    "hello viola",
    "viola",
    "hey there viola",
    "wake up please viola",
]


@dataclass
class Corpus:
    name: str
    samples: np.ndarray
    labels: list[str]
    note: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=root / "wasm" / "dist" / "violawake.js",
        help="Built ESM bundle to audit.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "console" / "frontend" / "public" / "wasm" / "models",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Optional directory of real WAV/FLAC/MP3 samples. First 10 are used.",
    )
    parser.add_argument(
        "--prefer-edge-tts",
        action="store_true",
        help="Generate 10 spoken samples with edge-tts when no corpus-dir is supplied.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON details.")
    return parser.parse_args()


def pad_or_trim(audio: np.ndarray, target_frames: int = SAMPLE_FRAMES) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < target_frames:
        audio = np.pad(audio, (0, target_frames - audio.size))
    elif audio.size > target_frames:
        audio = audio[:target_frames]
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def load_audio_file(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy import signal

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        target_len = int(round(data.shape[0] * SAMPLE_RATE / sample_rate))
        data = signal.resample(data, target_len).astype(np.float32)
    return pad_or_trim(data)


def load_corpus_dir(corpus_dir: Path) -> Corpus:
    suffixes = {".wav", ".flac", ".mp3", ".ogg"}
    paths = sorted(path for path in corpus_dir.rglob("*") if path.suffix.lower() in suffixes)
    if len(paths) < 10:
        raise RuntimeError(f"Need at least 10 audio files in {corpus_dir}; found {len(paths)}")
    chosen = paths[:10]
    samples = np.stack([load_audio_file(path) for path in chosen])
    return Corpus(
        name="real_audio_dir",
        samples=samples,
        labels=[path.name for path in chosen],
        note=f"Loaded first 10 audio files from {corpus_dir}",
    )


async def generate_edge_tts_corpus() -> Corpus:
    import edge_tts
    from pydub import AudioSegment

    generated = []
    labels = []
    for idx, (voice, phrase) in enumerate(zip(RECORDING_VOICES, PHRASES, strict=True)):
        communicate = edge_tts.Communicate(phrase, voice)
        audio_bytes = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.write(chunk["data"])
        payload = audio_bytes.getvalue()
        if len(payload) < 100:
            raise RuntimeError(f"edge-tts returned an empty sample for {voice}")
        segment = AudioSegment.from_mp3(io.BytesIO(payload))
        segment = segment.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
        int16 = np.array(segment.get_array_of_samples(), dtype=np.float32)
        generated.append(pad_or_trim(int16 / 32768.0))
        labels.append(f"{idx:02d}-{voice}-{phrase}")

    return Corpus(
        name="edge_tts_spoken_subset",
        samples=np.stack(generated),
        labels=labels,
        note="Generated 10 spoken samples with edge-tts using repo golden-path voices.",
    )


def generate_deterministic_speech_like_corpus() -> Corpus:
    rng = np.random.default_rng(20260603)
    t = np.arange(SAMPLE_FRAMES, dtype=np.float32) / SAMPLE_RATE
    samples = []
    labels = []
    for idx in range(10):
        f0 = 105.0 + idx * 13.0
        audio = np.zeros_like(t)
        syllable_count = 5 + (idx % 3)
        for syllable in range(syllable_count):
            start = 0.12 + syllable * (SAMPLE_SECONDS - 0.35) / syllable_count
            width = 0.16 + 0.03 * ((idx + syllable) % 4)
            envelope = np.exp(-0.5 * ((t - start) / width) ** 2)
            wobble = 1.0 + 0.02 * np.sin(2 * np.pi * (3.0 + syllable) * t)
            formants = [
                f0 * wobble,
                (520 + 30 * idx + 20 * syllable) * wobble,
                (1250 + 45 * idx - 10 * syllable) * wobble,
                (2400 - 25 * idx + 15 * syllable) * wobble,
            ]
            weights = [0.55, 0.30, 0.20, 0.10]
            for weight, freq in zip(weights, formants, strict=True):
                audio += weight * envelope * np.sin(2 * np.pi * freq * t)
        audio += 0.012 * rng.standard_normal(t.shape).astype(np.float32)
        audio *= 0.35 / max(float(np.max(np.abs(audio))), 1e-6)
        samples.append(audio.astype(np.float32))
        labels.append(f"deterministic-speech-like-{idx:02d}")
    return Corpus(
        name="deterministic_speech_like",
        samples=np.stack(samples),
        labels=labels,
        note="Fallback generated corpus; use --corpus-dir or --prefer-edge-tts for real/spoken audio.",
    )


async def choose_corpus(args: argparse.Namespace) -> Corpus:
    if args.corpus_dir is not None:
        return load_corpus_dir(args.corpus_dir)
    if args.prefer_edge_tts:
        try:
            return await generate_edge_tts_corpus()
        except Exception as exc:
            fallback = generate_deterministic_speech_like_corpus()
            fallback.note = f"edge-tts failed ({exc}); {fallback.note}"
            return fallback
    return generate_deterministic_speech_like_corpus()


def run_python_reference(samples: np.ndarray, classifier_path: Path) -> dict[str, Any]:
    root = repo_root()
    sys.path.insert(0, str(root / "src"))
    from violawake_sdk.wake_detector import WakeDetector

    detector = WakeDetector(
        model=str(classifier_path),
        backend="onnx",
        providers=["CPUExecutionProvider"],
        cooldown_s=0,
    )
    scores: list[list[float]] = []
    latencies: list[float] = []
    try:
        for sample in samples:
            detector.reset()
            sample_scores = []
            for offset in range(0, sample.shape[0], FRAME_SAMPLES):
                frame = sample[offset : offset + FRAME_SAMPLES].astype(np.float32, copy=False)
                start = time.perf_counter()
                sample_scores.append(float(detector.process(frame)))
                latencies.append((time.perf_counter() - start) * 1000)
            scores.append(sample_scores)
    finally:
        detector.close()

    return {
        "scores": scores,
        "frame_latency": summarize(latencies),
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0}
    sorted_values = sorted(values)

    def pct(p: float) -> float:
        index = min(len(sorted_values) - 1, math.ceil((p / 100.0) * len(sorted_values)) - 1)
        return sorted_values[index]

    return {
        "count": len(values),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": max(sorted_values),
    }


def run_node_scores(
    *,
    corpus_path: Path,
    bundle: Path,
    model_dir: Path,
    sample_count: int,
    sample_frames: int,
    frame_size: int = FRAME_SAMPLES,
    classifier: str = "temporal_cnn.onnx",
) -> subprocess.CompletedProcess[str]:
    root = repo_root()
    runner = root / "wasm" / "scripts" / "run_wasm_scores.mjs"
    ort_wasm_dir = root / "wasm" / "node_modules" / "onnxruntime-web" / "dist"
    command = [
        "node",
        str(runner),
        "--corpus",
        str(corpus_path),
        "--bundle",
        str(bundle),
        "--model-dir",
        str(model_dir),
        "--ort-wasm-dir",
        str(ort_wasm_dir),
        "--sample-count",
        str(sample_count),
        "--sample-frames",
        str(sample_frames),
        "--frame-size",
        str(frame_size),
        "--classifier",
        classifier,
    ]
    return subprocess.run(
        command,
        cwd=str(root / "wasm"),
        text=True,
        capture_output=True,
        check=False,
    )


def max_abs_diff(python_scores: list[list[float]], node_scores: list[list[float]]) -> float:
    max_diff = 0.0
    for sample_idx, (py_sample, node_sample) in enumerate(zip(python_scores, node_scores)):
        if len(py_sample) != len(node_sample):
            raise RuntimeError(
                f"Score length mismatch for sample {sample_idx}: "
                f"python={len(py_sample)} node={len(node_sample)}"
            )
        for py_score, node_score in zip(py_sample, node_sample):
            max_diff = max(max_diff, abs(float(py_score) - float(node_score)))
    return max_diff


def bundle_metrics(bundle: Path, model_dir: Path) -> dict[str, int]:
    files = {
        "bundle_js_bytes": bundle.stat().st_size,
        "melspectrogram_onnx_bytes": (model_dir / "melspectrogram.onnx").stat().st_size,
        "embedding_model_onnx_bytes": (model_dir / "embedding_model.onnx").stat().st_size,
        "temporal_cnn_onnx_bytes": (model_dir / "temporal_cnn.onnx").stat().st_size,
    }
    files["model_asset_total_bytes"] = (
        files["melspectrogram_onnx_bytes"]
        + files["embedding_model_onnx_bytes"]
        + files["temporal_cnn_onnx_bytes"]
    )
    return files


def print_summary(result: dict[str, Any]) -> None:
    print("Lane 3 WASM parity audit")
    print(f"corpus={result['corpus']['name']} samples={result['corpus']['sample_count']}")
    print(f"tolerance={result['tolerance']}")
    print(f"max_linf={result['parity']['max_linf']:.9g}")
    print(f"parity_pass={result['parity']['pass']}")
    print(f"node_load_ms={result['node']['load_ms']:.3f}")
    print(f"node_first_frame_score_ms={result['node']['first_frame_score_ms']:.3f}")
    print(f"node_first_temporal_score_audio_ms={result['node']['first_temporal_score_audio_ms']}")
    print(f"node_frame_latency={json.dumps(result['node']['frame_latency'], sort_keys=True)}")
    print(f"python_frame_latency={json.dumps(result['python']['frame_latency'], sort_keys=True)}")
    print(f"bundle_metrics={json.dumps(result['bundle_metrics'], sort_keys=True)}")
    print("negative_probes:")
    for probe in result["negative_probes"]:
        print(
            f"  {probe['name']}: caught={probe['caught']} "
            f"returncode={probe['returncode']} excerpt={probe['excerpt']}"
        )


async def async_main() -> int:
    args = parse_args()
    root = repo_root()
    model_dir = args.model_dir.resolve()
    bundle = args.bundle.resolve()
    required = [
        bundle,
        model_dir / "melspectrogram.onnx",
        model_dir / "embedding_model.onnx",
        model_dir / "temporal_cnn.onnx",
        root / "wasm" / "node_modules" / "onnxruntime-web" / "dist",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required WASM audit assets: {missing}")

    corpus = await choose_corpus(args)

    with tempfile.TemporaryDirectory(prefix="violawake-wasm-parity-") as tmp:
        corpus_path = Path(tmp) / "corpus.f32"
        corpus.samples.astype("<f4", copy=False).tofile(corpus_path)

        python_result = run_python_reference(corpus.samples, model_dir / "temporal_cnn.onnx")
        node_run = run_node_scores(
            corpus_path=corpus_path,
            bundle=bundle,
            model_dir=model_dir,
            sample_count=corpus.samples.shape[0],
            sample_frames=corpus.samples.shape[1],
        )
        if node_run.returncode != 0:
            raise RuntimeError(f"Node WASM scoring failed:\n{node_run.stderr}")
        node_result = json.loads(node_run.stdout)
        max_diff = max_abs_diff(python_result["scores"], node_result["scores"])

        wrong_stride = run_node_scores(
            corpus_path=corpus_path,
            bundle=bundle,
            model_dir=model_dir,
            sample_count=1,
            sample_frames=corpus.samples.shape[1],
            frame_size=160,
        )
        wrong_model = run_node_scores(
            corpus_path=corpus_path,
            bundle=bundle,
            model_dir=model_dir,
            sample_count=1,
            sample_frames=corpus.samples.shape[1],
            classifier="embedding_model.onnx",
        )

    negative_probes = [
        {
            "name": "wrong_frame_stride_160_samples",
            "caught": wrong_stride.returncode != 0
            and "exactly 320 samples" in wrong_stride.stderr,
            "returncode": wrong_stride.returncode,
            "excerpt": (wrong_stride.stderr or wrong_stride.stdout).strip().splitlines()[0:3],
        },
        {
            "name": "wrong_classifier_model_embedding_model",
            "caught": wrong_model.returncode != 0
            and "Classifier model" in wrong_model.stderr,
            "returncode": wrong_model.returncode,
            "excerpt": (wrong_model.stderr or wrong_model.stdout).strip().splitlines()[0:3],
        },
    ]

    result = {
        "tolerance": args.tolerance,
        "corpus": {
            "name": corpus.name,
            "note": corpus.note,
            "sample_count": int(corpus.samples.shape[0]),
            "sample_frames": int(corpus.samples.shape[1]),
            "sample_rate": SAMPLE_RATE,
            "labels": corpus.labels,
        },
        "parity": {
            "max_linf": max_diff,
            "pass": max_diff <= args.tolerance,
        },
        "python": {
            "frame_latency": python_result["frame_latency"],
        },
        "node": {
            key: value
            for key, value in node_result.items()
            if key not in {"scores"}
        },
        "bundle_metrics": bundle_metrics(bundle, model_dir),
        "negative_probes": negative_probes,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_summary(result)

    if not result["parity"]["pass"] or not all(probe["caught"] for probe in negative_probes):
        return 1
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
