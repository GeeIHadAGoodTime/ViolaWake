#!/usr/bin/env python3
"""Measure batch-gate vs runtime-streaming silence parity for wake models (#1487).

The training quality gate scores its SILENCE subgrade in BATCH mode
(``train._extract_temporal_embeddings`` -> ``preprocessor.embed_clips`` on a
1.5s center-crop, one max over the clip's windows). The RUNTIME
(``WakeDetector.process`` -> ``oww_backbone.push_audio``) scores hundreds of
windows over continuous audio with subtly different embeddings due to OWW's
internal streaming state (train.py's own docstring warns of this). The two are
~uncorrelated, so the batch silence subgrade is a weak oracle for the runtime
silence behavior it claims to protect.

This tool scores the SAME silence / near-silence audio through BOTH real code
paths on one or more deployed temporal_cnn models and reports the divergence. It
is the evidence harness the #1487 acceptance oracle asks for ("a test that reds
if the training-time silence probe diverges from the runtime scoring path beyond
tolerance"): run it across >=3 real models spanning grades on the training box
before deciding whether to make the silence subgrade streaming-aware or retire
it. It exits non-zero if any model's near-silence divergence exceeds
``--tolerance``.

Usage:
    python tools/silence_streaming_parity.py MODEL.onnx [MODEL2.onnx ...] \
        [--tolerance 0.15] [--mel PATH --embedding PATH]

The mel/embedding backbone models default to OpenWakeWord's installed
``resources/models``; pass ``--mel``/``--embedding`` to point at explicit copies
(e.g. ``console/frontend/public/wasm/models/{melspectrogram,embedding_model}.onnx``).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# Resolve the SDK from a source checkout when not installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from violawake_sdk.tools.train import (  # noqa: E402
    _load_training_audio,
    _prepare_audio_for_oww,
    _temporal_windows_from_frame_embeddings,
)
from violawake_sdk.wake_detector import (  # noqa: E402
    EMBEDDING_DIM,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    WakeDetector,
)

# Near-silence probe identical to the training gate's fallback probe
# (train._run_quality_gate): RMS ~= 1e-4 * full-scale white noise, ~80 dB below
# speech. Its int16 RMS (~3.3) sits ABOVE the runtime RMS floor (1.0), so unlike
# true digital zeros it is NOT rejected by wake_detector Gate 1 at runtime.
_NEAR_SILENCE_SCALE = 1e-4
_PROBE_SECONDS = 10


def _save_wav(audio_f32: np.ndarray, path: Path) -> None:
    i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(i16.tobytes())


def _streaming_max(det: WakeDetector, audio_f32: np.ndarray) -> tuple[float, int]:
    """Max raw model score over the audio via the runtime streaming path.

    This is the score BEFORE the runtime RMS floor / threshold gates, i.e. the
    number the model actually produces on the audio (what #1487 calls the
    'streams 0.945' value).
    """
    det._embedding_buffer.clear()
    scores: list[float] = []
    n = len(audio_f32)
    for i in range(0, n - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        scores.append(det.process(audio_f32[i : i + FRAME_SAMPLES].astype(np.float32)))
    return (float(max(scores)) if scores else float("nan"), len(scores))


def _batch_max(det: WakeDetector, preprocessor, wav_path: Path) -> tuple[float, int]:
    """Max model score via the training gate's batch embedding path."""
    seq = det._temporal_seq_len
    audio = _load_training_audio(wav_path)
    audio_i16 = _prepare_audio_for_oww(audio, clip_name=wav_path.name, verbose=False)
    if audio_i16 is None:
        return (float("nan"), 0)
    frame_emb = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
    embs, _src, _tags = _temporal_windows_from_frame_embeddings(
        frame_emb[0], source_id=0, tag="probe", seq_len=seq
    )
    if not embs:
        return (float("nan"), 0)
    X = np.array(embs).reshape(-1, seq, EMBEDDING_DIM).astype(np.float32)
    out = det._mlp_session.run(None, {det._mlp_input_name: X})[0].flatten()
    return (float(out.max()), len(embs))


def _build_preprocessor(mel: str | None, embedding: str | None):
    from openwakeword.utils import AudioFeatures

    kwargs = {"inference_framework": "onnx"}
    if mel and embedding:
        kwargs["melspec_model_path"] = mel
        kwargs["embedding_model_path"] = embedding
    return AudioFeatures(**kwargs)


def measure_model(model_path: Path, preprocessor, rng: np.random.Generator) -> dict[str, float]:
    det = WakeDetector(model=str(model_path), threshold=0.80)
    zeros = np.zeros(SAMPLE_RATE * _PROBE_SECONDS, dtype=np.float32)
    near = rng.standard_normal(SAMPLE_RATE * _PROBE_SECONDS).astype(np.float32) * _NEAR_SILENCE_SCALE
    with tempfile.TemporaryDirectory() as td:
        zeros_wav = Path(td) / "zeros.wav"
        near_wav = Path(td) / "near.wav"
        _save_wav(zeros, zeros_wav)
        _save_wav(near, near_wav)
        zeros_batch, _zbw = _batch_max(det, preprocessor, zeros_wav)
        near_batch, _nbw = _batch_max(det, preprocessor, near_wav)
        zeros_stream, _zsf = _streaming_max(det, zeros)
        near_stream, _nsf = _streaming_max(det, near)
    return {
        "zeros_batch": zeros_batch,
        "zeros_stream": zeros_stream,
        "near_batch": near_batch,
        "near_stream": near_stream,
        # Divergence is measured on the near-silence probe: it is the case that
        # survives the runtime RMS floor, so its batch-vs-stream gap is the real
        # oracle mismatch. (Pure zeros are rejected at runtime by Gate 1.)
        "near_divergence": abs(near_stream - near_batch),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="Deployed temporal_cnn .onnx model(s).")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Max allowed near-silence divergence.")
    parser.add_argument("--mel", default=None, help="melspectrogram.onnx path (default: OWW resources).")
    parser.add_argument("--embedding", default=None, help="embedding_model.onnx path (default: OWW resources).")
    args = parser.parse_args(argv)

    preprocessor = _build_preprocessor(args.mel, args.embedding)
    rng = np.random.default_rng(seed=42)

    print(f"Silence batch-vs-streaming parity (tolerance={args.tolerance}); RMS floor=1.0, threshold=0.80\n")
    worst = 0.0
    for model_path in args.models:
        r = measure_model(model_path, preprocessor, rng)
        worst = max(worst, r["near_divergence"])
        print(f"{model_path}")
        print(f"  pure zeros      batch={r['zeros_batch']:.4f}  stream={r['zeros_stream']:.4f}  (zeros rejected at runtime by RMS floor)")
        print(f"  near-silence    batch={r['near_batch']:.4f}  stream={r['near_stream']:.4f}  divergence={r['near_divergence']:.4f}")
        print()

    if worst > args.tolerance:
        print(f"RESULT: FAIL — worst near-silence divergence {worst:.4f} > tolerance {args.tolerance}.")
        print("The batch silence subgrade does not predict the runtime streaming score (#1487).")
        return 1
    print(f"RESULT: PASS — worst near-silence divergence {worst:.4f} <= tolerance {args.tolerance}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
