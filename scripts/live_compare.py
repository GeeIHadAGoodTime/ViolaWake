#!/usr/bin/env python
"""
Live Head-to-Head Wake Word Model Comparison
==============================================

Scores live microphone audio through up to 4 OWW-based models simultaneously.
Supports both MLP (mean-pooled 96-dim) and temporal (9x96 frame sequence) models.

Adapted from Viola's compare_wake_models.py for the OWW embedding pipeline.

Usage:
    python scripts/live_compare.py
    python scripts/live_compare.py --models r3_10x_s42 temporal_cnn viola_mlp_oww temporal_convgru
    python scripts/live_compare.py --threshold 0.80

Keyboard controls:
    1 = silence       2 = music_low     3 = music_high
    4 = speaking      5 = talking       q = quit

The OWW pipeline:
    int16-range audio → melspectrogram.onnx → mel/10+2 → embedding_model.onnx → 96-dim embeddings
    MLP models: mean-pool embeddings → MLP → score
    Temporal models: stack 9 frames → temporal model → score
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyaudio

# ---------------------------------------------------------------------------
# Resolve OWW backbone models
# ---------------------------------------------------------------------------
def _find_oww_models() -> tuple[Path, Path]:
    """Find melspectrogram.onnx and embedding_model.onnx from openwakeword."""
    try:
        import openwakeword
        oww_dir = Path(openwakeword.__file__).parent / "resources"
    except ImportError:
        # Fallback: check common locations
        candidates = [
            Path.home() / ".local" / "lib" / "python3.11" / "site-packages" / "openwakeword" / "resources",
            Path(sys.prefix) / "Lib" / "site-packages" / "openwakeword" / "resources",
        ]
        oww_dir = None
        for c in candidates:
            if c.exists():
                oww_dir = c
                break
        if oww_dir is None:
            print("ERROR: openwakeword not found. Install with: pip install openwakeword", file=sys.stderr)
            sys.exit(1)

    mel_files = list(oww_dir.rglob("melspectrogram*.onnx"))
    emb_files = list(oww_dir.rglob("embedding_model*.onnx"))
    if not mel_files or not emb_files:
        print(f"ERROR: OWW model files not found in {oww_dir}", file=sys.stderr)
        sys.exit(1)
    return mel_files[0], emb_files[0]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE * 2       # 2s ring buffer (32000 samples)
CHUNK_SIZE = 1280                     # 80ms at 16kHz
PROCESS_EVERY = 5                     # Score every 5 chunks (~400ms)
MEL_BINS = 32
MEL_FRAMES_PER_EMBEDDING = 76
MEL_STRIDE = 8
EMBEDDING_DIM = 96

AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1

CONDITIONS = {
    "1": "silence",
    "2": "music_low",
    "3": "music_high",
    "4": "speaking",
    "5": "talking",
}

# Where to find MLP/temporal .onnx models
WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [
    WAKEWORD_ROOT / "models",
    WAKEWORD_ROOT / "experiments" / "models",
    WAKEWORD_ROOT / "experiments" / "models" / "j5_temporal",
    Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA/violawake_data/trained_models"),
]

# Default 4 models for head-to-head
# "oww:alexa" means OWW's own pre-trained model (different inference path)
DEFAULT_MODELS = ["temporal_cnn", "oww:alexa", "operator_v2", "viola_mlp_oww_maxpool"]


# ---------------------------------------------------------------------------
# Non-blocking keyboard input
# ---------------------------------------------------------------------------
try:
    import msvcrt

    def _get_key() -> str | None:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):
                msvcrt.getch()
                return None
            return ch.decode("utf-8", errors="ignore")
        return None

except ImportError:
    import select

    def _get_key() -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------
def find_model(name: str) -> Path:
    """Resolve a model name to its .onnx file path."""
    # Direct path
    p = Path(name)
    if p.exists() and p.suffix == ".onnx":
        return p
    if not name.endswith(".onnx"):
        name_onnx = name + ".onnx"
    else:
        name_onnx = name

    for d in MODEL_DIRS:
        candidate = d / name_onnx
        if candidate.exists():
            return candidate
    print(f"ERROR: Model '{name}' not found in any model directory", file=sys.stderr)
    print(f"  Searched: {[str(d) for d in MODEL_DIRS]}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# OWW Embedding Backbone (shared across all models)
# ---------------------------------------------------------------------------
class OWWBackbone:
    """Extracts 96-dim embeddings from raw audio using OWW's 2-model pipeline.

    CRITICAL: mel model expects int16-range float32, output needs mel/10+2.
    """

    def __init__(self) -> None:
        mel_path, emb_path = _find_oww_models()
        self._mel_sess = ort.InferenceSession(str(mel_path), providers=["CPUExecutionProvider"])
        self._emb_sess = ort.InferenceSession(str(emb_path), providers=["CPUExecutionProvider"])
        self._mel_inp = self._mel_sess.get_inputs()[0].name
        self._emb_inp = self._emb_sess.get_inputs()[0].name

    def extract(self, audio_f32: np.ndarray) -> np.ndarray:
        """Extract embeddings from float32 audio ([-1, 1] range).

        Returns: (N, 96) array of embedding frames.
        """
        # Convert to int16 range (CRITICAL normalization)
        audio_int16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.float32)

        # Mel spectrogram
        mel_out = self._mel_sess.run(None, {self._mel_inp: audio_int16.reshape(1, -1)})[0]
        mel_raw = mel_out.squeeze().reshape(-1, MEL_BINS).astype(np.float32)

        # Critical transform: mel/10 + 2
        mel = mel_raw / 10.0 + 2.0

        # Extract embeddings with stride 8
        embeddings = []
        for start in range(0, mel.shape[0] - MEL_FRAMES_PER_EMBEDDING + 1, MEL_STRIDE):
            chunk = mel[start:start + MEL_FRAMES_PER_EMBEDDING].reshape(1, MEL_FRAMES_PER_EMBEDDING, MEL_BINS, 1)
            emb = self._emb_sess.run(None, {self._emb_inp: chunk})[0].flatten()
            embeddings.append(emb)

        if not embeddings:
            return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        return np.stack(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Model wrapper (handles both MLP and temporal)
# ---------------------------------------------------------------------------
class ModelWrapper:
    """Wraps an ONNX classifier model. Auto-detects MLP vs temporal from input shape."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0]
        self._inp_name = self._inp.name
        self._shape = self._inp.shape

        # Detect model type from input shape
        if len(self._shape) == 3:
            # Temporal: (batch, frames, dim) e.g. (batch, 9, 96)
            self.model_type = "temporal"
            self.n_frames = self._shape[1] if isinstance(self._shape[1], int) else 9
        else:
            # MLP: (batch, dim) e.g. (batch, 96)
            self.model_type = "mlp"
            self.n_frames = 0

        size_kb = path.stat().st_size / 1024
        self.label = f"{name} ({self.model_type}, {size_kb:.0f}KB)"

    def score(self, embeddings: np.ndarray) -> float:
        """Score from (N, 96) embeddings. Returns probability [0, 1]."""
        if embeddings.shape[0] == 0:
            return 0.0

        if self.model_type == "mlp":
            # Mean-pool all frames → single 96-dim vector
            pooled = embeddings.mean(axis=0).reshape(1, -1).astype(np.float32)
            out = self._sess.run(None, {self._inp_name: pooled})[0]
        else:
            # Temporal: need exactly n_frames frames
            n = embeddings.shape[0]
            if n >= self.n_frames:
                # Take last n_frames
                temporal = embeddings[-self.n_frames:].reshape(1, self.n_frames, EMBEDDING_DIM).astype(np.float32)
            else:
                # Pad with zeros
                temporal = np.zeros((1, self.n_frames, EMBEDDING_DIM), dtype=np.float32)
                temporal[0, -n:, :] = embeddings
            out = self._sess.run(None, {self._inp_name: temporal})[0]

        return float(np.asarray(out).flatten()[0])


class OWWModelWrapper:
    """Wraps OWW's own pre-trained model (e.g. alexa). Different inference path.

    OWW models use their own internal preprocessing pipeline rather than
    the shared OWWBackbone. They take raw int16 audio via predict().
    """

    def __init__(self, oww_name: str) -> None:
        from openwakeword.model import Model as OWWModel
        self.name = f"oww:{oww_name}"
        self._oww_name = oww_name
        self._model = OWWModel(wakeword_models=[oww_name])
        self.model_type = "oww_pretrained"
        self.n_frames = 0
        self.label = f"oww:{oww_name} (pre-trained)"
        # OWW accumulates state internally; feed chunks to predict()
        self._chunk_size = 1280  # OWW's native 80ms chunk

    def score(self, embeddings: np.ndarray) -> float:
        """Not used for OWW — see score_raw()."""
        return 0.0

    def score_raw(self, audio_int16: np.ndarray) -> float:
        """Score raw int16 audio through OWW's own pipeline.

        Feed the 2s buffer in 80ms chunks, return the max score from the
        last batch. This matches how OWW processes streaming audio.
        """
        # Reset OWW's internal state for fresh scoring of this window
        self._model.reset()

        best = 0.0
        for start in range(0, len(audio_int16) - self._chunk_size + 1, self._chunk_size):
            chunk = audio_int16[start:start + self._chunk_size]
            result = self._model.predict(chunk)
            s = result.get(self._oww_name, 0.0)
            if s > best:
                best = s
        return best


# ---------------------------------------------------------------------------
# Per-condition stats tracker
# ---------------------------------------------------------------------------
class ConditionStats:
    def __init__(self, n_models: int) -> None:
        self.frames: int = 0
        self.scores: list[list[float]] = [[] for _ in range(n_models)]
        self.triggers: list[int] = [0] * n_models
        self.total_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------
def run(model_names: list[str], threshold: float) -> None:
    # ---- Load OWW backbone ----
    print("Loading OWW backbone (melspectrogram + embedding_model)...")
    backbone = OWWBackbone()
    print("  Backbone loaded.\n")

    # ---- Load classifier models ----
    models: list[ModelWrapper | OWWModelWrapper] = []
    for name in model_names:
        if name.startswith("oww:"):
            oww_name = name.split(":", 1)[1]
            m = OWWModelWrapper(oww_name)
        else:
            path = find_model(name)
            m = ModelWrapper(name, path)
        models.append(m)
        print(f"  [{len(models)}] {m.label}")
    n_models = len(models)
    print()

    # ---- Color codes for each model slot ----
    COLORS = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
    RESET = "\033[0m"

    # ---- Output CSV ----
    output_dir = WAKEWORD_ROOT / "data" / "live_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"compare_{ts_str}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    header = ["timestamp"] + [f"{m.name}_score" for m in models] + [f"{m.name}_trigger" for m in models] + ["condition"]
    writer.writerow(header)

    # ---- Stats ----
    stats: dict[str, ConditionStats] = {c: ConditionStats(n_models) for c in CONDITIONS.values()}
    current_condition = "silence"
    condition_start = time.time()

    # Speaking session tracking
    speaking_sessions = 0
    current_speaking_session = -1
    detected_sessions: list[set[int]] = [set() for _ in range(n_models)]

    # ---- Audio setup ----
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
        input=True, frames_per_buffer=CHUNK_SIZE,
    )

    # Ring buffer
    ring_buf = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    write_idx = 0
    frame_count = 0
    total_scored = 0
    start_time = time.time()
    last_stats = start_time

    # ---- Banner ----
    print("=" * 80)
    print("  LIVE WAKE WORD HEAD-TO-HEAD COMPARISON")
    print("=" * 80)
    for i, m in enumerate(models):
        color = COLORS[i % len(COLORS)]
        print(f"  {color}[{i+1}] {m.label}{RESET}")
    print(f"\n  Threshold: {threshold}   Sample rate: {SAMPLE_RATE}")
    print(f"  Ring buffer: {CLIP_SAMPLES/SAMPLE_RATE:.1f}s   Score every: {PROCESS_EVERY * CHUNK_SIZE / SAMPLE_RATE * 1000:.0f}ms")
    print(f"  CSV: {csv_path}")
    print("=" * 80)
    print("Keys: 1=silence 2=music_low 3=music_high 4=speaking 5=talking q=quit")
    print(f"[condition: {current_condition}]")
    print("-" * 80)

    try:
        while True:
            # ---- Keyboard ----
            key = _get_key()
            if key == "q":
                break
            if key in CONDITIONS:
                now = time.time()
                stats[current_condition].total_seconds += now - condition_start
                condition_start = now
                new_cond = CONDITIONS[key]
                if new_cond != current_condition:
                    current_condition = new_cond
                    print(f"\n>>> Condition: {current_condition}")
                    if current_condition == "speaking":
                        speaking_sessions += 1
                        current_speaking_session = speaking_sessions

            # ---- Read audio ----
            try:
                raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except OSError:
                continue
            audio_int16 = np.frombuffer(raw, dtype=np.int16)
            audio_f32 = audio_int16.astype(np.float32) / 32768.0

            # ---- Ring buffer write ----
            n = len(audio_f32)
            end = write_idx + n
            if end <= CLIP_SAMPLES:
                ring_buf[write_idx:end] = audio_f32
            else:
                first = CLIP_SAMPLES - write_idx
                ring_buf[write_idx:] = audio_f32[:first]
                ring_buf[:end - CLIP_SAMPLES] = audio_f32[first:]
            write_idx = end % CLIP_SAMPLES
            frame_count += 1

            # ---- Score every PROCESS_EVERY chunks ----
            if frame_count % PROCESS_EVERY != 0:
                continue

            # Reconstruct contiguous 2s window
            contiguous = np.concatenate([ring_buf[write_idx:], ring_buf[:write_idx]])
            rms = float(np.sqrt(np.mean(contiguous ** 2)))

            if rms < 0.005:
                # Silence gate — show zeros
                scores = [0.0] * n_models
                triggers = [False] * n_models
            else:
                # Extract embeddings (shared backbone, single extraction)
                t0 = time.perf_counter()
                embeddings = backbone.extract(contiguous)
                emb_ms = (time.perf_counter() - t0) * 1000

                # Reconstruct int16 for OWW pre-trained models
                contiguous_int16 = (contiguous * 32767).clip(-32768, 32767).astype(np.int16)

                # Score each model
                scores = []
                triggers = []
                for m in models:
                    if isinstance(m, OWWModelWrapper):
                        s = m.score_raw(contiguous_int16)
                    else:
                        s = m.score(embeddings)
                    scores.append(s)
                    triggers.append(s >= threshold)

            total_scored += 1
            ts = datetime.now().strftime("%H:%M:%S")

            # ---- Record stats ----
            cstats = stats[current_condition]
            cstats.frames += 1
            for i in range(n_models):
                cstats.scores[i].append(scores[i])
                if triggers[i]:
                    cstats.triggers[i] += 1

            # Track speaking detections
            if current_condition == "speaking" and current_speaking_session >= 0:
                for i in range(n_models):
                    if triggers[i]:
                        detected_sessions[i].add(current_speaking_session)

            # ---- CSV ----
            row = [datetime.now().isoformat()]
            row += [f"{s:.4f}" for s in scores]
            row += [int(t) for t in triggers]
            row += [current_condition]
            writer.writerow(row)

            # ---- Display ----
            score_parts = []
            trigger_marks = []
            for i, m in enumerate(models):
                color = COLORS[i % len(COLORS)]
                tag = m.name[:12].ljust(12)
                score_parts.append(f"{tag}:{scores[i]:.3f}")
                if triggers[i]:
                    trigger_marks.append(f"  {color}<< {m.name} TRIGGER{RESET}")

            line = f"[{ts}] " + "  ".join(score_parts) + f"  | {current_condition}"
            print(line + "".join(trigger_marks))

            # ---- Periodic stats (every 30s) ----
            now = time.time()
            if now - last_stats >= 30:
                last_stats = now
                elapsed = now - start_time
                print()
                print(f"--- Stats at {elapsed:.0f}s ({total_scored} frames scored) ---")
                for cond, st in stats.items():
                    if st.frames == 0:
                        continue
                    parts = []
                    for i, m in enumerate(models):
                        avg = float(np.mean(st.scores[i])) if st.scores[i] else 0.0
                        mx = float(np.max(st.scores[i])) if st.scores[i] else 0.0
                        parts.append(f"{m.name[:8]}:avg={avg:.3f} max={mx:.3f} trg={st.triggers[i]}")
                    print(f"  {cond:12s}: {st.frames:4d}fr | {'  '.join(parts)}")
                print()

    except KeyboardInterrupt:
        pass
    finally:
        stats[current_condition].total_seconds += time.time() - condition_start
        stream.stop_stream()
        stream.close()
        pa.terminate()
        csv_file.close()

    # ---- Summary ----
    elapsed = time.time() - start_time
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60

    print()
    print("=" * 80)
    print("  COMPARISON RESULTS")
    print("=" * 80)
    print(f"Duration: {minutes}m {seconds:02d}s  |  Frames scored: {total_scored}")
    print(f"Threshold: {threshold}")
    print(f"CSV: {csv_path}")
    print()

    # Header
    name_width = max(len(m.name) for m in models)
    print(f"{'Model':<{name_width}}  {'Type':8s}  {'Avg':>6s}  {'Max':>6s}  {'Triggers':>8s}  {'FP/hr':>6s}")
    print("-" * (name_width + 45))

    for cond, st in stats.items():
        if st.frames == 0:
            continue
        hours = st.total_seconds / 3600 if st.total_seconds > 0 else 0.001

        cond_min = int(st.total_seconds) // 60
        cond_sec = int(st.total_seconds) % 60
        print(f"\n  [{cond}] ({cond_min}m {cond_sec:02d}s, {st.frames} frames)")

        if cond == "speaking" and speaking_sessions > 0:
            for i, m in enumerate(models):
                det = len(detected_sessions[i])
                recall = det / speaking_sessions * 100
                avg = float(np.mean(st.scores[i])) if st.scores[i] else 0.0
                mx = float(np.max(st.scores[i])) if st.scores[i] else 0.0
                print(f"    {m.name:<{name_width}}  {m.model_type:8s}  {avg:6.3f}  {mx:6.3f}  {det}/{speaking_sessions} = {recall:.0f}% recall")
        else:
            for i, m in enumerate(models):
                avg = float(np.mean(st.scores[i])) if st.scores[i] else 0.0
                mx = float(np.max(st.scores[i])) if st.scores[i] else 0.0
                trg = st.triggers[i]
                fph = trg / hours if hours > 0 else 0
                fp_label = f"{fph:.1f}" if cond != "silence" else "—"
                print(f"    {m.name:<{name_width}}  {m.model_type:8s}  {avg:6.3f}  {mx:6.3f}  {trg:>8d}  {fp_label:>6s}")

    print()
    print("=" * 80)
    print("  VERDICT: Compare FP triggers during music/talking (lower = better)")
    print("           Compare recall during speaking (higher = better)")
    print("           The best model has high recall + zero FP triggers.")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live head-to-head wake word model comparison (OWW pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/live_compare.py
  python scripts/live_compare.py --models r3_10x_s42 temporal_cnn
  python scripts/live_compare.py --models r3_10x_s42 temporal_cnn viola_mlp_oww --threshold 0.70

Keys during test:
  1=silence  2=music_low  3=music_high  4=speaking  5=talking  q=quit

  Press 4 before saying "Viola" to track recall.
  Press 2/3 while music plays to track false positives.
        """,
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Model names or paths (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Detection threshold (default: 0.80)",
    )
    args = parser.parse_args()
    run(args.models, args.threshold)


if __name__ == "__main__":
    main()
