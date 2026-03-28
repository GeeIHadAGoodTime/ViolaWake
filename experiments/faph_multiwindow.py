"""
Multi-Window Confirmation FAPH Test
====================================

Instead of triggering on a single window exceeding threshold,
requires N-of-M consecutive windows above threshold before triggering.

This is industry standard (Amazon Alexa, Google Home use similar patterns).
Expected: 40-70% FAPH reduction with minimal TP impact since real wake words
produce sustained high scores across multiple consecutive windows.

Configurations tested:
  1-of-1: baseline (current behavior)
  2-of-2: 2 consecutive windows above threshold
  2-of-3: 2 of any 3 consecutive windows above threshold
  3-of-3: 3 consecutive windows above threshold
  2-of-4: 2 of any 4 consecutive windows above threshold
  3-of-5: 3 of any 5 consecutive windows above threshold

Usage:
    python experiments/faph_multiwindow.py
    python experiments/faph_multiwindow.py --model experiments/models/faph_hardened_s43.onnx
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# ---------------------------------------------------------------------------
WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = WAKEWORD_ROOT / "experiments" / "models" / "faph_hardened_s43.onnx"
LIBRISPEECH_DIR = WAKEWORD_ROOT / "corpus" / "librispeech" / "LibriSpeech" / "dev-clean"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000       # 1.5s
STEP_SAMPLES = 1600        # 100ms
DEBOUNCE_SECONDS = 2.0
DEBOUNCE_SAMPLES = int(DEBOUNCE_SECONDS * SAMPLE_RATE)

THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]

# Multi-window configs: (name, n_required, window_size)
CONFIGS = [
    ("1-of-1", 1, 1),
    ("2-of-2", 2, 2),
    ("2-of-3", 2, 3),
    ("3-of-3", 3, 3),
    ("2-of-4", 2, 4),
    ("3-of-5", 3, 5),
]


def init_oww_preprocessor():
    from openwakeword.utils import AudioFeatures
    return AudioFeatures()


def extract_embedding(preprocessor, audio_int16: np.ndarray) -> np.ndarray:
    embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    return embeddings.mean(axis=1)[0].astype(np.float32)


def load_model(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def score_embedding(session: ort.InferenceSession, embedding: np.ndarray) -> float:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: embedding.reshape(1, -1)})
    return float(result[0][0][0])


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def find_flac_files(d: Path) -> list[Path]:
    files = sorted(d.rglob("*.flac"))
    print(f"Found {len(files)} .flac files in {d}")
    return files


class MultiWindowDetector:
    """Tracks N-of-M consecutive windows above threshold with debounce."""

    def __init__(self, n_required: int, window_size: int, debounce_samples: int):
        self.n_required = n_required
        self.window_size = window_size
        self.debounce_samples = debounce_samples
        self.reset_counts()
        self.reset()

    def reset(self):
        """Reset per-file state (history + debounce), but keep cumulative counts."""
        self.history: deque[bool] = deque(maxlen=self.window_size)
        self.last_trigger_pos = -999999

    def reset_counts(self):
        """Reset cumulative counts (only call at init)."""
        self.triggers_raw = 0
        self.triggers_debounced = 0

    def update(self, above_threshold: bool, pos: int) -> bool:
        """Feed one window result. Returns True if trigger fires (debounced)."""
        self.history.append(above_threshold)

        if len(self.history) == self.window_size:
            n_above = sum(self.history)
            if n_above >= self.n_required:
                self.triggers_raw += 1
                if pos - self.last_trigger_pos >= self.debounce_samples:
                    self.triggers_debounced += 1
                    self.last_trigger_pos = pos
                    return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-window confirmation FAPH test")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--librispeech-dir", type=Path, default=LIBRISPEECH_DIR)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "faph_multiwindow_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("ViolaWake Multi-Window Confirmation FAPH Test")
    print("=" * 70)
    print(f"Model: {args.model.name}")
    print(f"Corpus: {args.librispeech_dir}")
    print(f"Configs: {', '.join(c[0] for c in CONFIGS)}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    flac_files = find_flac_files(args.librispeech_dir)
    session = load_model(args.model)

    print("Initializing OWW preprocessor...")
    preprocessor = init_oww_preprocessor()

    # Create detectors for each (config, threshold) combination
    detectors: dict[tuple[str, float], MultiWindowDetector] = {}
    for cfg_name, n_req, win_size in CONFIGS:
        for t in THRESHOLDS:
            detectors[(cfg_name, t)] = MultiWindowDetector(n_req, win_size, DEBOUNCE_SAMPLES)

    # Temporal analysis: track consecutive-above-threshold streaks
    streak_tracker: dict[float, list[int]] = {t: [] for t in THRESHOLDS}
    current_streak: dict[float, int] = {t: 0 for t in THRESHOLDS}

    # Top-50 window tracking for temporal context analysis
    TOP_K = 50
    top_windows: list[tuple[float, str, float]] = []  # (score, rel_path, time_sec)
    min_top = 0.0
    # Store per-file score arrays for later context lookup
    file_score_map: dict[str, list[float]] = {}

    total_windows = 0
    total_audio_samples = 0
    t_start = time.time()

    for fi, fpath in enumerate(flac_files):
        try:
            audio, sr = sf.read(fpath, dtype="float32")
        except Exception as e:
            print(f"  WARN: Failed to load {fpath.name}: {e}")
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        n_samples = len(audio)
        total_audio_samples += n_samples

        # Reset all detectors for new file
        for det in detectors.values():
            det.reset()
        for t in THRESHOLDS:
            if current_streak[t] > 0:
                streak_tracker[t].append(current_streak[t])
                current_streak[t] = 0

        rel_path = str(fpath.relative_to(args.librispeech_dir))
        file_scores_list: list[float] = []

        # Slide windows
        pos = 0
        while pos + CLIP_SAMPLES <= n_samples:
            window = audio[pos:pos + CLIP_SAMPLES]
            window_int16 = float_to_int16(window)

            embedding = extract_embedding(preprocessor, window_int16)
            score = score_embedding(session, embedding)
            total_windows += 1
            file_scores_list.append(score)

            # Track top-K
            if score > min_top or len(top_windows) < TOP_K:
                t_sec = pos / SAMPLE_RATE
                top_windows.append((score, rel_path, t_sec))
                top_windows.sort(key=lambda x: x[0], reverse=True)
                if len(top_windows) > TOP_K:
                    top_windows = top_windows[:TOP_K]
                min_top = top_windows[-1][0] if top_windows else 0.0

            for t in THRESHOLDS:
                above = score >= t

                # Track streaks for temporal analysis
                if above:
                    current_streak[t] += 1
                else:
                    if current_streak[t] > 0:
                        streak_tracker[t].append(current_streak[t])
                        current_streak[t] = 0

                # Update all detectors for this threshold
                for cfg_name, n_req, win_size in CONFIGS:
                    detectors[(cfg_name, t)].update(above, pos)

            pos += STEP_SAMPLES

        file_score_map[rel_path] = file_scores_list

        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t_start
            hours = total_audio_samples / SAMPLE_RATE / 3600
            print(
                f"  [{fi+1}/{len(flac_files)}] "
                f"{hours:.2f}h audio, {total_windows} windows, "
                f"elapsed {elapsed:.0f}s"
            )

    # Flush final streaks
    for t in THRESHOLDS:
        if current_streak[t] > 0:
            streak_tracker[t].append(current_streak[t])

    elapsed = time.time() - t_start
    total_hours = total_audio_samples / SAMPLE_RATE / 3600

    # ── Results ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total audio: {total_hours:.3f} hours")
    print(f"Total windows: {total_windows}")
    print(f"Total files: {len(flac_files)}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Main results table
    results_data = {}
    for t in THRESHOLDS:
        print(f"\n--- Threshold {t:.2f} ---")
        print(f"{'Config':<20} {'Raw Trig':>10} {'Raw FAPH':>10} {'Deb Trig':>10} {'Deb FAPH':>10} {'% Reduct':>10}")
        print("-" * 72)

        baseline_faph = None
        for cfg_name, n_req, win_size in CONFIGS:
            det = detectors[(cfg_name, t)]
            faph_raw = det.triggers_raw / total_hours if total_hours > 0 else 0
            faph_deb = det.triggers_debounced / total_hours if total_hours > 0 else 0

            if baseline_faph is None:
                baseline_faph = faph_deb
                reduction = 0.0
            else:
                reduction = (1.0 - faph_deb / baseline_faph) * 100 if baseline_faph > 0 else 0.0

            print(
                f"{cfg_name:<20} {det.triggers_raw:>10} {faph_raw:>10.2f} "
                f"{det.triggers_debounced:>10} {faph_deb:>10.2f} {reduction:>9.1f}%"
            )

            key = f"{cfg_name}@{t}"
            results_data[key] = {
                "config": cfg_name,
                "threshold": t,
                "n_required": n_req,
                "window_size": win_size,
                "triggers_raw": det.triggers_raw,
                "faph_raw": round(faph_raw, 2),
                "triggers_debounced": det.triggers_debounced,
                "faph_debounced": round(faph_deb, 2),
                "reduction_pct": round(reduction, 1),
            }

    # ── Temporal Analysis ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEMPORAL ANALYSIS: False Alarm Streak Distribution")
    print(f"{'='*70}")

    temporal_data = {}
    for t in THRESHOLDS:
        streaks = streak_tracker[t]
        if not streaks:
            print(f"  Threshold {t:.2f}: no false alarm events")
            continue

        arr = np.array(streaks)
        n_transient = int((arr == 1).sum())
        n_sustained = int((arr >= 2).sum())
        n_long = int((arr >= 3).sum())
        total = len(arr)

        print(f"  Threshold {t:.2f}: {total} events")
        print(f"    Transient (1 window):  {n_transient} ({n_transient/total*100:.1f}%)")
        print(f"    Sustained (2+ windows): {n_sustained} ({n_sustained/total*100:.1f}%)")
        print(f"    Long (3+ windows):     {n_long} ({n_long/total*100:.1f}%)")
        print(f"    Mean streak: {arr.mean():.2f}, Max: {int(arr.max())}, Median: {np.median(arr):.1f}")

        temporal_data[str(t)] = {
            "total_events": total,
            "transient_1": n_transient,
            "sustained_2plus": n_sustained,
            "long_3plus": n_long,
            "transient_pct": round(n_transient / total * 100, 1),
            "mean_streak": round(float(arr.mean()), 2),
            "max_streak": int(arr.max()),
            "median_streak": round(float(np.median(arr)), 1),
        }

    # ── Top-50 False Alarm Temporal Context ──────────────────────────
    print(f"\n{'='*70}")
    print("TOP-50 FALSE ALARM TEMPORAL CONTEXT")
    print(f"{'='*70}")

    context_radius = 5
    temporal_context_list = []
    pattern_counts = {"TRANSIENT": 0, "BRIEF": 0, "SUSTAINED": 0}

    for rank, (sc, fp, tsec) in enumerate(top_windows, 1):
        scores = file_score_map.get(fp, [])
        widx = round(tsec / (STEP_SAMPLES / SAMPLE_RATE))
        before = scores[max(0, widx - context_radius):widx]
        after = scores[widx + 1:widx + 1 + context_radius]
        # Pad before if near start of file
        before = [0.0] * (context_radius - len(before)) + before

        # Classify: count consecutive windows >= 0.80 around center
        region = before + [sc] + after
        max_consec = 0
        cur = 0
        for s in region:
            if s >= 0.80:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        if max_consec >= 3:
            pat = "SUSTAINED"
        elif max_consec == 2:
            pat = "BRIEF"
        else:
            pat = "TRANSIENT"
        pattern_counts[pat] += 1

        before_str = "  ".join(f"{s:.2f}" for s in before)
        after_str = "  ".join(f"{s:.2f}" for s in after[:context_radius])

        if rank <= 20:
            print(f"\n  #{rank}: {fp} @ {tsec:.1f}s")
            print(f"    {before_str}  [{sc:.4f}]  {after_str}")
            print(f"    Pattern: {pat} ({max_consec} consecutive >= 0.80)")

        temporal_context_list.append({
            "rank": rank, "score": round(sc, 6), "file": fp,
            "time_sec": round(tsec, 2), "pattern": pat,
            "max_consecutive_above_80": max_consec,
            "before": [round(s, 4) for s in before],
            "after": [round(s, 4) for s in after[:context_radius]],
        })

    total_fa = len(top_windows)
    print(f"\n  Pattern distribution (top {total_fa}):")
    for p in ["TRANSIENT", "BRIEF", "SUSTAINED"]:
        c = pattern_counts[p]
        pct = c / total_fa * 100 if total_fa > 0 else 0
        print(f"    {p:<12}: {c:>3} ({pct:.0f}%)")
    print(f"  2-of-3 would eliminate: {pattern_counts['TRANSIENT']}/{total_fa} "
          f"({pattern_counts['TRANSIENT']/total_fa*100:.0f}%) of top false alarms")

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "model": args.model.name,
        "corpus": str(args.librispeech_dir),
        "total_hours": round(total_hours, 4),
        "total_windows": total_windows,
        "total_files": len(flac_files),
        "elapsed_seconds": round(elapsed, 1),
        "configs": {c[0]: {"n_required": c[1], "window_size": c[2]} for c in CONFIGS},
        "results": results_data,
        "temporal_analysis": temporal_data,
        "top_fa_context": temporal_context_list,
        "pattern_counts": pattern_counts,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # ── Positive TP Evaluation ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("POSITIVE AUDIO TP EVALUATION")
    print(f"{'='*70}")

    pos_dirs = [
        WAKEWORD_ROOT / "experiments" / "eval_fresh" / "positives",
        WAKEWORD_ROOT / "experiments" / "training_data" / "viola_contexts",
    ]
    pos_files: list[Path] = []
    for pd in pos_dirs:
        if pd.exists():
            for ext in ("*.wav", "*.flac"):
                pos_files.extend(sorted(pd.glob(ext)))

    if pos_files:
        # Sample up to 30 diverse files
        if len(pos_files) > 30:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pos_files), 30, replace=False)
            pos_files = [pos_files[i] for i in sorted(idx)]

        print(f"Testing {len(pos_files)} positive files")

        # Reset detectors for positive eval
        pos_detectors: dict[tuple[str, float], MultiWindowDetector] = {}
        for cfg_name, n_req, win_size in CONFIGS:
            for t in THRESHOLDS:
                pos_detectors[(cfg_name, t)] = MultiWindowDetector(n_req, win_size, DEBOUNCE_SAMPLES)

        pos_detection = {cfg: {t: 0 for t in THRESHOLDS} for cfg, _, _ in CONFIGS}
        pos_max_scores = []

        for pf in pos_files:
            try:
                audio, sr = sf.read(pf, dtype="float32")
            except Exception as e:
                print(f"  WARN: {pf.name}: {e}")
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

            n_samp = len(audio)
            if n_samp < CLIP_SAMPLES:
                audio = np.pad(audio, (0, CLIP_SAMPLES - n_samp))
                n_samp = len(audio)

            # Reset detectors per file
            for det in pos_detectors.values():
                det.reset()

            file_scores = []
            pos_s = 0
            while pos_s + CLIP_SAMPLES <= n_samp:
                w = audio[pos_s:pos_s + CLIP_SAMPLES]
                w16 = float_to_int16(w)
                emb = extract_embedding(preprocessor, w16)
                sc = score_embedding(session, emb)
                file_scores.append(sc)

                for t in THRESHOLDS:
                    above = sc >= t
                    for cfg_name, n_req, win_size in CONFIGS:
                        pos_detectors[(cfg_name, t)].update(above, pos_s)

                pos_s += STEP_SAMPLES

            if file_scores:
                ms = max(file_scores)
                above_80 = sum(1 for s in file_scores if s >= 0.8)
                pos_max_scores.append((pf.name, ms, len(file_scores), above_80))

            # Check if each config triggered at least once
            for cfg_name, n_req, win_size in CONFIGS:
                for t in THRESHOLDS:
                    if pos_detectors[(cfg_name, t)].triggers_debounced > 0:
                        pos_detection[cfg_name][t] += 1

        n_pos = len(pos_max_scores)
        pos_max_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\nPer-file max scores:")
        print(f"{'File':<55} {'Max':>8} {'Win':>5} {'>0.8':>5}")
        print("-" * 75)
        for fn, ms, nw, a80 in pos_max_scores[:30]:
            print(f"{fn:<55} {ms:>8.4f} {nw:>5} {a80:>5}")

        print(f"\nDetection rates ({n_pos} positive files):")
        header = f"{'Config':<12}"
        for t in THRESHOLDS:
            header += f"{'@'+str(t):<14}"
        print(header)
        print("-" * (12 + 14 * len(THRESHOLDS)))
        pos_results = {}
        for cfg_name, _, _ in CONFIGS:
            row = f"{cfg_name:<12}"
            cfg_data = {}
            for t in THRESHOLDS:
                d = pos_detection[cfg_name][t]
                rate = d / n_pos * 100 if n_pos > 0 else 0
                row += f"{d}/{n_pos} ({rate:.0f}%)   "
                cfg_data[str(t)] = {"detected": d, "rate_pct": round(rate, 1)}
            print(row)
            pos_results[cfg_name] = cfg_data

        output["positive_eval"] = {"total_files": n_pos, "configs": pos_results}

        # Re-save with positive eval
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: FAPH by config at threshold 0.80")
    print(f"{'='*70}")
    for cfg_name, n_req, win_size in CONFIGS:
        key = f"{cfg_name}@0.8"
        if key in results_data:
            r = results_data[key]
            print(f"  {cfg_name:<20}: {r['faph_debounced']:.2f} FAPH ({r['reduction_pct']:.0f}% reduction)")


if __name__ == "__main__":
    main()
