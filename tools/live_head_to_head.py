#!/usr/bin/env python3
"""Live head-to-head comparison of 4 wake word models on the same microphone.

Models compared:
  1. ViolaWake Temporal CNN  — "viola" (production)
  2. OWW Alexa               — "alexa" (pre-trained baseline)
  3. ViolaWake Operator v2   — "operator" (custom-trained proof-of-concept)
  4. ViolaWake MLP (old)     — "viola" (deprecated r3_10x_s42)

Usage:
    python tools/live_head_to_head.py
    python tools/live_head_to_head.py --threshold 0.70   # more sensitive
    python tools/live_head_to_head.py --device 1          # specific mic

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape codes on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Constants ────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
VW_FRAME_SAMPLES = 320      # 20ms — ViolaWake's native frame size
OWW_FRAME_SAMPLES = 1_280   # 80ms — OWW's native chunk size
# Read OWW-sized chunks from mic, split into 4x ViolaWake sub-frames
MIC_FRAME_SAMPLES = OWW_FRAME_SAMPLES
SUB_FRAMES_PER_READ = OWW_FRAME_SAMPLES // VW_FRAME_SAMPLES  # 4

# ANSI colors
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE  = "\033[94m"
MAGENTA = "\033[95m"
CYAN  = "\033[96m"
WHITE = "\033[97m"

MODEL_COLORS = {
    "ViolaWake Temporal CNN (viola)": GREEN,
    "OWW Alexa (pre-trained)":       YELLOW,
    "ViolaWake Operator v2":         CYAN,
    "ViolaWake MLP old (viola)":     MAGENTA,
}


def rms_bar(rms: float, width: int = 30) -> str:
    """Render a simple RMS meter bar."""
    # rms is in int16 scale (0–32768)
    level = min(rms / 8000.0, 1.0)  # normalize: speech ~2000-6000
    filled = int(level * width)
    if level < 0.05:
        color = DIM
    elif level < 0.3:
        color = GREEN
    elif level < 0.7:
        color = YELLOW
    else:
        color = RED
    bar = f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"
    return f"[{bar}] {rms:6.0f}"


def format_detection(name: str, score: float, threshold: float) -> str:
    """Format a detection event with color."""
    color = MODEL_COLORS.get(name, WHITE)
    marker = f"{BOLD}{color}★ DETECTED{RESET}"
    return (
        f"  {marker}  {color}{name}{RESET}  "
        f"score={BOLD}{score:.4f}{RESET}  "
        f"(threshold={threshold:.2f})"
    )


def format_score_line(scores: dict[str, float], thresholds: dict[str, float]) -> str:
    """Format current scores for all models in one line."""
    parts = []
    for name, score in scores.items():
        color = MODEL_COLORS.get(name, WHITE)
        thresh = thresholds[name]
        if score >= thresh:
            parts.append(f"{BOLD}{color}{score:.3f}{RESET}")
        elif score >= thresh * 0.7:
            parts.append(f"{color}{score:.3f}{RESET}")
        else:
            parts.append(f"{DIM}{score:.3f}{RESET}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live head-to-head wake word comparison")
    parser.add_argument("--threshold", "-t", type=float, default=0.50,
                        help="ViolaWake threshold (default: 0.50)")
    parser.add_argument("--oww-threshold", type=float, default=0.50,
                        help="OWW alexa threshold (default: 0.50)")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--show-scores", action="store_true", default=True,
                        help="Show live score readout (default: on)")
    parser.add_argument("--no-scores", dest="show_scores", action="store_false",
                        help="Hide live score readout")
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────
    print(f"\n{BOLD}Loading 4 models for head-to-head comparison...{RESET}\n")

    # 1. ViolaWake Temporal CNN (production "viola")
    print(f"  {GREEN}[1/4]{RESET} ViolaWake Temporal CNN (viola)...", end=" ", flush=True)
    from violawake_sdk import WakeDetector
    det_tcnn = WakeDetector(model="temporal_cnn", threshold=args.threshold, cooldown_s=1.5)
    print("OK")

    # 2. OWW Alexa (pre-trained)
    print(f"  {YELLOW}[2/4]{RESET} OWW Alexa (pre-trained)...", end=" ", flush=True)
    from openwakeword.model import Model as OWWModel  # type: ignore[import]
    oww_model = OWWModel(wakeword_models=["alexa"])
    print("OK")

    # 3. ViolaWake Operator v2 (custom-trained)
    print(f"  {CYAN}[3/4]{RESET} ViolaWake Operator v2...", end=" ", flush=True)
    operator_path = Path(__file__).resolve().parent.parent / "models" / "operator_v2.onnx"
    if not operator_path.exists():
        print(f"{RED}NOT FOUND{RESET} at {operator_path}")
        print("  Train with: violawake-train --word operator --output models/operator_v2.onnx")
        sys.exit(1)
    det_operator = WakeDetector(model=str(operator_path), threshold=args.threshold, cooldown_s=1.5)
    print("OK")

    # 4. ViolaWake MLP old (deprecated "viola")
    print(f"  {MAGENTA}[4/4]{RESET} ViolaWake MLP old (viola)...", end=" ", flush=True)
    det_mlp = WakeDetector(model="r3_10x_s42", threshold=args.threshold, cooldown_s=1.5)
    print("OK")

    # ── Open microphone ──────────────────────────────────────────────
    print(f"\n{BOLD}Opening microphone...{RESET}", end=" ", flush=True)
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=MIC_FRAME_SAMPLES,
            input_device_index=args.device,
        )
    except Exception as e:
        pa.terminate()
        print(f"{RED}FAILED{RESET}: {e}")
        sys.exit(1)
    print("OK")

    # ── Print header ─────────────────────────────────────────────────
    thresholds = {
        "ViolaWake Temporal CNN (viola)": args.threshold,
        "OWW Alexa (pre-trained)":       args.oww_threshold,
        "ViolaWake Operator v2":         args.threshold,
        "ViolaWake MLP old (viola)":     args.threshold,
    }

    print(f"\n{'═' * 72}")
    print(f"{BOLD}  LIVE HEAD-TO-HEAD WAKE WORD COMPARISON{RESET}")
    print(f"{'═' * 72}")
    print(f"  {GREEN}●{RESET} ViolaWake Temporal CNN  — say {BOLD}\"Viola\"{RESET}    (threshold={args.threshold:.2f})")
    print(f"  {YELLOW}●{RESET} OWW Alexa              — say {BOLD}\"Alexa\"{RESET}    (threshold={args.oww_threshold:.2f})")
    print(f"  {CYAN}●{RESET} ViolaWake Operator v2  — say {BOLD}\"Operator\"{RESET} (threshold={args.threshold:.2f})")
    print(f"  {MAGENTA}●{RESET} ViolaWake MLP old      — say {BOLD}\"Viola\"{RESET}    (threshold={args.threshold:.2f})")
    print(f"{'─' * 72}")
    print(f"  {DIM}Speak naturally. Detections print in real-time. Ctrl+C to stop.{RESET}")
    if args.show_scores:
        print(f"  {DIM}Scores: {GREEN}TCNN{RESET} | {YELLOW}OWW{RESET} | {CYAN}OP{RESET} | {MAGENTA}MLP{RESET}")
    print(f"{'─' * 72}\n")

    # ── Detection stats ──────────────────────────────────────────────
    stats = {name: {"detections": 0, "max_score": 0.0} for name in MODEL_COLORS}
    start_time = time.time()
    frame_count = 0

    # ── Main loop ────────────────────────────────────────────────────
    try:
        while True:
            # Read 1280 samples (80ms) — OWW's native chunk
            raw = stream.read(MIC_FRAME_SAMPLES, exception_on_overflow=False)
            frame_count += 1
            pcm = np.frombuffer(raw, dtype=np.int16)
            rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))

            # ── ViolaWake: split into 4x 320-sample (20ms) sub-frames ──
            # ViolaWake expects exactly 320 samples per detect() call.
            # Feeding oversized frames corrupts the temporal embedding buffer.
            tcnn_hit = False
            tcnn_score = 0.0
            operator_hit = False
            operator_score = 0.0
            mlp_hit = False
            mlp_score = 0.0

            for i in range(SUB_FRAMES_PER_READ):
                start = i * VW_FRAME_SAMPLES
                end = start + VW_FRAME_SAMPLES
                sub_raw = pcm[start:end].tobytes()

                if det_tcnn.detect(sub_raw):
                    tcnn_hit = True
                tcnn_score = max(tcnn_score, det_tcnn._last_score)

                if det_operator.detect(sub_raw):
                    operator_hit = True
                operator_score = max(operator_score, det_operator._last_score)

                if det_mlp.detect(sub_raw):
                    mlp_hit = True
                mlp_score = max(mlp_score, det_mlp._last_score)

            # ── OWW: feed full 1280-sample chunk ──
            oww_result = oww_model.predict(pcm)
            oww_score = oww_result.get("alexa", 0.0)
            oww_hit = oww_score >= args.oww_threshold

            # Update stats
            scores = {
                "ViolaWake Temporal CNN (viola)": tcnn_score,
                "OWW Alexa (pre-trained)":       oww_score,
                "ViolaWake Operator v2":         operator_score,
                "ViolaWake MLP old (viola)":     mlp_score,
            }

            for name, score in scores.items():
                if score > stats[name]["max_score"]:
                    stats[name]["max_score"] = score

            # Print detections (these are the events that matter)
            any_detection = False
            if tcnn_hit:
                stats["ViolaWake Temporal CNN (viola)"]["detections"] += 1
                print(format_detection("ViolaWake Temporal CNN (viola)", tcnn_score, args.threshold))
                any_detection = True
            if oww_hit:
                stats["OWW Alexa (pre-trained)"]["detections"] += 1
                print(format_detection("OWW Alexa (pre-trained)", oww_score, args.oww_threshold))
                any_detection = True
            if operator_hit:
                stats["ViolaWake Operator v2"]["detections"] += 1
                print(format_detection("ViolaWake Operator v2", operator_score, args.threshold))
                any_detection = True
            if mlp_hit:
                stats["ViolaWake MLP old (viola)"]["detections"] += 1
                print(format_detection("ViolaWake MLP old (viola)", mlp_score, args.threshold))
                any_detection = True

            if any_detection:
                print()  # blank line after detection group

            # Live score readout (every 5 reads = 400ms to reduce flicker)
            if args.show_scores and frame_count % 5 == 0 and not any_detection:
                meter = rms_bar(rms)
                score_line = format_score_line(scores, thresholds)
                elapsed = time.time() - start_time
                sys.stdout.write(
                    f"\r  {DIM}{elapsed:6.1f}s{RESET}  {meter}  {score_line}  "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        det_tcnn.close()
        det_operator.close()
        det_mlp.close()

    # ── Print summary ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n\n{'═' * 72}")
    print(f"{BOLD}  SESSION SUMMARY  ({elapsed:.1f}s, {frame_count} frames){RESET}")
    print(f"{'═' * 72}")
    for name, color in MODEL_COLORS.items():
        s = stats[name]
        print(
            f"  {color}●{RESET} {name:<35}  "
            f"detections={BOLD}{s['detections']}{RESET}  "
            f"max_score={s['max_score']:.4f}"
        )
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
