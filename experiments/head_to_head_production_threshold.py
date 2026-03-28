"""Quick production-threshold evaluation at threshold=0.80."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAKEWORD / "src"))

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
CLIP_SAMPLES = 24000

TEST_DIR = WAKEWORD / "eval_clean"
POS_DIR = TEST_DIR / "positives"
NEG_DIR = TEST_DIR / "negatives"


def load_audio_clip(path: Path) -> np.ndarray | None:
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        else:
            audio = audio[:CLIP_SAMPLES]
        return (audio * 32767).clip(-32768, 32767).astype(np.int16)
    except Exception:
        return None


def score_clip_streaming(detector, audio_int16: np.ndarray) -> float:
    detector.reset()
    max_score = 0.0
    for start in range(0, len(audio_int16) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = audio_int16[start:start + FRAME_SAMPLES].tobytes()
        score = detector.process(frame)
        max_score = max(max_score, score)
    return max_score


def main():
    from violawake_sdk.wake_detector import WakeDetector

    pos_files = sorted(list(POS_DIR.rglob("*.wav")) + list(POS_DIR.rglob("*.flac")))
    neg_files = sorted(list(NEG_DIR.rglob("*.wav")) + list(NEG_DIR.rglob("*.flac")))

    temporal_cnn_path = str(WAKEWORD / "experiments" / "models" / "j5_temporal" / "temporal_cnn.onnx")
    r3_path = str(WAKEWORD / "experiments" / "models" / "r3_10x_s42.onnx")

    for threshold in [0.50, 0.70, 0.80, 0.90]:
        print(f"\n{'='*60}")
        print(f"THRESHOLD = {threshold}")
        print(f"{'='*60}")

        for model_name, model_path in [("temporal_cnn", temporal_cnn_path), ("r3_10x_s42", r3_path)]:
            detector = WakeDetector(model=model_path, threshold=0.01, cooldown_s=0.0)

            pos_scores = []
            for f in pos_files:
                audio = load_audio_clip(f)
                if audio is None:
                    continue
                pos_scores.append(score_clip_streaming(detector, audio))

            neg_scores = []
            for f in neg_files:
                audio = load_audio_clip(f)
                if audio is None:
                    continue
                neg_scores.append(score_clip_streaming(detector, audio))

            pos_arr = np.array(pos_scores)
            neg_arr = np.array(neg_scores)

            tp = int(np.sum(pos_arr >= threshold))
            fn = int(np.sum(pos_arr < threshold))
            fp = int(np.sum(neg_arr >= threshold))
            tn = int(np.sum(neg_arr < threshold))

            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (tp + fn) if (tp + fn) > 0 else 0
            neg_hours = len(neg_arr) * 1.5 / 3600
            far_per_hour = fp / neg_hours if neg_hours > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  {model_name:20s}: FAR={far:.4f} FRR={frr:.4f} FAR/hr={far_per_hour:.1f} "
                  f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()
