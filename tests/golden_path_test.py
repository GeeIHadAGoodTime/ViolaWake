"""Golden path test: generate 10 recordings, train TemporalCNN, benchmark.

Proves the full pipeline works end-to-end after the AudioContext fix.
Uses edge-tts to simulate 10 diverse user recordings.
"""

import asyncio
import io
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

# 10 diverse voices to simulate 10 different recording sessions
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

WAKE_WORD = "big chungus"


def _save_wav(audio_int16: np.ndarray, path: Path, sr: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


async def generate_recordings(output_dir: Path) -> list[Path]:
    """Generate 10 simulated user recordings via edge-tts."""
    import edge_tts
    from pydub import AudioSegment

    output_dir.mkdir(parents=True, exist_ok=True)
    recordings = []

    for i, voice in enumerate(RECORDING_VOICES):
        out_path = output_dir / f"recording_{i:02d}.wav"
        if out_path.exists() and out_path.stat().st_size > 1000:
            recordings.append(out_path)
            print(f"  [{i+1}/10] {voice}: cached")
            continue

        try:
            communicate = edge_tts.Communicate(WAKE_WORD, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])

            mp3_data = buf.getvalue()
            if len(mp3_data) < 100:
                print(f"  [{i+1}/10] {voice}: SKIP (empty response)")
                continue

            seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            seg.export(str(out_path), format="wav")
            recordings.append(out_path)

            # Verify non-silent
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            rms = np.sqrt(np.mean(samples ** 2))
            print(f"  [{i+1}/10] {voice}: OK ({out_path.stat().st_size:,} bytes, RMS={rms:.1f})")

        except Exception as e:
            print(f"  [{i+1}/10] {voice}: FAIL ({e})")

    return recordings


def verify_recordings(recordings: list[Path]) -> bool:
    """Verify all recordings have actual audio data (not silent)."""
    print(f"\n=== Recording Verification ===")
    all_ok = True
    for path in recordings:
        with wave.open(str(path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(samples ** 2))
            max_val = np.max(np.abs(samples))
            is_silent = rms < 10.0  # Same threshold as server-side check

            status = "SILENT!" if is_silent else "OK"
            if is_silent:
                all_ok = False
            print(f"  {path.name}: RMS={rms:.1f}, max={max_val:.0f}, {status}")

    return all_ok


def run_training(recordings: list[Path], output_dir: Path) -> Path:
    """Run full TemporalCNN training pipeline."""
    from violawake_sdk.tools.train import (
        _generate_confusable_negatives,
        _generate_speech_negatives,
        _train_temporal_cnn,
    )

    corpus_dir = output_dir / "corpus"
    model_path = output_dir / "golden_path_model.onnx"
    neg_tag_map: dict[str, list[Path]] = {}

    # Confusable negatives
    print("\n=== Generating Confusable Negatives ===")
    conf = _generate_confusable_negatives(
        WAKE_WORD, corpus_dir / "confusables",
        n_confusables=20, voices_per_word=8, verbose=True,
    )
    if conf:
        neg_tag_map["neg_confusable"] = conf
        print(f"  Generated: {len(conf)} files")

    # Speech negatives
    print("\n=== Generating Speech Negatives ===")
    speech = _generate_speech_negatives(
        corpus_dir / "speech", n_voices=5, verbose=True,
    )
    if speech:
        neg_tag_map["neg_speech"] = speech
        print(f"  Generated: {len(speech)} files")

    all_neg = []
    for files in neg_tag_map.values():
        all_neg.extend(files)

    print(f"\n=== Training TemporalCNN ===")
    print(f"  Positives: {len(recordings)}")
    print(f"  Negatives: {len(all_neg)}")
    print(f"  Architecture: TemporalCNN(96, 9)")

    start = time.monotonic()

    def on_progress(info):
        epoch = info.get("epoch", 0)
        total = info.get("total_epochs", 80)
        tl = info.get("train_loss", 0)
        vl = info.get("val_loss", 0)
        if epoch % 10 == 0 or epoch == total:
            print(f"  Epoch {epoch}/{total} -- train: {tl:.4f}, val: {vl:.4f}")
            sys.stdout.flush()

    _train_temporal_cnn(
        pos_files=recordings,
        neg_files=all_neg,
        output_path=model_path,
        wake_word=WAKE_WORD,
        epochs=80,
        augment=True,
        eval_dir=None,
        verbose=True,
        progress_callback=on_progress,
        neg_tags=neg_tag_map,
    )

    elapsed = time.monotonic() - start
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Model: {model_path} ({model_path.stat().st_size:,} bytes)")

    return model_path


def report_results(model_path: Path) -> dict:
    """Read config and report quality gate results."""
    config_path = model_path.with_suffix(".config.json")
    if not config_path.exists():
        print("  WARNING: No config.json found")
        return {}

    cfg = json.loads(config_path.read_text())

    print(f"\n{'='*60}")
    print(f"  GOLDEN PATH TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Wake word:        {WAKE_WORD}")
    print(f"  Architecture:     TemporalCNN(96, 9)")
    print(f"  Model size:       {model_path.stat().st_size:,} bytes")
    print(f"  d-prime:          {cfg.get('d_prime', 'N/A')}")
    print(f"  Threshold:        {cfg.get('threshold', 'N/A')}")

    # Quality gate
    qg = cfg.get("quality_gate", {})
    if qg:
        print(f"  Quality Grade:    {qg.get('grade', 'N/A')}")
        print(f"  Silence Max:      {qg.get('silence_max_score', 'N/A')}")
        print(f"  Speech FP Rate:   {qg.get('speech_fp_rate', 'N/A')}")
        print(f"  Confusable FP:    {qg.get('confusable_fp_rate', 'N/A')}")

    # Training config
    tc = cfg.get("training_config", {})
    if tc:
        print(f"  Epochs:           {tc.get('epochs', 'N/A')}")
        print(f"  Positives:        {tc.get('n_positives', 'N/A')}")
        print(f"  Negatives:        {tc.get('n_negatives', 'N/A')}")

    grade = qg.get("grade", "F")
    if grade in ("A", "B"):
        print(f"\n  PASS -- Golden path produces grade {grade} model")
    elif grade == "C":
        print(f"\n  MARGINAL -- Grade C, acceptable for testing")
    else:
        print(f"\n  FAIL -- Grade {grade}, pipeline needs investigation")

    return cfg


def main():
    print("=" * 60)
    print("  VIOLAWAKE GOLDEN PATH TEST")
    print(f"  Wake word: '{WAKE_WORD}'")
    print("=" * 60)

    work_dir = Path(tempfile.mkdtemp(prefix="violawake_golden_"))
    rec_dir = work_dir / "recordings"

    # Step 1: Generate 10 recordings
    print("\n=== Step 1: Generate 10 Recordings (edge-tts) ===")
    recordings = asyncio.run(generate_recordings(rec_dir))
    print(f"  Generated: {len(recordings)} recordings")

    if len(recordings) < 5:
        print("FATAL: Not enough recordings generated", file=sys.stderr)
        sys.exit(1)

    # Step 2: Verify recordings are not silent
    if not verify_recordings(recordings):
        print("FATAL: Silent recordings detected!", file=sys.stderr)
        sys.exit(1)

    # Step 3: Train
    model_path = run_training(recordings, work_dir)

    # Step 4: Report
    cfg = report_results(model_path)

    # Step 5: Quick inference test
    print("\n=== Inference Test ===")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f"  Model loads: YES")
        print(f"  Input: {input_name} shape={input_shape}")

        # Feed random input to verify model runs
        dummy = np.random.randn(*[s if isinstance(s, int) else 1 for s in input_shape]).astype(np.float32)
        result = sess.run(None, {input_name: dummy})
        print(f"  Inference: OK (output shape={result[0].shape})")
    except Exception as e:
        print(f"  Inference: FAIL ({e})")

    print(f"\n  Work directory: {work_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
