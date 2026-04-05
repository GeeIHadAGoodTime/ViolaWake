"""Full production temporal CNN training with TTS voice diversity.

Replicates the exact CLI production pipeline for Console users:
1. TTS positive generation (20 voices x 3 phrases + noise/reverb variants)
2. Confusable negative generation (phonetically similar words)
3. Speech negative generation (common phrases)
4. Universal corpus (LibriSpeech, MUSAN) if available
5. TemporalCNN training with FocalLoss, AdamW, EMA

Usage:
    python scripts/train_full_pipeline.py \
        --positives "data/recordings/563/big chungus" \
        --wake-word "big chungus" \
        --output "data/models/563/big_chungus_production.onnx"
"""

import argparse
import asyncio
import io
import json
import random
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


# -- TTS positive generation using edge-tts directly --------------------------

EDGE_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AmberNeural", "en-US-AnaNeural",
    "en-US-AndrewNeural", "en-US-BrandonNeural", "en-US-ChristopherNeural",
    "en-US-CoraNeural", "en-US-ElizabethNeural", "en-US-EricNeural",
    "en-US-JacobNeural", "en-US-MichelleNeural", "en-US-MonicaNeural",
    "en-US-RogerNeural", "en-US-SteffanNeural", "en-GB-RyanNeural",
    "en-GB-SoniaNeural", "en-AU-NatashaNeural",
]


def _save_wav(audio: np.ndarray, path: Path, sr: int = 16000) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


async def _synthesize_one(text: str, voice: str) -> bytes:
    """Synthesize text with edge-tts, return MP3 bytes."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _mp3_to_wav16k(mp3_data: bytes, out_path: Path) -> bool:
    """Convert MP3 bytes to mono 16kHz WAV."""
    from pydub import AudioSegment
    seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
    seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    seg.export(str(out_path), format="wav")
    return out_path.exists()


def generate_tts_positives(wake_word: str, output_dir: Path, n_voices: int = 20) -> list[Path]:
    """Generate TTS positives: n_voices x 3 phrases + noise + reverb variants."""
    from violawake_sdk.training.augment import apply_additive_noise, rir_augment
    from violawake_sdk.audio import load_audio

    output_dir.mkdir(parents=True, exist_ok=True)
    phrases = [wake_word, f"hey {wake_word}", f"ok {wake_word}"]
    voices = EDGE_VOICES[:n_voices]
    generated: list[Path] = []
    rng = np.random.default_rng(42)

    total = len(voices) * len(phrases)
    print(f"  Generating: {len(voices)} voices x {len(phrases)} phrases = {total} clean samples")

    async def _batch_synth():
        results = []
        for vi, voice in enumerate(voices):
            for pi, phrase in enumerate(phrases):
                clean_path = output_dir / f"tts_{vi:02d}_{pi}_{voice}.wav"
                if clean_path.exists():
                    results.append(clean_path)
                    continue
                try:
                    mp3 = await _synthesize_one(phrase, voice)
                    if len(mp3) > 100 and _mp3_to_wav16k(mp3, clean_path):
                        results.append(clean_path)
                except Exception as e:
                    print(f"    WARN: {voice}/{phrase}: {e}", file=sys.stderr)
            if (vi + 1) % 5 == 0:
                print(f"    {vi + 1}/{len(voices)} voices done ({len(results)} files)")
        return results

    clean_files = asyncio.run(_batch_synth())
    generated.extend(clean_files)
    print(f"  Clean TTS files: {len(clean_files)}")

    # Generate noisy + reverb variants for each clean file
    n_aug = 0
    for i, clean_path in enumerate(clean_files):
        audio = load_audio(clean_path)
        if audio is None or len(audio) == 0:
            continue

        # Noisy variant (SNR 10-15 dB)
        noisy = apply_additive_noise(audio, snr_db=12.0, rng=rng)
        noisy_path = clean_path.with_name(clean_path.stem + "_noisy.wav")
        _save_wav(noisy, noisy_path)
        generated.append(noisy_path)
        n_aug += 1

        # Reverb variant
        reverbed = rir_augment(audio, rng=rng)
        reverb_path = clean_path.with_name(clean_path.stem + "_reverb.wav")
        _save_wav(reverbed, reverb_path)
        generated.append(reverb_path)
        n_aug += 1

    print(f"  + {n_aug} augmented variants = {len(generated)} total TTS positives")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Full production temporal CNN training")
    parser.add_argument("--positives", required=True, help="Directory with user's WAV recordings")
    parser.add_argument("--wake-word", required=True, help="Wake word text")
    parser.add_argument("--output", required=True, help="Output ONNX model path")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs")
    parser.add_argument("--n-voices", type=int, default=20, help="Number of TTS voices")
    args = parser.parse_args()

    pos_dir = Path(args.positives)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    user_pos = sorted(list(pos_dir.glob("*.wav")) + list(pos_dir.glob("*.flac")))
    print(f"=== ViolaWake TemporalCNN — Full Production Pipeline ===")
    print(f"Wake word: {args.wake_word}")
    print(f"User positives: {len(user_pos)}")
    print(f"Output: {output_path}")
    print()

    # Use a temp dir on the same drive as the project to avoid filling the system drive
    _tmp_base = Path(args.output).parent / "tmp"
    _tmp_base.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(tempfile.mkdtemp(prefix="violawake_full_", dir=str(_tmp_base)))

    # Step 1: TTS positive voice diversity
    print("Step 1: TTS positive generation (voice diversity + noise + reverb)...")
    tts_files = generate_tts_positives(args.wake_word, corpus_dir / "tts_pos", n_voices=args.n_voices)
    all_pos = user_pos + tts_files
    print(f"  Total positives: {len(all_pos)} ({len(user_pos)} user + {len(tts_files)} TTS)")
    print()

    # Step 2: Confusable negatives (2 rounds, matching production)
    print("Step 2: Confusable negatives (phonetically similar words)...")
    from violawake_sdk.tools.train import _generate_confusable_negatives
    neg_tag_map: dict[str, list[Path]] = {}

    conf_r1 = _generate_confusable_negatives(
        args.wake_word, corpus_dir / "conf_r1",
        n_confusables=30, voices_per_word=10, verbose=True,
    )
    if conf_r1:
        neg_tag_map["neg_confusable_r1"] = conf_r1

    conf_r2 = _generate_confusable_negatives(
        args.wake_word, corpus_dir / "conf_r2",
        n_confusables=16, voices_per_word=10, verbose=True,
    )
    if conf_r2:
        neg_tag_map["neg_confusable_r2"] = conf_r2
    print()

    # Step 3: Speech negatives
    print("Step 3: Speech negatives (common phrases)...")
    from violawake_sdk.tools.train import _generate_speech_negatives
    speech = _generate_speech_negatives(corpus_dir / "speech", n_voices=5, verbose=True)
    if speech:
        neg_tag_map["neg_speech"] = speech
    print()

    # Step 4: Universal corpus
    print("Step 4: Universal corpus (LibriSpeech, MUSAN)...")
    rng = random.Random(42)
    search_paths = [
        Path(__file__).resolve().parent.parent.parent / "corpus",  # repo root
        Path.home() / ".violawake" / "corpus",
        Path("corpus"),
    ]
    subdirs = {
        "neg_librispeech": ("librispeech",),
        "neg_musan_speech": ("musan/musan/speech", "musan/speech"),
        "neg_musan_music": ("musan/musan/music", "musan/music"),
        "neg_musan_noise": ("musan/musan/noise", "musan/noise"),
    }
    for tag, sds in subdirs.items():
        for root in search_paths:
            if not root.exists():
                continue
            for sd in sds:
                cand = root / sd
                if cand.exists():
                    cf = sorted(list(cand.rglob("*.wav")) + list(cand.rglob("*.flac")))
                    if cf:
                        if len(cf) > 2000:
                            cf = sorted(rng.sample(cf, 2000))
                        neg_tag_map[tag] = cf
                        print(f"  [{tag}]: {len(cf)} files")
                        break
            if tag in neg_tag_map:
                break

    all_neg: list[Path] = []
    for files in neg_tag_map.values():
        all_neg.extend(files)
    print(f"\n  Total negatives: {len(all_neg)}")
    print(f"  Tags: {list(neg_tag_map.keys())}")
    print()

    if len(all_neg) < 5:
        print("ERROR: Not enough negatives generated", file=sys.stderr)
        sys.exit(1)

    # Step 5: Train TemporalCNN
    print("Step 5: Training TemporalCNN...")
    print(f"  Positives: {len(all_pos)} (will be 10x augmented -> ~{len(all_pos) * 11})")
    print(f"  Negatives: {len(all_neg)}")
    print(f"  Architecture: TemporalCNN(96, 9)")
    print(f"  FocalLoss + AdamW + cosine annealing + EMA")
    print()

    from violawake_sdk.tools.train import _train_temporal_cnn

    start = time.monotonic()

    def on_progress(info):
        epoch = info.get("epoch", 0)
        total = info.get("total_epochs", args.epochs)
        tl = info.get("train_loss", 0)
        vl = info.get("val_loss", 0)
        if epoch % 10 == 0 or epoch == total:
            print(f"  Epoch {epoch}/{total} — train: {tl:.4f}, val: {vl:.4f}")
            sys.stdout.flush()

    _train_temporal_cnn(
        pos_files=all_pos,
        neg_files=all_neg,
        output_path=output_path,
        wake_word=args.wake_word,
        epochs=args.epochs,
        augment=True,
        eval_dir=None,
        verbose=True,
        progress_callback=on_progress,
        neg_tags=neg_tag_map,
        augment_source_files=user_pos,
    )

    elapsed = time.monotonic() - start
    print(f"\n=== Training Complete ({elapsed:.1f}s) ===")
    print(f"Model: {output_path} ({output_path.stat().st_size:,} bytes)")

    config_path = output_path.with_suffix(".config.json")
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        print(f"d_prime: {cfg.get('d_prime', 'N/A')}")
        print(f"threshold: {cfg.get('threshold', 'N/A')}")


if __name__ == "__main__":
    main()
