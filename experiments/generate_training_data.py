#!/usr/bin/env python3
"""Generate TTS training data for wake word model improvement experiments.

Part 1: Confusable-word hard negatives (phonetically similar to "viola")
Part 2: Diverse TTS positives with augmentation (noise + reverb)

All audio is output as 16kHz mono 16-bit WAV.

Voice selection: Only uses voices verified to produce audio via the Edge TTS
API as of 2026-03-26.  All 22 eval-set voices are excluded.  Confusable and
positive voice pools are disjoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent / "training_data"
CONFUSABLE_DIR = BASE_DIR / "confusable_negatives"
POSITIVE_DIR = BASE_DIR / "diverse_positives"

SAMPLE_RATE = 16000
MAX_RETRIES = 3
RETRY_DELAY = 2.0          # seconds between retries
INTER_REQUEST_DELAY = 0.4   # seconds between successful requests

# 22 eval-set voices — NEVER use these for training
EVAL_VOICES = {
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-US-AnaNeural", "en-US-AndrewNeural", "en-US-AriaNeural", "en-US-BrianNeural",
    "en-US-ChristopherNeural", "en-US-EmmaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
}

# Part 1 voices — confusable negatives (13 verified-working voices)
CONFUSABLE_VOICES = [
    "en-US-AvaNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-GB-MaisieNeural",
    "en-GB-MiaNeural",
    "en-KE-AsiliaNeural",
    "en-KE-ChilembaNeural",
    "en-NG-AbeoNeural",
    "en-NG-EzinneNeural",
]

# Part 2 voices — diverse positives (13 verified-working voices, disjoint from above)
POSITIVE_VOICES = [
    "en-AU-WilliamMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-HK-SamNeural",
    "en-HK-YanNeural",
    "en-NZ-MitchellNeural",
    "en-NZ-MollyNeural",
    "en-PH-JamesNeural",
    "en-PH-RosaNeural",
    "en-SG-LunaNeural",
    "en-SG-WayneNeural",
    "en-TZ-ElimuNeural",
    "en-TZ-ImaniNeural",
]

# Confusable words / phrases
CONFUSABLE_WORDS = [
    # Single words phonetically close to "viola"
    "vanilla", "villa", "violet", "vinyl", "villain", "village",
    "viper", "vista", "vivid",
    "valor", "valley", "valid", "venture", "vintage", "visual",
    "vital", "volume",
    # Short triggers
    "via",
    # Instrument usage in a sentence (contains "viola" but NOT as a wake word)
    "she plays the viola in the orchestra",
    "the viola section sounds beautiful",
    "hand me that viola please",
    # Common speech that might false-trigger
    "I'll be over",
    "I'll be over there",
    "fly over",
    "buy a villa",
]

# Positive phrases
POSITIVE_PHRASES = ["viola", "hey viola", "ok viola"]

# Augmentation parameters
NOISE_SNR_MIN_DB = 15
NOISE_SNR_MAX_DB = 25


# ---------------------------------------------------------------------------
# Safety check — no voice overlap between sets or with eval
# ---------------------------------------------------------------------------
def validate_voice_sets():
    conf_set = set(CONFUSABLE_VOICES)
    pos_set = set(POSITIVE_VOICES)

    overlap_eval_conf = conf_set & EVAL_VOICES
    overlap_eval_pos = pos_set & EVAL_VOICES
    overlap_conf_pos = conf_set & pos_set

    errors = []
    if overlap_eval_conf:
        errors.append(f"Confusable voices overlap with eval: {overlap_eval_conf}")
    if overlap_eval_pos:
        errors.append(f"Positive voices overlap with eval: {overlap_eval_pos}")
    if overlap_conf_pos:
        errors.append(f"Confusable and positive voices overlap: {overlap_conf_pos}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Voice set validation passed (conf={len(CONFUSABLE_VOICES)}, "
          f"pos={len(POSITIVE_VOICES)}, eval_excluded={len(EVAL_VOICES)}).")


# ---------------------------------------------------------------------------
# TTS generation with retry
# ---------------------------------------------------------------------------
async def generate_tts(text: str, voice: str, output_mp3: str) -> bool:
    """Synthesize text with Edge TTS, saving to MP3. Retries on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_mp3)
            # Verify file has content
            if os.path.getsize(output_mp3) > 100:
                return True
            else:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  FAILED ({attempt}/{MAX_RETRIES}): {voice} / {text!r} -- {e}")
            else:
                await asyncio.sleep(RETRY_DELAY)
    return False


def mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """Convert MP3 to 16kHz mono 16-bit WAV using pydub."""
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"  FAILED mp3->wav: {mp3_path} -- {e}")
        return False


def sanitize_filename(text: str) -> str:
    """Convert text to a safe filename component."""
    safe = text.lower().strip()
    safe = safe.replace("'", "")
    safe = safe.replace(" ", "_")
    # Remove anything that isn't alphanumeric or underscore
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    # Truncate long sentences
    if len(safe) > 60:
        safe = safe[:60]
    return safe


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def add_noise(signal: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add white noise at the specified SNR."""
    rms_signal = np.sqrt(np.mean(signal ** 2))
    if rms_signal < 1e-10:
        return signal
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.default_rng(seed).normal(0, rms_noise, len(signal))
    return (signal + noise).astype(np.float32)


def add_reverb(signal: np.ndarray, sr: int) -> np.ndarray:
    """Add simple synthetic reverb via exponential-decay impulse response."""
    # Create a synthetic room impulse response (~100ms)
    ir_len = int(0.1 * sr)  # 100ms
    t = np.arange(ir_len) / sr
    # Exponential decay with a few early reflections
    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0  # direct
    ir[int(0.012 * sr)] = 0.4   # early reflection 12ms
    ir[int(0.025 * sr)] = 0.25  # early reflection 25ms
    ir[int(0.040 * sr)] = 0.15  # early reflection 40ms
    # Late reverb tail
    decay = np.exp(-6.0 * t)
    rng = np.random.default_rng(123)
    late = rng.normal(0, 0.05, ir_len).astype(np.float32) * decay.astype(np.float32)
    ir += late
    ir = ir / np.max(np.abs(ir))  # normalize IR

    # Convolve
    wet = fftconvolve(signal, ir, mode="full")[:len(signal)]
    # Mix 70% dry + 30% wet
    mixed = 0.7 * signal + 0.3 * wet.astype(np.float32)
    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def augment_wav(wav_path: str, out_dir: Path, base_name: str,
                file_idx: int = 0) -> list[dict]:
    """Read a clean WAV and produce noisy + reverb variants. Returns metadata entries."""
    data, sr = sf.read(wav_path, dtype="float32")
    entries = []

    # Noisy variant — use file_idx as part of the seed for variety
    rng = np.random.default_rng(file_idx * 7 + 1)
    snr = rng.uniform(NOISE_SNR_MIN_DB, NOISE_SNR_MAX_DB)
    noisy = add_noise(data, snr, seed=file_idx * 7 + 1)
    noisy_path = out_dir / f"{base_name}_noisy.wav"
    sf.write(str(noisy_path), noisy, sr, subtype="PCM_16")
    entries.append({
        "file": str(noisy_path.relative_to(BASE_DIR)),
        "augmentation": "noise",
        "snr_db": round(float(snr), 1),
    })

    # Reverb variant
    reverbed = add_reverb(data, sr)
    reverb_path = out_dir / f"{base_name}_reverb.wav"
    sf.write(str(reverb_path), reverbed, sr, subtype="PCM_16")
    entries.append({
        "file": str(reverb_path.relative_to(BASE_DIR)),
        "augmentation": "reverb",
    })

    return entries


# ---------------------------------------------------------------------------
# Part 1: Confusable-word hard negatives
# ---------------------------------------------------------------------------
async def generate_confusables() -> tuple[int, int, list[dict]]:
    """Generate confusable-word negative samples. Returns (success, fail, manifest_entries)."""
    CONFUSABLE_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    fail = 0
    entries = []
    total = len(CONFUSABLE_VOICES) * len(CONFUSABLE_WORDS)
    done = 0

    for voice in CONFUSABLE_VOICES:
        for word in CONFUSABLE_WORDS:
            done += 1
            safe_word = sanitize_filename(word)
            base_name = f"{voice}_{safe_word}"
            wav_path = CONFUSABLE_DIR / f"{base_name}.wav"

            if wav_path.exists():
                success += 1
                entries.append({
                    "file": str(wav_path.relative_to(BASE_DIR)),
                    "voice": voice,
                    "text": word,
                    "type": "confusable_negative",
                    "augmentation": "clean",
                })
                if done % 50 == 0:
                    print(f"  [{done}/{total}] (skipped existing)")
                continue

            # Generate via TTS
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                mp3_path = tmp.name

            ok = await generate_tts(word, voice, mp3_path)
            if ok:
                ok = mp3_to_wav(mp3_path, str(wav_path))

            # Cleanup temp
            try:
                os.unlink(mp3_path)
            except OSError:
                pass

            if ok:
                success += 1
                entries.append({
                    "file": str(wav_path.relative_to(BASE_DIR)),
                    "voice": voice,
                    "text": word,
                    "type": "confusable_negative",
                    "augmentation": "clean",
                })
            else:
                fail += 1

            if done % 25 == 0:
                print(f"  [{done}/{total}] ...")

            await asyncio.sleep(INTER_REQUEST_DELAY)

    return success, fail, entries


# ---------------------------------------------------------------------------
# Part 2: Diverse TTS positives with augmentation
# ---------------------------------------------------------------------------
async def generate_positives() -> tuple[int, int, int, list[dict]]:
    """Generate positive samples + augmented variants.
    Returns (clean_count, augmented_count, fail_count, manifest_entries).
    """
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    clean_ok = 0
    aug_count = 0
    fail = 0
    entries = []
    total = len(POSITIVE_VOICES) * len(POSITIVE_PHRASES)
    done = 0

    for voice in POSITIVE_VOICES:
        for phrase in POSITIVE_PHRASES:
            done += 1
            safe_phrase = sanitize_filename(phrase)
            base_name = f"{voice}_{safe_phrase}"
            wav_path = POSITIVE_DIR / f"{base_name}.wav"

            already_existed = wav_path.exists()

            if not already_existed:
                # Generate via TTS
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    mp3_path = tmp.name

                ok = await generate_tts(phrase, voice, mp3_path)
                if ok:
                    ok = mp3_to_wav(mp3_path, str(wav_path))

                try:
                    os.unlink(mp3_path)
                except OSError:
                    pass

                if not ok:
                    fail += 1
                    continue

            clean_ok += 1
            entries.append({
                "file": str(wav_path.relative_to(BASE_DIR)),
                "voice": voice,
                "text": phrase,
                "type": "positive",
                "augmentation": "clean",
            })

            # Augment (noise + reverb)
            aug_entries = augment_wav(str(wav_path), POSITIVE_DIR, base_name,
                                     file_idx=done)
            for ae in aug_entries:
                ae["voice"] = voice
                ae["text"] = phrase
                ae["type"] = "positive"
            entries.extend(aug_entries)
            aug_count += len(aug_entries)

            if done % 10 == 0:
                print(f"  [{done}/{total}] ...")

            await asyncio.sleep(INTER_REQUEST_DELAY)

    return clean_ok, aug_count, fail, entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    t0 = time.time()
    validate_voice_sets()

    print()
    print("=" * 60)
    print("Part 1: Confusable-word hard negatives")
    print("=" * 60)
    print(f"  Voices: {len(CONFUSABLE_VOICES)}")
    print(f"  Words/phrases: {len(CONFUSABLE_WORDS)}")
    print(f"  Expected files: {len(CONFUSABLE_VOICES) * len(CONFUSABLE_WORDS)}")
    print()

    conf_ok, conf_fail, conf_entries = await generate_confusables()

    print()
    print("=" * 60)
    print("Part 2: Diverse TTS positives")
    print("=" * 60)
    print(f"  Voices: {len(POSITIVE_VOICES)}")
    print(f"  Phrases: {len(POSITIVE_PHRASES)}")
    print(f"  Expected clean: {len(POSITIVE_VOICES) * len(POSITIVE_PHRASES)}")
    print(f"  Expected augmented: {len(POSITIVE_VOICES) * len(POSITIVE_PHRASES) * 2}")
    print()

    pos_clean, pos_aug, pos_fail, pos_entries = await generate_positives()

    # Write manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_rate": SAMPLE_RATE,
        "format": "16kHz mono 16-bit WAV",
        "eval_voices_excluded": sorted(EVAL_VOICES),
        "confusable_negatives": {
            "voices": CONFUSABLE_VOICES,
            "words": CONFUSABLE_WORDS,
            "total_files": conf_ok,
            "failures": conf_fail,
        },
        "diverse_positives": {
            "voices": POSITIVE_VOICES,
            "phrases": POSITIVE_PHRASES,
            "clean_files": pos_clean,
            "augmented_files": pos_aug,
            "total_files": pos_clean + pos_aug,
            "failures": pos_fail,
        },
        "files": conf_entries + pos_entries,
    }

    manifest_path = BASE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Compute total size
    total_bytes = 0
    for entry in manifest["files"]:
        fp = BASE_DIR / entry["file"]
        if fp.exists():
            total_bytes += fp.stat().st_size

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Confusable negatives: {conf_ok} generated, {conf_fail} failed")
    print(f"  Positive clean:       {pos_clean} generated, {pos_fail} failed")
    print(f"  Positive augmented:   {pos_aug} generated")
    print(f"  Total files:          {conf_ok + pos_clean + pos_aug}")
    print(f"  Total size:           {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Manifest:             {manifest_path}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
