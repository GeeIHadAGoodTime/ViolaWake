"""Fast TTS data generation for experiments B/C/D.

Generates confusable-word negatives and diverse positives using Edge TTS.
Processes voices in parallel batches for speed.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent / "training_data"
CONFUSABLE_DIR = BASE_DIR / "confusable_negatives"
POSITIVE_DIR = BASE_DIR / "diverse_positives"

# Eval voices — NEVER use for training
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

# Confusable negative voices (15 voices, diverse accents)
NEG_VOICES = [
    "en-US-DavisNeural", "en-US-JaneNeural", "en-US-JasonNeural",
    "en-US-SaraNeural", "en-US-TonyNeural",
    "en-US-NancyNeural", "en-US-AmberNeural", "en-US-AshleyNeural",
    "en-US-BrandonNeural", "en-US-RogerNeural",
    "en-GB-AbbiNeural", "en-GB-AlfieNeural", "en-GB-BellaNeural",
    "en-GB-ElliotNeural", "en-GB-NoahNeural",
]

# Diverse positive voices (15 voices, different from eval and neg)
POS_VOICES = [
    "en-AU-AnnetteNeural", "en-AU-CarlyNeural", "en-AU-DarrenNeural",
    "en-AU-FreyaNeural", "en-AU-KenNeural",
    "en-GB-HollieNeural", "en-GB-MaisieNeural", "en-GB-OliverNeural",
    "en-GB-OliviaNeural", "en-GB-MiaNeural",
    "en-NZ-MollyNeural", "en-NZ-MitchellNeural",
    "en-PH-RosaNeural", "en-SG-LunaNeural", "en-HK-YanNeural",
]

CONFUSABLE_WORDS = [
    "vanilla", "villa", "violet", "vinyl", "villain", "village",
    "viper", "vista", "vivid", "valor", "valley", "valid",
    "venture", "vintage", "visual", "vital", "volume", "via",
    "she plays the viola in the orchestra",
    "the viola section sounds beautiful",
    "hand me that viola please",
    "buy a villa", "fly over", "I'll be over",
]

POSITIVE_PHRASES = ["viola", "hey viola", "ok viola"]


async def generate_tts(text: str, voice: str, output_path: Path) -> bool:
    """Generate a single TTS clip, save as 16kHz mono WAV."""
    if output_path.exists():
        return True  # skip if already generated
    try:
        import edge_tts
        import tempfile
        from pydub import AudioSegment

        # Use save() — more reliable than streaming for short text
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        communicate = edge_tts.Communicate(text + ".", voice)
        await communicate.save(tmp_path)

        # Convert MP3 to 16kHz mono WAV
        audio = AudioSegment.from_mp3(tmp_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(output_path), format="wav")

        import os
        os.unlink(tmp_path)
        return True
    except Exception as e:
        print(f"  FAIL: {voice} '{text[:30]}': {e}", file=sys.stderr)
        return False


def add_noise(audio_arr: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white noise at given SNR."""
    rms = np.sqrt(np.mean(audio_arr ** 2))
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.default_rng(42).normal(0, noise_rms, len(audio_arr))
    return np.clip(audio_arr + noise, -1.0, 1.0).astype(np.float32)


def add_reverb(audio_arr: np.ndarray, decay: float = 0.3, delay_ms: int = 30) -> np.ndarray:
    """Simple reverb via comb filter."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    ir = np.zeros(delay_samples * 4, dtype=np.float32)
    ir[0] = 1.0
    for i in range(1, 4):
        ir[i * delay_samples] = decay ** i
    from scipy.signal import fftconvolve
    result = fftconvolve(audio_arr, ir, mode="full")[:len(audio_arr)]
    return np.clip(result, -1.0, 1.0).astype(np.float32)


async def generate_confusable_negatives():
    """Generate confusable-word negatives across all voices."""
    CONFUSABLE_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    failed = 0

    for vi, voice in enumerate(NEG_VOICES):
        print(f"  Negatives: voice {vi+1}/{len(NEG_VOICES)}: {voice}")
        for word in CONFUSABLE_WORDS:
            safe_word = word.replace(" ", "_").replace("'", "")[:40]
            out_path = CONFUSABLE_DIR / f"{voice}_{safe_word}.wav"
            ok = await generate_tts(word, voice, out_path)
            if ok:
                total += 1
            else:
                failed += 1

    print(f"  Confusable negatives: {total} generated, {failed} failed")
    return total


async def generate_diverse_positives():
    """Generate diverse TTS positives with augmentation."""
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    failed = 0

    import soundfile as sf

    for vi, voice in enumerate(POS_VOICES):
        print(f"  Positives: voice {vi+1}/{len(POS_VOICES)}: {voice}")
        for phrase in POSITIVE_PHRASES:
            safe_phrase = phrase.replace(" ", "_")
            # Clean version
            clean_path = POSITIVE_DIR / f"{voice}_{safe_phrase}.wav"
            ok = await generate_tts(phrase, voice, clean_path)
            if not ok:
                failed += 1
                continue
            total += 1

            # Read clean audio for augmentation
            try:
                audio, sr = sf.read(str(clean_path), dtype="float32")
                if sr != SAMPLE_RATE:
                    # Resample if needed
                    from scipy.signal import resample
                    audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype(np.float32)

                # Noisy variant
                noisy = add_noise(audio, snr_db=18.0)
                noisy_path = POSITIVE_DIR / f"{voice}_{safe_phrase}_noisy.wav"
                sf.write(str(noisy_path), noisy, SAMPLE_RATE)
                total += 1

                # Reverb variant
                reverb = add_reverb(audio)
                reverb_path = POSITIVE_DIR / f"{voice}_{safe_phrase}_reverb.wav"
                sf.write(str(reverb_path), reverb, SAMPLE_RATE)
                total += 1
            except Exception as e:
                print(f"  Augmentation failed for {clean_path}: {e}")
                failed += 1

    print(f"  Diverse positives: {total} generated, {failed} failed")
    return total


async def main():
    t0 = time.time()
    print("Generating training data for experiments B/C/D...")
    print(f"Target: ~{len(NEG_VOICES) * len(CONFUSABLE_WORDS)} confusable negatives")
    print(f"Target: ~{len(POS_VOICES) * len(POSITIVE_PHRASES) * 3} diverse positives")
    print()

    # Generate both in sequence (Edge TTS doesn't handle parallel well)
    n_neg = await generate_confusable_negatives()
    n_pos = await generate_diverse_positives()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Total: {n_neg} confusable negatives + {n_pos} diverse positives")


if __name__ == "__main__":
    asyncio.run(main())
