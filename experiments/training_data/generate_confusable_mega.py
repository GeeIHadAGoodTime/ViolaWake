"""
Generate massive confusable-word TTS negatives for ViolaWake training.

Produces 1000+ negative examples across 20+ voices, 3 variants each (clean, noisy, rate-shifted).
Also generates synthetic noise negatives (tones, white/pink noise, silence, chords).
"""

import asyncio
import json
import math
import os
import struct
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "confusable_mega"
NOISE_DIR = Path(__file__).parent / "noise_negatives"
TEMP_DIR = OUTPUT_DIR / "_temp_mp3"

SAMPLE_RATE = 16000
CHANNELS = 1

# Confusable words grouped by danger level
CONFUSABLE_WORDS = {
    # HIGH danger - phonetically very close
    "voila": "voilà",
    "hola": "hola",
    "vanilla": "vanilla",
    "villa": "villa",
    "viola_instrument": "viola",
    "via": "via",
    "valley": "valley",
    "violet": "violet",
    "violin": "violin",
    "viola_da_gamba": "viola da gamba",
    "vuela": "vuela",
    "violas": "violas",
    # MEDIUM danger - partial overlap
    "hey_violet": "hey violet",
    "how_are_you": "how are you",
    "what_time_is_it": "what time is it",
    "good_morning": "good morning",
    "hello": "hello",
    # LOW danger - other wake words (should reject)
    "alexa": "alexa",
    "hey_siri": "hey siri",
    "hey_google": "hey google",
    "okay_google": "okay google",
}

# Additional sentence contexts for high-danger words (embeds confusable in natural speech)
SENTENCE_CONTEXTS = {
    "voila_sentence1": "And voilà, the trick is done!",
    "voila_sentence2": "Voilà, here it is.",
    "hola_sentence1": "Hola, cómo estás?",
    "hola_sentence2": "She said hola to everyone.",
    "vanilla_sentence1": "I'd like a vanilla ice cream please.",
    "vanilla_sentence2": "The vanilla extract smells amazing.",
    "villa_sentence1": "We rented a villa in Tuscany.",
    "villa_sentence2": "The villa has a swimming pool.",
    "viola_sentence1": "She plays the viola in the orchestra.",
    "viola_sentence2": "The viola section sounds beautiful today.",
    "violin_sentence1": "He practiced violin for three hours.",
    "valley_sentence1": "The valley was covered in morning mist.",
    "violet_sentence1": "Her favorite color is violet.",
    "via_sentence1": "We traveled via the coastal route.",
    "how_are_you_sentence": "Hey, how are you doing today?",
}

# Voices - diverse selection covering required locales
VOICES = [
    # US English (6 voices - most important)
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
    "en-US-SaraNeural",
    "en-US-TonyNeural",
    # GB English (3)
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-GB-LibbyNeural",
    # AU English (2)
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    # CA English (2)
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",
    # IN English (2)
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    # IE English (1)
    "en-IE-EmilyNeural",
    # ZA English (1)
    "en-ZA-LeahNeural",
    # Non-English accents speaking English words (critical for confusables)
    "es-MX-DaliaNeural",       # Mexican Spanish
    "es-ES-ElviraNeural",      # Spain Spanish
    "fr-FR-DeniseNeural",      # French
    "fr-FR-HenriNeural",       # French male
    "de-DE-KatjaNeural",       # German
    "it-IT-IsabellaNeural",    # Italian
    "it-IT-DiegoNeural",       # Italian male
    "ja-JP-NanamiNeural",      # Japanese
]

# SSML rate variants for pitch/speed variation
RATE_VARIANTS = [
    ("clean", "medium"),    # Normal speed
    ("slow", "slow"),       # Slower
    ("fast", "fast"),       # Faster
]

# Noise levels for noisy variants
NOISE_AMPLITUDES = [0.01, 0.02, 0.03]

# Concurrency control
MAX_CONCURRENT_TTS = 10
MAX_CONCURRENT_CONVERT = 20


# ── Helpers ────────────────────────────────────────────────────────────────

def make_ssml(text: str, voice: str, rate: str = "medium") -> str:
    """Build SSML with rate control."""
    return f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="{voice}">
        <prosody rate="{rate}">{text}</prosody>
    </voice>
</speak>"""


async def generate_tts(text: str, voice: str, output_mp3: Path, rate: str = "medium") -> bool:
    """Generate TTS audio via edge_tts. Returns True on success."""
    import edge_tts
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="+0%" if rate == "medium" else ("-20%" if rate == "slow" else "+30%"),
        )
        await communicate.save(str(output_mp3))
        # Verify file exists and has content
        if output_mp3.exists() and output_mp3.stat().st_size > 100:
            return True
        return False
    except Exception as e:
        return False


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert mp3 to 16kHz mono WAV using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp3_path),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(wav_path),
            ],
            capture_output=True,
            timeout=30,
        )
        return wav_path.exists() and wav_path.stat().st_size > 100
    except Exception:
        return False


def add_noise_to_wav(input_wav: Path, output_wav: Path, noise_amplitude: float) -> bool:
    """Add Gaussian noise to a WAV file."""
    try:
        with wave.open(str(input_wav), 'rb') as wf:
            params = wf.getparams()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert to numpy
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # Add noise
        noise = np.random.normal(0, noise_amplitude * 32767, len(samples))
        noisy = np.clip(samples + noise, -32768, 32767).astype(np.int16)

        with wave.open(str(output_wav), 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(noisy.tobytes())

        return True
    except Exception:
        return False


def generate_sine_wav(output_path: Path, freq: float, duration: float = 2.0, amplitude: float = 0.8):
    """Generate a sine tone WAV."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    samples = (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def generate_white_noise_wav(output_path: Path, duration: float = 2.0, amplitude: float = 0.3):
    """Generate white noise WAV."""
    samples = (amplitude * 32767 * np.random.randn(int(SAMPLE_RATE * duration))).astype(np.int16)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def generate_pink_noise_wav(output_path: Path, duration: float = 2.0, amplitude: float = 0.3):
    """Generate pink noise (1/f) WAV."""
    n_samples = int(SAMPLE_RATE * duration)
    white = np.random.randn(n_samples)
    # Simple pink noise via cumulative filter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    try:
        pink = lfilter(b, a, white)
    except ImportError:
        # Fallback: simple approximation
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
    pink = pink / (np.max(np.abs(pink)) + 1e-10)
    samples = (amplitude * 32767 * pink).astype(np.int16)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def generate_silence_wav(output_path: Path, duration: float = 2.0):
    """Generate silence WAV."""
    samples = np.zeros(int(SAMPLE_RATE * duration), dtype=np.int16)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def generate_chord_wav(output_path: Path, duration: float = 2.0, amplitude: float = 0.5):
    """Generate a music-like chord (C major: C4, E4, G4)."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    samples = (amplitude * 32767 * signal).astype(np.int16)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


# ── Main Generation Pipeline ──────────────────────────────────────────────

async def generate_all_confusables():
    """Main async pipeline for generating all confusable negatives."""

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Merge words and sentence contexts
    all_items = {}
    all_items.update(CONFUSABLE_WORDS)
    all_items.update(SENTENCE_CONTEXTS)

    # Build task list: (word_key, text, voice, rate_label, rate_value)
    tasks = []
    for word_key, text in all_items.items():
        for voice in VOICES:
            for rate_label, rate_value in RATE_VARIANTS:
                tasks.append((word_key, text, voice, rate_label, rate_value))

    print(f"Total TTS generation tasks: {len(tasks)}")
    print(f"Words/phrases: {len(all_items)}")
    print(f"Voices: {len(VOICES)}")
    print(f"Rate variants: {len(RATE_VARIANTS)}")
    print()

    # Semaphore for concurrency control
    sem = asyncio.Semaphore(MAX_CONCURRENT_TTS)
    convert_sem = asyncio.Semaphore(MAX_CONCURRENT_CONVERT)

    stats = {
        "generated": 0,
        "failed": 0,
        "converted": 0,
        "noisy_variants": 0,
        "per_word": defaultdict(int),
        "per_voice": defaultdict(int),
    }

    progress_lock = asyncio.Lock()
    progress_count = 0
    total_tasks = len(tasks)

    async def process_one(word_key, text, voice, rate_label, rate_value):
        nonlocal progress_count
        voice_short = voice.split("-")[-1].replace("Neural", "")
        locale = "-".join(voice.split("-")[:2])
        safe_word = word_key.replace(" ", "_")

        base_name = f"{safe_word}__{locale}_{voice_short}__{rate_label}"
        mp3_path = TEMP_DIR / f"{base_name}.mp3"
        wav_path = OUTPUT_DIR / f"{base_name}.wav"

        async with sem:
            success = await generate_tts(text, voice, mp3_path, rate_value)

        if not success:
            async with progress_lock:
                stats["failed"] += 1
                progress_count += 1
                if progress_count % 100 == 0:
                    print(f"  Progress: {progress_count}/{total_tasks} (failed this one: {word_key}/{voice})")
            return

        # Convert mp3 -> wav
        async with convert_sem:
            converted = await asyncio.get_event_loop().run_in_executor(
                None, convert_mp3_to_wav, mp3_path, wav_path
            )

        if not converted:
            async with progress_lock:
                stats["failed"] += 1
                progress_count += 1
            return

        async with progress_lock:
            stats["generated"] += 1
            stats["converted"] += 1
            stats["per_word"][word_key] += 1
            stats["per_voice"][voice] += 1
            progress_count += 1
            if progress_count % 100 == 0:
                print(f"  Progress: {progress_count}/{total_tasks}")

        # Generate noisy variant (pick a random noise level)
        noise_amp = np.random.choice(NOISE_AMPLITUDES)
        noisy_path = OUTPUT_DIR / f"{base_name}__noisy{noise_amp:.2f}.wav"
        noisy_ok = await asyncio.get_event_loop().run_in_executor(
            None, add_noise_to_wav, wav_path, noisy_path, noise_amp
        )
        if noisy_ok:
            async with progress_lock:
                stats["noisy_variants"] += 1
                stats["per_word"][word_key] += 1
                stats["per_voice"][voice] += 1

        # Clean up mp3
        try:
            mp3_path.unlink()
        except Exception:
            pass

    # Run all tasks
    print("Starting TTS generation...")
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(
            *[process_one(*t) for t in batch],
            return_exceptions=True,
        )
        print(f"  Batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size} complete")

    return stats


def generate_noise_negatives():
    """Generate synthetic non-speech negatives."""
    NOISE_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    # Sine tones at various frequencies
    for freq in [500, 1000, 2000, 4000]:
        for dur in [1.0, 2.0, 3.0]:
            for amp in [0.3, 0.6, 0.9]:
                path = NOISE_DIR / f"sine_{freq}hz_{dur}s_amp{amp:.1f}.wav"
                generate_sine_wav(path, freq, dur, amp)
                count += 1

    # White noise at various amplitudes and durations
    for dur in [1.0, 2.0, 3.0]:
        for amp in [0.1, 0.3, 0.5, 0.8]:
            path = NOISE_DIR / f"white_noise_{dur}s_amp{amp:.1f}.wav"
            generate_white_noise_wav(path, dur, amp)
            count += 1

    # Pink noise
    for dur in [1.0, 2.0, 3.0]:
        for amp in [0.1, 0.3, 0.5]:
            path = NOISE_DIR / f"pink_noise_{dur}s_amp{amp:.1f}.wav"
            generate_pink_noise_wav(path, dur, amp)
            count += 1

    # Silence
    for dur in [1.0, 2.0, 3.0, 5.0]:
        path = NOISE_DIR / f"silence_{dur}s.wav"
        generate_silence_wav(path, dur)
        count += 1

    # Chords
    for dur in [1.0, 2.0, 3.0]:
        for amp in [0.3, 0.6, 0.9]:
            path = NOISE_DIR / f"chord_cmaj_{dur}s_amp{amp:.1f}.wav"
            generate_chord_wav(path, dur, amp)
            count += 1

    # Frequency sweeps (chirp)
    for dur in [2.0, 3.0]:
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        freq = np.linspace(200, 4000, len(t))
        signal = (0.5 * 32767 * np.sin(2 * np.pi * freq * t / SAMPLE_RATE * np.cumsum(np.ones_like(t)) / SAMPLE_RATE)).astype(np.int16)
        # Simpler chirp
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        signal = (0.5 * 32767 * np.sin(phase)).astype(np.int16)
        path = NOISE_DIR / f"chirp_200_4000hz_{dur}s.wav"
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(signal.tobytes())
        count += 1

    print(f"Generated {count} noise negatives in {NOISE_DIR}")
    return count


async def main():
    print("=" * 70)
    print("ViolaWake Confusable Mega Negative Generator")
    print("=" * 70)
    print()

    # Phase 1: Noise negatives (fast, synchronous)
    print("Phase 1: Generating noise negatives...")
    noise_count = generate_noise_negatives()
    print()

    # Phase 2: TTS confusables (async, slow)
    print("Phase 2: Generating TTS confusables...")
    stats = await generate_all_confusables()
    print()

    # Clean up temp directory
    try:
        import shutil
        temp = OUTPUT_DIR / "_temp_mp3"
        if temp.exists():
            shutil.rmtree(temp)
    except Exception:
        pass

    # ── Report ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("GENERATION REPORT")
    print("=" * 70)
    print()

    print("Files per word/phrase:")
    for word, count in sorted(stats["per_word"].items(), key=lambda x: -x[1]):
        print(f"  {word:30s} {count:5d}")

    print()
    print("Files per voice:")
    for voice, count in sorted(stats["per_voice"].items(), key=lambda x: -x[1]):
        print(f"  {voice:45s} {count:5d}")

    total_confusable = sum(stats["per_word"].values())
    print()
    print(f"TTS clean variants generated:   {stats['generated']}")
    print(f"TTS noisy variants generated:    {stats['noisy_variants']}")
    print(f"TTS failures (skipped):          {stats['failed']}")
    print(f"Total confusable files:          {total_confusable}")
    print(f"Total noise negatives:           {noise_count}")
    print(f"GRAND TOTAL:                     {total_confusable + noise_count}")
    print()
    print(f"Output: {OUTPUT_DIR}")
    print(f"Noise:  {NOISE_DIR}")

    # Write manifest
    manifest = {
        "total_confusable": total_confusable,
        "total_noise": noise_count,
        "grand_total": total_confusable + noise_count,
        "per_word": dict(stats["per_word"]),
        "per_voice": dict(stats["per_voice"]),
        "failed": stats["failed"],
        "voices_used": VOICES,
        "words_used": list(CONFUSABLE_WORDS.keys()),
        "sentence_contexts": list(SENTENCE_CONTEXTS.keys()),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
