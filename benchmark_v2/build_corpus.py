#!/usr/bin/env python3
"""
Build the shared corpus for ViolaWake vs OpenWakeWord benchmark v2.

Fixes all methodological flaws from v1:
1. SAME negative corpus for both systems
2. No "Alexa" in negatives (was contaminating OWW's negative set)
3. Adversarial negatives for BOTH wake words
4. Matched positive corpora (same voices, same augmentations, same count)

Outputs:
  benchmark_v2/corpus/
    positives/viola/    -- 180 files: 20 voices x 3 phrases x 3 augmentations
    positives/alexa/    -- 180 files: 20 voices x 3 phrases x 3 augmentations
    negatives/speech/   -- 200+ general speech files
    negatives/noise/    -- 20+ noise files
    negatives/adversarial_viola/   -- confusables for "viola"
    negatives/adversarial_alexa/   -- confusables for "alexa"
"""
from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

try:
    import edge_tts
except ImportError:
    print("ERROR: edge_tts not installed. Run: pip install edge-tts")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)


# ── Configuration ──

BASE_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_v2/corpus")
EXISTING_NEGATIVES = Path("J:/CLAUDE/PROJECTS/Wakeword/eval_clean/negatives")

SAMPLE_RATE = 16_000

# 20 voices shared for both systems (diverse: US, GB, AU, CA, IN, IE, ZA)
VOICES = [
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",
    "en-GB-LibbyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-GB-ThomasNeural",
    "en-IE-EmilyNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    "en-US-AndrewNeural",
    "en-US-AriaNeural",
    "en-US-AvaNeural",
    "en-US-BrianNeural",
    "en-US-ChristopherNeural",
    "en-US-EmmaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-ZA-LeahNeural",
]

# Positive phrases
VIOLA_PHRASES = {
    "viola": "Viola",
    "hey_viola": "Hey Viola",
    "ok_viola": "OK Viola",
}

ALEXA_PHRASES = {
    "alexa": "Alexa",
    "hey_alexa": "Hey Alexa",
    "ok_alexa": "OK Alexa",
}

# Adversarial words for "viola" -- phonetically similar
VIOLA_ADVERSARIAL = [
    "vanilla",
    "villa",
    "violet",
    "violin",
    "volume",
    "She plays viola in the orchestra",  # viola the instrument in context
    "valley",
    "valid",
    "value",
    "voila",
    "villain",
    "violence",
    "via",
    "violent",
    "valet",
]

# Adversarial words for "alexa" -- phonetically similar
ALEXA_ADVERSARIAL = [
    "Alexis",
    "election",
    "a lecture",
    "a lexicon",
    "Alex",
    "elect",
    "Alexander",
    "electric",
    "elegant",
    "a legacy",
    "electrode",
    "a lesson",
    "Alexandra",
    "elective",
    "a letter",
]

# General speech sentences (NOT containing either wake word)
GENERAL_SPEECH = [
    "What time is it right now?",
    "Please turn off the lights in the kitchen.",
    "Set a timer for fifteen minutes.",
    "Play some music from the nineteen eighties.",
    "Good morning, how are you today?",
    "Can you tell me the weather forecast?",
    "I need to call my mother later.",
    "Remind me to pick up groceries.",
    "What's the capital of France?",
    "Tell me a joke about cats.",
    "The quick brown fox jumps over the lazy dog.",
    "How far is the nearest gas station?",
    "I'd like to order a large pizza with mushrooms.",
    "Can you play the next episode?",
    "Schedule a meeting for tomorrow at three.",
    "Open the garage door please.",
    "Thank you very much for your help.",
    "Hello, nice to meet you.",
    "Read me the latest news headlines.",
    "Turn up the thermostat to seventy two degrees.",
]

# 5 adversarial-generation voices (subset, for efficiency)
ADVERSARIAL_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
    "en-IN-NeerjaNeural",
    "en-CA-LiamNeural",
    "en-US-BrianNeural",
]


# ── Audio utilities ──

def mp3_bytes_to_wav_int16(mp3_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert MP3 bytes to int16 PCM at target sample rate using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd, input=mp3_bytes, capture_output=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.int16)


def save_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save int16 PCM as WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())


def add_noise(audio: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Add white Gaussian noise at specified SNR."""
    rms_signal = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms_signal < 1.0:
        return audio
    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    noise = np.random.default_rng(42).normal(0, rms_noise, len(audio))
    return np.clip(audio.astype(np.float64) + noise, -32768, 32767).astype(np.int16)


def add_reverb(audio: np.ndarray, decay: float = 0.3, delay_ms: int = 50) -> np.ndarray:
    """Add simple reverb (delayed copy with decay)."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    result = audio.astype(np.float64).copy()
    if delay_samples < len(audio):
        result[delay_samples:] += decay * audio[:len(audio) - delay_samples].astype(np.float64)
    return np.clip(result, -32768, 32767).astype(np.int16)


async def tts_to_pcm(voice: str, text: str) -> np.ndarray:
    """Generate TTS audio and return as int16 PCM at 16kHz."""
    communicate = edge_tts.Communicate(text, voice)
    mp3_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])
    if not mp3_chunks:
        raise RuntimeError(f"No audio from edge_tts for voice={voice}, text={text!r}")
    mp3_data = b"".join(mp3_chunks)
    return mp3_bytes_to_wav_int16(mp3_data)


# ── Corpus builders ──

async def build_positives(
    phrases: dict[str, str],
    output_dir: Path,
    label: str,
) -> int:
    """Generate positive samples: 20 voices x 3 phrases x 3 augmentations = 180."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total = len(VOICES) * len(phrases) * 3
    for vi, voice in enumerate(VOICES):
        for phrase_key, phrase_text in phrases.items():
            try:
                audio = await tts_to_pcm(voice, phrase_text)
            except Exception as e:
                print(f"  WARN: TTS failed for {voice}/{phrase_key}: {e}")
                continue

            # Clean
            save_wav(output_dir / f"{voice}_{phrase_key}.wav", audio)
            count += 1

            # Noisy
            noisy = add_noise(audio, snr_db=15.0)
            save_wav(output_dir / f"{voice}_{phrase_key}_noisy.wav", noisy)
            count += 1

            # Reverb
            reverbed = add_reverb(audio, decay=0.3, delay_ms=50)
            save_wav(output_dir / f"{voice}_{phrase_key}_reverb.wav", reverbed)
            count += 1

        print(f"  [{label}] Voice {vi+1}/{len(VOICES)}: {voice} done ({count}/{total})")

    return count


async def build_adversarial(
    words: list[str],
    output_dir: Path,
    label: str,
) -> int:
    """Generate adversarial negatives: N words x 7 voices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total = len(words) * len(ADVERSARIAL_VOICES)
    for word in words:
        # Create a safe filename
        safe_name = word.lower().replace(" ", "_").replace(",", "").replace("'", "")
        # Truncate long sentences
        if len(safe_name) > 40:
            safe_name = safe_name[:40]
        for voice in ADVERSARIAL_VOICES:
            try:
                audio = await tts_to_pcm(voice, word)
            except Exception as e:
                print(f"  WARN: TTS failed for {voice}/{word}: {e}")
                continue
            voice_short = voice.split("-")[-1].replace("Neural", "")
            save_wav(output_dir / f"{voice_short}_{safe_name}.wav", audio)
            count += 1

        if count % 20 == 0:
            print(f"  [{label}] {count}/{total} adversarial samples generated...")

    return count


async def build_general_speech(output_dir: Path) -> int:
    """Generate general speech negatives: 20 sentences x 10 voices = 200."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    # Use 10 diverse voices for general speech
    speech_voices = VOICES[:10]
    total = len(GENERAL_SPEECH) * len(speech_voices)
    for si, sentence in enumerate(GENERAL_SPEECH):
        safe_name = sentence.lower()[:50].replace(" ", "_").replace("?", "").replace(".", "").replace("'", "").replace(",", "")
        for voice in speech_voices:
            try:
                audio = await tts_to_pcm(voice, sentence)
            except Exception as e:
                print(f"  WARN: TTS failed for {voice}: {e}")
                continue
            voice_short = voice.split("-")[-1].replace("Neural", "")
            save_wav(output_dir / f"{voice_short}_{safe_name}.wav", audio)
            count += 1

        if (si + 1) % 5 == 0:
            print(f"  [speech] {count}/{total} general speech samples generated...")

    return count


def build_noise(output_dir: Path) -> int:
    """Generate synthetic noise samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 0
    duration_s = 2.0
    n_samples = int(SAMPLE_RATE * duration_s)

    noise_types = {
        "white_noise": lambda: rng.normal(0, 3000, n_samples).astype(np.int16),
        "pink_noise": lambda: _pink_noise(rng, n_samples, amplitude=3000),
        "brown_noise": lambda: _brown_noise(rng, n_samples, amplitude=3000),
        "silence": lambda: np.zeros(n_samples, dtype=np.int16),
        "low_hum": lambda: (2000 * np.sin(2 * np.pi * 60 * np.arange(n_samples) / SAMPLE_RATE)).astype(np.int16),
    }

    for name, gen_fn in noise_types.items():
        for level_idx, amplitude_scale in enumerate([0.5, 1.0, 2.0, 4.0]):
            audio = gen_fn()
            audio = np.clip(audio.astype(np.float64) * amplitude_scale, -32768, 32767).astype(np.int16)
            save_wav(output_dir / f"{name}_level{level_idx}.wav", audio)
            count += 1

    print(f"  [noise] {count} noise samples generated")
    return count


def _pink_noise(rng: np.random.Generator, n: int, amplitude: float = 3000) -> np.ndarray:
    """Generate approximate pink noise using Voss-McCartney algorithm."""
    white = rng.normal(0, 1, n)
    # Simple 1/f approximation: cumsum + highpass
    pink = np.cumsum(white)
    pink -= np.mean(pink)
    if np.max(np.abs(pink)) > 0:
        pink = pink / np.max(np.abs(pink)) * amplitude
    return pink.astype(np.int16)


def _brown_noise(rng: np.random.Generator, n: int, amplitude: float = 3000) -> np.ndarray:
    """Generate brown noise (1/f^2)."""
    white = rng.normal(0, 1, n)
    brown = np.cumsum(white)
    brown -= np.mean(brown)
    if np.max(np.abs(brown)) > 0:
        brown = brown / np.max(np.abs(brown)) * amplitude
    return brown.astype(np.int16)


def copy_existing_speech_negatives(output_dir: Path) -> int:
    """Copy existing speech negatives from eval_clean, filtering out wake words."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = EXISTING_NEGATIVES / "speech"
    count = 0
    if source_dir.exists():
        for wav in sorted(source_dir.glob("*.wav")):
            fname = wav.stem.lower()
            # Skip any files containing actual wake words
            if "alexa" in fname or "viola" in fname:
                print(f"  SKIP (contains wake word): {wav.name}")
                continue
            dest = output_dir / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            count += 1
    return count


# ── Main ──

async def main():
    print("=" * 70)
    print("Building Benchmark v2 Corpus")
    print("=" * 70)

    # Step 1: Positives
    print("\n[1/6] Building viola positives (20 voices x 3 phrases x 3 augmentations)...")
    n_viola = await build_positives(
        VIOLA_PHRASES, BASE_DIR / "positives" / "viola", "viola"
    )
    print(f"  -> {n_viola} viola positives")

    print("\n[2/6] Building alexa positives (20 voices x 3 phrases x 3 augmentations)...")
    n_alexa = await build_positives(
        ALEXA_PHRASES, BASE_DIR / "positives" / "alexa", "alexa"
    )
    print(f"  -> {n_alexa} alexa positives")

    # Step 2: Negatives - adversarial
    print("\n[3/6] Building viola-adversarial negatives (15 words x 7 voices)...")
    n_adv_viola = await build_adversarial(
        VIOLA_ADVERSARIAL, BASE_DIR / "negatives" / "adversarial_viola", "viola-adv"
    )
    print(f"  -> {n_adv_viola} viola adversarial negatives")

    print("\n[4/6] Building alexa-adversarial negatives (15 words x 7 voices)...")
    n_adv_alexa = await build_adversarial(
        ALEXA_ADVERSARIAL, BASE_DIR / "negatives" / "adversarial_alexa", "alexa-adv"
    )
    print(f"  -> {n_adv_alexa} alexa adversarial negatives")

    # Step 3: Negatives - general speech (TTS-generated)
    print("\n[5/6] Building general speech negatives (20 sentences x 10 voices)...")
    n_speech = await build_general_speech(BASE_DIR / "negatives" / "speech")
    print(f"  -> {n_speech} general speech negatives")

    # Also copy existing speech negatives (excluding wake words)
    n_existing = copy_existing_speech_negatives(BASE_DIR / "negatives" / "speech_existing")
    print(f"  -> {n_existing} existing speech negatives copied")

    # Step 4: Negatives - noise
    print("\n[6/6] Building noise negatives...")
    n_noise = build_noise(BASE_DIR / "negatives" / "noise")
    print(f"  -> {n_noise} noise negatives")

    # Summary
    total_neg = n_adv_viola + n_adv_alexa + n_speech + n_existing + n_noise
    print("\n" + "=" * 70)
    print("CORPUS SUMMARY")
    print("=" * 70)
    print(f"Positives:")
    print(f"  viola:  {n_viola} files")
    print(f"  alexa:  {n_alexa} files")
    print(f"Negatives (shared):")
    print(f"  adversarial_viola:  {n_adv_viola} files")
    print(f"  adversarial_alexa:  {n_adv_alexa} files")
    print(f"  general speech:     {n_speech} files (TTS-generated)")
    print(f"  existing speech:    {n_existing} files (from eval_clean)")
    print(f"  noise:              {n_noise} files")
    print(f"  TOTAL negatives:    {total_neg} files")
    print(f"\nCorpus written to: {BASE_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
