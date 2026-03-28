#!/usr/bin/env python3
"""
Generate TTS positive samples for OWW's "alexa" wake word.
Uses the same voices and methodology as ViolaWake's eval_clean positives.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import wave
from pathlib import Path

import edge_tts
import numpy as np

OUT_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_oww/oww_positives")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same voices used in ViolaWake eval
VOICES = [
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-US-AriaNeural", "en-US-GuyNeural",
    "en-US-JennyNeural", "en-US-BrianNeural",
    "en-US-EmmaNeural", "en-US-AndrewNeural",
    "en-US-AvaNeural", "en-US-ChristopherNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural",
    "en-GB-LibbyNeural", "en-GB-ThomasNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-ZA-LeahNeural", "en-IE-EmilyNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
]

# OWW wake word phrases - "alexa" in various forms
# OWW's "alexa" model is trained on standalone "alexa"
PHRASES = [
    "alexa",
    "hey alexa",
    "ok alexa",
]

# Augmentation: add noisy and reverb variants
def add_noise(samples: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Add white noise at given SNR."""
    rms_signal = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.default_rng(42).normal(0, rms_noise, samples.shape)
    return np.clip(samples + noise, -32768, 32767).astype(np.int16)


def add_reverb(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simple convolution reverb approximation."""
    # Exponentially decaying impulse response
    ir_len = int(0.3 * sr)  # 300ms
    ir = np.exp(-np.linspace(0, 5, ir_len))
    ir[0] = 1.0
    ir = ir / np.sum(ir)
    convolved = np.convolve(samples.astype(np.float64), ir, mode='full')[:len(samples)]
    max_val = np.max(np.abs(convolved))
    if max_val > 0:
        convolved = convolved * (32767 / max_val) * 0.9
    return convolved.astype(np.int16)


def convert_to_16k_wav(mp3_path: str, wav_path: str):
    """Convert mp3 to 16kHz mono WAV using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        wav_path
    ], capture_output=True, check=True)


async def generate_one(voice: str, phrase: str) -> str | None:
    """Generate a single TTS sample."""
    safe_phrase = phrase.replace(" ", "_")
    base_name = f"{voice}_{safe_phrase}"
    mp3_path = str(OUT_DIR / f"{base_name}.mp3")
    wav_path = str(OUT_DIR / f"{base_name}.wav")

    if os.path.exists(wav_path):
        return wav_path

    try:
        comm = edge_tts.Communicate(phrase, voice)
        await comm.save(mp3_path)
        convert_to_16k_wav(mp3_path, wav_path)
        os.remove(mp3_path)

        # Generate noisy variant
        with wave.open(wav_path, 'rb') as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        noisy = add_noise(data)
        noisy_path = str(OUT_DIR / f"{base_name}_noisy.wav")
        with wave.open(noisy_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(noisy.tobytes())

        reverb = add_reverb(data)
        reverb_path = str(OUT_DIR / f"{base_name}_reverb.wav")
        with wave.open(reverb_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(reverb.tobytes())

        return wav_path
    except Exception as e:
        print(f"FAILED: {voice} / {phrase}: {e}")
        return None


async def main():
    total = 0
    for voice in VOICES:
        for phrase in PHRASES:
            result = await generate_one(voice, phrase)
            if result:
                total += 1
                # +2 for noisy and reverb variants
                print(f"  Generated: {voice} / {phrase} (+ noisy + reverb)")

    # Count total files
    wav_count = len(list(OUT_DIR.glob("*.wav")))
    print(f"\nTotal WAV files generated: {wav_count}")
    print(f"  ({len(VOICES)} voices x {len(PHRASES)} phrases x 3 variants = {len(VOICES) * len(PHRASES) * 3} expected)")


if __name__ == "__main__":
    asyncio.run(main())
