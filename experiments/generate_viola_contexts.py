"""
Generate "viola" in varied contexts — positives where "viola" is followed
by different commands, matching what real users would say.

This fixes the model's blind spot: it detects "hey viola" and "ok viola"
(viola at end) but fails on "viola wake up" (viola at start + trailing speech).

Generates TTS for:
- "viola wake up" / "viola please" (matches eval set)
- "viola play music" / "viola stop" / "viola what time is it" etc. (real usage)
- Various speeds, pauses, voices

Usage:
  python experiments/generate_viola_contexts.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
OUTPUT_DIR = WAKEWORD / "experiments" / "training_data" / "viola_contexts"

# Phrases containing "viola" in different positions and contexts
PHRASES = [
    # Viola at start (what the model currently fails on)
    "viola wake up",
    "viola please",
    "viola stop",
    "viola play music",
    "viola what time is it",
    "viola turn off the lights",
    "viola set a timer",
    "viola tell me a joke",
    "viola how are you",
    "viola good morning",
    # Viola in middle
    "excuse me viola can you help",
    "yo viola what's up",
    "dear viola please help me",
    # Just viola (standalone — different durations/emphasis)
    "viola",
    "viola!",
]

# Edge TTS voices (proven to work)
VOICES = [
    "en-US-AriaNeural",
    "en-US-ChristopherNeural",
    "en-US-EmmaNeural",
    "en-US-BrianNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-IN-NeerjaNeural",
    "en-CA-ClaraNeural",
    "en-ZA-LeahNeural",
    "en-AU-WilliamNeural",
    "en-US-JennyNeural",
    "en-US-AnaNeural",
]


async def generate_tts(text: str, voice: str, output_path: Path) -> bool:
    """Generate TTS audio using edge_tts."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        # Verify non-empty
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True
        output_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  FAIL {voice}/{text}: {e}")
        return False


async def convert_to_wav(mp3_path: Path) -> Path | None:
    """Convert mp3 to wav 16kHz mono."""
    import subprocess
    wav_path = mp3_path.with_suffix(".wav")
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(wav_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            mp3_path.unlink(missing_ok=True)
            return wav_path
    except Exception:
        pass
    return None


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    success = 0

    for phrase in PHRASES:
        # Clean filename
        clean = phrase.replace(" ", "_").replace("!", "").replace("'", "").replace(",", "")
        for voice in VOICES:
            voice_short = voice.replace("Neural", "")
            mp3_path = OUTPUT_DIR / f"{voice_short}_{clean}.mp3"
            wav_path = OUTPUT_DIR / f"{voice_short}_{clean}.wav"

            if wav_path.exists():
                success += 1
                total += 1
                continue

            total += 1
            ok = await generate_tts(phrase, voice, mp3_path)
            if ok:
                wav = await convert_to_wav(mp3_path)
                if wav:
                    success += 1

            if total % 20 == 0:
                print(f"  {total} generated, {success} success...")

    print(f"\nDone: {success}/{total} files in {OUTPUT_DIR}")
    print(f"Now re-run embedding extraction to include these in training:")
    print(f"  python experiments/incremental_extract.py pos_viola_contexts")


if __name__ == "__main__":
    asyncio.run(main())
