"""
violawake-generate CLI -- Generate wake word training samples via TTS.

Entry point: ``violawake-generate`` (declared in pyproject.toml).

Headless alternative to ``violawake-collect``: synthesises positive wake word
samples using Edge TTS voices, with optional augmentation (noise, reverb) and
confusable-negative generation.  Requires no microphone -- works on CI servers,
headless VMs, and developer laptops.

Usage::

    violawake-generate --word "operator" --output data/operator/positives/

    # Clean samples only, fewer voices
    violawake-generate --word "jarvis" --output data/jarvis/positives/ \\
        --count 60 --voices 10 --no-augment

    # With confusable negatives
    violawake-generate --word "viola" --output data/viola/positives/ \\
        --negatives --neg-count 300

    # Custom phrase templates
    violawake-generate --word "alexa" --output data/alexa/positives/ \\
        --phrases "WORD,hey WORD,ok WORD,hi WORD"
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Audio conversion helpers (no ffmpeg required)
# ---------------------------------------------------------------------------

def _mp3_bytes_to_pcm(mp3_data: bytes, target_sr: int) -> bytes | None:
    """Convert MP3 bytes to 16-bit mono PCM at *target_sr*.

    Tries pydub first (pure-Python MP3 decode via audioop), then falls back
    to a helpful error message.  Does NOT require ffmpeg when pydub is built
    with its bundled decoder.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        print(
            "ERROR: pydub is required for MP3-to-WAV conversion. "
            "Install with: pip install pydub\n"
            "If pydub still fails to decode MP3, install ffmpeg as well.",
            file=sys.stderr,
        )
        return None

    try:
        seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        return seg.raw_data
    except Exception as exc:
        print(
            f"WARNING: MP3 decode failed ({exc}). "
            "Install ffmpeg for reliable MP3 support.",
            file=sys.stderr,
        )
        return None


def _save_wav(pcm_data: bytes, path: Path, sample_rate: int = 16_000) -> None:
    """Save raw 16-bit mono PCM bytes as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ---------------------------------------------------------------------------
# Augmentation helpers (numpy only -- no heavy pipeline import)
# ---------------------------------------------------------------------------

def _pcm_to_float(pcm: bytes) -> "np.ndarray":
    """Convert 16-bit PCM bytes to float32 numpy array in [-1, 1]."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def _float_to_pcm(arr: "np.ndarray") -> bytes:
    """Convert float32 numpy array back to 16-bit PCM bytes."""
    import numpy as np

    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _add_noise(pcm: bytes, snr_db: float) -> bytes:
    """Add white noise at the given SNR (dB)."""
    import numpy as np

    signal = _pcm_to_float(pcm)
    rms_signal = max(np.sqrt(np.mean(signal ** 2)), 1e-10)
    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    noise = np.random.default_rng().normal(0, rms_noise, signal.shape).astype(
        np.float32
    )
    return _float_to_pcm(signal + noise)


def _add_reverb(pcm: bytes, sample_rate: int) -> bytes:
    """Apply simple synthetic reverb via convolution with a decaying impulse."""
    import numpy as np

    signal = _pcm_to_float(pcm)
    # Synthetic room impulse response: 100ms exponential decay
    rir_len = int(sample_rate * 0.1)
    rir = np.random.default_rng(42).normal(0, 1, rir_len).astype(np.float32)
    rir *= np.exp(-np.linspace(0, 6, rir_len)).astype(np.float32)
    rir[0] = 1.0
    rir /= max(np.sqrt(np.sum(rir ** 2)), 1e-10)

    convolved = np.convolve(signal, rir, mode="full")[: len(signal)]
    return _float_to_pcm(convolved)


# ---------------------------------------------------------------------------
# Edge TTS helpers
# ---------------------------------------------------------------------------

async def _get_english_voices(max_voices: int) -> list[dict]:
    """Return up to *max_voices* English Edge TTS voices."""
    try:
        import edge_tts
    except ImportError:
        print(
            "ERROR: edge-tts is required for TTS generation. "
            "Install with: pip install edge-tts",
            file=sys.stderr,
        )
        sys.exit(1)

    all_voices = await edge_tts.list_voices()
    english = [v for v in all_voices if v.get("Locale", "").startswith("en")]
    # Shuffle deterministically so repeated runs get the same set
    import hashlib

    english.sort(key=lambda v: hashlib.md5(v["ShortName"].encode()).hexdigest())
    return english[:max_voices]


async def _synthesize_one(
    voice_name: str,
    text: str,
    target_sr: int,
) -> bytes | None:
    """Synthesize *text* with *voice_name* and return 16-bit mono PCM bytes."""
    try:
        import edge_tts
    except ImportError:
        return None

    communicate = edge_tts.Communicate(text, voice_name)
    mp3_chunks: list[bytes] = []
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
    except Exception:
        # Some voices intermittently fail -- skip gracefully
        return None

    if not mp3_chunks:
        return None

    mp3_data = b"".join(mp3_chunks)
    return _mp3_bytes_to_pcm(mp3_data, target_sr)


def _safe_filename(text: str) -> str:
    """Turn an arbitrary phrase into a filesystem-safe string."""
    safe = text.lower().replace(" ", "_")
    return "".join(c for c in safe if c.isalnum() or c == "_")


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

async def _generate_positives(
    word: str,
    output_dir: Path,
    count: int,
    voices: int,
    phrases: list[str],
    augment: bool,
    sample_rate: int,
    quiet: bool,
) -> int:
    """Generate positive TTS samples.  Returns number of files saved."""
    voice_list = await _get_english_voices(voices)
    if not voice_list:
        print("ERROR: No English voices found from Edge TTS.", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the full list of (voice, phrase) pairs, cycling until we reach count
    tasks: list[tuple[str, str]] = []
    idx = 0
    while len(tasks) < count:
        voice = voice_list[idx % len(voice_list)]
        phrase = phrases[idx % len(phrases)]
        tasks.append((voice["ShortName"], phrase))
        idx += 1

    saved = 0
    for i, (voice_name, phrase) in enumerate(tasks):
        text = phrase.replace("WORD", word)
        voice_short = voice_name.split("-")[-1].replace("Neural", "").strip()
        base_name = f"{voice_short}_{_safe_filename(text)}"

        pcm = await _synthesize_one(voice_name, text, sample_rate)
        if pcm is None:
            if not quiet:
                print(f"  [{i + 1}/{count}] SKIP {voice_name} ({text}) -- TTS failed")
            continue

        # Save clean sample
        clean_path = output_dir / f"{base_name}.wav"
        _save_wav(pcm, clean_path, sample_rate)
        saved += 1

        if augment:
            # Noisy variant (random SNR between 10-20 dB)
            import random

            snr = random.uniform(10.0, 20.0)
            noisy_pcm = _add_noise(pcm, snr)
            _save_wav(noisy_pcm, output_dir / f"{base_name}_noisy.wav", sample_rate)
            saved += 1

            # Reverb variant
            reverb_pcm = _add_reverb(pcm, sample_rate)
            _save_wav(reverb_pcm, output_dir / f"{base_name}_reverb.wav", sample_rate)
            saved += 1

        if not quiet:
            print(f"  Generated {saved} samples...", end="\r", flush=True)

    if not quiet:
        print()

    return saved


async def _generate_negatives(
    word: str,
    output_dir: Path,
    neg_count: int,
    voices: int,
    sample_rate: int,
    quiet: bool,
) -> int:
    """Generate confusable-negative TTS samples.  Returns number of files saved."""
    from violawake_sdk.tools.confusables import generate_confusables

    confusables = generate_confusables(word, count=neg_count)
    if not confusables:
        if not quiet:
            print("  No confusables generated -- skipping negatives.")
        return 0

    neg_dir = output_dir / ".." / "negatives" / "confusables"
    neg_dir = neg_dir.resolve()
    neg_dir.mkdir(parents=True, exist_ok=True)

    voice_list = await _get_english_voices(voices)
    if not voice_list:
        print("ERROR: No English voices found from Edge TTS.", file=sys.stderr)
        return 0

    saved = 0
    total = min(neg_count, len(confusables) * len(voice_list))

    tasks: list[tuple[str, str]] = []
    idx = 0
    while len(tasks) < neg_count and idx < len(confusables) * len(voice_list):
        confusable = confusables[idx % len(confusables)]
        voice = voice_list[idx % len(voice_list)]
        tasks.append((voice["ShortName"], confusable))
        idx += 1

    for i, (voice_name, text) in enumerate(tasks):
        voice_short = voice_name.split("-")[-1].replace("Neural", "").strip()
        base_name = f"{voice_short}_{_safe_filename(text)}"

        pcm = await _synthesize_one(voice_name, text, sample_rate)
        if pcm is None:
            continue

        _save_wav(pcm, neg_dir / f"{base_name}.wav", sample_rate)
        saved += 1

        if not quiet:
            print(f"  Generated {saved}/{neg_count} negatives...", end="\r", flush=True)

    if not quiet:
        print()

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _async_main(args: argparse.Namespace) -> None:
    """Async body called by main()."""
    output_dir = Path(args.output)
    phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]
    if not phrases:
        phrases = ["WORD", "hey WORD", "ok WORD"]

    if not args.quiet:
        print(f"Generating TTS samples for wake word: {args.word!r}")
        print(f"Output directory: {output_dir}")
        print(f"Target count: {args.count} | Voices: {args.voices}")
        print(f"Phrases: {', '.join(phrases)}")
        print(f"Augment: {args.augment} | Sample rate: {args.sample_rate} Hz")
        print()

    # --- Positives ---
    if not args.quiet:
        print("Generating positive samples...")

    pos_count = await _generate_positives(
        word=args.word,
        output_dir=output_dir,
        count=args.count,
        voices=args.voices,
        phrases=phrases,
        augment=args.augment,
        sample_rate=args.sample_rate,
        quiet=args.quiet,
    )

    if not args.quiet:
        print(f"Saved {pos_count} positive samples to {output_dir}")

    # --- Negatives ---
    if args.negatives:
        if not args.quiet:
            print()
            print("Generating confusable negatives...")

        neg_dir = (output_dir / ".." / "negatives" / "confusables").resolve()
        neg_count = await _generate_negatives(
            word=args.word,
            output_dir=output_dir,
            neg_count=args.neg_count,
            voices=args.voices,
            sample_rate=args.sample_rate,
            quiet=args.quiet,
        )

        if not args.quiet:
            print(f"Saved {neg_count} confusable negatives to {neg_dir}")

    # --- Summary ---
    if not args.quiet:
        print()
        print("Done.")
        if pos_count < 20:
            print(
                "TIP: For good model accuracy, generate at least 50 clean samples. "
                "Increase --count or --voices."
            )


def main() -> None:
    """CLI entry point for violawake-generate."""
    parser = argparse.ArgumentParser(
        prog="violawake-generate",
        description="Generate wake word training samples via TTS (headless).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--word",
        required=True,
        metavar="WORD",
        help="The wake word to generate samples for",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory to save WAV files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=180,
        metavar="N",
        help="Target number of samples (default: 180)",
    )
    parser.add_argument(
        "--voices",
        type=int,
        default=20,
        metavar="N",
        help="Number of Edge TTS voices to use (default: 20)",
    )
    parser.add_argument(
        "--phrases",
        type=str,
        default="WORD,hey WORD,ok WORD",
        metavar="TEMPLATES",
        help='Comma-separated phrase templates (default: "WORD,hey WORD,ok WORD")',
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        dest="augment",
        help="Generate noisy and reverb variants (default: True)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        help="Clean samples only, no augmentation",
    )
    parser.add_argument(
        "--negatives",
        action="store_true",
        default=False,
        help="Also generate confusable negatives",
    )
    parser.add_argument(
        "--neg-count",
        type=int,
        default=300,
        metavar="N",
        help="Number of confusable negatives to generate (default: 300)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        metavar="HZ",
        help="Output sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate early
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.voices < 1:
        parser.error("--voices must be at least 1")
    if args.neg_count < 1:
        parser.error("--neg-count must be at least 1")

    # Check edge-tts availability before doing anything
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print(
            "ERROR: edge-tts is required for TTS-based sample generation.\n"
            "Install with: pip install edge-tts",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check pydub availability (needed for MP3->WAV)
    try:
        import pydub  # noqa: F401
    except ImportError:
        print(
            "WARNING: pydub is not installed. MP3-to-WAV conversion may fail.\n"
            "Install with: pip install pydub\n"
            "For best results, also install ffmpeg.",
            file=sys.stderr,
        )

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
