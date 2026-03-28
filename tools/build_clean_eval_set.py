#!/usr/bin/env python
"""
Build a clean evaluation dataset with ZERO overlap with training data.

Generates:
  - Positives: multi-voice TTS wake word utterances + augmented variants
  - Negatives: adversarial confusables, generic speech, silence/noise

Usage:
    python tools/build_clean_eval_set.py --output eval_clean/ [--skip-librispeech] [--voices 24] [--seed 42]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
CLIP_DURATION = 1.5  # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

WAKE_PHRASES = ["viola", "hey viola", "ok viola", "viola wake up"]

EDGE_TTS_VOICES = [
    # en-US (10)
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
    "en-US-AmberNeural",
    "en-US-AnaNeural",
    "en-US-AndrewNeural",
    "en-US-EmmaNeural",
    "en-US-BrianNeural",
    "en-US-ChristopherNeural",
    # en-GB (4)
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-GB-LibbyNeural",
    "en-GB-ThomasNeural",
    # en-AU (2)
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    # en-IN (2)
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    # en-ZA (2)
    "en-ZA-LeahNeural",
    "en-ZA-LukeNeural",
    # en-IE (2)
    "en-IE-EmilyNeural",
    "en-IE-ConnorNeural",
    # en-CA (2)
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",
]

ADVERSARIAL_WORDS = [
    # Confusable with "viola"
    "violet",
    "violin",
    "violence",
    "violent",
    "villa",
    "vanilla",
    "valley",
    "volume",
    "hola",
    "voila",
    "via",
    "valet",
    "villain",
    # Confusable with "hey viola" / other assistants
    "hey violet",
    "hey violin",
    "ok Google",
    "hey Siri",
    "Alexa",
    # Common phrases (should not trigger)
    "hello",
    "good morning",
    "what time is it",
    "play music",
    "turn off the lights",
    "set a timer",
    "how are you",
    "thank you",
]

# 10 diverse voices for adversarial negatives
ADVERSARIAL_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-IN-NeerjaNeural",
    "en-ZA-LeahNeural",
    "en-IE-ConnorNeural",
    "en-CA-LiamNeural",
]

# Additional generic speech phrases for speech/ negatives (non-adversarial)
GENERIC_SPEECH_PHRASES = [
    "The weather today is partly cloudy with a chance of rain.",
    "I need to pick up groceries on the way home.",
    "Can you send me the report by end of day?",
    "The meeting has been moved to three o'clock.",
    "Let me check my calendar for availability.",
    "We should go to the park this weekend.",
    "Please remember to lock the door when you leave.",
    "I'm running about ten minutes late.",
    "What do you want for dinner tonight?",
    "The traffic on the highway is terrible right now.",
    "Have you seen the latest episode of that show?",
    "I'll call you back in a few minutes.",
    "The package should arrive by tomorrow morning.",
    "Don't forget to water the plants.",
    "I think we need to replace the batteries.",
    "Could you help me move this table?",
    "The restaurant opens at eleven thirty.",
    "I'm looking for a new apartment downtown.",
    "She said the project deadline was extended.",
    "Let's take a different route today.",
    "My phone battery is almost dead.",
    "The kids have soccer practice at four.",
    "I heard there's a sale at the electronics store.",
    "Can we reschedule our appointment?",
    "The flight leaves at seven in the morning.",
    "I need to renew my driver's license.",
    "The new coffee shop on Main Street is great.",
    "Do you have any recommendations for a good book?",
    "We should plan something for the holiday weekend.",
    "The software update will take about fifteen minutes.",
]

GENERIC_SPEECH_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
    "en-IN-PrabhatNeural",
    "en-ZA-LukeNeural",
    "en-IE-EmilyNeural",
    "en-CA-ClaraNeural",
]


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write float32 audio as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())


def read_wav(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Read a WAV file and return float32 mono at target_sr."""
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            if sr != target_sr:
                from scipy.signal import resample

                new_len = int(len(audio) * target_sr / sr)
                audio = resample(audio, new_len).astype(np.float32)
            return audio
    except Exception:
        return None


def read_audio_file(path: Path) -> np.ndarray | None:
    """Read any audio file (WAV, MP3, FLAC, OGG) via soundfile, return 16kHz float32."""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample

            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = resample(audio, new_len).astype(np.float32)
        return audio
    except Exception:
        # Fallback to stdlib wave
        return read_wav(path)


# ---------------------------------------------------------------------------
# Audio augmentations
# ---------------------------------------------------------------------------


def generate_synthetic_ir(
    sr: int = SAMPLE_RATE, rt60: float = 0.3, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a simple synthetic room impulse response.

    Creates an exponentially decaying noise IR that simulates a small room.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    duration = rt60
    n_samples = int(sr * duration)
    noise = rng.standard_normal(n_samples).astype(np.float32)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    decay = np.exp(-6.9 * t / rt60)  # -60dB at rt60
    ir = noise * decay
    ir /= np.sqrt(np.sum(ir**2) + 1e-12)
    return ir


def apply_reverb(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Apply convolution reverb using a pre-computed impulse response."""
    from scipy.signal import fftconvolve

    wet = fftconvolve(audio, ir, mode="full")[: len(audio)]
    mixed = 0.7 * audio + 0.3 * wet.astype(np.float32)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def generate_pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink noise (1/f) using Voss-McCartney algorithm."""
    n_octaves = 16
    white = rng.standard_normal(length).astype(np.float32)
    pink = np.zeros(length, dtype=np.float32)
    for i in range(n_octaves):
        step = 2**i
        row_noise = rng.standard_normal((length + step - 1) // step).astype(np.float32)
        repeated = np.repeat(row_noise, step)[:length]
        pink += repeated
    pink += white
    std = pink.std()
    if std > 1e-9:
        pink /= std
    return pink


def add_noise_at_snr(
    audio: np.ndarray, snr_db: float, rng: np.random.Generator, noise_type: str = "pink"
) -> np.ndarray:
    """Add noise at a specified SNR in dB."""
    if noise_type == "pink":
        noise = generate_pink_noise(len(audio), rng)
    else:
        noise = rng.standard_normal(len(audio)).astype(np.float32)

    sig_power = np.mean(audio**2)
    if sig_power < 1e-10:
        return audio.copy()
    noise_power = np.mean(noise**2)
    if noise_power < 1e-10:
        return audio.copy()

    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    result = audio + noise * scale
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_length: int = CLIP_SAMPLES) -> np.ndarray:
    """Pad (center) or trim audio to exact length."""
    if len(audio) < target_length:
        padding = target_length - len(audio)
        left = padding // 2
        right = padding - left
        audio = np.pad(audio, (left, right), mode="constant")
    elif len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        audio = audio[start : start + target_length]
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    """Track generation statistics."""

    positives_original: int = 0
    positives_reverb: int = 0
    positives_noisy: int = 0
    negatives_adversarial: int = 0
    negatives_speech: int = 0
    negatives_noise: int = 0
    negatives_librispeech: int = 0
    edge_tts_voices_used: set = field(default_factory=set)
    pyttsx3_voices_used: int = 0
    kokoro_voices_used: int = 0
    errors: list = field(default_factory=list)
    skipped_edge_tts: int = 0

    @property
    def total_positives(self) -> int:
        return self.positives_original + self.positives_reverb + self.positives_noisy

    @property
    def total_negatives(self) -> int:
        return (
            self.negatives_adversarial
            + self.negatives_speech
            + self.negatives_noise
            + self.negatives_librispeech
        )


# ---------------------------------------------------------------------------
# Edge TTS generation
# ---------------------------------------------------------------------------


async def generate_edge_tts_sample(text: str, voice: str, output_path: Path) -> bool:
    """Generate a single Edge TTS sample. Returns True on success."""
    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        mp3_path = output_path.with_suffix(".mp3")
        await communicate.save(str(mp3_path))

        audio = read_audio_file(mp3_path)
        if audio is None:
            mp3_path.unlink(missing_ok=True)
            return False

        audio = pad_or_trim(audio, CLIP_SAMPLES)
        write_wav(output_path, audio)
        mp3_path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


async def generate_edge_tts_batch(
    items: list[tuple[str, str, Path]],
    concurrency: int = 5,
    stats: Stats | None = None,
) -> list[Path]:
    """Generate Edge TTS samples with limited concurrency.

    items: list of (text, voice, output_path)
    Returns list of successfully created paths.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Path] = []
    lock = asyncio.Lock()

    async def _gen(text: str, voice: str, path: Path) -> None:
        async with semaphore:
            ok = await generate_edge_tts_sample(text, voice, path)
            if ok:
                async with lock:
                    results.append(path)
            else:
                if stats:
                    async with lock:
                        stats.skipped_edge_tts += 1

    tasks = [_gen(text, voice, path) for text, voice, path in items]
    await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# pyttsx3 generation
# ---------------------------------------------------------------------------


def _pyttsx3_worker(voice_id: str, phrase: str, out_path_str: str) -> bool:
    """Run pyttsx3 synthesis in a subprocess to avoid blocking.

    pyttsx3's runAndWait() blocks the event loop on Windows and can hang
    indefinitely. Running in a separate process with a timeout is safer.
    """
    import subprocess

    code = f"""
import pyttsx3, sys
engine = pyttsx3.init()
engine.setProperty('voice', {voice_id!r})
engine.save_to_file({phrase!r}, {out_path_str!r})
engine.runAndWait()
engine.stop()
sys.exit(0)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=15,
            capture_output=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def generate_pyttsx3_samples(
    output_dir: Path, stats: Stats, rng: np.random.Generator
) -> list[Path]:
    """Generate positive samples using pyttsx3 (Windows SAPI5 voices)."""
    try:
        import pyttsx3
    except ImportError:
        print("  [SKIP] pyttsx3 not installed")
        return []

    results: list[Path] = []
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.stop()
        del engine

        if not voices:
            print("  [SKIP] pyttsx3: no voices found")
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  pyttsx3: found {len(voices)} SAPI voices")

        for vi, voice_obj in enumerate(voices):
            voice_name = voice_obj.name.replace(" ", "_").replace("/", "_")
            voice_ok = False

            for phrase in WAKE_PHRASES:
                safe_phrase = phrase.replace(" ", "_")
                out_path = output_dir / f"sapi_{voice_name}_{safe_phrase}.wav"

                try:
                    ok = _pyttsx3_worker(voice_obj.id, phrase, str(out_path))
                    if ok and out_path.exists():
                        audio = read_wav(out_path)
                        if audio is not None and len(audio) > 0:
                            audio = pad_or_trim(audio, CLIP_SAMPLES)
                            write_wav(out_path, audio)
                            results.append(out_path)
                            stats.positives_original += 1
                            voice_ok = True
                        else:
                            out_path.unlink(missing_ok=True)
                    else:
                        out_path.unlink(missing_ok=True)
                except Exception as e:
                    stats.errors.append(f"pyttsx3 {voice_name}/{phrase}: {e}")

            if voice_ok:
                stats.pyttsx3_voices_used += 1

    except Exception as e:
        stats.errors.append(f"pyttsx3 init failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Kokoro TTS generation
# ---------------------------------------------------------------------------


def generate_kokoro_samples(output_dir: Path, stats: Stats) -> list[Path]:
    """Generate positive samples using Kokoro TTS from the SDK."""
    try:
        # Add the SDK src to the path if needed
        sdk_src = Path(__file__).resolve().parent.parent / "src"
        if str(sdk_src) not in sys.path:
            sys.path.insert(0, str(sdk_src))
        from violawake_sdk.tts import TTSEngine, AVAILABLE_VOICES
    except ImportError:
        print("  [SKIP] Kokoro TTS not available (violawake_sdk.tts import failed)")
        return []

    results: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for voice_name in AVAILABLE_VOICES:
        try:
            tts = TTSEngine(voice=voice_name, sample_rate=SAMPLE_RATE)
        except Exception as e:
            stats.errors.append(f"Kokoro voice {voice_name}: {e}")
            continue

        voice_ok = False
        for phrase in WAKE_PHRASES:
            safe_phrase = phrase.replace(" ", "_")
            out_path = output_dir / f"kokoro_{voice_name}_{safe_phrase}.wav"

            try:
                audio = tts.synthesize(phrase)
                if audio is not None and len(audio) > 0:
                    audio = pad_or_trim(audio, CLIP_SAMPLES)
                    write_wav(out_path, audio)
                    results.append(out_path)
                    stats.positives_original += 1
                    voice_ok = True
            except Exception as e:
                stats.errors.append(f"Kokoro {voice_name}/{phrase}: {e}")

        if voice_ok:
            stats.kokoro_voices_used += 1

    return results


# ---------------------------------------------------------------------------
# LibriSpeech negatives
# ---------------------------------------------------------------------------


def generate_librispeech_negatives(
    output_dir: Path, stats: Stats, rng: np.random.Generator, n_clips: int = 300
) -> list[Path]:
    """Extract random clips from LibriSpeech test-clean."""
    results: list[Path] = []

    try:
        import torchaudio

        print("  Downloading/loading LibriSpeech test-clean via torchaudio...")
        data_root = Path("./data")
        data_root.mkdir(exist_ok=True)
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(data_root), url="test-clean", download=True
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        n_total = len(dataset)
        if n_total == 0:
            print("  [WARN] LibriSpeech dataset empty")
            return []

        indices = rng.choice(n_total, size=min(n_clips, n_total), replace=False)
        for i, idx in enumerate(indices):
            waveform, sr, *_ = dataset[int(idx)]
            audio = waveform.squeeze().numpy()
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                audio = resampler(waveform).squeeze().numpy()

            if len(audio) > CLIP_SAMPLES:
                start = int(rng.integers(0, len(audio) - CLIP_SAMPLES))
                audio = audio[start : start + CLIP_SAMPLES]
            else:
                audio = pad_or_trim(audio, CLIP_SAMPLES)

            out_path = output_dir / f"librispeech_{i:04d}.wav"
            write_wav(out_path, audio)
            results.append(out_path)
            stats.negatives_librispeech += 1

        return results

    except (ImportError, Exception) as e:
        print(f"  [SKIP] LibriSpeech via torchaudio failed: {e}")
        print("  Falling back to TTS-based speech negatives only.")
        return []


# ---------------------------------------------------------------------------
# Noise negatives
# ---------------------------------------------------------------------------


def generate_noise_negatives(
    output_dir: Path, stats: Stats, rng: np.random.Generator
) -> list[Path]:
    """Generate silence, white noise, pink noise, and room tone samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    samples: dict[str, np.ndarray] = {
        "silence": np.zeros(CLIP_SAMPLES, dtype=np.float32),
        "white_noise_loud": np.clip(
            rng.standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.3, -1.0, 1.0
        ),
        "white_noise_quiet": np.clip(
            rng.standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.01, -1.0, 1.0
        ),
        "pink_noise_loud": np.clip(
            generate_pink_noise(CLIP_SAMPLES, rng) * 0.3, -1.0, 1.0
        ).astype(np.float32),
        "pink_noise_quiet": np.clip(
            generate_pink_noise(CLIP_SAMPLES, rng) * 0.01, -1.0, 1.0
        ).astype(np.float32),
    }

    # Room tone: very low-level pink noise
    for i in range(5):
        level = rng.uniform(0.001, 0.01)
        samples[f"room_tone_{i:02d}"] = np.clip(
            generate_pink_noise(CLIP_SAMPLES, rng) * level, -1.0, 1.0
        ).astype(np.float32)

    # Pure tones (hum, buzz)
    for freq, name in [(60, "hum_60hz"), (120, "hum_120hz"), (1000, "tone_1khz")]:
        t = np.linspace(0, CLIP_DURATION, CLIP_SAMPLES, dtype=np.float32)
        samples[name] = (np.sin(2 * np.pi * freq * t) * 0.1).astype(np.float32)

    # Broadband clicks
    clicks = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    click_positions = rng.integers(0, CLIP_SAMPLES, size=10)
    clicks[click_positions] = rng.uniform(0.3, 0.8, size=10).astype(np.float32)
    samples["clicks"] = clicks

    # Static (clipped white noise)
    samples["static"] = np.clip(
        rng.standard_normal(CLIP_SAMPLES).astype(np.float32) * 2.0, -1.0, 1.0
    ).astype(np.float32)

    # Brownian noise (integrated white noise)
    white = rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
    brown = np.cumsum(white)
    brown_max = np.max(np.abs(brown))
    if brown_max > 0:
        brown = (brown / brown_max * 0.3).astype(np.float32)
    samples["brown_noise"] = brown

    for name, audio in samples.items():
        out_path = output_dir / f"{name}.wav"
        write_wav(out_path, audio)
        results.append(out_path)
        stats.negatives_noise += 1

    return results


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def write_manifest(
    output_dir: Path, stats: Stats, elapsed: float, args: argparse.Namespace
) -> None:
    """Write MANIFEST.md documenting the eval set."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = textwrap.dedent(f"""\
    # Clean Evaluation Set Manifest

    **Generated**: {now}
    **Script**: `tools/build_clean_eval_set.py`
    **Seed**: {args.seed}
    **Generation time**: {elapsed:.1f}s

    ## Summary

    | Category | Count |
    |----------|-------|
    | Positives (original) | {stats.positives_original} |
    | Positives (reverb variant) | {stats.positives_reverb} |
    | Positives (noisy variant) | {stats.positives_noisy} |
    | **Total positives** | **{stats.total_positives}** |
    | Negatives (adversarial TTS) | {stats.negatives_adversarial} |
    | Negatives (generic speech) | {stats.negatives_speech} |
    | Negatives (LibriSpeech) | {stats.negatives_librispeech} |
    | Negatives (noise/silence) | {stats.negatives_noise} |
    | **Total negatives** | **{stats.total_negatives}** |
    | **Grand total** | **{stats.total_positives + stats.total_negatives}** |

    ## Voices Used

    | Engine | Unique Voices |
    |--------|---------------|
    | Edge TTS | {len(stats.edge_tts_voices_used)} |
    | pyttsx3 (SAPI5) | {stats.pyttsx3_voices_used} |
    | Kokoro | {stats.kokoro_voices_used} |

    ## Methodology

    ### Zero Training Overlap Guarantee

    This eval set has **zero overlap** with training data:
    - All positives are freshly synthesized via TTS (Edge TTS, pyttsx3, Kokoro)
    - No recorded audio from any training corpus is included
    - LibriSpeech test-clean is a held-out partition never used in ViolaWake training
    - Adversarial negatives are TTS-generated confusable words
    - Noise samples are procedurally generated with a fixed seed

    ### Positive Samples

    **Phrases**: {', '.join(f'"{p}"' for p in WAKE_PHRASES)}

    **Augmentation variants** (2 per original):
    - **Reverb**: Convolution with a synthetic exponentially-decaying IR (RT60=0.3s),
      70% dry / 30% wet mix. Simulates small room acoustics.
    - **Noisy**: Pink noise added at 20 dB SNR. Simulates moderate background noise.

    All samples are 16kHz mono, 16-bit PCM WAV, padded/trimmed to {CLIP_DURATION}s ({CLIP_SAMPLES} samples).

    ### Negative Samples

    **Adversarial TTS**: Confusable words synthesized with {len(ADVERSARIAL_VOICES)} diverse voices.
    Words include: {', '.join(f'"{w}"' for w in ADVERSARIAL_WORDS[:10])}... ({len(ADVERSARIAL_WORDS)} total)

    **Generic speech**: {len(GENERIC_SPEECH_PHRASES)} common English phrases synthesized with
    {len(GENERIC_SPEECH_VOICES)} voices. These test that the model does not trigger on general speech.

    **LibriSpeech test-clean**: Random 1.5s clips from the standard ASR held-out test set
    (different speakers, read speech). {'Included.' if stats.negatives_librispeech > 0 else 'Skipped (--skip-librispeech or torchaudio unavailable).'}

    **Noise/silence**: Procedurally generated silence, white noise, pink noise, room tone,
    hum, clicks, static, and Brownian noise. These confirm the model does not trigger on
    non-speech audio.

    ## Audio Format

    - Sample rate: {SAMPLE_RATE} Hz
    - Channels: 1 (mono)
    - Bit depth: 16-bit signed integer PCM
    - Duration: {CLIP_DURATION}s ({CLIP_SAMPLES} samples)
    - Container: WAV

    ## Reproducibility

    All random operations use numpy with seed={args.seed}. Re-running with the same
    seed and same TTS engine versions will produce identical noise/augmentation variants.
    TTS outputs themselves may vary slightly across engine versions.
    """)

    if stats.errors:
        content += "\n## Errors Encountered\n\n"
        for err in stats.errors[:50]:
            content += f"- {err}\n"
        if len(stats.errors) > 50:
            content += f"\n... and {len(stats.errors) - 50} more.\n"

    manifest_path = output_dir / "MANIFEST.md"
    manifest_path.write_text(content, encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def build_eval_set(args: argparse.Namespace) -> None:
    """Main pipeline: generate the full clean eval set."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    stats = Stats()
    start_time = time.time()

    n_voices = min(args.voices, len(EDGE_TTS_VOICES))
    selected_voices = EDGE_TTS_VOICES[:n_voices]

    # Precompute IR for reverb augmentation
    ir = generate_synthetic_ir(SAMPLE_RATE, rt60=0.3, rng=np.random.default_rng(args.seed + 1))

    # ==================================================================
    # 1. POSITIVES: Edge TTS
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Generating positive samples (Edge TTS)")
    print(f"{'='*60}")
    print(f"  Voices: {n_voices}, Phrases: {len(WAKE_PHRASES)}")
    print(f"  Expected originals: {n_voices * len(WAKE_PHRASES)}")

    edge_pos_dir = output_dir / "positives" / "edge_tts"
    edge_pos_dir.mkdir(parents=True, exist_ok=True)

    edge_items: list[tuple[str, str, Path]] = []
    for voice in selected_voices:
        for phrase in WAKE_PHRASES:
            safe_phrase = phrase.replace(" ", "_")
            out_path = edge_pos_dir / f"{voice}_{safe_phrase}.wav"
            edge_items.append((phrase, voice, out_path))

    print(f"  Synthesizing {len(edge_items)} samples (concurrency=5)...")
    created_paths = await generate_edge_tts_batch(edge_items, concurrency=5, stats=stats)
    stats.positives_original += len(created_paths)
    for voice in selected_voices:
        if any(voice in str(p) for p in created_paths):
            stats.edge_tts_voices_used.add(voice)

    print(f"  Created: {len(created_paths)} originals")

    # Augment: reverb + noisy
    print("  Generating augmented variants (reverb + noisy)...")
    aug_rng = np.random.default_rng(args.seed + 2)
    for orig_path in created_paths:
        audio = read_wav(orig_path)
        if audio is None:
            continue

        reverbed = apply_reverb(audio, ir)
        reverb_path = orig_path.with_name(orig_path.stem + "_reverb.wav")
        write_wav(reverb_path, reverbed)
        stats.positives_reverb += 1

        noisy = add_noise_at_snr(audio, snr_db=20.0, rng=aug_rng, noise_type="pink")
        noisy_path = orig_path.with_name(orig_path.stem + "_noisy.wav")
        write_wav(noisy_path, noisy)
        stats.positives_noisy += 1

    print(f"  Reverb variants: {stats.positives_reverb}")
    print(f"  Noisy variants: {stats.positives_noisy}")

    # ==================================================================
    # 2. POSITIVES: pyttsx3 (Windows SAPI5)
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Generating positive samples (pyttsx3)")
    print(f"{'='*60}")

    pyttsx3_dir = output_dir / "positives" / "pyttsx3"
    pyttsx3_paths = generate_pyttsx3_samples(pyttsx3_dir, stats, rng)

    if pyttsx3_paths:
        print(f"  Created: {len(pyttsx3_paths)} originals, augmenting...")
        aug_rng2 = np.random.default_rng(args.seed + 3)
        for orig_path in pyttsx3_paths:
            audio = read_wav(orig_path)
            if audio is None:
                continue
            reverbed = apply_reverb(audio, ir)
            write_wav(orig_path.with_name(orig_path.stem + "_reverb.wav"), reverbed)
            stats.positives_reverb += 1
            noisy = add_noise_at_snr(audio, snr_db=20.0, rng=aug_rng2, noise_type="pink")
            write_wav(orig_path.with_name(orig_path.stem + "_noisy.wav"), noisy)
            stats.positives_noisy += 1

    # ==================================================================
    # 3. POSITIVES: Kokoro TTS (if available)
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Generating positive samples (Kokoro)")
    print(f"{'='*60}")

    kokoro_dir = output_dir / "positives" / "kokoro"
    kokoro_paths = generate_kokoro_samples(kokoro_dir, stats)

    if kokoro_paths:
        print(f"  Created: {len(kokoro_paths)} originals, augmenting...")
        aug_rng3 = np.random.default_rng(args.seed + 4)
        for orig_path in kokoro_paths:
            audio = read_wav(orig_path)
            if audio is None:
                continue
            reverbed = apply_reverb(audio, ir)
            write_wav(orig_path.with_name(orig_path.stem + "_reverb.wav"), reverbed)
            stats.positives_reverb += 1
            noisy = add_noise_at_snr(audio, snr_db=20.0, rng=aug_rng3, noise_type="pink")
            write_wav(orig_path.with_name(orig_path.stem + "_noisy.wav"), noisy)
            stats.positives_noisy += 1

    # ==================================================================
    # 4. NEGATIVES: Adversarial TTS
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 4: Generating adversarial negatives (Edge TTS)")
    print(f"{'='*60}")
    print(f"  Voices: {len(ADVERSARIAL_VOICES)}, Words: {len(ADVERSARIAL_WORDS)}")

    adv_dir = output_dir / "negatives" / "adversarial_tts"
    adv_dir.mkdir(parents=True, exist_ok=True)

    adv_items: list[tuple[str, str, Path]] = []
    for voice in ADVERSARIAL_VOICES:
        for word in ADVERSARIAL_WORDS:
            safe_word = word.replace(" ", "_").replace("'", "")
            out_path = adv_dir / f"{voice}_{safe_word}.wav"
            adv_items.append((word, voice, out_path))

    print(f"  Synthesizing {len(adv_items)} adversarial samples...")
    adv_created = await generate_edge_tts_batch(adv_items, concurrency=5, stats=stats)
    stats.negatives_adversarial += len(adv_created)
    print(f"  Created: {len(adv_created)} adversarial negatives")

    # ==================================================================
    # 5. NEGATIVES: Generic speech (TTS)
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 5: Generating generic speech negatives (Edge TTS)")
    print(f"{'='*60}")

    speech_dir = output_dir / "negatives" / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)

    speech_items: list[tuple[str, str, Path]] = []
    for vi, voice in enumerate(GENERIC_SPEECH_VOICES):
        for pi, phrase in enumerate(GENERIC_SPEECH_PHRASES):
            safe_phrase = f"phrase_{pi:03d}"
            out_path = speech_dir / f"{voice}_{safe_phrase}.wav"
            speech_items.append((phrase, voice, out_path))

    print(f"  Synthesizing {len(speech_items)} speech samples...")
    speech_created = await generate_edge_tts_batch(speech_items, concurrency=5, stats=stats)
    stats.negatives_speech += len(speech_created)
    print(f"  Created: {len(speech_created)} speech negatives")

    # ==================================================================
    # 6. NEGATIVES: LibriSpeech test-clean (optional)
    # ==================================================================
    if not args.skip_librispeech:
        print(f"\n{'='*60}")
        print("PHASE 6: LibriSpeech test-clean negatives")
        print(f"{'='*60}")
        libri_dir = output_dir / "negatives" / "speech"
        generate_librispeech_negatives(libri_dir, stats, rng, n_clips=300)
        print(f"  Created: {stats.negatives_librispeech} LibriSpeech clips")
    else:
        print(f"\n{'='*60}")
        print("PHASE 6: LibriSpeech SKIPPED (--skip-librispeech)")
        print(f"{'='*60}")

    # ==================================================================
    # 7. NEGATIVES: Noise / silence
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 7: Generating noise negatives")
    print(f"{'='*60}")

    noise_dir = output_dir / "negatives" / "noise"
    generate_noise_negatives(noise_dir, stats, rng)
    print(f"  Created: {stats.negatives_noise} noise samples")

    # ==================================================================
    # Write manifest
    # ==================================================================
    elapsed = time.time() - start_time
    write_manifest(output_dir, stats, elapsed, args)

    # ==================================================================
    # Final report
    # ==================================================================
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total positives:  {stats.total_positives}")
    print(f"    - Originals:    {stats.positives_original}")
    print(f"    - Reverb:       {stats.positives_reverb}")
    print(f"    - Noisy:        {stats.positives_noisy}")
    print(f"  Total negatives:  {stats.total_negatives}")
    print(f"    - Adversarial:  {stats.negatives_adversarial}")
    print(f"    - Speech:       {stats.negatives_speech}")
    print(f"    - LibriSpeech:  {stats.negatives_librispeech}")
    print(f"    - Noise:        {stats.negatives_noise}")
    print(f"  Grand total:      {stats.total_positives + stats.total_negatives}")
    print(f"  Unique voices:    Edge={len(stats.edge_tts_voices_used)} pyttsx3={stats.pyttsx3_voices_used} Kokoro={stats.kokoro_voices_used}")
    print(f"  Edge TTS skipped: {stats.skipped_edge_tts}")
    print(f"  Errors:           {len(stats.errors)}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"  Output:           {output_dir}")

    if stats.errors:
        print(f"\n  First 5 errors:")
        for e in stats.errors[:5]:
            print(f"    - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean evaluation dataset with ZERO training overlap."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_clean/",
        help="Output directory (default: eval_clean/)",
    )
    parser.add_argument(
        "--skip-librispeech",
        action="store_true",
        help="Skip LibriSpeech download (uses TTS speech negatives only)",
    )
    parser.add_argument(
        "--voices",
        type=int,
        default=24,
        help="Number of Edge TTS voices to use (default: 24)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    asyncio.run(build_eval_set(args))


if __name__ == "__main__":
    main()
