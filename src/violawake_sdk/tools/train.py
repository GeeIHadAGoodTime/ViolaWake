"""
violawake-train CLI -- Train a custom wake word model.

Entry point: ``violawake-train`` (declared in pyproject.toml).

Requires: ``pip install "violawake[training]"``.

Architecture: TemporalCNN classifier head on top of frozen OpenWakeWord (OWW)
audio embeddings. Operates on 9-frame windows of 96-dim OWW embeddings (~25K
params). This is the same architecture as the production Viola model.

Training pipeline:
  - Auto-generates TTS positives if user provides fewer than 100 samples
  - Auto-generates confusable negatives (phonetically similar words)
  - Auto-generates speech negatives (common phrases via TTS)
  - FocalLoss for class imbalance handling
  - AdamW optimizer with cosine annealing LR schedule
  - Exponential Moving Average (EMA) of model weights
  - 80/20 group-aware train/validation split with early stopping
  - Post-training quality gate (speech FP check)

Data pipeline (matches production golden path):
  A. Positives: user-provided + auto-TTS (edge-tts, len(EDGE_TTS_VOICES) voices x 3 phrases x 3 conditions)
  B. Confusable negatives round 1: 30 phonetically similar words x 10 voices
  C. Confusable negatives round 2: 16 tighter variants x 10 voices
  D. Speech negatives: common phrases via TTS (100+ phrases x 5 voices)
  E. Shared universal corpus: LibriSpeech, MUSAN speech/music/noise (auto-discovered)
  F. User-provided negatives via --negatives directory (if any)

Usage::

    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --output models/jarvis.onnx \\
      --epochs 80

    # With real negative samples:
    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --negatives data/jarvis/negatives/ \\
      --output models/jarvis.onnx

Minimum: 5 positive samples (auto-TTS fills to ~200). Recommended: 50+.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when programmatic training cannot continue safely."""


class ModelQualityGateError(TrainingError):
    """Raised when a freshly trained model fails the deployment quality gate
    (grade F) and ONNX export is blocked.

    This is an EXPECTED outcome, not a bug: the model scored at or above the
    deployment detection threshold on no-wake audio (silence, speech, or
    confusable words), i.e. it would false-fire on the wrong sound, so it is not
    shipped. This is usually run-to-run training variance rather than a problem
    with the user's recordings (root cause CL-20260714-4c23 / #1184 / #1465), so
    a retrain with the same recordings often passes. The job is correctly marked
    failed and the user is told, but consumers (the Console backend's error
    classifier) treat this distinctly from an unexpected error so it does not
    page ops via Sentry (GlitchTip violawake issue 28)."""


# Module-level temp directory override. When set, all tempfile operations use
# this instead of the OS default (which may be on a small system drive).
# Set by _train_temporal_cnn() via its tmp_dir parameter.
_TMP_DIR: str | None = None
_LAST_EDGE_TTS_ERROR: str | None = None
_REPORTED_EDGE_TTS_ERRORS: set[tuple[str, str]] = set()
_EDGE_TTS_MAX_ATTEMPTS = 3
_EDGE_TTS_RETRY_BASE_SECONDS = 0.75
_EDGE_TTS_RETRY_MAX_SECONDS = 4.0
_EDGE_TTS_RETRY_RNG = Random()

# ---------------------------------------------------------------------------
# Edge-TTS voice pool for diverse positive and negative generation
# ---------------------------------------------------------------------------

# Verified live against edge_tts.list_voices() on 2026-07-15 (GlitchTip
# violawake issues 34/38, #1768). Microsoft has retired seven of the twenty
# voices this list originally shipped with -- DavisNeural, AmberNeural,
# BrandonNeural, CoraNeural, ElizabethNeural, JacobNeural, MonicaNeural no
# longer exist server-side. Requesting a retired ShortName still completes
# the edge-tts WebSocket handshake (101 Switching Protocols) but the server
# never sends audio frames, so every attempt exhausts all retries with
# NoAudioReceived -- a permanent, deterministic failure, not throttling or a
# network/egress issue (other voices in this same list synthesize instantly
# from the same process). Replaced the seven dead entries with valid
# same-locale voices (AvaNeural, EmmaNeural, BrianNeural) to keep the pool at
# reasonable diversity without inflating it with untested multilingual
# variants. If a future voice silently goes dead again, `_edge_tts_fail`
# below now reports each (voice, text) failure independently instead of
# reporting only the first one seen per process lifetime, so it won't take a
# live investigation to notice.
EDGE_TTS_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-AnaNeural",
    "en-US-AndrewNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-US-AvaNeural",
    "en-US-EmmaNeural",
    "en-US-BrianNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
]

# Common phrases for speech negative generation
SPEECH_NEGATIVE_PHRASES = [
    "what time is it",
    "play some music",
    "turn off the lights",
    "set an alarm for seven",
    "how is the weather today",
    "call mom",
    "send a message",
    "open the door",
    "good morning",
    "good night",
    "thank you very much",
    "what is the news",
    "order a pizza",
    "find a restaurant",
    "navigate to home",
    "play the next song",
    "stop the music",
    "volume up",
    "volume down",
    "skip this track",
    "read my emails",
    "remind me tomorrow",
    "schedule a meeting",
    "take a note",
    "search the web",
    "tell me a joke",
    "translate hello to spanish",
    "what is the temperature",
    "start a timer",
    "cancel the alarm",
    "shuffle my playlist",
    "add to shopping list",
    "where is the nearest store",
    "how far is the airport",
    "book a flight",
    "check my calendar",
    "do not disturb",
    "answer the phone",
    "reject the call",
    "pair bluetooth",
    "connect to wifi",
    "take a screenshot",
    "lock the screen",
    "battery level",
    "airplane mode on",
    "increase brightness",
    "mute the microphone",
    "record a video",
    "scan this document",
    "convert dollars to euros",
    "the quick brown fox jumps over the lazy dog",
    "once upon a time in a land far far away",
    "i need to go to the grocery store",
    "can you help me with something",
    "that sounds like a great idea",
    "i am not sure about that",
    "let me think about it for a moment",
    "we should probably leave soon",
    "have you seen my keys anywhere",
    "it is raining outside right now",
    "i will be there in five minutes",
    "please close the window",
    "the meeting starts at three",
    "dinner is almost ready",
    "happy birthday to you",
    "excuse me could you repeat that",
    "nice to meet you",
    "see you later",
    "how much does it cost",
    "where did you put it",
    "i forgot my password",
    "the package arrived today",
    "she said hello yesterday",
    "they went to the park",
    "he is coming home soon",
    "we are running late",
    "it was a beautiful day",
    "the cat sat on the mat",
    "please pass the salt",
    "i love this song",
    "turn left at the corner",
    "the train departs at noon",
    "water the plants please",
    "feed the dog",
    "empty the dishwasher",
    "check the mailbox",
    "pick up the groceries",
    "wash the car tomorrow",
    "vacuum the living room",
    "fold the laundry",
    "take out the trash",
    "clean the kitchen",
    "organize the closet",
    "paint the bedroom",
    "fix the leaky faucet",
    "mow the lawn this weekend",
    "trim the hedges",
    "shovel the driveway",
    "water the garden",
    "prune the roses",
    "rake the leaves",
]

ProgressCallback = Callable[[dict[str, Any]], None]

_TRAINING_SAMPLE_RATE = 16_000
_TRAINING_FRAME_MS = 20
_TRAINING_FRAME_SAMPLES = _TRAINING_SAMPLE_RATE * _TRAINING_FRAME_MS // 1000


def _check_cancelled(check_cancelled: Callable[[], None] | None) -> None:
    """Raise immediately when the caller indicates cancellation."""
    if check_cancelled is not None:
        check_cancelled()


def _sleep_with_cancel(delay_seconds: float, check_cancelled: Callable[[], None] | None) -> None:
    """Sleep in short slices so cancellation can interrupt retry backoff."""
    if delay_seconds <= 0:
        return

    deadline = time.monotonic() + delay_seconds
    while True:
        _check_cancelled(check_cancelled)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


# ---------------------------------------------------------------------------
# Utility: ONNX runtime provider auto-detection
# ---------------------------------------------------------------------------


def get_best_provider(device: str | None = None) -> str:
    """Auto-detect the best ONNX Runtime execution provider.

    Priority order: CUDA > DirectML > CPU.

    Args:
        device: Optional manual override. One of "cuda", "directml", "cpu",
            or a full provider name like "CUDAExecutionProvider".

    Returns:
        An ONNX Runtime execution provider string.
    """
    import onnxruntime as ort

    if device is not None:
        _SHORTHAND = {
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        provider = _SHORTHAND.get(device.lower(), device)
        available = ort.get_available_providers()
        if provider in available:
            return provider
        print(
            f"WARNING: Requested provider '{provider}' not available "
            f"(have: {available}). Falling back to auto-detection.",
            file=sys.stderr,
        )

    available = ort.get_available_providers()
    for provider in [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if provider in available:
            return provider
    return "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# Edge-TTS audio synthesis helpers (async -> sync bridge)
# ---------------------------------------------------------------------------


def _edge_tts_fail(text: str, voice: str, detail: str | BaseException) -> bool:
    """Record and log an edge-tts failure while preserving the bool API."""
    global _LAST_EDGE_TTS_ERROR

    summary = f"{type(detail).__name__}: {detail}" if isinstance(detail, BaseException) else detail
    _LAST_EDGE_TTS_ERROR = summary

    # Dedup key is (voice, summary), NOT summary alone. A missing decoder or a
    # broken conversion toolchain causes hundreds of *identical* per-sample
    # failures for one job -- log the actual exception once per voice, then
    # the generator summary logs the zero/partial count. Keying on summary
    # alone (pre-#1768) collapsed unrelated failures too: the exhausted-
    # retries message never includes voice/text, so a dead voice (e.g.
    # en-US-DavisNeural, retired by Microsoft -- GlitchTip 34/38) silently
    # ate the report slot for every *other* voice's failures for the rest of
    # the process lifetime. Reproduced live: two different (voice, text)
    # pairs failing in the same process produced only one log line before
    # this fix. `reset_edge_tts_reporting()` additionally clears this per
    # training job so failures are never masked across jobs/customers either.
    dedup_key = (voice, summary)
    if dedup_key not in _REPORTED_EDGE_TTS_ERRORS:
        _REPORTED_EDGE_TTS_ERRORS.add(dedup_key)
        logger.error(
            "edge-tts synthesis failed for voice %s text %.80r: %s",
            voice,
            text,
            summary,
        )
    return False


def reset_edge_tts_reporting() -> None:
    """Clear per-run edge-tts failure state at a training job boundary.

    ``_REPORTED_EDGE_TTS_ERRORS`` and ``_LAST_EDGE_TTS_ERROR`` are process-
    lifetime globals so within-job log spam stays deduped even though the
    training worker process runs for days across many jobs. Without this
    reset, a (voice, summary) pair reported once for job N stays silently
    suppressed for every later job that hits the exact same failure --
    hiding a permanently-broken voice from every customer after the first
    one. Call this once at the start of each training job.
    """
    global _LAST_EDGE_TTS_ERROR
    _REPORTED_EDGE_TTS_ERRORS.clear()
    _LAST_EDGE_TTS_ERROR = None


def _edge_tts_synthesize(
    text: str,
    voice: str,
    output_path: Path,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> bool:
    """Synthesize a single phrase with edge-tts and save as WAV at 16kHz.

    Returns True on success, False on failure.
    """
    import asyncio
    import io
    import tempfile

    global _LAST_EDGE_TTS_ERROR
    _LAST_EDGE_TTS_ERROR = None

    try:
        import edge_tts
    except ImportError as exc:
        message = "edge-tts is not installed. Install with: pip install edge-tts"
        if exc:
            message = f"{message} ({type(exc).__name__}: {exc})"
        return _edge_tts_fail(
            text,
            voice,
            message,
        )

    async def _synth():
        communicate = edge_tts.Communicate(text, voice)
        mp3_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])
        return mp3_buf.getvalue()

    def _run_synth() -> bytes:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.run(_synth())

        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(_synth())).result(timeout=30)
        return loop.run_until_complete(_synth())

    mp3_data: bytes | None = None
    max_attempts = max(1, _EDGE_TTS_MAX_ATTEMPTS)
    for attempt in range(1, max_attempts + 1):
        _check_cancelled(check_cancelled)
        try:
            mp3_data = _run_synth()
            break
        except Exception as exc:
            if attempt >= max_attempts:
                return _edge_tts_fail(
                    text,
                    voice,
                    f"edge-tts failed after {attempt} attempts: {type(exc).__name__}: {exc}",
                )
            delay = min(
                _EDGE_TTS_RETRY_MAX_SECONDS,
                _EDGE_TTS_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            delay += _EDGE_TTS_RETRY_RNG.uniform(0.0, _EDGE_TTS_RETRY_BASE_SECONDS)
            logger.warning(
                "edge-tts synthesis attempt %s/%s failed for voice %s text %.80r: "
                "%s: %s; retrying in %.2fs",
                attempt,
                max_attempts,
                voice,
                text,
                type(exc).__name__,
                exc,
                delay,
            )
            _sleep_with_cancel(delay, check_cancelled)

    if not mp3_data or len(mp3_data) < 100:
        return _edge_tts_fail(
            text,
            voice,
            f"edge-tts returned too little audio data ({len(mp3_data) if mp3_data else 0} bytes)",
        )

    conversion_errors: list[str] = []

    # First try libsndfile via soundfile. The backend image already gets this
    # through the training stack, and it avoids a hard ffmpeg dependency.
    try:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(mp3_data), dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("decoded MP3 contained no audio samples")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = _resample_audio(audio, sr, 16000)
        _save_wav(audio, output_path, 16000)
        return output_path.exists() and output_path.stat().st_size > 44
    except ImportError as exc:
        conversion_errors.append(f"soundfile unavailable: {type(exc).__name__}: {exc}")
    except Exception as exc:
        conversion_errors.append(f"soundfile decode failed: {type(exc).__name__}: {exc}")

    # Fallback: pydub with ffmpeg/ffprobe when available.
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        seg.export(str(output_path), format="wav")
        return output_path.exists() and output_path.stat().st_size > 44
    except ImportError as exc:
        conversion_errors.append(f"pydub unavailable: {type(exc).__name__}: {exc}")
    except Exception as exc:
        conversion_errors.append(f"pydub decode failed: {type(exc).__name__}: {exc}")

    # Fallback: write MP3 to temp, load with torchaudio.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3", dir=_TMP_DIR)
    try:
        os.write(tmp_fd, mp3_data)
    finally:
        os.close(tmp_fd)
    os.chmod(tmp_path, 0o600)

    try:
        import torchaudio

        waveform, sr = torchaudio.load(tmp_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        torchaudio.save(str(output_path), waveform, 16000)
        return output_path.exists() and output_path.stat().st_size > 44
    except ImportError as exc:
        conversion_errors.append(f"torchaudio unavailable: {type(exc).__name__}: {exc}")
    except Exception as exc:
        conversion_errors.append(f"torchaudio decode failed: {type(exc).__name__}: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return _edge_tts_fail(
        text,
        voice,
        "MP3-to-WAV conversion failed; " + "; ".join(conversion_errors),
    )


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio while keeping float32 output."""
    import numpy as np
    from scipy.signal import resample_poly

    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)

    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return np.asarray(resample_poly(audio, up, down), dtype=np.float32)


def _kokoro_tts_synthesize(
    text: str,
    voice: str,
    output_path: Path,
    *,
    engine: Any | None = None,
) -> bool:
    """Synthesize a single phrase with Kokoro and save as WAV at 16kHz."""
    import numpy as np

    try:
        from violawake_sdk.tts import TTS_SAMPLE_RATE, TTSEngine
    except ImportError:
        return False

    try:
        kokoro_engine = engine
        if kokoro_engine is None:
            kokoro_engine = TTSEngine(voice=voice, sample_rate=TTS_SAMPLE_RATE)
        else:
            kokoro_engine.voice = voice

        audio = np.asarray(kokoro_engine.synthesize(text), dtype=np.float32)
        if audio.size == 0:
            return False
        if int(kokoro_engine.sample_rate) != 16000:
            audio = _resample_audio(audio, int(kokoro_engine.sample_rate), 16000)
        _save_wav(audio, output_path, sample_rate=16000)
        return True
    except Exception:
        return False


class _KokoroFallback:
    """Lazily-initialized, per-sample Kokoro TTS fallback for edge-tts callers.

    Shared by every ``EDGE_TTS_VOICES`` generator (#1768) so a single dead or
    transiently-flaky edge-tts voice only ever loses ITS OWN sample to
    Kokoro, never the whole run. Readiness (import + engine construction) is
    checked at most once per generator call and cached either way, so a
    missing/broken Kokoro install costs one probe, not one probe per sample.
    """

    def __init__(self) -> None:
        self._checked = False
        self._available = False
        self._engine: Any | None = None
        self._voices: list[str] = []

    def ready(self) -> bool:
        if self._checked:
            return self._available
        self._checked = True
        try:
            from violawake_sdk.tts import AVAILABLE_VOICES, TTS_SAMPLE_RATE, TTSEngine
        except ImportError:
            return False

        self._voices = list(AVAILABLE_VOICES)
        if not self._voices:
            return False
        try:
            self._engine = TTSEngine(voice=self._voices[0], sample_rate=TTS_SAMPLE_RATE)
        except Exception:
            self._engine = None
            return False
        print("Kokoro TTS fallback ready (used per-sample when edge-tts fails)")
        self._available = True
        return True

    def synthesize(self, text: str, output_path: Path, *, rotate_index: int) -> bool:
        """Synthesize with a Kokoro voice picked deterministically from `rotate_index`.

        Caller must have already confirmed ``ready()`` returned True.
        """
        if not self._voices:
            return False
        voice = self._voices[rotate_index % len(self._voices)]
        return _kokoro_tts_synthesize(text, voice, output_path, engine=self._engine)


def _generate_tts_positives(
    wake_word: str,
    output_dir: Path,
    verbose: bool = True,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> list[Path]:
    """Generate diverse TTS positive samples using Edge TTS with Kokoro fallback.

    Produces: len(EDGE_TTS_VOICES) voices x 3 phrases (WORD, hey WORD, ok WORD) clean files.
    Then augmentation (noisy + reverb) multiplies to ~180 total.

    Returns list of generated WAV file paths.
    """
    import numpy as np

    from violawake_sdk.training.augment import (
        rir_augment,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    phrases = [wake_word, f"hey {wake_word}", f"ok {wake_word}"]
    generated: list[Path] = []
    # NOTE (#1768): this used to be a *sticky* switch -- once any single
    # edge-tts call failed, a `kokoro_fallback = True` flag made every
    # subsequent (voice, phrase) in this job route straight to Kokoro, even
    # for the 15+ other edge-tts voices that were perfectly valid. One dead
    # voice (en-US-DavisNeural, retired by Microsoft) was silently
    # collapsing ~80% of a job's "diverse edge-tts voices" down to Kokoro's
    # much smaller voice set for every job that hit it in voice-list order.
    # `_KokoroFallback` retries edge-tts independently for every (voice,
    # phrase); Kokoro only ever substitutes for the one sample that actually
    # failed, so a single bad voice can no longer silently erase the
    # diversity the other voices provide.
    kokoro = _KokoroFallback()

    if verbose:
        total = len(EDGE_TTS_VOICES) * len(phrases)
        print(
            f"  Generating TTS positives: {len(EDGE_TTS_VOICES)} voices x {len(phrases)} phrases = {total} clean samples..."
        )

    for voice_idx, voice in enumerate(EDGE_TTS_VOICES):
        _check_cancelled(check_cancelled)
        for phrase_idx, phrase in enumerate(phrases):
            _check_cancelled(check_cancelled)
            clean_path = output_dir / f"tts_pos_{voice_idx:02d}_{phrase_idx}_{voice}.wav"
            if clean_path.exists():
                generated.append(clean_path)
                continue

            ok = _edge_tts_synthesize(
                phrase,
                voice,
                clean_path,
                check_cancelled=check_cancelled,
            )
            if not ok and kokoro.ready():
                ok = kokoro.synthesize(phrase, clean_path, rotate_index=voice_idx)
            if ok and clean_path.exists():
                generated.append(clean_path)

                # Generate noisy variant
                try:
                    _check_cancelled(check_cancelled)
                    from violawake_sdk.audio import load_audio
                    from violawake_sdk.training.augment import apply_additive_noise

                    audio = load_audio(clean_path)
                    if audio is not None and len(audio) > 0:
                        rng = np.random.default_rng(voice_idx * 100 + phrase_idx)

                        # Noisy variant (SNR 10-15 dB)
                        noisy = apply_additive_noise(audio, snr_db=12.0, rng=rng)
                        noisy_path = (
                            output_dir / f"tts_pos_{voice_idx:02d}_{phrase_idx}_{voice}_noisy.wav"
                        )
                        _save_wav(noisy, noisy_path)
                        generated.append(noisy_path)

                        # Reverb variant
                        reverbed = rir_augment(audio, rng=rng)
                        reverb_path = (
                            output_dir / f"tts_pos_{voice_idx:02d}_{phrase_idx}_{voice}_reverb.wav"
                        )
                        _save_wav(reverbed, reverb_path)
                        generated.append(reverb_path)
                except Exception:
                    pass  # Augmented variants are best-effort

        if verbose and (voice_idx + 1) % 5 == 0:
            print(
                f"    {voice_idx + 1}/{len(EDGE_TTS_VOICES)} voices done ({len(generated)} files)"
            )

    if verbose:
        print(f"  TTS positives generated: {len(generated)} files")

    return generated


def _generate_confusable_negatives(
    wake_word: str,
    output_dir: Path,
    n_confusables: int = 30,
    voices_per_word: int = 10,
    verbose: bool = True,
    *,
    progress_callback: ProgressCallback | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> list[Path]:
    """Generate confusable negative samples via TTS.

    Uses the confusables generator to find phonetically similar words,
    then synthesizes each with multiple TTS voices.

    Returns list of generated WAV file paths.
    """
    from violawake_sdk.tools.confusables import generate_confusables

    output_dir.mkdir(parents=True, exist_ok=True)
    confusable_words = generate_confusables(wake_word, count=n_confusables)

    if verbose:
        print(f"  Generated {len(confusable_words)} confusable words for '{wake_word}'")
        if confusable_words[:5]:
            print(f"    Top 5: {', '.join(confusable_words[:5])}")
        total = len(confusable_words) * voices_per_word
        print(
            f"  Synthesizing: {len(confusable_words)} words x {voices_per_word} voices = {total} samples..."
        )

    voices_subset = EDGE_TTS_VOICES[:voices_per_word]
    generated: list[Path] = []
    total_samples = len(confusable_words) * len(voices_subset)
    completed_samples = 0
    # #1768: negatives had NO fallback at all -- a dead/flaky edge-tts voice
    # just silently dropped that sample, shrinking the negative-sample pool
    # with no recovery. Give it the same per-sample Kokoro fallback the
    # positives path uses.
    kokoro = _KokoroFallback()

    for word_idx, word in enumerate(confusable_words):
        _check_cancelled(check_cancelled)
        for voice_idx, voice in enumerate(voices_subset):
            _check_cancelled(check_cancelled)
            safe_word = word.replace(" ", "_")[:30]
            out_path = output_dir / f"confusable_{word_idx:03d}_{voice_idx}_{safe_word}.wav"
            if out_path.exists():
                generated.append(out_path)
                continue

            ok = _edge_tts_synthesize(
                word,
                voice,
                out_path,
                check_cancelled=check_cancelled,
            )
            if not ok and kokoro.ready():
                ok = kokoro.synthesize(word, out_path, rotate_index=voice_idx)
            if ok and out_path.exists():
                generated.append(out_path)

            completed_samples += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_word": word,
                        "word_index": word_idx + 1,
                        "total_words": len(confusable_words),
                        "voice_index": voice_idx + 1,
                        "total_voices": len(voices_subset),
                        "completed_samples": completed_samples,
                        "total_samples": total_samples,
                        "generated_files": len(generated),
                    }
                )

        if verbose and (word_idx + 1) % 10 == 0:
            print(f"    {word_idx + 1}/{len(confusable_words)} words done ({len(generated)} files)")

    if not generated and confusable_words and voices_subset:
        logger.error(
            "edge-tts confusable negative generation produced 0 files for wake word %.80r "
            "after %s attempts; last error: %s",
            wake_word,
            len(confusable_words) * len(voices_subset),
            _LAST_EDGE_TTS_ERROR or "unknown",
        )

    if verbose:
        print(f"  Confusable negatives generated: {len(generated)} files")

    return generated


def _generate_speech_negatives(
    output_dir: Path,
    n_voices: int = 5,
    verbose: bool = True,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> list[Path]:
    """Deprecated for production training: generate speech negatives via TTS.

    Production training should use the shared LibriSpeech/MUSAN corpus for
    generic speech negatives. This helper remains for legacy CLI experiments.

    Returns list of generated WAV file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    voices_subset = EDGE_TTS_VOICES[:n_voices]
    generated: list[Path] = []

    total = len(SPEECH_NEGATIVE_PHRASES) * n_voices
    if verbose:
        print(
            f"  Generating speech negatives: {len(SPEECH_NEGATIVE_PHRASES)} phrases x {n_voices} voices = {total} samples..."
        )

    for phrase_idx, phrase in enumerate(SPEECH_NEGATIVE_PHRASES):
        _check_cancelled(check_cancelled)
        for voice_idx, voice in enumerate(voices_subset):
            _check_cancelled(check_cancelled)
            safe_phrase = phrase.replace(" ", "_")[:40]
            out_path = output_dir / f"speech_neg_{phrase_idx:03d}_{voice_idx}_{safe_phrase}.wav"
            if out_path.exists():
                generated.append(out_path)
                continue

            ok = _edge_tts_synthesize(
                phrase,
                voice,
                out_path,
                check_cancelled=check_cancelled,
            )
            if ok and out_path.exists():
                generated.append(out_path)

        if verbose and (phrase_idx + 1) % 25 == 0:
            print(
                f"    {phrase_idx + 1}/{len(SPEECH_NEGATIVE_PHRASES)} phrases done ({len(generated)} files)"
            )

    if not generated and SPEECH_NEGATIVE_PHRASES and voices_subset:
        logger.error(
            "edge-tts speech negative generation produced 0 files after %s attempts; "
            "last error: %s",
            len(SPEECH_NEGATIVE_PHRASES) * len(voices_subset),
            _LAST_EDGE_TTS_ERROR or "unknown",
        )

    if verbose:
        print(f"  Speech negatives generated: {len(generated)} files")

    return generated


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int = 16000) -> None:
    """Save float32 audio to a WAV file."""
    import wave

    import numpy as np

    audio = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def _validate_training_audio_array(audio: np.ndarray, sample_rate: int, path: Path) -> np.ndarray:
    """Validate the training audio contract and return mono float32 samples."""
    import numpy as np

    if sample_rate != _TRAINING_SAMPLE_RATE:
        raise TrainingError(
            f"Training audio must be {_TRAINING_SAMPLE_RATE} Hz mono; {path} is {sample_rate} Hz."
        )

    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim == 1:
        mono = audio_array
    elif audio_array.ndim == 2:
        if audio_array.shape[1] != 1:
            raise TrainingError(
                f"Training audio must be mono; {path} has {audio_array.shape[1]} channels."
            )
        mono = audio_array[:, 0]
    else:
        raise TrainingError(f"Training audio must be 1D or mono 2D audio; {path} is invalid.")

    if mono.size == 0:
        raise TrainingError(f"Training audio is empty: {path}")

    return mono.astype(np.float32, copy=False)


def _load_training_audio(path: Path) -> np.ndarray:
    """Load training audio without resampling or channel mixing.

    Generic SDK loading may resample for user convenience. Training cannot:
    a 22 kHz or stereo clip in the corpus is an audio-contract breach, so it
    must fail before embeddings are extracted.
    """
    import wave

    import numpy as np

    path = Path(path)
    load_errors: list[str] = []

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return _validate_training_audio_array(audio, int(sample_rate), path)
    except TrainingError:
        raise
    except Exception as exc:
        load_errors.append(f"soundfile: {type(exc).__name__}: {exc}")

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())

            if channels != 1:
                raise TrainingError(f"Training audio must be mono; {path} has {channels} channels.")
            if sample_rate != _TRAINING_SAMPLE_RATE:
                raise TrainingError(
                    f"Training audio must be {_TRAINING_SAMPLE_RATE} Hz mono; "
                    f"{path} is {sample_rate} Hz."
                )

            if sample_width == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise TrainingError(
                    f"Training WAV must use 16-bit or 32-bit PCM; {path} has {sample_width} bytes."
                )
            return _validate_training_audio_array(audio, sample_rate, path)
        except TrainingError:
            raise
        except Exception as exc:
            load_errors.append(f"wave: {type(exc).__name__}: {exc}")

    raise TrainingError(f"Failed to load training audio {path}: {'; '.join(load_errors)}")


# ---------------------------------------------------------------------------
# Positive augmentation and temporal embedding helpers
# ---------------------------------------------------------------------------


def _augment_positives(
    raw_audio_arrays: list[np.ndarray],
    *,
    sample_rate: int = 16000,
    copies_per_clip: int = 21,
    seed: int = 42,
) -> list[np.ndarray]:
    """Augment positive clips with the roadmap audiomentations chain.

    This operates on raw waveform arrays before OWW embedding extraction and
    returns only augmented copies (the originals remain unchanged).
    """
    import numpy as np

    try:
        from audiomentations import (
            Compose,
            Gain,
            Mp3Compression,
            PitchShift,
            TimeMask,
            TimeStretch,
        )
    except ImportError as e:
        raise RuntimeError(
            "audiomentations is required for positive augmentation. "
            "Install with: pip install 'violawake[training]'"
        ) from e

    if not raw_audio_arrays:
        return []

    augmenter = Compose(
        [
            Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.8),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.5),
            Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.3),
            TimeMask(min_band_part=0.0, max_band_part=0.1, p=0.3),
        ],
        shuffle=False,
    )

    augmented: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for audio in raw_audio_arrays:
        base_audio = np.asarray(audio, dtype=np.float32)
        for _ in range(copies_per_clip):
            # audiomentations reads numpy's global RNG internally.
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            augmented_audio = augmenter(samples=base_audio.copy(), sample_rate=sample_rate)
            augmented.append(np.asarray(augmented_audio, dtype=np.float32))

    return augmented


def _prepare_audio_for_oww(
    audio: np.ndarray,
    *,
    clip_name: str,
    verbose: bool,
) -> np.ndarray | None:
    """Center-crop/pad an audio clip and convert it to int16 for OWW."""
    import numpy as np

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if audio_f32.size == 0:
        return None

    audio_rms = float(np.sqrt(np.mean(audio_f32**2)))
    if audio_rms < 1e-6:
        if verbose:
            print(f"    WARNING: Skipping zero-energy clip: {clip_name}")
        return None

    audio_f32 = center_crop(audio_f32, CLIP_SAMPLES)
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767).astype(np.int16)

    if len(audio_i16) < CLIP_SAMPLES:
        audio_i16 = np.pad(audio_i16, (0, CLIP_SAMPLES - len(audio_i16)))
    else:
        audio_i16 = audio_i16[:CLIP_SAMPLES]

    return audio_i16


# ---------------------------------------------------------------------------
# Streaming-parity scoring for the silence subgrade (#1487 / #2611)
# ---------------------------------------------------------------------------
#
# The silence subgrade used to score audio through _extract_temporal_embeddings:
# ONE batch preprocessor.embed_clips() call on a SINGLE 1.5s center-crop
# (_prepare_audio_for_oww -> center_crop(audio, CLIP_SAMPLES)). The runtime
# (WakeDetector.process -> OpenWakeWordBackbone.push_audio, once per 20ms frame)
# instead streams the FULL continuous input through persistent ring/mel buffers
# that accumulate state across calls. _extract_temporal_embeddings' own docstring
# warns "streaming push_audio() produces subtly different embeddings due to
# internal state accumulation" -- the silence subgrade was the one place that
# warning was never heeded.
#
# Measured on six real deployed models (#2611, 2026-07-24, wakeword-backend-1):
# batch-vs-streaming divergence up to 0.368 on the same near-silence audio, i.e.
# the batch number did not predict the runtime number it claimed to protect.
def _extract_streaming_temporal_windows(
    audio_clips: list[np.ndarray],
    seq_len: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract temporal embedding windows via the real runtime streaming path.

    Unlike ``_extract_temporal_embeddings`` (one batch ``embed_clips`` call on a
    center-cropped 1.5s excerpt), this feeds each FULL, uncropped clip through
    ``OpenWakeWordBackbone.push_audio`` -- the exact call ``WakeDetector.process``
    makes at runtime -- one 20ms frame (``wake_detector.FRAME_SAMPLES``) at a
    time, and windows the resulting embedding sequence with the same sliding-
    window helper used everywhere else in this module. A fresh backbone is reset
    per clip so probes do not leak streaming state into one another.

    Returns ``(windows, source_indices)``.
    """
    import numpy as np

    from violawake_sdk.backends.onnx_backend import OnnxBackend
    from violawake_sdk.oww_backbone import OpenWakeWordBackbone
    from violawake_sdk.wake_detector import FRAME_SAMPLES

    backbone = OpenWakeWordBackbone(OnnxBackend())

    all_windows: list[np.ndarray] = []
    all_source_idx: list[int] = []
    for clip_idx, clip in enumerate(audio_clips):
        audio_f32 = np.asarray(clip, dtype=np.float32).reshape(-1)
        if audio_f32.size == 0:
            continue
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)

        backbone.reset()
        frame_embeddings: list[np.ndarray] = []
        n_usable = len(audio_i16) - (len(audio_i16) % FRAME_SAMPLES)
        for i in range(0, n_usable, FRAME_SAMPLES):
            produced, embedding = backbone.push_audio(audio_i16[i : i + FRAME_SAMPLES])
            if produced and embedding is not None:
                frame_embeddings.append(embedding.astype(np.float32))
        if not frame_embeddings:
            continue

        windows, source_idx, _tags = _temporal_windows_from_frame_embeddings(
            np.stack(frame_embeddings), source_id=clip_idx, tag="stream", seq_len=seq_len
        )
        all_windows.extend(windows)
        all_source_idx.extend(source_idx)

    return all_windows, all_source_idx


# Room-tone probe extraction (#2611).
#
# The gate's old silence probe was synthetic white noise at float RMS 1e-4 --
# int16 RMS 3.29. Measured against real recorded audio (LibriSpeech quiet
# windows, and real user recordings on the box): real room tone sits at int16
# RMS 224-3782. The synthetic probe was therefore ~100-1000x quieter than the
# quietest sound any microphone actually produces, in a regime the model never
# saw in training, where its output is arbitrary. The runtime's own RMS floor
# comment (wake_detector.py: "speech ~= 500-5000") agrees on the scale.
#
# The user's own recordings always contain the real room tone of the real
# microphone that will run this model -- the most predictive no-wake probe
# available. These constants select those segments.
_ROOM_TONE_WINDOW = 4800  # 300ms energy window
# Quiet := window RMS below this fraction of the clip's OWN loudest window (i.e. of
# the spoken wake word in that same recording). Referencing the clip's peak rather
# than its mean keeps the split stable no matter how much of the clip is speech.
_ROOM_TONE_MAX_FRACTION = 0.25
_ROOM_TONE_MIN_SAMPLES = 16000  # need >=1s of room tone from a clip to use it
_RUNTIME_RMS_FLOOR = 1.0  # wake_detector.py Gate 1 -- below this the runtime never scores


def _int16_rms(audio: np.ndarray) -> float:
    """RMS on the int16 scale the runtime's RMS floor is calibrated against."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a * 32767.0) ** 2)))


def _extract_room_tone(audio: np.ndarray) -> np.ndarray | None:
    """Pull the real room-tone (non-speech) segments out of one recording.

    Keeps 300ms windows whose energy is far below the clip's own average (so the
    spoken wake word itself is excluded) but still above the runtime RMS floor
    (so it is audio the runtime would actually score). Returns None when the clip
    yields too little room tone to be a useful probe.
    """
    import numpy as np

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size < _ROOM_TONE_WINDOW * 2:
        return None

    windows = [
        a[start : start + _ROOM_TONE_WINDOW]
        for start in range(0, len(a) - _ROOM_TONE_WINDOW, _ROOM_TONE_WINDOW)
    ]
    window_rms = [_int16_rms(w) for w in windows]
    speech_level = max(window_rms, default=0.0)
    if speech_level <= 0.0:
        return None
    quiet_bar = _ROOM_TONE_MAX_FRACTION * speech_level

    keep = [
        w
        for w, rms in zip(windows, window_rms, strict=True)
        if _RUNTIME_RMS_FLOOR < rms < quiet_bar
    ]
    if not keep:
        return None
    room_tone = np.concatenate(keep)
    return room_tone if len(room_tone) >= _ROOM_TONE_MIN_SAMPLES else None


def _temporal_windows_from_frame_embeddings(
    frame_embeddings: np.ndarray,
    *,
    source_id: int,
    tag: str,
    seq_len: int,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Convert one clip's frame embeddings into seq_len temporal windows."""
    import numpy as np

    if len(frame_embeddings.shape) == 1:
        frame_embeddings = frame_embeddings.reshape(1, -1)

    n_frames = frame_embeddings.shape[0]
    windows: list[np.ndarray] = []
    source_indices: list[int] = []
    tags: list[str] = []

    if n_frames >= seq_len:
        for i in range(n_frames - seq_len + 1):
            window = frame_embeddings[i : i + seq_len].astype(np.float32)
            windows.append(window)
            source_indices.append(source_id)
            tags.append(tag)
    elif n_frames > 0:
        padded = np.zeros((seq_len, frame_embeddings.shape[1]), dtype=np.float32)
        padded[:n_frames] = frame_embeddings
        for j in range(n_frames, seq_len):
            padded[j] = frame_embeddings[-1]
        windows.append(padded)
        source_indices.append(source_id)
        tags.append(tag)

    return windows, source_indices, tags


def _extract_temporal_windows_from_audio(
    audio_clips: list[np.ndarray],
    source_ids: list[int],
    tag: str,
    verbose: bool = True,
    seq_len: int = 9,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Extract temporal OWW embedding windows from in-memory audio arrays."""

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        raise TrainingError(f"openwakeword required: {e}") from e

    if len(audio_clips) != len(source_ids):
        raise ValueError("audio_clips and source_ids must have the same length")

    # Pin ONNX backend explicitly. openwakeword defaults to TFLite when both are
    # present, but the bundled tflite_runtime in our backend image rejects the
    # current openwakeword .tflite schema with "Could not open ...". ONNX path
    # is the canonical production target anyway.
    oww = OWWModel(inference_framework="onnx")
    preprocessor = oww.preprocessor

    all_embeddings: list[np.ndarray] = []
    all_source_idx: list[int] = []
    all_tags: list[str] = []
    failures = 0

    for clip_idx, audio in enumerate(audio_clips):
        audio_i16 = _prepare_audio_for_oww(
            audio,
            clip_name=f"{tag}_{clip_idx:04d}",
            verbose=verbose and failures == 0,
        )
        if audio_i16 is None:
            failures += 1
            continue

        try:
            frame_embeddings_3d = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
            clip_embeddings, clip_sources, clip_tags = _temporal_windows_from_frame_embeddings(
                frame_embeddings_3d[0],
                source_id=source_ids[clip_idx],
                tag=tag,
                seq_len=seq_len,
            )
            all_embeddings.extend(clip_embeddings)
            all_source_idx.extend(clip_sources)
            all_tags.extend(clip_tags)
        except Exception:
            failures += 1

        if verbose and (clip_idx + 1) % 100 == 0:
            print(f"    {clip_idx + 1}/{len(audio_clips)} clips -> {len(all_embeddings)} windows")

    if verbose:
        print(
            f"  [{tag}] {len(audio_clips)} clips -> {len(all_embeddings)} temporal windows "
            f"({failures} failures)"
        )

    return all_embeddings, all_source_idx, all_tags


# ---------------------------------------------------------------------------
# Temporal embedding extraction (9-frame windows from OWW backbone)
# ---------------------------------------------------------------------------


def _extract_temporal_embeddings(
    audio_files: list[Path],
    tag: str,
    verbose: bool = True,
    seq_len: int = 9,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Extract 9-frame temporal OWW embedding windows from audio files.

    Uses OWW's preprocessor.embed_clips (batch mode) — the same embedding
    extraction method used to train the production temporal_cnn model.
    This is critical for pipeline equivalence: streaming push_audio() produces
    subtly different embeddings due to internal state accumulation.

    Files are processed one at a time so long corpus negatives do not force the
    trainer to hold hours of decoded raw audio in memory before the 1.5s crop.
    Each file is loaded, center-cropped to CLIP_SAMPLES, embedded, converted to
    temporal windows, and then discarded before the next file is loaded.

    Returns:
        embeddings: List of (seq_len, 96) numpy arrays.
        source_indices: Source file index for each embedding (for group-aware split).
        tags: Tag string for each embedding.
    """
    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        raise TrainingError(f"openwakeword required: {e}") from e

    oww = OWWModel(inference_framework="onnx")
    preprocessor = oww.preprocessor

    all_embeddings: list[np.ndarray] = []
    all_source_idx: list[int] = []
    all_tags: list[str] = []
    failures = 0

    for file_idx, wav_path in enumerate(audio_files):
        audio = _load_training_audio(wav_path)

        audio_i16 = _prepare_audio_for_oww(
            audio,
            clip_name=wav_path.name,
            verbose=verbose and failures == 0,
        )
        if audio_i16 is None:
            failures += 1
            continue

        try:
            frame_embeddings_3d = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
            file_embeddings, file_source_ids, file_tags = _temporal_windows_from_frame_embeddings(
                frame_embeddings_3d[0],
                source_id=file_idx,
                tag=tag,
                seq_len=seq_len,
            )
            all_embeddings.extend(file_embeddings)
            all_source_idx.extend(file_source_ids)
            all_tags.extend(file_tags)
        except Exception:
            failures += 1

        if verbose and (file_idx + 1) % 100 == 0:
            print(f"    {file_idx + 1}/{len(audio_files)} files -> {len(all_embeddings)} windows")

    if verbose:
        print(
            f"  [{tag}] {len(audio_files)} files -> {len(all_embeddings)} temporal windows "
            f"({failures} failures)"
        )

    return all_embeddings, all_source_idx, all_tags


# ---------------------------------------------------------------------------
# MLP single-frame embedding extraction (legacy path)
# ---------------------------------------------------------------------------


def _extract_mlp_embeddings(
    audio_files: list[Path],
    tag: str,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Extract mean-pooled OWW embeddings for legacy MLP architecture.

    Returns:
        embeddings: List of (96,) numpy arrays.
        source_indices: Source file index for each embedding.
        tags: Tag string for each embedding.
    """
    import numpy as np

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        print(f"ERROR: openwakeword required: {e}", file=sys.stderr)
        sys.exit(1)

    oww = OWWModel(inference_framework="onnx")
    preprocessor = oww.preprocessor

    all_embeddings: list[np.ndarray] = []
    all_source_idx: list[int] = []
    all_tags: list[str] = []
    failures = 0

    for file_idx, wav_path in enumerate(audio_files):
        audio = _load_training_audio(wav_path)

        # Guard against zero-energy files (corrupted or silent recordings).
        # If these slip through upload validation, they corrupt training:
        # the model learns silence = wake word.
        audio_rms = float(np.sqrt(np.mean(audio**2)))
        if audio_rms < 1e-6:
            if verbose and failures == 0:
                print(f"    WARNING: Skipping zero-energy file: {wav_path.name}")
            failures += 1
            continue

        audio = center_crop(audio, CLIP_SAMPLES)
        audio_i16 = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767).astype(np.int16)

        if len(audio_i16) < CLIP_SAMPLES:
            audio_i16 = np.pad(audio_i16, (0, CLIP_SAMPLES - len(audio_i16)))
        else:
            audio_i16 = audio_i16[:CLIP_SAMPLES]

        try:
            embeddings = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
            emb = embeddings.mean(axis=1)[0].astype(np.float32)
            all_embeddings.append(emb)
            all_source_idx.append(file_idx)
            all_tags.append(tag)
        except Exception:
            failures += 1

        if verbose and (file_idx + 1) % 100 == 0:
            print(
                f"    {file_idx + 1}/{len(audio_files)} files -> {len(all_embeddings)} embeddings"
            )

    if verbose:
        print(
            f"  [{tag}] {len(audio_files)} files -> {len(all_embeddings)} embeddings "
            f"({failures} failures)"
        )

    return all_embeddings, all_source_idx, all_tags


# ---------------------------------------------------------------------------
# Group-aware train/val split
# ---------------------------------------------------------------------------


def _group_aware_split(
    labels: np.ndarray,
    source_idx: np.ndarray,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware stratified train/val split.

    All embeddings from the same source file go to the same split
    to prevent data leakage from augmented variants.

    Returns (train_indices, val_indices) as numpy arrays.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_sources = sorted(set(source_idx[pos_mask].tolist()))
    neg_sources = sorted(set(source_idx[neg_mask].tolist()))

    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)

    n_val_pos = max(1, int(len(pos_sources) * val_fraction))
    n_val_neg = max(1, int(len(neg_sources) * val_fraction))

    val_pos_sources = set(pos_sources[:n_val_pos])
    val_neg_sources = set(neg_sources[:n_val_neg])

    val_mask = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if (
            labels[i] == 1
            and source_idx[i] in val_pos_sources
            or labels[i] == 0
            and source_idx[i] in val_neg_sources
        ):
            val_mask[i] = True

    train_indices = np.where(~val_mask)[0]
    val_indices = np.where(val_mask)[0]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Core training: TemporalCNN (production architecture)
# ---------------------------------------------------------------------------


def _train_temporal_cnn(
    pos_files: list[Path],
    neg_files: list[Path],
    output_path: Path,
    wake_word: str = "custom",
    epochs: int = 80,
    augment: bool = True,
    eval_dir: Path | None = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    device: str | None = None,
    ema_decay: float = 0.999,
    seq_len: int = 9,
    neg_tags: dict[str, list[Path]] | None = None,
    augment_source_files: list[Path] | None = None,
    tmp_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train a TemporalCNN on 9-frame OWW embedding windows.

    This replicates the proven production training recipe:
    - TemporalCNN(96, 9) architecture (~25K params)
    - FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    - AdamW + cosine annealing LR
    - EMA weight averaging
    - Group-aware split
    - Early stopping

    Args:
        pos_files: Positive audio file paths.
        neg_files: All negative audio file paths (flat list).
        output_path: Where to save the ONNX model.
        wake_word: Wake word name (for config).
        epochs: Max training epochs.
        augment: Whether to augment positives before extraction.
        eval_dir: Optional eval directory.
        batch_size: Mini-batch size.
        lr: Learning rate.
        patience: Early stopping patience (default 15, matching J5 proven recipe).
        verbose: Print progress.
        progress_callback: Optional callback for UI.
        device: Torch device hint.
        ema_decay: EMA decay factor.
        seq_len: Number of frames per temporal window.
        neg_tags: Optional dict mapping tag -> file list, for tagged negatives.
        augment_source_files: Optional subset of positives to augment. Defaults
            to all positives when omitted.

    Returns:
        Config dict with training results.
    """
    training_start = time.monotonic()

    # -- Direct temp files to a non-system drive when requested --------------
    global _TMP_DIR  # noqa: PLW0603
    if tmp_dir is not None:
        _TMP_DIR = str(tmp_dir)
        Path(_TMP_DIR).mkdir(parents=True, exist_ok=True)

    # -- Lazy imports --------------------------------------------------------
    try:
        import numpy as np
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise TrainingError(
            f"PyTorch required for training: {e}. Install with: pip install 'violawake[training]'"
        ) from e

    from violawake_sdk.training.losses import FocalLoss
    from violawake_sdk.training.temporal_model import (
        TemporalCNN,
        count_parameters,
        export_temporal_onnx,
    )
    from violawake_sdk.training.weight_averaging import (
        EMATracker,
        auto_select_averaging,
    )

    # -- Deterministic seeding (matches production) --------------------------
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    EMBEDDING_DIM = 96
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Load and augment positives before embedding extraction ---------------
    from violawake_sdk._constants import SAMPLE_RATE

    validation_fraction = 0.2
    raw_pos_audio: list[np.ndarray] = []
    raw_pos_source_ids: list[int] = []
    augment_candidates: list[np.ndarray] = []
    augment_candidate_source_ids: list[int] = []
    augment_target_paths = set(augment_source_files or pos_files)

    for file_idx, wav_path in enumerate(pos_files):
        audio = _load_training_audio(wav_path)
        raw_pos_audio.append(audio)
        raw_pos_source_ids.append(file_idx)
        if wav_path in augment_target_paths:
            augment_candidates.append(audio)
            augment_candidate_source_ids.append(file_idx)

    original_pos_clip_count = len(raw_pos_audio)
    n_augmented = 0
    augmented_pos_audio: list[np.ndarray] = []
    augmented_pos_source_ids: list[int] = []

    if augment and augment_candidates:
        if verbose:
            print("\nStep 2: Augmenting positive audio arrays with audiomentations...")

        min_augmented_total = 210
        copies_per_clip = max(1, math.ceil(min_augmented_total / len(augment_candidates)))
        augmented_pos_audio = _augment_positives(
            augment_candidates,
            sample_rate=SAMPLE_RATE,
            copies_per_clip=copies_per_clip,
            seed=SEED,
        )
        augmented_pos_source_ids = [
            source_id for source_id in augment_candidate_source_ids for _ in range(copies_per_clip)
        ]
        n_augmented = len(augmented_pos_audio)
        raw_pos_audio.extend(augmented_pos_audio)
        raw_pos_source_ids.extend(augmented_pos_source_ids)

        if verbose:
            print(
                f"  {original_pos_clip_count} original clips + {n_augmented} augmented clips "
                f"= {len(raw_pos_audio)} positive clips before embeddings"
            )
    elif verbose and not augment:
        print("\nStep 2: Positive augmentation disabled; using original clips only.")
    elif verbose:
        print("\nStep 2: No positive clips available for augmentation; using originals only.")

    # -- Extract temporal embeddings -----------------------------------------
    if verbose:
        print(f"\nStep 3: Extracting {seq_len}-frame temporal OWW embeddings...")
        print(f"  Processing {len(raw_pos_audio)} positive clips...")

    pos_embs, pos_src, pos_tags = _extract_temporal_windows_from_audio(
        raw_pos_audio,
        raw_pos_source_ids,
        "pos",
        verbose=verbose,
        seq_len=seq_len,
    )

    if len(pos_embs) < 5:
        raise TrainingError(
            f"Only {len(pos_embs)} positive embeddings extracted. Need at least 5. "
            "Check audio files."
        )

    if verbose:
        print(f"\n  Processing {len(neg_files)} negative files...")

    # Extract negatives with tags if provided
    all_neg_embs: list[np.ndarray] = []
    all_neg_src: list[int] = []
    all_neg_tags: list[str] = []
    source_offset = 0

    if neg_tags:
        for ntag, nfiles in neg_tags.items():
            if not nfiles:
                continue
            embs, srcs, tags = _extract_temporal_embeddings(
                nfiles, ntag, verbose=verbose, seq_len=seq_len
            )
            # Offset source indices to avoid collisions across tag groups
            all_neg_embs.extend(embs)
            all_neg_src.extend([s + source_offset for s in srcs])
            all_neg_tags.extend(tags)
            source_offset += len(nfiles) + 1
    else:
        all_neg_embs, all_neg_src, all_neg_tags = _extract_temporal_embeddings(
            neg_files, "neg", verbose=verbose, seq_len=seq_len
        )

    corpus_tags = {
        "neg_librispeech",
        "neg_musan_speech",
        "neg_musan_music",
        "neg_musan_noise",
    }
    corpus_found = bool(
        neg_tags and any(tag in corpus_tags and files for tag, files in neg_tags.items())
    )

    if len(all_neg_embs) < 5:
        raise TrainingError(
            f"Only {len(all_neg_embs)} negative embeddings extracted. Need at least 5."
        )

    # -- Build dataset -------------------------------------------------------
    n_pos = len(pos_embs)
    n_neg = len(all_neg_embs)

    X_data = np.array(pos_embs + all_neg_embs, dtype=np.float32)  # (N, 9, 96)
    labels = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
    source_idx = np.array(pos_src + [s + max(pos_src) + 1 for s in all_neg_src], dtype=np.int32)
    tags = np.array(pos_tags + all_neg_tags)

    if verbose:
        print(f"\nDataset: {n_pos} pos + {n_neg} neg = {n_pos + n_neg} total")
        print(f"  Temporal shape: ({seq_len} frames, {EMBEDDING_DIM}-dim)")
        print(f"  corpus_found: {corpus_found}")

        # Show tag breakdown
        unique_tags = sorted(set(tags.tolist()))
        for t in unique_tags:
            count = int((tags == t).sum())
            print(f"    {t}: {count}")

    # -- Group-aware split ---------------------------------------------------
    train_idx, val_idx = _group_aware_split(
        labels,
        source_idx,
        seed=SEED,
        val_fraction=validation_fraction,
    )

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    train_pos_count = int(y_train.sum().item())
    val_pos_count = int(y_val.sum().item())

    if verbose:
        print(
            f"\nSplit: {len(train_idx)} train ({train_pos_count} pos / "
            f"{len(train_idx) - train_pos_count} neg) | "
            f"{len(val_idx)} val ({val_pos_count} pos / "
            f"{len(val_idx) - val_pos_count} neg)"
        )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -- Build TemporalCNN ---------------------------------------------------
    model = TemporalCNN(embedding_dim=EMBEDDING_DIM, seq_len=seq_len)
    model = model.to(torch_device)
    n_params = count_parameters(model)

    if verbose:
        print(f"\nModel: TemporalCNN ({n_params:,} params)")

    criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMATracker(model, decay=ema_decay)

    # -- Training loop -------------------------------------------------------
    if verbose:
        print(f"\nTraining TemporalCNN for up to {epochs} epochs (patience={patience})...")
        print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Best':>10} {'LR':>10}")
        print("-" * 50)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None
    best_ema_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(torch_device), by.to(torch_device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(torch_device), by.to(torch_device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ema_state = ema.state_dict()
        else:
            no_improve += 1

        # Progress reporting
        current_lr = scheduler.get_last_lr()[0]

        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "best_val_loss": best_val_loss,
                    "lr": current_lr,
                }
            )

        if verbose and (epoch % 10 == 0 or epoch == 1 or no_improve == 0):
            marker = " *" if epoch == best_epoch else ""
            print(
                f"{epoch:>6} {avg_train:>10.4f} {avg_val:>10.4f} "
                f"{best_val_loss:>10.4f} {current_lr:>10.6f}{marker}"
            )

        if no_improve >= patience:
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs). "
                    f"Best epoch: {best_epoch}"
                )
            break

    # -- Restore best weights and select averaging ---------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(torch_device)
    if best_ema_state is not None:
        ema.load_state_dict(best_ema_state)

    # Evaluate EMA
    ema.apply()
    model.eval()
    ema_val_loss = 0.0
    n_ema = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(torch_device), by.to(torch_device)
            pred = model(bx)
            loss = criterion(pred, by)
            ema_val_loss += loss.item()
            n_ema += 1
    ema_val_loss = ema_val_loss / max(n_ema, 1)
    ema.restore()

    method = auto_select_averaging(
        raw_val_loss=best_val_loss,
        ema_val_loss=ema_val_loss,
        swa_val_loss=None,
    )
    if method == "ema":
        ema.apply()

    training_duration = time.monotonic() - training_start

    if verbose:
        print(f"\nWeight averaging: {method} (raw={best_val_loss:.4f}, ema={ema_val_loss:.4f})")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Training duration: {training_duration:.1f}s")

    # -- Post-training quality gate ------------------------------------------
    from violawake_sdk._constants import DEFAULT_THRESHOLD, get_feature_config
    from violawake_sdk.oww_backbone import get_openwakeword_backbone_hashes

    deployment_threshold = float(DEFAULT_THRESHOLD)

    if verbose:
        print("\nStep 5: Post-training quality gate (speech/confusable/silence)...")

    quality_grade, quality_gate = _run_quality_gate(
        model,
        torch_device,
        seq_len,
        EMBEDDING_DIM,
        wake_word=wake_word,
        deployment_threshold=deployment_threshold,
        positive_files=pos_files,
        verbose=verbose,
    )

    # Optional test-mode bypass. Setting VIOLAWAKE_SKIP_QUALITY_GATE=1 exports
    # the model regardless of grade, with a loud warning. This exists for E2E
    # tests + dev iterations where verifying the full export+download chain
    # matters more than blocking a low-quality model. NEVER set this in a
    # customer-facing deploy — it would let unfit models ship.
    skip_gate = os.environ.get("VIOLAWAKE_SKIP_QUALITY_GATE", "").lower() in ("1", "true", "yes")

    if quality_grade == "F":
        print(
            "\n" + "!" * 72 + "\nQUALITY GATE FAILED: model is not ready for deployment.\n"
            f"  Speech FP rate:     {quality_gate['speech_fp_rate'] * 100:.1f}%\n"
            f"  Confusable FP rate: {quality_gate['confusable_fp_rate'] * 100:.1f}%\n"
            f"  Silence FP rate:    {_format_silence_fp(quality_gate)}\n"
            "Recommended fixes:\n"
            "  - Add more diverse speech negatives via --negatives or keep --auto-corpus enabled.\n"
            f"  - Expand confusable negatives for '{wake_word}' and retrain.\n"
            "  - Audit mislabeled positives/negatives and remove noisy clips.\n"
            "  - Raise the deployment threshold only after checking recall on eval data.\n"
            + "!"
            * 72
        )
        if skip_gate:
            print(
                "\n" + "*" * 72 + "\n"
                "WARNING: VIOLAWAKE_SKIP_QUALITY_GATE=1 — exporting failing model anyway.\n"
                "         This is for E2E testing only. NEVER set this in production.\n" + "*" * 72
            )

    model_exported = (quality_grade != "F") or skip_gate
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Export to ONNX ------------------------------------------------------
    if model_exported:
        if verbose:
            print(f"\nExporting model to ONNX: {output_path}")

        export_temporal_onnx(model, str(output_path), seq_len=seq_len, embedding_dim=EMBEDDING_DIM)
    elif verbose:
        print("\nSkipping ONNX export because the quality gate failed.")

    # -- Evaluate if test set provided ---------------------------------------
    d_prime_result: float | None = None
    if model_exported and eval_dir and eval_dir.exists():
        if verbose:
            print(f"\nEvaluating on test set: {eval_dir}")
        try:
            from violawake_sdk.training.evaluate import evaluate_onnx_model

            results = evaluate_onnx_model(output_path, eval_dir)
            d_prime_result = results["d_prime"]
            far = results["far_per_hour"]
            frr = results["frr"] * 100
            print(f"d': {d_prime_result:.2f}  FAR: {far:.2f}/hr  FRR: {frr:.1f}%")
        except Exception as e:
            print(f"Evaluation failed: {e}")
    elif quality_grade == "F" and verbose and eval_dir and eval_dir.exists():
        print("Skipping eval because no ONNX model was exported after the failed quality gate.")

    # -- Save config ---------------------------------------------------------
    config = get_feature_config()
    config.update(
        {
            "architecture": "temporal_cnn",
            "model_class": "TemporalCNN",
            "embedding_dim": EMBEDDING_DIM,
            "seq_len": seq_len,
            "n_params": n_params,
            "n_pos_samples": n_pos,
            "n_neg_samples": n_neg,
            "n_original_pos_clips": original_pos_clip_count,
            "n_augmented_pos_clips": n_augmented,
            "augmented": augment,
            "epochs_trained": min(epoch, epochs),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "ema_val_loss": float(ema_val_loss),
            "averaging_method": method,
            "ema_decay": ema_decay,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "validation_split": validation_fraction,
            "early_stopped": no_improve >= patience,
            "training_duration_s": round(training_duration, 2),
            "wake_word": wake_word,
            "deployment_threshold": deployment_threshold,
            "quality_grade": quality_grade,
            "quality_gate": quality_gate,
            "quality_gate_blocked_export": quality_grade == "F",
            "neg_corpus_breakdown": {tag: len(files) for tag, files in neg_tags.items()}
            if neg_tags
            else {},
            "corpus_found": corpus_found,
        }
    )
    config.update(get_openwakeword_backbone_hashes("onnx"))
    if d_prime_result is not None:
        config["d_prime"] = round(d_prime_result, 2)
    elif quality_gate.get("d_prime") is not None:
        # No external eval set was provided. Surface the d-prime computed by
        # the post-training quality gate so the Console + dashboard can render
        # a deployment grade without requiring an eval_dir.
        config["d_prime"] = round(float(quality_gate["d_prime"]), 2)

    config_path = output_path.with_suffix(".config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nConfig saved: {config_path}")
        if model_exported:
            print(f"Model saved: {output_path}")
            print(f"Load with:  WakeDetector(model='{output_path}')")

    if quality_grade == "F" and not skip_gate:
        # User-facing message: no internal paths (str(exc) is published to the
        # client via the Console's job_queue), accurate about WHY (post-#1465 a
        # grade-F means the model scored at/above the detection threshold on
        # no-wake audio -- a real false-fire risk -- not that the recording was
        # bad), and actionable (training has run-to-run variance, so a retrain
        # with the same recordings usually passes). The full per-axis metrics are
        # in the logged quality-gate block above for operators.
        raise ModelQualityGateError(
            "Your wake word didn't pass the quality check, so it wasn't saved. On "
            "no-wake audio (silence, everyday speech, or similar-sounding words) the "
            f"model scored at or above the {deployment_threshold:.2f} detection "
            "threshold, which means it would trigger on the wrong sound. Wake-word "
            "training varies run to run, so the quickest fix is to train again with "
            "the same recordings. If it keeps failing, add a few more clear "
            "recordings of your wake word, said a little differently each time."
        )

    return config


# ---------------------------------------------------------------------------
# Post-training quality gate
# ---------------------------------------------------------------------------

# Silence-subgrade safety-margin tiers, expressed as FRACTIONS of the deployment
# threshold so the whole silence ladder tracks the real false-fire line instead of
# a hardcoded constant. The load-bearing C->F cliff is the deployment threshold
# itself: the silence subgrade measures the model's worst score on no-wake (near-
# silence) audio, and a score at/above the deployment threshold is exactly a model
# that would fire on quiet input at deployment (a real false-fire) -- so it fails;
# a score below the threshold cannot fire at deployment and must not be forced to
# grade F on silence grounds (it is then graded on the speech/confusable axes,
# which already measure against the same deployment threshold via _fp_rate).
#
# Before #1465 the cliff was a hardcoded 0.50, disconnected from the 0.80
# deployment threshold (DEFAULT_THRESHOLD, _constants.py): it blocked ~75% of real
# models whose near-silence score sat in 0.53-0.79 -- below 0.80, unable to fire at
# deployment -- for run-to-run training variance, not real risk (root cause
# CL-20260714-4c23 / #1184; founder decision #1465, aligned at the deployment
# threshold given the independent inference-time RMS silence guard at
# wake_detector.py Gate 1). The A/B tiers preserve their historical margins
# (0.20 / 0.30 at threshold 0.80) but are now derived, so if the deployment
# threshold ever moves the entire ladder moves with it.
# The silence subgrade is a false-fire RATE, on the same tiers as the speech
# subgrade (#2611). It used to be a single max draw against the deployment
# threshold, which was invalid three ways at once -- see the measurement note in
# _run_quality_gate. A rate over many runtime windows is the same shape as the
# speech/confusable subgrades this function already grades, so all three
# subgrades now answer one consistent question: on no-wake audio, how often would
# this model fire at the deployment threshold?
_SILENCE_A_RATE = 0.02  # A: <2% of no-wake windows would fire
_SILENCE_B_RATE = 0.05  # B: <5%
_SILENCE_C_RATE = 0.10  # C: <10%; at/above => grade F


def _format_silence_fp(quality_gate: dict[str, Any]) -> str:
    """Render the silence subgrade for operator output ('n/a' when unmeasurable)."""
    rate = quality_gate.get("silence_fp_rate")
    if rate is None:
        return "n/a (no room tone available to measure)"
    return f"{rate * 100:.1f}% ({quality_gate.get('silence_window_count', 0)} room-tone windows)"


def _grade_quality(
    speech_fp_rate: float,
    confusable_fp_rate: float,
    silence_fp_rate: float | None,
    deployment_threshold: float,
) -> str:
    """Grade a model A/B/C/F from its no-wake false-fire measurements.

    All three subgrades are false-positive RATES measured at the SAME deployment
    threshold: how often the model would fire on everyday speech, on
    similar-sounding words, and on real no-wake room tone.

    ``silence_fp_rate`` is ``None`` when no real quiet audio could be measured for
    this model (e.g. the user's recordings yielded no room tone). In that case the
    model is graded on the speech and confusable axes alone rather than being
    failed for our missing measurement -- both remaining axes are measured against
    the same threshold, and the runtime keeps an independent RMS floor
    (wake_detector.py Gate 1) for genuinely quiet input.

    ``deployment_threshold`` is retained because every rate above is computed at
    that threshold by the caller; the tiers themselves are rate bars, so no score
    bar is hardcoded here.
    """
    silence = 0.0 if silence_fp_rate is None else silence_fp_rate
    if speech_fp_rate < 0.02 and confusable_fp_rate < 0.05 and silence < _SILENCE_A_RATE:
        return "A"
    if speech_fp_rate < 0.05 and confusable_fp_rate < 0.10 and silence < _SILENCE_B_RATE:
        return "B"
    if speech_fp_rate < 0.10 and confusable_fp_rate < 0.20 and silence < _SILENCE_C_RATE:
        return "C"
    return "F"


def _run_quality_gate(
    model: Any,
    torch_device: str,
    seq_len: int,
    embedding_dim: int,
    wake_word: str,
    deployment_threshold: float = 0.80,
    positive_files: list[Path] | None = None,
    verbose: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Run a post-training quality gate on speech, confusables, and silence.

    Returns:
        Tuple of ``(grade, metrics)`` where grade is one of ``A/B/C/F``.
    """
    import tempfile

    import numpy as np
    import torch

    from violawake_sdk.tools.confusables import generate_confusables

    del embedding_dim  # Signature kept for compatibility with existing caller.

    def _score_files(audio_files: list[Path], tag: str) -> np.ndarray:
        if not audio_files:
            return np.array([], dtype=np.float32)

        embs, source_indices, _ = _extract_temporal_embeddings(
            audio_files, tag, verbose=False, seq_len=seq_len
        )
        if not embs:
            return np.array([], dtype=np.float32)

        X_qc = torch.tensor(np.array(embs), dtype=torch.float32).to(torch_device)
        with torch.no_grad():
            window_scores = model(X_qc).cpu().numpy().flatten()

        clip_scores: dict[int, float] = {}
        for idx, source_idx in enumerate(source_indices):
            score = float(window_scores[idx])
            clip_scores[source_idx] = max(score, clip_scores.get(source_idx, float("-inf")))

        return np.array(
            [clip_scores[i] for i in sorted(clip_scores)],
            dtype=np.float32,
        )

    def _score_windows_streaming(audio_clips: list[np.ndarray]) -> np.ndarray:
        """Score raw audio clips via the runtime streaming path (#1487 / #2611).

        Returns EVERY window's score (not one max per clip): the silence subgrade
        is a false-fire rate, so it needs the full distribution the runtime would
        see, not an extreme value over an unknown number of samples.
        """
        if not audio_clips:
            return np.array([], dtype=np.float32)

        windows, _source_indices = _extract_streaming_temporal_windows(audio_clips, seq_len)
        if not windows:
            return np.array([], dtype=np.float32)

        X_qc = torch.tensor(np.array(windows), dtype=torch.float32).to(torch_device)
        with torch.no_grad():
            return model(X_qc).cpu().numpy().flatten()

    def _fp_rate(scores: np.ndarray) -> float:
        if len(scores) == 0:
            return 1.0
        return float((scores >= deployment_threshold).mean())

    def _grade_label(grade: str) -> str:
        return {
            "A": "EXCELLENT",
            "B": "GOOD",
            "C": "CAUTION",
            "F": "FAIL",
        }[grade]

    model.eval()
    model = model.to(torch_device)

    quality_phrases = SPEECH_NEGATIVE_PHRASES[:50]
    voice = EDGE_TTS_VOICES[0]  # Single voice keeps the gate fast and deterministic.

    with tempfile.TemporaryDirectory(prefix="violawake_qc_", dir=_TMP_DIR) as tmp_dir:
        quality_dir = Path(tmp_dir)

        speech_files: list[Path] = []
        if verbose:
            print(f"  Generating {len(quality_phrases)} speech phrases for quality check...")
        for i, phrase in enumerate(quality_phrases):
            out_path = quality_dir / f"qc_speech_{i:03d}.wav"
            ok = _edge_tts_synthesize(phrase, voice, out_path)
            if ok and out_path.exists():
                speech_files.append(out_path)

        raw_confusables = generate_confusables(wake_word, count=40)
        confusable_words: list[str] = []
        seen_confusables: set[str] = set()
        normalized_wake_word = " ".join(wake_word.lower().split())
        for word in raw_confusables:
            normalized_word = " ".join(word.lower().split())
            if not normalized_word or normalized_word == normalized_wake_word:
                continue
            if normalized_word in seen_confusables:
                continue
            seen_confusables.add(normalized_word)
            confusable_words.append(word)
            if len(confusable_words) == 20:
                break

        confusable_files: list[Path] = []
        if verbose:
            print(f"  Generating {len(confusable_words)} confusable words for quality check...")
        for i, word in enumerate(confusable_words):
            safe_word = word.replace(" ", "_")[:30]
            out_path = quality_dir / f"qc_confusable_{i:03d}_{safe_word}.wav"
            ok = _edge_tts_synthesize(word, voice, out_path)
            if ok and out_path.exists():
                confusable_files.append(out_path)

        # ------------------------------------------------------------------
        # Silence subgrade probes (#2611).
        #
        # The old probe was ONE fixed-seed (42) white-noise clip at float RMS
        # 1e-4, center-cropped to 1.5s and scored in batch mode -- reproduced on
        # the box 2026-07-24, it yielded silence_window_count == 1, i.e. the
        # "max over windows" was a single forward pass on a single arbitrary
        # out-of-distribution input. Its run-to-run spread across probe draws was
        # +-0.2 straddling the 0.80 cliff, so on six real deployed models that had
        # PASSED the gate, 10-37% of alternative probe seeds would have failed
        # them. It was also invalid in the other direction: models with an 82-90%
        # runtime false-fire rate on broadband noise passed it. Every production
        # training failure in the post-recalibration window (12 of 21 jobs) was
        # caused by this one number, with speech FP and confusable FP both 0.0%.
        #
        # Replaced by REAL no-wake audio -- the room tone of the user's own
        # microphone, taken from their own recordings -- scored through the real
        # runtime streaming path over the FULL clip, and graded as a rate.
        room_tone_clips: list[np.ndarray] = []
        for pos_path in list(positive_files or []):
            try:
                room_tone = _extract_room_tone(_load_training_audio(pos_path))
            except Exception:
                continue
            if room_tone is not None:
                room_tone_clips.append(room_tone)

        speech_scores = _score_files(speech_files, "qc_speech")
        confusable_scores = _score_files(confusable_files, "qc_confusable")
        silence_window_scores = _score_windows_streaming(room_tone_clips)
        positive_scores = (
            _score_files(list(positive_files), "qc_positive")
            if positive_files
            else np.array([], dtype=np.float32)
        )

    speech_fp_rate = _fp_rate(speech_scores)
    confusable_fp_rate = _fp_rate(confusable_scores)
    # Silence subgrade: the false-fire rate on the user's own room tone, scored
    # through the runtime streaming path. When no room tone could be extracted
    # (e.g. every recording is wall-to-wall speech, or recordings were purged),
    # the rate is None and _grade_quality grades on speech/confusable alone --
    # the old code forced grade F here, which failed the user for OUR missing
    # measurement. Genuinely quiet input is independently handled at runtime by
    # the RMS floor (wake_detector.py Gate 1).
    if len(silence_window_scores) > 0:
        silence_fp_rate: float | None = float(
            (silence_window_scores >= deployment_threshold).mean()
        )
        silence_max_score = float(silence_window_scores.max())
        silence_source = "room_tone"
        silence_window_count = int(len(silence_window_scores))
    else:
        silence_fp_rate = None
        silence_max_score = 0.0
        silence_source = "unmeasurable"
        silence_window_count = 0
    grade = _grade_quality(
        speech_fp_rate, confusable_fp_rate, silence_fp_rate, deployment_threshold
    )

    # Pool every non-positive score we collected into the negative distribution
    # used for d-prime.
    neg_pool_parts = [speech_scores, confusable_scores]
    if len(silence_window_scores) > 0:
        neg_pool_parts.append(silence_window_scores)
    negative_scores_pool = (
        np.concatenate(neg_pool_parts)
        if any(len(p) > 0 for p in neg_pool_parts)
        else np.array([], dtype=np.float32)
    )

    d_prime: float | None = None
    if len(positive_scores) >= 2 and len(negative_scores_pool) >= 2:
        pos_mean = float(positive_scores.mean())
        neg_mean = float(negative_scores_pool.mean())
        # Pooled std with ddof=1 (sample variance). Floor variance at 1e-6 so a
        # degenerate model (all-same-score) doesn't divide by zero.
        pooled_var = max(
            (float(positive_scores.var(ddof=1)) + float(negative_scores_pool.var(ddof=1))) / 2.0,
            1e-6,
        )
        d_prime = (pos_mean - neg_mean) / (pooled_var**0.5)

    metrics: dict[str, Any] = {
        "grade": grade,
        "deployment_threshold": float(deployment_threshold),
        "speech_fp_rate": speech_fp_rate,
        "speech_sample_count": int(len(speech_scores)),
        "confusable_fp_rate": confusable_fp_rate,
        "confusable_sample_count": int(len(confusable_scores)),
        "silence_fp_rate": silence_fp_rate,
        "silence_max_score": silence_max_score,
        "silence_window_count": silence_window_count,
        "silence_source": silence_source,
        # Which code path produced the silence score. "runtime_streaming" is the
        # real WakeDetector path; the retired "batch_crop" path scored a 1.5s
        # center-crop through embed_clips and did not predict it (#1487).
        "silence_scoring_path": ("runtime_streaming" if silence_window_count else "n/a"),
        "d_prime": (round(float(d_prime), 4) if d_prime is not None else None),
        "positive_scores": [round(float(s), 6) for s in positive_scores.tolist()],
        "negative_scores": [round(float(s), 6) for s in negative_scores_pool.tolist()],
        "positive_sample_count": int(len(positive_scores)),
        "negative_sample_count": int(len(negative_scores_pool)),
    }

    print(f"Model Quality Grade: {grade} ({_grade_label(grade)})")
    print(
        f"  Speech FP rate:     {speech_fp_rate * 100:4.1f}% "
        f"({len(speech_scores)} phrases, threshold={deployment_threshold:.2f})"
    )
    print(
        f"  Confusable FP rate: {confusable_fp_rate * 100:4.1f}% "
        f"({len(confusable_scores)} words, threshold={deployment_threshold:.2f})"
    )
    if silence_fp_rate is None:
        print("  Silence FP rate:    n/a (no room tone in the recordings to measure)")
    else:
        print(
            f"  Silence FP rate:    {silence_fp_rate * 100:4.1f}% "
            f"({silence_window_count} room-tone windows, max={silence_max_score:.2f}, "
            f"threshold={deployment_threshold:.2f})"
        )
    if d_prime is not None:
        print(
            f"  d-prime:            {d_prime:.2f} "
            f"(pos={len(positive_scores)}, neg={len(negative_scores_pool)})"
        )

    if verbose and len(speech_scores) < len(quality_phrases):
        print(
            f"  WARNING: Only {len(speech_scores)}/{len(quality_phrases)} speech phrases "
            "were scored in the quality gate."
        )
    if verbose and len(confusable_scores) < 20:
        print(
            f"  WARNING: Only {len(confusable_scores)}/20 confusable words "
            "were scored in the quality gate."
        )
    if verbose and silence_window_count == 0:
        print(
            "  NOTE: No room tone could be extracted from the recordings, so the "
            "silence subgrade was not measured; graded on speech/confusable only."
        )

    return grade, metrics


# ---------------------------------------------------------------------------
# Legacy MLP training (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _train_mlp_on_oww(
    positives_dir: Path,
    output_path: Path,
    epochs: int = 50,
    augment: bool = True,
    eval_dir: Path | None = None,
    negatives_dir: Path | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    neg_ratio: int = 5,
    patience: int = 10,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    device: str | None = None,
    ema_decay: float = 0.999,
    swa_epochs: int = 10,
    swa_lr: float | None = None,
    save_raw_model: bool = False,
) -> None:
    """Removed legacy MLP training entry point."""
    raise RuntimeError(
        "Legacy MLP training has been removed. Use the production TemporalCNN pipeline via violawake_sdk.tools.train."
    )

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        print(f"ERROR: PyTorch required for training: {e}", file=sys.stderr)
        print("Install with: pip install 'violawake[training]'", file=sys.stderr)
        sys.exit(1)

    from violawake_sdk._constants import CLIP_SAMPLES, get_feature_config
    from violawake_sdk.audio import center_crop, load_audio
    from violawake_sdk.oww_backbone import get_openwakeword_backbone_hashes
    from violawake_sdk.training.augment import AugmentationPipeline
    from violawake_sdk.training.losses import FocalLoss
    from violawake_sdk.training.weight_averaging import (
        EMATracker,
        SWACollector,
        auto_select_averaging,
    )

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        print(f"ERROR: openwakeword required: {e}", file=sys.stderr)
        sys.exit(1)

    # -- Collect files -------------------------------------------------------
    pos_files = sorted(list(positives_dir.rglob("*.wav")) + list(positives_dir.rglob("*.flac")))
    if len(pos_files) < 5:
        print(f"ERROR: Found only {len(pos_files)} positive samples.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Found {len(pos_files)} positive samples")

    # -- Embedding extraction ------------------------------------------------
    if verbose:
        print("Loading OpenWakeWord backbone...")

    oww = OWWModel(inference_framework="onnx")
    preprocessor = oww.preprocessor

    def _audio_to_embedding(audio_f32):
        audio = center_crop(audio_f32, CLIP_SAMPLES)
        audio_i16 = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767).astype(np.int16)
        if len(audio_i16) < CLIP_SAMPLES:
            audio_i16 = np.pad(audio_i16, (0, CLIP_SAMPLES - len(audio_i16)))
        else:
            audio_i16 = audio_i16[:CLIP_SAMPLES]
        try:
            embeddings = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
            return embeddings.mean(axis=1)[0].astype(np.float32)
        except Exception:
            return None

    # Extract positives
    pos_embeddings = []
    pos_source_file_idx = []

    if augment:
        pipeline = AugmentationPipeline(seed=42)
        augment_factor = 10
        for file_idx, f in enumerate(pos_files):
            audio = load_audio(f)
            if audio is None:
                continue
            emb = _audio_to_embedding(audio)
            if emb is not None:
                pos_embeddings.append(emb)
                pos_source_file_idx.append(file_idx)
            for variant in pipeline.augment_clip(audio, factor=augment_factor):
                emb = _audio_to_embedding(variant)
                if emb is not None:
                    pos_embeddings.append(emb)
                    pos_source_file_idx.append(file_idx)
    else:
        for file_idx, f in enumerate(pos_files):
            audio = load_audio(f)
            if audio is None:
                continue
            emb = _audio_to_embedding(audio)
            if emb is not None:
                pos_embeddings.append(emb)
                pos_source_file_idx.append(file_idx)

    if len(pos_embeddings) < 5:
        print("ERROR: Too few positive embeddings.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  {len(pos_embeddings)} positive embeddings")

    # Extract negatives
    n_negatives = len(pos_embeddings) * neg_ratio
    neg_embeddings = []
    neg_source_file_idx = []

    if negatives_dir and negatives_dir.exists():
        neg_files = sorted(list(negatives_dir.rglob("*.wav")) + list(negatives_dir.rglob("*.flac")))
        for file_idx, f in enumerate(neg_files):
            audio = load_audio(f)
            if audio is None:
                continue
            emb = _audio_to_embedding(audio)
            if emb is not None:
                neg_embeddings.append(emb)
                neg_source_file_idx.append(file_idx)

    if len(neg_embeddings) < 5:
        # Synthetic fallback
        if verbose:
            print(f"  Generating {n_negatives} synthetic negatives (legacy MLP mode)...")
        rng_synth = np.random.default_rng(42)
        for i in range(n_negatives):
            clip = rng_synth.standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.1
            emb = _audio_to_embedding(clip)
            if emb is not None:
                neg_embeddings.append(emb)
                neg_source_file_idx.append(i)

    if verbose:
        print(f"  {len(neg_embeddings)} negative embeddings")

    # -- Build dataset and train ---------------------------------------------
    X = torch.tensor(np.array(pos_embeddings + neg_embeddings), dtype=torch.float32)
    y = torch.tensor(
        [1.0] * len(pos_embeddings) + [0.0] * len(neg_embeddings), dtype=torch.float32
    ).unsqueeze(1)
    embedding_dim = X.shape[1]

    labels_np = np.array([1] * len(pos_embeddings) + [0] * len(neg_embeddings))
    source_np = np.array(
        pos_source_file_idx + [s + max(pos_source_file_idx) + 1 for s in neg_source_file_idx]
    )
    train_idx, val_idx = _group_aware_split(labels_np, source_np)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, 1),
        nn.Sigmoid(),
    )

    criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMATracker(model, decay=ema_decay)

    swa = SWACollector(n_epochs=swa_epochs) if swa_epochs > 0 else None
    swa_start_epoch = max(1, epochs - swa_epochs + 1) if swa_epochs > 0 else epochs + 1

    if verbose:
        print(f"\nTraining MLP for up to {epochs} epochs (patience={patience})...")
        print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Best':>10} {'LR':>10}")
        print("-" * 50)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None
    best_ema_state_mlp = None
    training_start = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()
            train_loss += float(loss.item())
            n_train_batches += 1
        scheduler.step()
        avg_train_loss = train_loss / max(n_train_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += float(loss.item())
                n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)

        if swa is not None and epoch >= swa_start_epoch:
            swa.collect(model, val_loss=avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema_state_mlp = ema.state_dict()
        else:
            epochs_without_improvement += 1

        current_lr = scheduler.get_last_lr()[0]
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    "lr": current_lr,
                }
            )

        if verbose and (epoch % 10 == 0 or epoch == 1 or epochs_without_improvement == 0):
            marker = " *" if epoch == best_epoch else ""
            print(
                f"{epoch:>6} {avg_train_loss:>10.4f} {avg_val_loss:>10.4f} "
                f"{best_val_loss:>10.4f} {current_lr:>10.6f}{marker}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Restore and average
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_ema_state_mlp is not None:
        ema.load_state_dict(best_ema_state_mlp)

    ema.apply()
    model.eval()
    ema_val_loss = 0.0
    n_ema = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            ema_val_loss += float(loss.item())
            n_ema += 1
    ema_val_loss = ema_val_loss / max(n_ema, 1)
    ema.restore()

    swa_val_loss = None
    if swa is not None and swa.n_collected > 0:
        swa_backup = {k: v.clone() for k, v in model.state_dict().items()}
        swa.apply(model)
        model.eval()
        swa_total = 0.0
        n_swa = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                swa_total += float(loss.item())
                n_swa += 1
        swa_val_loss = swa_total / max(n_swa, 1)
        model.load_state_dict(swa_backup)

    averaging_method = auto_select_averaging(best_val_loss, ema_val_loss, swa_val_loss)
    if averaging_method == "ema":
        ema.apply()
    elif averaging_method == "swa" and swa is not None:
        swa.apply(model)

    training_duration = time.monotonic() - training_start

    if verbose:
        print(f"\nAveraging: {averaging_method}")
        print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Duration: {training_duration:.1f}s")

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.zeros(1, embedding_dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["embedding"],
        output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )

    # Config
    config = get_feature_config()
    config.update(
        {
            "architecture": "mlp_on_oww",
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "n_pos_samples": len(pos_embeddings),
            "n_neg_samples": len(neg_embeddings),
            "augmented": augment,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "training_duration_s": round(training_duration, 2),
            "averaging_method": averaging_method,
        }
    )
    config.update(get_openwakeword_backbone_hashes("onnx"))
    config_path = output_path.with_suffix(".config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nModel saved: {output_path}")


# ---------------------------------------------------------------------------
# Checkpoint averaging (utility, kept from original)
# ---------------------------------------------------------------------------


def average_checkpoints(checkpoint_paths: list[str], output_path: str) -> None:
    """Average weights of multiple ONNX model checkpoints (SWA).

    Args:
        checkpoint_paths: List of paths to ONNX model files to average.
        output_path: Path to save the averaged model.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints to average")

    models = [onnx.load(p) for p in checkpoint_paths]
    base = models[0]

    for tensor in base.graph.initializer:
        weights = []
        for m in models:
            matching = [t for t in m.graph.initializer if t.name == tensor.name]
            if matching:
                weights.append(numpy_helper.to_array(matching[0]))
        if len(weights) == len(models):
            avg = np.mean(weights, axis=0)
            tensor.CopyFrom(numpy_helper.from_array(avg, tensor.name))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(base, output_path)


def _copy_eval_files(files: list[Path], target_dir: Path) -> None:
    """Copy held-out files into a flat eval directory without name collisions."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(files):
        dst = target_dir / f"{idx:05d}_{src.name}"
        shutil.copy2(src, dst)


def _held_out_count(n_files: int) -> int:
    """Reserve 20% for test while keeping at least one training file."""
    if n_files <= 1:
        return 0
    return min(n_files - 1, max(5, n_files // 5))


def _auto_eval_verdict(eer_percent: float) -> str:
    if eer_percent < 10.0:
        return "GOOD (EER < 10%)"
    if eer_percent <= 15.0:
        return "ACCEPTABLE (EER <= 15%)"
    if eer_percent <= 25.0:
        return "WARNING (EER > 15%)"
    return "CRITICAL (EER > 25%)"


def _update_auto_eval_config(config_path: Path, auto_eval: dict[str, Any]) -> None:
    """Merge auto-eval results into the saved model config."""
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                config = loaded
    config["auto_eval"] = auto_eval
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-train",
        description=(
            "Train a custom wake word model.\n\n"
            "Default: TemporalCNN on 9-frame OWW embedding windows (production architecture).\n"
            "Auto-generates TTS positives, confusable negatives, and speech negatives."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--word",
        required=True,
        metavar="WORD",
        help="The wake word (e.g. 'jarvis', 'viola', 'hey computer')",
    )
    parser.add_argument(
        "--positives",
        metavar="DIR",
        default=None,
        help="Directory containing positive WAV/FLAC samples of the wake word. "
        "If fewer than 100 samples, auto-generated TTS positives fill the gap.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for the trained ONNX model (e.g., models/jarvis.onnx)",
    )
    parser.add_argument(
        "--negatives",
        metavar="DIR",
        default=None,
        help="Optional directory of negative WAV/FLAC files (speech, music, etc.). "
        "Added on top of auto-generated negatives.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        metavar="N",
        help="Maximum training epochs (default: 80)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="RATE",
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        metavar="N",
        help="Early stopping patience (default: 15)",
    )
    parser.set_defaults(augment=True)
    parser.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Enable audio-level data augmentation (default: True)",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable audio-level augmentation (TTS generation still runs)",
    )
    parser.add_argument(
        "--auto-corpus",
        action="store_true",
        default=True,
        dest="auto_corpus",
        help="Auto-generate TTS positives, confusables, and speech negatives (default: True)",
    )
    parser.add_argument(
        "--no-auto-corpus",
        action="store_false",
        dest="auto_corpus",
        help="Disable auto-generation of TTS corpus. Only use --positives and --negatives.",
    )
    parser.add_argument(
        "--eval-dir",
        metavar="DIR",
        help="Optional test set directory for evaluation after training. "
        "Must contain positives/ and negatives/ subdirectories.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress output",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    eval_dir = Path(args.eval_dir) if args.eval_dir else None
    positives_dir = Path(args.positives) if args.positives else None
    negatives_dir = Path(args.negatives) if args.negatives else None
    verbose = not args.quiet

    if positives_dir and not positives_dir.exists():
        print(f"ERROR: Positives directory not found: {positives_dir}", file=sys.stderr)
        sys.exit(1)

    if negatives_dir and not negatives_dir.exists():
        print(f"ERROR: Negatives directory not found: {negatives_dir}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("=" * 70)
        print(f"ViolaWake Training: '{args.word}'")
        print("=" * 70)
        print("Architecture:       temporal_cnn")
        print(f"Auto corpus:        {'enabled' if args.auto_corpus else 'disabled'}")
        if positives_dir:
            print(f"Positives dir:      {positives_dir}")
        if negatives_dir:
            print(f"Negatives dir:      {negatives_dir}")
        print(f"Output:             {output_path}")
        print(f"Epochs:             {args.epochs} (patience={args.patience})")
        print(f"Batch size:         {args.batch_size}")
        print(f"Learning rate:      {args.lr}")
        print(f"Augmentation:       {'enabled' if args.augment else 'disabled'}")
        if eval_dir:
            print(f"Eval set:           {eval_dir}")
        print()

    # ======================================================================
    # TemporalCNN path (production architecture)
    # ======================================================================

    # -- Step 1: Collect and auto-generate corpus ----------------------------
    corpus_dir = output_path.parent / "_training_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Collect user-provided positive files
    user_pos_files: list[Path] = []
    if positives_dir and positives_dir.exists():
        user_pos_files = sorted(
            list(positives_dir.rglob("*.wav")) + list(positives_dir.rglob("*.flac"))
        )
        if verbose:
            print(f"Found {len(user_pos_files)} user-provided positive samples")

    # Auto-generate TTS positives if needed
    tts_pos_files: list[Path] = []
    if args.auto_corpus and len(user_pos_files) < 100:
        if verbose:
            print(
                f"\nStep 1a: Auto-generating TTS positives (have {len(user_pos_files)}, need ~100+)..."
            )
        tts_pos_dir = corpus_dir / "tts_positives"
        tts_pos_files = _generate_tts_positives(args.word, tts_pos_dir, verbose=verbose)

    all_pos_files = user_pos_files + tts_pos_files
    if len(all_pos_files) < 5:
        print(
            f"ERROR: Only {len(all_pos_files)} positive files total. "
            "Provide at least 5 via --positives or enable --auto-corpus.",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(
            f"\nTotal positive files: {len(all_pos_files)} "
            f"({len(user_pos_files)} user + {len(tts_pos_files)} TTS)"
        )

    # Collect negative files from multiple sources
    neg_tag_map: dict[str, list[Path]] = {}

    # Source 1: User-provided negatives
    user_neg_files: list[Path] = []
    if negatives_dir and negatives_dir.exists():
        user_neg_files = sorted(
            list(negatives_dir.rglob("*.wav")) + list(negatives_dir.rglob("*.flac"))
        )
        if user_neg_files:
            neg_tag_map["neg_user"] = user_neg_files
            if verbose:
                print(f"Found {len(user_neg_files)} user-provided negative samples")

    # Source 2: Auto-generated confusable negatives (2 rounds, matching production)
    # Round 1: 30 confusables x 10 voices (broad phonetic coverage)
    # Round 2: 16 confusables x 10 voices (tighter variants for hard negatives)
    confusable_files: list[Path] = []
    if args.auto_corpus:
        if verbose:
            print("\nStep 1b: Auto-generating confusable negatives (round 1: broad)...")
        confusable_dir_r1 = corpus_dir / "confusables_r1"
        confusable_r1 = _generate_confusable_negatives(
            args.word,
            confusable_dir_r1,
            n_confusables=30,
            voices_per_word=10,
            verbose=verbose,
        )
        if confusable_r1:
            neg_tag_map["neg_confusable_r1"] = confusable_r1
            confusable_files.extend(confusable_r1)

        if verbose:
            print("\nStep 1b2: Auto-generating confusable negatives (round 2: tight variants)...")
        confusable_dir_r2 = corpus_dir / "confusables_r2"
        confusable_r2 = _generate_confusable_negatives(
            args.word,
            confusable_dir_r2,
            n_confusables=16,
            voices_per_word=10,
            verbose=verbose,
        )
        if confusable_r2:
            neg_tag_map["neg_confusable_r2"] = confusable_r2
            confusable_files.extend(confusable_r2)

    # Source 3: Auto-generated speech negatives
    speech_neg_files: list[Path] = []
    if args.auto_corpus:
        if verbose:
            print("\nStep 1c: Auto-generating speech negatives...")
        speech_neg_dir = corpus_dir / "speech_negatives"
        speech_neg_files = _generate_speech_negatives(
            speech_neg_dir,
            n_voices=5,
            verbose=verbose,
        )
        if speech_neg_files:
            neg_tag_map["neg_speech"] = speech_neg_files

    # Source 4: Shared universal negative corpus (LibriSpeech, MUSAN, etc.)
    # These are word-agnostic negatives that every wake word model needs.
    # Without them, models only learn to distinguish the wake word from a
    # tiny auto-generated set and false-trigger on any real-world speech.
    _CORPUS_SEARCH_PATHS = [
        Path(__file__).resolve().parent.parent.parent.parent / "corpus",  # repo root
        Path.home() / ".violawake" / "corpus",
        Path("corpus"),
    ]
    _CORPUS_SUBDIRS = {
        "neg_librispeech": "librispeech",
        "neg_musan_speech": ("musan/musan/speech", "musan/speech"),
        "neg_musan_music": ("musan/musan/music", "musan/music"),
        "neg_musan_noise": ("musan/musan/noise", "musan/noise"),
    }
    for tag, subdirs in _CORPUS_SUBDIRS.items():
        if isinstance(subdirs, str):
            subdirs = (subdirs,)
        for corpus_root in _CORPUS_SEARCH_PATHS:
            if not corpus_root.exists():
                continue
            for subdir in subdirs:
                candidate = corpus_root / subdir
                if candidate.exists():
                    corpus_files = sorted(
                        list(candidate.rglob("*.wav")) + list(candidate.rglob("*.flac"))
                    )
                    if corpus_files:
                        # Cap each source to avoid swamping the dataset
                        max_per_source = 2000
                        if len(corpus_files) > max_per_source:
                            import random

                            rng = random.Random(42)
                            corpus_files = sorted(rng.sample(corpus_files, max_per_source))
                        neg_tag_map[tag] = corpus_files
                        if verbose:
                            print(
                                f"  Shared corpus [{tag}]: {len(corpus_files)} files from {candidate}"
                            )
                        break  # found this tag, move to next
            if tag in neg_tag_map:
                break  # found in this root, move to next tag

    corpus_paths = {
        "neg_librispeech": "~/.violawake/corpus/librispeech/   (speech recordings)",
        "neg_musan_speech": "~/.violawake/corpus/musan/speech/  (MUSAN speech subset)",
        "neg_musan_music": "~/.violawake/corpus/musan/music/   (MUSAN music subset)",
        "neg_musan_noise": "~/.violawake/corpus/musan/noise/   (MUSAN noise subset)",
    }
    found_corpus_tags = [tag for tag in _CORPUS_SUBDIRS if neg_tag_map.get(tag)]
    missing_corpus_tags = [tag for tag in _CORPUS_SUBDIRS if tag not in found_corpus_tags]
    if not found_corpus_tags:
        print(
            "\nWARNING: No universal negative corpus found.\n"
            "Training with TTS-only negatives may produce a model with high\n"
            "false positive rates on real speech and music.\n"
            "\n"
            "Place audio files in one of these locations:\n"
            "  ~/.violawake/corpus/librispeech/   (speech recordings)\n"
            "  ~/.violawake/corpus/musan/speech/  (MUSAN speech subset)\n"
            "  ~/.violawake/corpus/musan/music/   (MUSAN music subset)\n"
            "  ~/.violawake/corpus/musan/noise/   (MUSAN noise subset)\n"
            "\n"
            "Or provide negatives via: --negatives <dir>\n"
        )
    elif missing_corpus_tags:
        print("\nNOTE: Universal negative corpus is incomplete.")
        print(f"Found {len(found_corpus_tags)}/{len(_CORPUS_SUBDIRS)} corpus sources; missing:")
        for tag in missing_corpus_tags:
            print(f"  {tag}: {corpus_paths[tag]}")
        print("Add files to the paths above or provide negatives via --negatives <dir>.")

    total_neg = sum(len(v) for v in neg_tag_map.values())
    if total_neg < 5:
        print(
            f"ERROR: Only {total_neg} negative files total. "
            "Enable --auto-corpus or provide negatives via --negatives.",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"\nTotal negative files: {total_neg}")
        for tag, files in neg_tag_map.items():
            print(f"  {tag}: {len(files)}")

    # Flatten for the training function
    all_neg_files: list[Path] = []
    for files in neg_tag_map.values():
        all_neg_files.extend(files)

    train_pos_files = all_pos_files
    train_neg_files = all_neg_files
    train_neg_tag_map = {tag: list(files) for tag, files in neg_tag_map.items()}
    eval_target_dir = eval_dir
    auto_eval_label = "user-provided eval set" if eval_dir else "held-out 20% test set"

    if eval_dir is None:
        if verbose:
            print("\nStep 1d: Creating held-out 20% test set...")

        rng = Random(42)
        pos_test_count = _held_out_count(len(all_pos_files))
        neg_test_count = _held_out_count(len(all_neg_files))
        test_pos = rng.sample(all_pos_files, pos_test_count)
        test_neg = rng.sample(all_neg_files, neg_test_count)

        test_pos_set = set(test_pos)
        test_neg_set = set(test_neg)
        train_pos_files = [f for f in all_pos_files if f not in test_pos_set]
        train_neg_files = [f for f in all_neg_files if f not in test_neg_set]
        train_neg_tag_map = {
            tag: [f for f in files if f not in test_neg_set] for tag, files in neg_tag_map.items()
        }

        if not train_pos_files or not train_neg_files:
            print(
                "ERROR: Held-out split left no training data. "
                "Provide more samples or use --eval-dir.",
                file=sys.stderr,
            )
            sys.exit(1)

        eval_target_dir = corpus_dir / "auto_test"
        shutil.rmtree(eval_target_dir, ignore_errors=True)
        _copy_eval_files(test_pos, eval_target_dir / "positives")
        _copy_eval_files(test_neg, eval_target_dir / "negatives")

        if verbose:
            print(f"  Train positives:    {len(train_pos_files)}")
            print(f"  Test positives:     {len(test_pos)}")
            print(f"  Train negatives:    {len(train_neg_files)}")
            print(f"  Test negatives:     {len(test_neg)}")
            print(f"  Auto-test dir:      {eval_target_dir}")

    # -- Step 2-5: Train TemporalCNN ----------------------------------------
    try:
        _train_temporal_cnn(
            pos_files=train_pos_files,
            neg_files=train_neg_files,
            output_path=output_path,
            wake_word=args.word,
            epochs=args.epochs,
            augment=args.augment,
            eval_dir=None,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            verbose=verbose,
            neg_tags=train_neg_tag_map,
            augment_source_files=user_pos_files or train_pos_files,
        )
    except TrainingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    auto_eval_payload: dict[str, Any] = {
        "source": "auto_holdout" if eval_dir is None else "user_eval_dir",
        "test_dir": str(eval_target_dir) if eval_target_dir else None,
        "status": "skipped",
    }
    config_path = output_path.with_suffix(".config.json")

    if eval_target_dir is not None:
        try:
            from violawake_sdk.tools.evaluate import evaluate_onnx_model

            results = evaluate_onnx_model(output_path, eval_target_dir)
            eer = results["eer_approx"] * 100
            roc_auc = results["roc_auc"]
            far = results["optimal_far"] * 100
            frr = results["optimal_frr"] * 100
            verdict = _auto_eval_verdict(eer)

            print(f"\n=== Auto-Evaluation ({auto_eval_label}) ===")
            print(f"EER:      {eer:.1f}%")
            print(f"ROC AUC:  {roc_auc:.3f}")
            print(f"FAR:      {far:.1f}%")
            print(f"FRR:      {frr:.1f}%")
            print(f"Verdict:  {verdict}")

            if eer > 25.0:
                print(
                    "CRITICAL: Held-out EER exceeds 25%. "
                    "Add more real positives, harder speech/background negatives, and retrain before deployment."
                )
            elif eer > 15.0:
                print(
                    "WARNING: Held-out EER exceeds 15%. "
                    "Add more speaker/environment diversity and harder negatives, then retrain."
                )

            auto_eval_payload.update(
                {
                    "status": "ok",
                    "architecture": results["architecture"],
                    "n_positives": results["n_positives"],
                    "n_negatives": results["n_negatives"],
                    "roc_auc": round(roc_auc, 4),
                    "eer_percent": round(eer, 2),
                    "far_percent": round(far, 2),
                    "frr_percent": round(frr, 2),
                    "optimal_threshold": round(results["optimal_threshold"], 4),
                    "verdict": verdict,
                }
            )
        except Exception as e:
            print(f"\nAuto-evaluation failed: {e}")
            auto_eval_payload.update(
                {
                    "status": "error",
                    "error": str(e),
                }
            )

    try:
        _update_auto_eval_config(config_path, auto_eval_payload)
    except Exception as e:
        print(f"WARNING: Failed to save auto-eval results to config: {e}")


if __name__ == "__main__":
    main()
