"""
violawake-train CLI -- Train a custom wake word model.

Entry point: ``violawake-train`` (declared in pyproject.toml).

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

Data pipeline:
  A. Positives: user-provided + auto-TTS (edge-tts, 20 voices x 3 phrases x 3 conditions)
  B. Confusable negatives: phonetically similar words via TTS (30 words x 10 voices)
  C. Speech negatives: common phrases via TTS (100+ phrases x 5 voices)
  D. User-provided negatives via --negatives directory (if any)

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

    # Legacy MLP mode:
    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --output models/jarvis.onnx \\
      --architecture mlp

Minimum: 5 positive samples (auto-TTS fills to ~200). Recommended: 50+.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Edge-TTS voice pool for diverse positive and negative generation
# ---------------------------------------------------------------------------

EDGE_TTS_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
    "en-US-AmberNeural",
    "en-US-AnaNeural",
    "en-US-AndrewNeural",
    "en-US-BrandonNeural",
    "en-US-ChristopherNeural",
    "en-US-CoraNeural",
    "en-US-ElizabethNeural",
    "en-US-EricNeural",
    "en-US-JacobNeural",
    "en-US-MichelleNeural",
    "en-US-MonicaNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
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

def _edge_tts_synthesize(text: str, voice: str, output_path: Path) -> bool:
    """Synthesize a single phrase with edge-tts and save as WAV at 16kHz.

    Returns True on success, False on failure.
    """
    import asyncio
    import io
    import tempfile

    try:
        import edge_tts
    except ImportError:
        print("WARNING: edge-tts not installed. pip install edge-tts", file=sys.stderr)
        return False

    async def _synth():
        communicate = edge_tts.Communicate(text, voice)
        mp3_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])
        return mp3_buf.getvalue()

    try:
        # Run the async synthesis
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    mp3_data = pool.submit(
                        lambda: asyncio.run(_synth())
                    ).result(timeout=30)
            else:
                mp3_data = loop.run_until_complete(_synth())
        except RuntimeError:
            mp3_data = asyncio.run(_synth())

        if not mp3_data or len(mp3_data) < 100:
            return False

        # Convert MP3 to WAV at 16kHz using pydub or ffmpeg
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            seg.export(str(output_path), format="wav")
            return True
        except ImportError:
            pass

        # Fallback: write MP3 to temp, load with torchaudio/scipy
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
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
            return True
        except Exception:
            pass
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return False
    except Exception:
        return False


def _generate_tts_positives(
    wake_word: str,
    output_dir: Path,
    verbose: bool = True,
) -> list[Path]:
    """Generate diverse TTS positive samples using edge-tts.

    Produces: 20 voices x 3 phrases (WORD, hey WORD, ok WORD) = 60 clean files.
    Then augmentation (noisy + reverb) multiplies to ~180 total.

    Returns list of generated WAV file paths.
    """
    import numpy as np

    from violawake_sdk.training.augment import (
        AugmentationPipeline,
        AugmentConfig,
        apply_additive_noise,
        rir_augment,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    phrases = [wake_word, f"hey {wake_word}", f"ok {wake_word}"]
    generated: list[Path] = []

    if verbose:
        total = len(EDGE_TTS_VOICES) * len(phrases)
        print(f"  Generating TTS positives: {len(EDGE_TTS_VOICES)} voices x {len(phrases)} phrases = {total} clean samples...")

    for voice_idx, voice in enumerate(EDGE_TTS_VOICES):
        for phrase_idx, phrase in enumerate(phrases):
            clean_path = output_dir / f"tts_pos_{voice_idx:02d}_{phrase_idx}_{voice}.wav"
            if clean_path.exists():
                generated.append(clean_path)
                continue

            ok = _edge_tts_synthesize(phrase, voice, clean_path)
            if ok and clean_path.exists():
                generated.append(clean_path)

                # Generate noisy variant
                try:
                    from violawake_sdk.audio import load_audio
                    audio = load_audio(clean_path)
                    if audio is not None and len(audio) > 0:
                        rng = np.random.default_rng(voice_idx * 100 + phrase_idx)

                        # Noisy variant (SNR 10-15 dB)
                        noisy = apply_additive_noise(audio, snr_db=12.0, rng=rng)
                        noisy_path = output_dir / f"tts_pos_{voice_idx:02d}_{phrase_idx}_{voice}_noisy.wav"
                        _save_wav(noisy, noisy_path)
                        generated.append(noisy_path)

                        # Reverb variant
                        reverbed = rir_augment(audio, rng=rng)
                        reverb_path = output_dir / f"tts_pos_{voice_idx:02d}_{phrase_idx}_{voice}_reverb.wav"
                        _save_wav(reverbed, reverb_path)
                        generated.append(reverb_path)
                except Exception:
                    pass  # Augmented variants are best-effort

        if verbose and (voice_idx + 1) % 5 == 0:
            print(f"    {voice_idx + 1}/{len(EDGE_TTS_VOICES)} voices done ({len(generated)} files)")

    if verbose:
        print(f"  TTS positives generated: {len(generated)} files")

    return generated


def _generate_confusable_negatives(
    wake_word: str,
    output_dir: Path,
    n_confusables: int = 30,
    voices_per_word: int = 10,
    verbose: bool = True,
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
        print(f"  Synthesizing: {len(confusable_words)} words x {voices_per_word} voices = {total} samples...")

    voices_subset = EDGE_TTS_VOICES[:voices_per_word]
    generated: list[Path] = []

    for word_idx, word in enumerate(confusable_words):
        for voice_idx, voice in enumerate(voices_subset):
            safe_word = word.replace(" ", "_")[:30]
            out_path = output_dir / f"confusable_{word_idx:03d}_{voice_idx}_{safe_word}.wav"
            if out_path.exists():
                generated.append(out_path)
                continue

            ok = _edge_tts_synthesize(word, voice, out_path)
            if ok and out_path.exists():
                generated.append(out_path)

        if verbose and (word_idx + 1) % 10 == 0:
            print(f"    {word_idx + 1}/{len(confusable_words)} words done ({len(generated)} files)")

    if verbose:
        print(f"  Confusable negatives generated: {len(generated)} files")

    return generated


def _generate_speech_negatives(
    output_dir: Path,
    n_voices: int = 5,
    verbose: bool = True,
) -> list[Path]:
    """Generate speech negative samples via TTS using common phrases.

    Returns list of generated WAV file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    voices_subset = EDGE_TTS_VOICES[:n_voices]
    generated: list[Path] = []

    total = len(SPEECH_NEGATIVE_PHRASES) * n_voices
    if verbose:
        print(f"  Generating speech negatives: {len(SPEECH_NEGATIVE_PHRASES)} phrases x {n_voices} voices = {total} samples...")

    for phrase_idx, phrase in enumerate(SPEECH_NEGATIVE_PHRASES):
        for voice_idx, voice in enumerate(voices_subset):
            safe_phrase = phrase.replace(" ", "_")[:40]
            out_path = output_dir / f"speech_neg_{phrase_idx:03d}_{voice_idx}_{safe_phrase}.wav"
            if out_path.exists():
                generated.append(out_path)
                continue

            ok = _edge_tts_synthesize(phrase, voice, out_path)
            if ok and out_path.exists():
                generated.append(out_path)

        if verbose and (phrase_idx + 1) % 25 == 0:
            print(f"    {phrase_idx + 1}/{len(SPEECH_NEGATIVE_PHRASES)} phrases done ({len(generated)} files)")

    if verbose:
        print(f"  Speech negatives generated: {len(generated)} files")

    return generated


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int = 16000) -> None:
    """Save float32 audio to a WAV file."""
    import numpy as np
    import wave

    audio = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


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

    For each audio file, pushes audio through the OWW backbone in chunks,
    collects all 96-dim embeddings, and builds sliding windows of `seq_len`
    consecutive embeddings. Each window is a (seq_len, 96) tensor.

    Returns:
        embeddings: List of (seq_len, 96) numpy arrays.
        source_indices: Source file index for each embedding (for group-aware split).
        tags: Tag string for each embedding.
    """
    import numpy as np

    from violawake_sdk.audio import load_audio
    from violawake_sdk.oww_backbone import (
        EMBEDDING_DIM,
        OWW_CHUNK_SAMPLES,
        OpenWakeWordBackbone,
    )

    # Load OWW backbone
    try:
        from violawake_sdk.backends.onnx_backend import OnnxBackend
        backend = OnnxBackend()
    except ImportError:
        # Fallback: use openwakeword directly
        from openwakeword.model import Model as OWWModel
        oww = OWWModel()
        return _extract_temporal_embeddings_oww_fallback(
            audio_files, tag, oww, verbose, seq_len
        )

    backbone = OpenWakeWordBackbone(backend)

    all_embeddings: list[np.ndarray] = []
    all_source_idx: list[int] = []
    all_tags: list[str] = []
    failures = 0

    for file_idx, wav_path in enumerate(audio_files):
        audio = load_audio(wav_path)
        if audio is None:
            failures += 1
            continue

        # Feed audio through backbone in chunks and collect embeddings
        backbone.reset()
        frame_embeddings: list[np.ndarray] = []

        # Convert to int16
        audio_i16 = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767).astype(np.int16)

        # Push in OWW_CHUNK_SAMPLES-sized chunks
        for start in range(0, len(audio_i16), OWW_CHUNK_SAMPLES):
            chunk = audio_i16[start : start + OWW_CHUNK_SAMPLES]
            if len(chunk) < OWW_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, OWW_CHUNK_SAMPLES - len(chunk)))
            has_new, embedding = backbone.push_audio(chunk)
            if has_new and embedding is not None:
                frame_embeddings.append(embedding.copy())

        # Build sliding windows of seq_len frames
        if len(frame_embeddings) >= seq_len:
            for i in range(len(frame_embeddings) - seq_len + 1):
                window = np.stack(frame_embeddings[i : i + seq_len], axis=0)
                all_embeddings.append(window)
                all_source_idx.append(file_idx)
                all_tags.append(tag)
        elif len(frame_embeddings) > 0:
            # Pad shorter clips to seq_len by repeating the last frame
            padded = list(frame_embeddings)
            while len(padded) < seq_len:
                padded.append(padded[-1].copy())
            window = np.stack(padded[:seq_len], axis=0)
            all_embeddings.append(window)
            all_source_idx.append(file_idx)
            all_tags.append(tag)

        if verbose and (file_idx + 1) % 100 == 0:
            print(f"    {file_idx + 1}/{len(audio_files)} files -> {len(all_embeddings)} windows")

    if verbose:
        print(
            f"  [{tag}] {len(audio_files)} files -> {len(all_embeddings)} temporal windows "
            f"({failures} failures)"
        )

    return all_embeddings, all_source_idx, all_tags


def _extract_temporal_embeddings_oww_fallback(
    audio_files: list[Path],
    tag: str,
    oww_model: Any,
    verbose: bool = True,
    seq_len: int = 9,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Fallback temporal extraction using openwakeword's embed_clips directly.

    Uses OWW's preprocessor.embed_clips which returns (n_clips, n_frames, 96).
    We build sliding windows from the frame axis.
    """
    import numpy as np

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    preprocessor = oww_model.preprocessor

    all_embeddings: list[np.ndarray] = []
    all_source_idx: list[int] = []
    all_tags: list[str] = []
    failures = 0

    for file_idx, wav_path in enumerate(audio_files):
        audio = load_audio(wav_path)
        if audio is None:
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
            # embed_clips returns (n_clips, n_frames, embedding_dim)
            frame_embeddings_3d = preprocessor.embed_clips(
                audio_i16.reshape(1, -1), ncpu=1
            )
            # Shape: (1, n_frames, 96) -> (n_frames, 96)
            frame_embeddings = frame_embeddings_3d[0]  # (n_frames, 96)

            if len(frame_embeddings.shape) == 1:
                # Single frame returned -- reshape to (1, 96)
                frame_embeddings = frame_embeddings.reshape(1, -1)

            n_frames = frame_embeddings.shape[0]

            if n_frames >= seq_len:
                for i in range(n_frames - seq_len + 1):
                    window = frame_embeddings[i : i + seq_len].astype(np.float32)
                    all_embeddings.append(window)
                    all_source_idx.append(file_idx)
                    all_tags.append(tag)
            elif n_frames > 0:
                # Pad to seq_len
                padded = np.zeros((seq_len, frame_embeddings.shape[1]), dtype=np.float32)
                padded[:n_frames] = frame_embeddings
                for j in range(n_frames, seq_len):
                    padded[j] = frame_embeddings[-1]
                all_embeddings.append(padded)
                all_source_idx.append(file_idx)
                all_tags.append(tag)

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
    from violawake_sdk.audio import center_crop, load_audio

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        print(f"ERROR: openwakeword required: {e}", file=sys.stderr)
        sys.exit(1)

    oww = OWWModel()
    preprocessor = oww.preprocessor

    all_embeddings: list[np.ndarray] = []
    all_source_idx: list[int] = []
    all_tags: list[str] = []
    failures = 0

    for file_idx, wav_path in enumerate(audio_files):
        audio = load_audio(wav_path)
        if audio is None:
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
            print(f"    {file_idx + 1}/{len(audio_files)} files -> {len(all_embeddings)} embeddings")

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
        if labels[i] == 1 and source_idx[i] in val_pos_sources:
            val_mask[i] = True
        elif labels[i] == 0 and source_idx[i] in val_neg_sources:
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
) -> dict[str, Any]:
    """Train a TemporalCNN on 9-frame OWW embedding windows.

    This replicates the proven production training recipe:
    - TemporalCNN(96, 9) architecture (~12K params)
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
        patience: Early stopping patience.
        verbose: Print progress.
        progress_callback: Optional callback for UI.
        device: Torch device hint.
        ema_decay: EMA decay factor.
        seq_len: Number of frames per temporal window.
        neg_tags: Optional dict mapping tag -> file list, for tagged negatives.

    Returns:
        Config dict with training results.
    """
    training_start = time.monotonic()

    # -- Lazy imports --------------------------------------------------------
    import logging

    logger = logging.getLogger("violawake.train")

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

    EMBEDDING_DIM = 96
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Augment positive files if requested ---------------------------------
    augmented_pos_files = list(pos_files)
    if augment:
        if verbose:
            print("\nStep 2: Augmenting positive audio files...")
        from violawake_sdk.audio import load_audio
        from violawake_sdk.training.augment import AugmentationPipeline

        pipeline = AugmentationPipeline(seed=42)
        aug_dir = output_path.parent / "_aug_positives"
        aug_dir.mkdir(parents=True, exist_ok=True)
        augment_factor = 3  # 3 augmented variants per file
        n_augmented = 0

        for file_idx, f in enumerate(pos_files):
            audio = load_audio(f)
            if audio is None:
                continue
            variants = pipeline.augment_clip(audio, factor=augment_factor)
            for var_idx, variant in enumerate(variants):
                aug_path = aug_dir / f"aug_{file_idx:04d}_{var_idx}.wav"
                _save_wav(variant, aug_path)
                augmented_pos_files.append(aug_path)
                n_augmented += 1

        if verbose:
            print(f"  {len(pos_files)} originals + {n_augmented} augmented = {len(augmented_pos_files)} total")

    # -- Extract temporal embeddings -----------------------------------------
    if verbose:
        print(f"\nStep 3: Extracting {seq_len}-frame temporal OWW embeddings...")
        print(f"  Processing {len(augmented_pos_files)} positive files...")

    pos_embs, pos_src, pos_tags = _extract_temporal_embeddings(
        augmented_pos_files, "pos", verbose=verbose, seq_len=seq_len
    )

    if len(pos_embs) < 5:
        print(
            f"ERROR: Only {len(pos_embs)} positive embeddings extracted. "
            "Need at least 5. Check audio files.",
            file=sys.stderr,
        )
        sys.exit(1)

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

    if len(all_neg_embs) < 5:
        print(
            f"ERROR: Only {len(all_neg_embs)} negative embeddings extracted. "
            "Need at least 5.",
            file=sys.stderr,
        )
        sys.exit(1)

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

        # Show tag breakdown
        unique_tags = sorted(set(tags.tolist()))
        for t in unique_tags:
            count = int((tags == t).sum())
            print(f"    {t}: {count}")

    # -- Group-aware split ---------------------------------------------------
    train_idx, val_idx = _group_aware_split(labels, source_idx, seed=42)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
            progress_callback({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "best_val_loss": best_val_loss,
                "lr": current_lr,
            })

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

    # -- Export to ONNX ------------------------------------------------------
    if verbose:
        print(f"\nExporting model to ONNX: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_temporal_onnx(
        model, str(output_path), seq_len=seq_len, embedding_dim=EMBEDDING_DIM
    )

    # -- Post-training quality gate ------------------------------------------
    if verbose:
        print("\nStep 5: Post-training quality gate (speech FP check)...")

    _run_quality_gate(model, torch_device, seq_len, EMBEDDING_DIM, verbose)

    # -- Evaluate if test set provided ---------------------------------------
    d_prime_result: float | None = None
    if eval_dir and eval_dir.exists():
        if verbose:
            print(f"\nEvaluating on test set: {eval_dir}")
        try:
            from violawake_sdk.training.evaluate import evaluate_onnx_model

            results = evaluate_onnx_model(output_path, eval_dir)
            d_prime_result = results["d_prime"]
            far = results["far_per_hour"]
            frr = results["frr"] * 100
            print(f"Cohen's d: {d_prime_result:.2f}  FAR: {far:.2f}/hr  FRR: {frr:.1f}%")
        except Exception as e:
            print(f"Evaluation failed: {e}")

    # -- Save config ---------------------------------------------------------
    from violawake_sdk._constants import get_feature_config

    config = get_feature_config()
    config.update({
        "architecture": "temporal_cnn",
        "model_class": "TemporalCNN",
        "embedding_dim": EMBEDDING_DIM,
        "seq_len": seq_len,
        "n_params": n_params,
        "n_pos_samples": n_pos,
        "n_neg_samples": n_neg,
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
        "training_duration_s": round(training_duration, 2),
        "wake_word": wake_word,
    })
    if d_prime_result is not None:
        config["d_prime"] = round(d_prime_result, 2)

    config_path = output_path.with_suffix(".config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nConfig saved: {config_path}")
        print(f"Model saved: {output_path}")
        print(f"Load with:  WakeDetector(model='{output_path}')")

    # Cleanup augmented files
    aug_dir = output_path.parent / "_aug_positives"
    if aug_dir.exists():
        import shutil
        shutil.rmtree(aug_dir, ignore_errors=True)

    return config


# ---------------------------------------------------------------------------
# Post-training quality gate
# ---------------------------------------------------------------------------

def _run_quality_gate(
    model: Any,
    torch_device: str,
    seq_len: int,
    embedding_dim: int,
    verbose: bool = True,
) -> None:
    """Score 50 speech phrases via TTS and warn if FP rate is high.

    Generates quick speech samples, extracts temporal embeddings, scores them,
    and prints a warning if more than 25% score above 0.5.
    """
    import tempfile

    import numpy as np
    import torch

    quality_phrases = SPEECH_NEGATIVE_PHRASES[:50]
    quality_dir = Path(tempfile.mkdtemp(prefix="violawake_qc_"))
    voice = EDGE_TTS_VOICES[0]  # Use a single voice for speed
    quality_files: list[Path] = []

    if verbose:
        print(f"  Generating {len(quality_phrases)} test phrases for quality check...")

    for i, phrase in enumerate(quality_phrases):
        out_path = quality_dir / f"qc_{i:03d}.wav"
        ok = _edge_tts_synthesize(phrase, voice, out_path)
        if ok and out_path.exists():
            quality_files.append(out_path)

    if len(quality_files) < 10:
        if verbose:
            print("  WARNING: Could not generate enough quality check samples. Skipping gate.")
        return

    # Extract temporal embeddings
    embs, _, _ = _extract_temporal_embeddings(
        quality_files, "qc", verbose=False, seq_len=seq_len
    )

    if len(embs) == 0:
        if verbose:
            print("  WARNING: No embeddings extracted for quality check. Skipping gate.")
        return

    # Score
    model.eval()
    model = model.to(torch_device)
    X_qc = torch.tensor(np.array(embs), dtype=torch.float32).to(torch_device)
    with torch.no_grad():
        scores = model(X_qc).cpu().numpy().flatten()

    fp_rate = float((scores > 0.5).mean())
    mean_score = float(scores.mean())
    max_score = float(scores.max())

    if verbose:
        print(f"  Quality gate: {len(scores)} speech samples scored")
        print(f"    Mean score: {mean_score:.4f}, Max: {max_score:.4f}")
        print(f"    FP rate (>0.5): {fp_rate * 100:.1f}%")

    if fp_rate > 0.25:
        print(
            "\n  *** WARNING: HIGH FALSE POSITIVE RATE ***\n"
            f"  {fp_rate * 100:.1f}% of speech test phrases scored above 0.5.\n"
            "  This model may trigger on regular speech.\n"
            "  Recommendation: add more speech negatives via --negatives\n"
            "  or ensure --auto-corpus is enabled.\n"
        )
    elif fp_rate > 0.10:
        print(
            f"\n  CAUTION: {fp_rate * 100:.1f}% of speech test phrases scored >0.5.\n"
            "  Consider adding more diverse speech negatives.\n"
        )
    else:
        if verbose:
            print("  Quality gate PASSED: low speech FP rate.")

    # Cleanup
    import shutil
    shutil.rmtree(quality_dir, ignore_errors=True)


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
    """Legacy MLP training on mean-pooled OWW embeddings.

    Kept for backward compatibility with --architecture mlp.
    See _train_temporal_cnn for the production architecture.
    """
    training_start = time.monotonic()

    import logging

    logger = logging.getLogger("violawake.train")

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
    pos_files = sorted(
        list(positives_dir.rglob("*.wav")) + list(positives_dir.rglob("*.flac"))
    )
    if len(pos_files) < 5:
        print(f"ERROR: Found only {len(pos_files)} positive samples.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Found {len(pos_files)} positive samples")

    # -- Embedding extraction ------------------------------------------------
    if verbose:
        print("Loading OpenWakeWord backbone...")

    oww = OWWModel()
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
        neg_files = sorted(
            list(negatives_dir.rglob("*.wav")) + list(negatives_dir.rglob("*.flac"))
        )
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
        from violawake_sdk.training.augment import apply_additive_noise
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
            progress_callback({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "lr": current_lr,
            })

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
        model, dummy_input, str(output_path),
        input_names=["embedding"], output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )

    # Config
    config = get_feature_config()
    config.update({
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
    })
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
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable audio-level data augmentation (default: True)",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable audio-level augmentation (TTS generation still runs)",
    )
    parser.add_argument(
        "--architecture",
        choices=["temporal_cnn", "mlp"],
        default="temporal_cnn",
        help="Model architecture (default: temporal_cnn). "
             "'mlp' is the legacy single-frame architecture.",
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
        "--neg-ratio",
        type=int,
        default=5,
        metavar="N",
        help="Negatives per positive (used in legacy MLP mode, default: 5)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        metavar="N",
        help="Hidden dim for legacy MLP (default: 64)",
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
        print(f"Architecture:       {args.architecture}")
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
    # Legacy MLP path
    # ======================================================================
    if args.architecture == "mlp":
        if positives_dir is None:
            print("ERROR: --positives is required for MLP architecture.", file=sys.stderr)
            sys.exit(1)
        if verbose:
            print("Using legacy MLP architecture (single-frame, mean-pooled embeddings).\n")
        _train_mlp_on_oww(
            positives_dir=positives_dir,
            output_path=output_path,
            epochs=args.epochs,
            augment=args.augment,
            eval_dir=eval_dir,
            negatives_dir=negatives_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            neg_ratio=args.neg_ratio,
            patience=args.patience,
            verbose=verbose,
        )
        return

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
            print(f"\nStep 1a: Auto-generating TTS positives (have {len(user_pos_files)}, need ~100+)...")
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
        print(f"\nTotal positive files: {len(all_pos_files)} "
              f"({len(user_pos_files)} user + {len(tts_pos_files)} TTS)")

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

    # Source 2: Auto-generated confusable negatives
    confusable_files: list[Path] = []
    if args.auto_corpus:
        if verbose:
            print(f"\nStep 1b: Auto-generating confusable negatives...")
        confusable_dir = corpus_dir / "confusables"
        confusable_files = _generate_confusable_negatives(
            args.word, confusable_dir, n_confusables=30, voices_per_word=10,
            verbose=verbose,
        )
        if confusable_files:
            neg_tag_map["neg_confusable"] = confusable_files

    # Source 3: Auto-generated speech negatives
    speech_neg_files: list[Path] = []
    if args.auto_corpus:
        if verbose:
            print(f"\nStep 1c: Auto-generating speech negatives...")
        speech_neg_dir = corpus_dir / "speech_negatives"
        speech_neg_files = _generate_speech_negatives(
            speech_neg_dir, n_voices=5, verbose=verbose,
        )
        if speech_neg_files:
            neg_tag_map["neg_speech"] = speech_neg_files

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
    all_neg_files = []
    for files in neg_tag_map.values():
        all_neg_files.extend(files)

    # -- Step 2-5: Train TemporalCNN ----------------------------------------
    _train_temporal_cnn(
        pos_files=all_pos_files,
        neg_files=all_neg_files,
        output_path=output_path,
        wake_word=args.word,
        epochs=args.epochs,
        augment=args.augment,
        eval_dir=eval_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        verbose=verbose,
        neg_tags=neg_tag_map,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
