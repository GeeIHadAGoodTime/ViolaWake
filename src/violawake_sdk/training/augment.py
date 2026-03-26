"""
Audio augmentation pipeline for training data expansion.

Provides numpy-only audio augmentations that operate at the waveform level
(before embedding extraction). Each augmentation takes a float32 audio array
at 16 kHz and returns a transformed copy.

Usage::

    from violawake_sdk.training.augment import AugmentationPipeline

    pipeline = AugmentationPipeline(seed=42)
    augmented_clips = pipeline.augment_batch(audio_clips, factor=10)

All transforms are designed for wake-word training: they preserve the
phonetic content of the utterance while varying acoustic conditions the
model will encounter in deployment (gain, noise, pitch, speed, timing).

Zero external dependencies beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    """Configuration for the augmentation pipeline.

    All ranges are inclusive [min, max]. Probabilities are per-augmentation
    chance of being applied to each sample (independently).

    Attributes:
        gain_db_range: Gain variation in dB.
        time_stretch_range: Speed factor range (1.0 = original).
        pitch_shift_semitone_range: Pitch shift in semitones.
        noise_snr_range_db: SNR range for additive noise in dB.
        time_shift_fraction: Maximum fraction of clip length to shift.
        p_gain: Probability of applying gain augmentation.
        p_time_stretch: Probability of applying time stretch.
        p_pitch_shift: Probability of applying pitch shift.
        p_noise: Probability of applying additive noise.
        p_time_shift: Probability of applying time shift.
    """

    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    time_stretch_range: tuple[float, float] = (0.9, 1.1)
    pitch_shift_semitone_range: tuple[float, float] = (-2.0, 2.0)
    noise_snr_range_db: tuple[float, float] = (5.0, 20.0)
    time_shift_fraction: float = 0.10
    p_gain: float = 0.8
    p_time_stretch: float = 0.5
    p_pitch_shift: float = 0.5
    p_noise: float = 0.7
    p_time_shift: float = 0.5


# ---------------------------------------------------------------------------
# Individual augmentation functions (stateless, pure numpy)
# ---------------------------------------------------------------------------


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in dB and clip to [-1, 1].

    Args:
        audio: Float32 waveform.
        gain_db: Gain in decibels (positive = louder).

    Returns:
        Gained audio, clipped to [-1, 1].
    """
    factor = 10.0 ** (gain_db / 20.0)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def apply_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch via linear interpolation (no phase vocoder needed).

    For the small stretch factors used in wake-word training (0.9-1.1),
    linear interpolation preserves formant structure adequately and avoids
    heavy dependencies.

    Args:
        audio: Float32 waveform.
        rate: Stretch factor. >1.0 = faster (shorter), <1.0 = slower (longer).

    Returns:
        Time-stretched audio (length changes by 1/rate).
    """
    if abs(rate - 1.0) < 1e-6:
        return audio.copy()

    original_len = len(audio)
    new_len = int(round(original_len / rate))
    if new_len < 2:
        return audio.copy()

    indices = np.linspace(0, original_len - 1, new_len)
    return np.interp(indices, np.arange(original_len), audio).astype(np.float32)


def apply_pitch_shift(
    audio: np.ndarray, semitones: float, sample_rate: int = 16000
) -> np.ndarray:
    """Pitch shift by resampling then time-correcting.

    Algorithm: resample to shift pitch, then stretch back to original
    duration. This is the classic resample-based pitch shift -- simple,
    fast, and adequate for +/-2 semitones.

    Args:
        audio: Float32 waveform.
        semitones: Shift amount (-2 to +2 typical).
        sample_rate: Sample rate (unused but kept for API consistency).

    Returns:
        Pitch-shifted audio at original length.
    """
    if abs(semitones) < 1e-6:
        return audio.copy()

    # Pitch shift factor: positive semitones = higher pitch
    factor = 2.0 ** (semitones / 12.0)
    original_len = len(audio)

    # Resample: to raise pitch by `factor`, we need fewer samples
    # (playback at original rate sounds higher)
    resampled_len = int(round(original_len / factor))
    if resampled_len < 2:
        return audio.copy()

    indices = np.linspace(0, original_len - 1, resampled_len)
    resampled = np.interp(indices, np.arange(original_len), audio)

    # Time-correct back to original length
    out_indices = np.linspace(0, resampled_len - 1, original_len)
    result = np.interp(out_indices, np.arange(resampled_len), resampled)

    return result.astype(np.float32)


def _generate_pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink noise (1/f spectrum) using the Voss-McCartney algorithm.

    Args:
        length: Number of samples.
        rng: Numpy random generator.

    Returns:
        Pink noise array, normalized to unit variance.
    """
    # Use 16 octaves of random generators
    n_octaves = 16
    # Pad length to next power of 2 for efficiency
    rows = n_octaves
    cols = length

    # White noise base
    white = rng.standard_normal(cols).astype(np.float32)

    # Accumulate lower-frequency random walks
    pink = np.zeros(cols, dtype=np.float32)
    for i in range(rows):
        step = 2**i
        # Hold-and-sample noise at progressively lower rates
        row_noise = rng.standard_normal((cols + step - 1) // step).astype(np.float32)
        # Repeat-upsample to full length
        repeated = np.repeat(row_noise, step)[:cols]
        pink += repeated

    pink += white
    std = pink.std()
    if std > 1e-9:
        pink /= std
    return pink


def apply_additive_noise(
    audio: np.ndarray, snr_db: float, rng: np.random.Generator, noise_type: str = "white"
) -> np.ndarray:
    """Add noise at a specified SNR.

    Args:
        audio: Float32 waveform (the signal).
        snr_db: Target signal-to-noise ratio in dB.
        rng: Numpy random generator.
        noise_type: "white" or "pink".

    Returns:
        Noisy audio, clipped to [-1, 1].
    """
    if noise_type == "pink":
        noise = _generate_pink_noise(len(audio), rng)
    else:
        noise = rng.standard_normal(len(audio)).astype(np.float32)

    # Compute signal and noise power
    sig_power = np.mean(audio**2)
    if sig_power < 1e-10:
        # Audio is near-silent; just return it
        return audio.copy()

    noise_power = np.mean(noise**2)
    if noise_power < 1e-10:
        return audio.copy()

    # Scale noise to achieve target SNR
    # SNR_dB = 10 * log10(sig_power / noise_power_scaled)
    # noise_power_scaled = sig_power / 10^(SNR_dB/10)
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)

    result = audio + noise * scale
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def apply_time_shift(
    audio: np.ndarray, shift_samples: int
) -> np.ndarray:
    """Shift audio in time by rolling with zero-fill.

    Args:
        audio: Float32 waveform.
        shift_samples: Number of samples to shift (positive = right).

    Returns:
        Time-shifted audio (zeros fill the vacated region).
    """
    if shift_samples == 0:
        return audio.copy()

    result = np.zeros_like(audio)
    if shift_samples > 0:
        if shift_samples < len(audio):
            result[shift_samples:] = audio[: len(audio) - shift_samples]
    else:
        abs_shift = abs(shift_samples)
        if abs_shift < len(audio):
            result[: len(audio) - abs_shift] = audio[abs_shift:]
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AugmentationPipeline:
    """Composable audio augmentation pipeline for wake-word training.

    Applies a random subset of augmentations to each input clip, producing
    ``factor`` augmented variants per original sample.

    Example::

        pipeline = AugmentationPipeline(seed=42)

        # Single clip -> list of augmented clips
        variants = pipeline.augment_clip(audio, factor=10)

        # Batch of clips -> flat list of all augmented clips
        all_variants = pipeline.augment_batch(clips, factor=10)
    """

    def __init__(
        self,
        config: AugmentConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def augment_clip(self, audio: np.ndarray, factor: int = 10) -> list[np.ndarray]:
        """Generate ``factor`` augmented variants of a single audio clip.

        Args:
            audio: Float32 waveform at 16 kHz.
            factor: Number of augmented variants to produce.

        Returns:
            List of augmented audio arrays (does NOT include the original).
        """
        cfg = self.config
        variants: list[np.ndarray] = []

        for _ in range(factor):
            aug = audio.copy()

            # Gain
            if self.rng.random() < cfg.p_gain:
                gain_db = self.rng.uniform(*cfg.gain_db_range)
                aug = apply_gain(aug, gain_db)

            # Time stretch
            if self.rng.random() < cfg.p_time_stretch:
                rate = self.rng.uniform(*cfg.time_stretch_range)
                aug = apply_time_stretch(aug, rate)

            # Pitch shift
            if self.rng.random() < cfg.p_pitch_shift:
                semitones = self.rng.uniform(*cfg.pitch_shift_semitone_range)
                aug = apply_pitch_shift(aug, semitones)

            # Additive noise (randomly pick white or pink)
            if self.rng.random() < cfg.p_noise:
                snr_db = self.rng.uniform(*cfg.noise_snr_range_db)
                noise_type = "pink" if self.rng.random() < 0.5 else "white"
                aug = apply_additive_noise(aug, snr_db, self.rng, noise_type)

            # Time shift
            if self.rng.random() < cfg.p_time_shift:
                max_shift = int(len(audio) * cfg.time_shift_fraction)
                shift = self.rng.integers(-max_shift, max_shift + 1)
                aug = apply_time_shift(aug, shift)

            variants.append(aug)

        return variants

    def augment_batch(
        self, clips: list[np.ndarray], factor: int = 10
    ) -> list[np.ndarray]:
        """Augment a batch of clips.

        Args:
            clips: List of float32 waveforms.
            factor: Number of augmented variants per original clip.

        Returns:
            Flat list of all augmented clips (len = len(clips) * factor).
        """
        all_variants: list[np.ndarray] = []
        for clip in clips:
            all_variants.extend(self.augment_clip(clip, factor=factor))
        return all_variants
