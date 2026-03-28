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

Zero external dependencies beyond numpy (scipy is optional for RIR convolution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
        p_rir: Probability of applying RIR convolution (requires scipy).
        rir_files: List of paths to RIR WAV files for convolution.
        p_spec_augment: Probability of applying SpecAugment to spectrograms.
        spec_freq_mask_param: Max frequency mask width for SpecAugment.
        spec_time_mask_param: Max time mask width for SpecAugment.
        spec_n_freq_masks: Number of frequency masks for SpecAugment.
        spec_n_time_masks: Number of time masks for SpecAugment.
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
    p_rir: float = 0.0
    rir_files: list[str] = field(default_factory=list)
    p_spec_augment: float = 0.0
    spec_freq_mask_param: int = 27
    spec_time_mask_param: int = 100
    spec_n_freq_masks: int = 1
    spec_n_time_masks: int = 1


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
# SpecAugment (J1) -- frequency and time masking for spectrograms
# ---------------------------------------------------------------------------


def spec_augment(
    spectrogram: np.ndarray,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    rng: np.random.Generator | None = None,
    *,
    n_freq_masks: int | None = None,
    n_time_masks: int | None = None,
) -> np.ndarray:
    """Apply SpecAugment (frequency and time masking) to a mel spectrogram.

    Pure numpy implementation of the SpecAugment paper (Park et al. 2019).
    Masks random contiguous bands along the frequency and time axes by
    zeroing them out. This forces the model to be robust to missing
    spectral information, improving generalization on unseen acoustic
    conditions.

    The mask widths are sampled uniformly from [0, mask_param], clamped
    to the actual axis length.

    Args:
        spectrogram: 2D array of shape (n_freq_bins, n_time_frames).
        freq_mask_param: Maximum width of each frequency mask (F in the paper).
            Default 27 matches the SpecAugment paper's LibriSpeech policy.
        time_mask_param: Maximum width of each time mask (T in the paper).
            Default 100. Clamped to n_time_frames internally.
        num_freq_masks: Number of frequency masks to apply (mF in paper).
        num_time_masks: Number of time masks to apply (mT in paper).
        rng: Optional numpy random Generator for reproducibility.
        n_freq_masks: Legacy alias for num_freq_masks.
        n_time_masks: Legacy alias for num_time_masks.

    Returns:
        Masked spectrogram (copy; original is not modified).
    """
    # Handle legacy aliases
    if n_freq_masks is not None:
        num_freq_masks = n_freq_masks
    if n_time_masks is not None:
        num_time_masks = n_time_masks

    spec = spectrogram.copy()
    n_freq, n_time = spec.shape

    # Clamp mask params to actual dimensions
    freq_mask_param = min(freq_mask_param, n_freq)
    time_mask_param = min(time_mask_param, n_time)

    _randint = rng.integers if rng is not None else np.random.randint

    for _ in range(num_freq_masks):
        f = int(_randint(0, freq_mask_param + 1))
        if f == 0:
            continue
        f0 = int(_randint(0, max(n_freq - f, 1)))
        spec[f0: f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = int(_randint(0, time_mask_param + 1))
        if t == 0:
            continue
        t0 = int(_randint(0, max(n_time - t, 1)))
        spec[:, t0: t0 + t] = 0.0

    return spec


# ---------------------------------------------------------------------------
# RIR convolution (J2) -- room impulse response augmentation
# ---------------------------------------------------------------------------


def generate_synthetic_rir(
    sample_rate: int = 16000,
    rt60: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic Room Impulse Response using exponential decay.

    Produces a simple but effective synthetic RIR by generating white noise
    shaped with an exponential decay envelope. The decay rate is controlled
    by the RT60 parameter (time for the impulse to decay by 60 dB).

    Args:
        sample_rate: Audio sample rate (default 16000 Hz).
        rt60: Reverberation time in seconds. If None, randomly sampled
            from [0.1, 0.8]s covering small offices to medium rooms.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        Float32 impulse response array, normalized to unit peak amplitude.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rt60 is None:
        rt60 = float(rng.uniform(0.1, 0.8))

    n_samples = int(sample_rate * rt60)
    if n_samples < 2:
        n_samples = 2

    noise = rng.standard_normal(n_samples).astype(np.float32)

    # Exponential decay: amplitude drops by 60 dB over RT60
    decay = np.exp(-np.arange(n_samples, dtype=np.float32) * np.log(1000.0) / n_samples)

    rir = noise * decay
    rir[0] = 1.0

    peak = np.abs(rir).max()
    if peak > 1e-10:
        rir = rir / peak

    return rir.astype(np.float32)


def load_rir_dataset(rir_dir: str | Path) -> list[np.ndarray]:
    """Load all RIR WAV files from a directory.

    Supports standard RIR dataset formats including MIT IR Survey and
    OpenAIR. Recursively scans for .wav files.

    Args:
        rir_dir: Path to directory containing RIR WAV files.

    Returns:
        List of float32 RIR arrays. Empty list if no valid files found.
    """
    rir_dir = Path(rir_dir)
    if not rir_dir.exists():
        return []

    rirs: list[np.ndarray] = []
    for wav_path in sorted(rir_dir.rglob("*.wav")):
        rir = _load_rir_file(wav_path)
        if rir is not None and len(rir) > 1:
            rirs.append(rir)

    return rirs


def rir_augment(
    audio: np.ndarray,
    rir_path: str | Path | None = None,
    rir: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Apply Room Impulse Response convolution to simulate room acoustics.

    Either convolves with a provided/loaded RIR, or generates a synthetic
    RIR if no real RIR is available.

    Args:
        audio: Float32 waveform.
        rir_path: Optional path to an RIR WAV file.
        rir: Optional pre-loaded RIR array. Takes precedence over rir_path.
        rng: Optional numpy random Generator for synthetic RIR generation.
        sample_rate: Audio sample rate for synthetic RIR generation.

    Returns:
        Reverberated audio at the same length and dtype as the input.
    """
    if rir is None and rir_path is not None:
        rir = _load_rir_file(rir_path)

    if rir is None:
        rir = generate_synthetic_rir(sample_rate=sample_rate, rng=rng)

    return apply_rir(audio, rir)


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve audio with a Room Impulse Response for realistic reverberation.

    Requires scipy (optional dependency). Falls back to returning the
    original audio unchanged if scipy is not installed.

    Args:
        audio: Float32 waveform.
        rir: Float32 room impulse response waveform.

    Returns:
        Reverberated audio at the same length and dtype as the input,
        or the original audio if scipy is unavailable.
    """
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        return audio

    convolved = fftconvolve(audio, rir, mode="full")[: len(audio)]
    peak = np.abs(audio).max()
    if peak > 0:
        convolved = convolved * (peak / max(np.abs(convolved).max(), 1e-10))
    return convolved.astype(audio.dtype)


def _load_rir_file(path: str | Path) -> np.ndarray | None:
    """Load an RIR WAV file and return as float32 mono.

    Returns None if the file cannot be loaded.
    """
    try:
        import wave

        path = Path(path)
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None

        if n_channels > 1:
            data = data.reshape(-1, n_channels)[:, 0]

        return data
    except Exception:
        return None


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
        # Pre-load RIR files if configured
        self._rir_cache: list[np.ndarray] = []
        if self.config.rir_files:
            for rir_path in self.config.rir_files:
                loaded_rir = _load_rir_file(rir_path)
                if loaded_rir is not None:
                    self._rir_cache.append(loaded_rir)

    def augment_spectrogram(
        self, spectrogram: np.ndarray
    ) -> np.ndarray:
        """Apply SpecAugment masking to a spectrogram if enabled.

        Uses the config's p_spec_augment probability and SpecAugment
        parameters. Call this on extracted mel/PCEN features before
        feeding to the model during training.

        Args:
            spectrogram: 2D array of shape (n_freq_bins, n_time_frames).

        Returns:
            Possibly masked spectrogram (copy if modified, original otherwise).
        """
        cfg = self.config
        if self.rng.random() < cfg.p_spec_augment:
            return spec_augment(
                spectrogram,
                freq_mask_param=cfg.spec_freq_mask_param,
                time_mask_param=cfg.spec_time_mask_param,
                num_freq_masks=cfg.spec_n_freq_masks,
                num_time_masks=cfg.spec_n_time_masks,
                rng=self.rng,
            )
        return spectrogram

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

            # RIR convolution (uses real RIRs if available, synthetic otherwise)
            if self.rng.random() < cfg.p_rir:
                if self._rir_cache:
                    rir_idx = self.rng.integers(0, len(self._rir_cache))
                    aug = apply_rir(aug, self._rir_cache[rir_idx])
                else:
                    aug = rir_augment(aug, rng=self.rng)

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
