"""
ViolaWake Configuration Constants
===================================

Constants for audio processing and model configuration.
These MUST match the values used during training.

Copied from Viola's violawake/config.py — Viola-specific imports removed.
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# AUDIO SETTINGS
# ============================================================

# Sample rate (must match training data)
SAMPLE_RATE: int = 16_000

# Clip duration in seconds
CLIP_DURATION: float = 1.5

# Number of samples per clip
CLIP_SAMPLES: int = int(SAMPLE_RATE * CLIP_DURATION)

# INT16 scale factor
AUDIO_INT16_SCALE: float = 32768.0

# ============================================================
# FEATURE EXTRACTION SETTINGS
# ============================================================

# Feature type selector: "linear" (legacy v2 model) or "mel" / "mel_pcen" (v3+)
# "linear" = scipy spectrogram, first 32 FFT bins (~0-1280 Hz)
# "mel" = librosa mel spectrogram with log compression (60-7800 Hz)
# "mel_pcen" = librosa mel spectrogram with PCEN compression (60-7800 Hz)
# MUST be "mel_pcen" to match v3 model training (pipeline audit 2026-03-02).
# Using "mel" with a PCEN-trained model produces inverted scores (0.003 on
# positives, 1.0 on negatives) — confirmed live production bug 2026-03-02.
FEATURE_TYPE: str = "mel_pcen"

# ============================================================
# LINEAR SPECTROGRAM SETTINGS (legacy, used by viola_v2.onnx)
# ============================================================

# Number of linear frequency bins (legacy — first N bins of scipy FFT)
N_MELS: int = 32

# FFT window size
N_FFT: int = 512

# Hop length (10ms at 16kHz)
HOP_LENGTH: int = 160

# Window length (25ms at 16kHz)
WIN_LENGTH: int = 400

# ============================================================
# MEL SPECTROGRAM V2 SETTINGS (for retraining with full speech coverage)
# ============================================================

# Number of mel frequency bins — 40 captures F1/F2/F3 formants
N_MELS_MEL: int = 40

# FFT window size for mel spectrogram
N_FFT_MEL: int = 512

# Hop length for mel spectrogram (10ms at 16kHz)
HOP_LENGTH_MEL: int = 160

# Window length for mel spectrogram (25ms at 16kHz)
WIN_LENGTH_MEL: int = 400

# Minimum frequency for mel filterbank (Hz) — captures fundamental F0
F_MIN: int = 60

# Maximum frequency for mel filterbank (Hz) — covers F3 + sibilants
F_MAX: int = 7800

# ============================================================
# PCEN (Per-Channel Energy Normalization) SETTINGS
# ============================================================
# PCEN replaces log compression with an adaptive normalization that is
# more robust to varying volume levels and background noise. Formula:
#   y = (mel / (eps + smoother))^gain + bias)^power - bias^power
# Reference: Wang et al. 2017, "Trainable Frontend for Robust and
# Far-Field Keyword Spotting"

# Whether to use PCEN instead of log compression for mel features.
# MUST be True when using v3 model (trained with PCEN). Setting False
# with a PCEN-trained model is a live bug — see pipeline audit 2026-03-02.
USE_PCEN: bool = True

# PCEN gain (alpha) — controls AGC strength; higher = more normalization
PCEN_GAIN: float = 0.98

# PCEN bias (delta) — stabilizes the ratio for near-silent frames
PCEN_BIAS: float = 2.0

# PCEN power (r) — root compression exponent; 0.5 = sqrt-like compression
PCEN_POWER: float = 0.5

# PCEN time constant (seconds) — IIR smoothing filter time constant
# At 16kHz/hop=160: ~100 frames/s, 0.06s = ~6 frame averaging window
PCEN_TIME_CONSTANT: float = 0.06

# PCEN epsilon — numerical stability floor
PCEN_EPS: float = 1e-6

# ============================================================
# MODEL SETTINGS
# ============================================================

# Hidden layer size in classifier
HIDDEN_SIZE: int = 64

# Default detection threshold
# Raised to 0.80 after false-positive flood (see ADR-002).
DEFAULT_THRESHOLD: float = 0.80

# ============================================================
# PATHS
# ============================================================

# SDK install root (computed dynamically)
_SDK_ROOT = Path(__file__).parent

# Default model cache directory
DEFAULT_MODEL_DIR = Path.home() / ".violawake" / "models"

# ============================================================
# INFERENCE SETTINGS
# ============================================================

# Debounce window in seconds (prevent multiple triggers)
DEBOUNCE_SECONDS: float = 2.0

# RMS silence gate threshold (float32 scale).
# Audio with RMS below this is skipped before inference.
# Lowered from 0.005 to 0.001 (2026-03-06): 0.005 rejected 29/113
# positive eval files (legitimate whisper/quiet speech), dropping
# d-prime from 1.45 to 0.55 and recall from 95% to 72%.
SILENCE_GATE_RMS: float = 0.001

# Minimum score to log as potential detection
LOG_THRESHOLD: float = 0.3


def get_feature_config() -> dict[str, object]:
    """Return the complete feature configuration dict.

    This dict is saved inside model checkpoints at training time
    and verified at inference time. If the checkpoint's feature
    config doesn't match the current config, inference will use
    the CHECKPOINT's config (not _constants.py) and log an error.

    Added 2026-03-02 after config drift bug caused inverted
    scores in production.
    """
    return {
        "feature_type": FEATURE_TYPE,
        "n_mels": N_MELS_MEL,
        "n_fft": N_FFT_MEL,
        "hop_length": HOP_LENGTH_MEL,
        "win_length": WIN_LENGTH_MEL,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "sample_rate": SAMPLE_RATE,
        "clip_samples": CLIP_SAMPLES,
        "use_pcen": USE_PCEN,
        "pcen_gain": PCEN_GAIN,
        "pcen_bias": PCEN_BIAS,
        "pcen_power": PCEN_POWER,
        "pcen_time_constant": PCEN_TIME_CONSTANT,
        "pcen_eps": PCEN_EPS,
    }
