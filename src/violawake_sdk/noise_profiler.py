"""Experimental: K4 noise robustness metrics and adaptive threshold.

Provides a ``NoiseProfiler`` that estimates ambient noise level from audio
and dynamically adjusts the detection threshold for better performance
in noisy environments.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_NOISE_WINDOW_S = 5.0  # Seconds of audio to average for noise floor
DEFAULT_MIN_THRESHOLD = 0.60  # Never drop threshold below this
DEFAULT_MAX_THRESHOLD = 0.95  # Never raise threshold above this
DEFAULT_SNR_BOOST_DB = 6.0  # SNR above which threshold can be lowered
DEFAULT_SNR_PENALTY_DB = 3.0  # SNR below which threshold should be raised


@dataclass(frozen=True)
class NoiseProfile:
    """Snapshot of the current noise estimation.

    Attributes:
        noise_rms: Estimated noise floor RMS (float32 scale, not int16).
        signal_rms: Estimated signal RMS (recent frame).
        snr_db: Estimated signal-to-noise ratio in dB.
        adjusted_threshold: The dynamically adjusted detection threshold.
        base_threshold: The original (unadjusted) threshold.
    """

    noise_rms: float
    signal_rms: float
    snr_db: float
    adjusted_threshold: float
    base_threshold: float


class NoiseProfiler:
    """Estimates ambient noise and adjusts detection threshold.

    The profiler maintains a rolling window of RMS energy measurements.
    The noise floor is estimated as the 10th percentile of recent RMS values
    (capturing the quietest frames, which are likely ambient noise).

    Threshold adjustment logic:
    - High SNR (signal clearly above noise): lower threshold slightly to
      improve sensitivity.
    - Low SNR (signal barely above noise): raise threshold to reduce
      false alarms from noise bursts.
    - The adjusted threshold is always clamped to
      ``[min_threshold, max_threshold]``.

    Adaptive threshold bounds:
    - ``min_threshold`` defaults to ``0.60``. In very quiet environments, or
      when the current frame is far above the estimated noise floor, the
      threshold may be lowered to improve sensitivity, but it will never be
      reduced below this floor.
    - ``max_threshold`` defaults to ``0.95``. In very noisy environments, or
      when the current frame is close to the estimated noise floor, the
      threshold may be raised to suppress false accepts, but it will never be
      increased above this ceiling.
    - Before enough history is collected (fewer than 10 RMS frames), the
      profiler returns ``base_threshold`` with no adaptation.

    Args:
        base_threshold: The default detection threshold (e.g. 0.80).
        noise_window_s: Seconds of audio history for noise estimation.
        min_threshold: Floor for adaptive threshold. Default 0.60.
        max_threshold: Ceiling for adaptive threshold. Default 0.95.
        snr_boost_db: SNR above this value enables threshold lowering.
        snr_penalty_db: SNR below this value enables threshold raising.
        frames_per_second: Expected audio frames per second (default 50 for 20ms).
    """

    def __init__(
        self,
        base_threshold: float = 0.80,
        noise_window_s: float = DEFAULT_NOISE_WINDOW_S,
        min_threshold: float = DEFAULT_MIN_THRESHOLD,
        max_threshold: float = DEFAULT_MAX_THRESHOLD,
        snr_boost_db: float = DEFAULT_SNR_BOOST_DB,
        snr_penalty_db: float = DEFAULT_SNR_PENALTY_DB,
        frames_per_second: float = 50.0,
    ) -> None:
        self._base_threshold = base_threshold
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._snr_boost_db = snr_boost_db
        self._snr_penalty_db = snr_penalty_db

        window_frames = max(1, int(noise_window_s * frames_per_second))
        self._rms_history: deque[float] = deque(maxlen=window_frames)
        self._current_rms: float = 0.0
        self._noise_floor_rms: float = 0.0

    @property
    def base_threshold(self) -> float:
        """The unadjusted detection threshold."""
        return self._base_threshold

    @property
    def noise_floor(self) -> float:
        """Current estimated noise floor RMS."""
        return self._noise_floor_rms

    def update(self, audio_frame: np.ndarray) -> float:
        """Update noise estimate with a new audio frame and return adjusted threshold.

        The audio should be float32 values. Both normalized [-1,1] and int16-range
        float32 are accepted; the profiler works on relative ratios so absolute
        scale doesn't matter.

        Args:
            audio_frame: 1-D float32 audio samples (any length).

        Returns:
            The adaptively adjusted detection threshold.
        """
        rms = float(np.sqrt(np.mean(audio_frame.astype(np.float64) ** 2)))
        self._current_rms = rms
        self._rms_history.append(rms)

        # Estimate noise floor as 10th percentile of recent RMS values
        if len(self._rms_history) >= 10:
            sorted_rms = sorted(self._rms_history)
            idx = max(0, int(len(sorted_rms) * 0.10))
            self._noise_floor_rms = sorted_rms[idx]
        elif self._rms_history:
            self._noise_floor_rms = min(self._rms_history)
        else:
            self._noise_floor_rms = 0.0

        return self._compute_adjusted_threshold()

    def _compute_adjusted_threshold(self) -> float:
        """Compute threshold adjustment based on current SNR estimate."""
        snr_db = self._estimate_snr_db()

        # No adjustment if we don't have enough data
        if len(self._rms_history) < 10:
            return self._base_threshold

        if snr_db > self._snr_boost_db:
            # High SNR: lower threshold proportionally (max 0.10 reduction)
            excess = snr_db - self._snr_boost_db
            reduction = min(0.10, excess * 0.01)
            adjusted = self._base_threshold - reduction
        elif snr_db < self._snr_penalty_db:
            # Low SNR: raise threshold proportionally (max 0.10 increase)
            deficit = self._snr_penalty_db - snr_db
            increase = min(0.10, deficit * 0.02)
            adjusted = self._base_threshold + increase
        else:
            adjusted = self._base_threshold

        return max(self._min_threshold, min(self._max_threshold, adjusted))

    def _estimate_snr_db(self) -> float:
        """Estimate signal-to-noise ratio in decibels.

        Uses the current frame RMS as signal and the noise floor as noise.
        Returns 0.0 if noise floor is effectively zero.
        """
        if self._noise_floor_rms < 1e-10:
            # No noise estimate yet — return a neutral value
            return self._snr_boost_db  # neutral, no adjustment
        if self._current_rms < 1e-10:
            return 0.0

        ratio = self._current_rms / self._noise_floor_rms
        return 20.0 * math.log10(max(ratio, 1e-10))

    def get_profile(self) -> NoiseProfile:
        """Return a snapshot of the current noise state.

        Returns:
            NoiseProfile with noise floor, signal RMS, SNR, and adjusted threshold.
        """
        snr_db = self._estimate_snr_db()
        adjusted = self._compute_adjusted_threshold()

        return NoiseProfile(
            noise_rms=self._noise_floor_rms,
            signal_rms=self._current_rms,
            snr_db=snr_db,
            adjusted_threshold=adjusted,
            base_threshold=self._base_threshold,
        )

    def reset(self) -> None:
        """Clear noise history and reset estimates."""
        self._rms_history.clear()
        self._current_rms = 0.0
        self._noise_floor_rms = 0.0
