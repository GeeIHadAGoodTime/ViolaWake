"""K7: Energy-aware processing / power management.

Provides duty cycling, silence skipping, and battery-aware inference
frequency reduction for embedded and mobile deployments.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PowerState:
    """Snapshot of the power manager state.

    Attributes:
        battery_percent: Battery level 0-100, or -1 if not detectable.
        is_on_battery: True if running on battery power.
        duty_cycle_n: Process every Nth frame (1 = every frame).
        frames_processed: Total frames actually processed.
        frames_skipped: Total frames skipped (duty cycle + silence).
        silence_skipped: Frames skipped specifically due to silence.
        effective_rate: Fraction of frames actually processed.
    """

    battery_percent: int
    is_on_battery: bool
    duty_cycle_n: int
    frames_processed: int
    frames_skipped: int
    silence_skipped: int
    effective_rate: float


def _get_battery_info() -> tuple[int, bool]:
    """Detect battery level and AC/battery state.

    Returns:
        (percent, is_on_battery) — percent is -1 if not detectable.
    """
    # Try psutil first (cross-platform)
    try:
        import psutil

        batt = psutil.sensors_battery()
        if batt is not None:
            return int(batt.percent), not batt.power_plugged
    except (ImportError, AttributeError):
        pass

    # Windows fallback via ctypes
    if platform.system() == "Windows":
        try:
            import ctypes

            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_byte),
                    ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            status = SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                pct = status.BatteryLifePercent
                if 0 <= pct <= 100:
                    return int(pct), status.ACLineStatus == 0
        except Exception:
            pass

    # Linux fallback: read /sys
    if platform.system() == "Linux":
        try:
            from pathlib import Path

            bat_path = Path("/sys/class/power_supply/BAT0")
            if bat_path.exists():
                capacity = int((bat_path / "capacity").read_text().strip())
                status = (bat_path / "status").read_text().strip()
                return capacity, status != "Charging" and status != "Full"
        except Exception:
            pass

    return -1, False


class PowerManager:
    """Energy-aware inference controller.

    Reduces inference frequency based on battery level, silence detection,
    and explicit duty cycling configuration.

    Modes of power saving:
    1. **Duty cycling**: Process every Nth frame when idle (no recent detections).
       When a score above ``activity_threshold`` is detected, switches to
       full-rate processing for ``active_window_s`` seconds.
    2. **Silence skipping**: Skip inference when audio RMS is below
       ``silence_rms`` (no speech possible).
    3. **Battery-aware**: When on battery and below ``battery_low_pct``,
       increase the duty cycle factor by ``battery_multiplier``.

    Args:
        duty_cycle_n: Base duty cycle (process every Nth frame). Default 1 (no skipping).
        silence_rms: RMS threshold in int16 scale (typical range 0-32768) below which
            frames are skipped. Default 10.0 filters near-silence.
        activity_threshold: Score above which the system enters "active" mode. Default 0.3.
        active_window_s: Seconds to stay in full-rate mode after activity. Default 3.0.
        battery_low_pct: Battery percent below which power saving kicks in. Default 20.
        battery_multiplier: Multiply duty_cycle_n by this when on low battery. Default 3.
        check_battery_interval_s: How often to re-check battery. Default 60.
    """

    def __init__(
        self,
        duty_cycle_n: int = 1,
        silence_rms: float = 10.0,
        activity_threshold: float = 0.3,
        active_window_s: float = 3.0,
        battery_low_pct: int = 20,
        battery_multiplier: int = 3,
        check_battery_interval_s: float = 60.0,
    ) -> None:
        if duty_cycle_n < 1:
            raise ValueError(f"duty_cycle_n must be >= 1, got {duty_cycle_n}")

        self._base_duty = duty_cycle_n
        self._silence_rms = silence_rms
        self._activity_threshold = activity_threshold
        self._active_window_s = active_window_s
        self._battery_low_pct = battery_low_pct
        self._battery_multiplier = battery_multiplier
        self._check_interval = check_battery_interval_s

        # Lock protects all mutable state below
        self._lock = threading.Lock()

        # State
        self._frame_counter = 0
        self._frames_processed = 0
        self._frames_skipped = 0
        self._silence_skipped = 0
        self._last_activity_time = 0.0
        self._is_active = False

        # Battery state (cached)
        self._battery_pct = -1
        self._is_on_battery = False
        self._last_battery_check = 0.0

    @property
    def effective_duty_cycle(self) -> int:
        """Current effective duty cycle considering battery and activity state."""
        with self._lock:
            return self._effective_duty_cycle_unlocked()

    def _effective_duty_cycle_unlocked(self) -> int:
        """Compute duty cycle without acquiring the lock (caller must hold it)."""
        if self._is_active:
            return 1  # Full rate when active

        base = self._base_duty

        # Battery scaling
        if self._is_on_battery and 0 <= self._battery_pct < self._battery_low_pct:
            base = base * self._battery_multiplier

        return max(1, base)

    def should_process(self, audio_frame: np.ndarray) -> bool:
        """Decide whether this frame should be processed or skipped.

        Call this before running inference. If it returns False, skip the
        frame to save CPU/power.

        Args:
            audio_frame: 1-D audio samples (int16-range float32 or actual int16).

        Returns:
            True if inference should run on this frame.
        """
        # Compute RMS outside the lock (pure computation on immutable input)
        rms = float(np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2)))

        with self._lock:
            self._frame_counter += 1

            # Periodically check battery
            now = time.monotonic()
            if now - self._last_battery_check > self._check_interval:
                self._battery_pct, self._is_on_battery = _get_battery_info()
                self._last_battery_check = now

            # Check if active window has expired
            if self._is_active and (now - self._last_activity_time > self._active_window_s):
                self._is_active = False

            # Silence gate: skip if audio is very quiet
            if rms < self._silence_rms:
                self._frames_skipped += 1
                self._silence_skipped += 1
                return False

            # Duty cycling: process every Nth frame
            duty = self._effective_duty_cycle_unlocked()
            if duty > 1 and (self._frame_counter % duty) != 0:
                self._frames_skipped += 1
                return False

            self._frames_processed += 1
            return True

    def report_score(self, score: float) -> None:
        """Report a detection score to the power manager.

        If the score is above the activity threshold, the manager switches
        to full-rate processing mode for ``active_window_s`` seconds.

        Args:
            score: Detection score from the model.
        """
        if score >= self._activity_threshold:
            with self._lock:
                self._is_active = True
                self._last_activity_time = time.monotonic()

    def get_state(self) -> PowerState:
        """Return current power management state snapshot."""
        with self._lock:
            total = self._frames_processed + self._frames_skipped
            # Returns 0.0 if no frames received (no data yet to measure).
            rate = self._frames_processed / total if total > 0 else 0.0

            return PowerState(
                battery_percent=self._battery_pct,
                is_on_battery=self._is_on_battery,
                duty_cycle_n=self._effective_duty_cycle_unlocked(),
                frames_processed=self._frames_processed,
                frames_skipped=self._frames_skipped,
                silence_skipped=self._silence_skipped,
                effective_rate=rate,
            )

    def reset(self) -> None:
        """Reset all counters and state."""
        with self._lock:
            self._frame_counter = 0
            self._frames_processed = 0
            self._frames_skipped = 0
            self._silence_skipped = 0
            self._last_activity_time = 0.0
            self._is_active = False
