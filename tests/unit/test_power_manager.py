"""Unit tests for K7: Energy-aware processing / power management.

Tests the PowerManager class with duty cycling, silence skipping,
and battery-aware behavior without requiring real battery hardware.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.power_manager import PowerManager, PowerState, _get_battery_info


class TestPowerManagerBasics:
    """Test basic PowerManager construction and validation."""

    def test_default_construction(self) -> None:
        pm = PowerManager()
        state = pm.get_state()
        assert state.duty_cycle_n == 1
        assert state.frames_processed == 0
        assert state.frames_skipped == 0
        assert state.effective_rate == 0.0  # No frames received yet

    def test_invalid_duty_cycle(self) -> None:
        with pytest.raises(ValueError, match="duty_cycle_n must be >= 1"):
            PowerManager(duty_cycle_n=0)

    def test_duty_cycle_one_processes_all(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=0.0)  # disable silence gate
        rng = np.random.default_rng(42)
        for _ in range(100):
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            assert pm.should_process(frame) is True
        state = pm.get_state()
        assert state.frames_processed == 100
        assert state.frames_skipped == 0


class TestDutyCycling:
    """Test duty cycle frame skipping."""

    def test_duty_cycle_skips_frames(self) -> None:
        pm = PowerManager(duty_cycle_n=3, silence_rms=0.0)
        rng = np.random.default_rng(42)
        results = []
        for _ in range(9):
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            results.append(pm.should_process(frame))
        # Every 3rd frame should be processed: frames 3, 6, 9 (1-indexed)
        assert sum(results) == 3
        state = pm.get_state()
        assert state.frames_processed == 3
        assert state.frames_skipped == 6

    def test_active_mode_overrides_duty_cycle(self) -> None:
        pm = PowerManager(
            duty_cycle_n=5, silence_rms=0.0,
            activity_threshold=0.3, active_window_s=10.0,
        )
        # Trigger active mode
        pm.report_score(0.5)
        rng = np.random.default_rng(42)
        # In active mode, every frame should be processed
        results = []
        for _ in range(10):
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            results.append(pm.should_process(frame))
        assert all(results)

    def test_active_mode_expires(self) -> None:
        pm = PowerManager(
            duty_cycle_n=3, silence_rms=0.0,
            activity_threshold=0.3, active_window_s=0.0,  # expires immediately
        )
        pm.report_score(0.5)  # trigger active
        # Force the active window to have expired by manipulating the timestamp
        pm._last_activity_time = time.monotonic() - 1.0
        rng = np.random.default_rng(42)
        processed = 0
        for _ in range(9):
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            if pm.should_process(frame):
                processed += 1
        # Should NOT process all 9 (duty cycling should kick in after expiry)
        assert processed < 9


class TestSilenceSkipping:
    """Test silence-based frame skipping."""

    def test_silence_skipped(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=10.0)
        # Feed a nearly-silent frame
        silence = np.zeros(320, dtype=np.float32)
        assert pm.should_process(silence) is False
        state = pm.get_state()
        assert state.silence_skipped == 1
        assert state.frames_skipped == 1

    def test_loud_frame_not_skipped(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=10.0)
        loud = np.full(320, 100.0, dtype=np.float32)
        assert pm.should_process(loud) is True

    def test_silence_rms_boundary(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=10.0)
        # RMS of constant 10.0 is exactly 10.0
        at_boundary = np.full(320, 10.0, dtype=np.float32)
        # RMS = 10.0, silence_rms = 10.0. 10.0 < 10.0 is False -> should process
        assert pm.should_process(at_boundary) is True


class TestBatteryAwareness:
    """Test battery-level-based power management."""

    def test_battery_multiplier_on_low_battery(self) -> None:
        pm = PowerManager(
            duty_cycle_n=2, silence_rms=0.0,
            battery_low_pct=20, battery_multiplier=3,
            check_battery_interval_s=0.0,  # check every call
        )
        # Mock low battery
        with patch(
            "violawake_sdk.power_manager._get_battery_info",
            return_value=(10, True),
        ):
            rng = np.random.default_rng(42)
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            pm.should_process(frame)  # triggers battery check
            # Effective duty = 2 * 3 = 6
            assert pm.effective_duty_cycle == 6

    def test_no_battery_multiplier_on_ac(self) -> None:
        pm = PowerManager(
            duty_cycle_n=2, silence_rms=0.0,
            battery_low_pct=20, battery_multiplier=3,
            check_battery_interval_s=0.0,
        )
        # Mock plugged in
        with patch(
            "violawake_sdk.power_manager._get_battery_info",
            return_value=(50, False),
        ):
            rng = np.random.default_rng(42)
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            pm.should_process(frame)
            assert pm.effective_duty_cycle == 2

    def test_no_battery_multiplier_above_threshold(self) -> None:
        pm = PowerManager(
            duty_cycle_n=2, silence_rms=0.0,
            battery_low_pct=20, battery_multiplier=3,
            check_battery_interval_s=0.0,
        )
        with patch(
            "violawake_sdk.power_manager._get_battery_info",
            return_value=(80, True),  # on battery but above threshold
        ):
            rng = np.random.default_rng(42)
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            pm.should_process(frame)
            assert pm.effective_duty_cycle == 2


class TestPowerState:
    """Test PowerState snapshot."""

    def test_state_after_mixed_processing(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=50.0)
        rng = np.random.default_rng(42)
        # Feed 5 loud frames and 5 silent frames
        for _ in range(5):
            loud = (rng.standard_normal(320) * 100).astype(np.float32)
            pm.should_process(loud)
        for _ in range(5):
            silence = np.zeros(320, dtype=np.float32)
            pm.should_process(silence)
        state = pm.get_state()
        assert state.frames_processed == 5
        assert state.silence_skipped == 5
        assert state.frames_skipped == 5
        assert 0.0 < state.effective_rate < 1.0

    def test_effective_rate_all_processed(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=0.0)
        rng = np.random.default_rng(42)
        for _ in range(10):
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            pm.should_process(frame)
        state = pm.get_state()
        assert abs(state.effective_rate - 1.0) < 1e-6


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_counters(self) -> None:
        pm = PowerManager(duty_cycle_n=1, silence_rms=0.0)
        rng = np.random.default_rng(42)
        for _ in range(10):
            frame = (rng.standard_normal(320) * 100).astype(np.float32)
            pm.should_process(frame)
        pm.reset()
        state = pm.get_state()
        assert state.frames_processed == 0
        assert state.frames_skipped == 0
        assert state.silence_skipped == 0


class TestGetBatteryInfo:
    """Test the battery detection utility."""

    def test_returns_tuple(self) -> None:
        result = _get_battery_info()
        assert isinstance(result, tuple)
        assert len(result) == 2
        pct, on_battery = result
        assert isinstance(pct, int)
        assert isinstance(on_battery, bool)
        # pct is either -1 (undetectable) or 0-100
        assert pct == -1 or 0 <= pct <= 100

    def test_psutil_battery_present_on_battery(self) -> None:
        """psutil reports battery present, discharging."""
        mock_batt = MagicMock()
        mock_batt.percent = 45
        mock_batt.power_plugged = False
        mock_psutil = MagicMock()
        mock_psutil.sensors_battery.return_value = mock_batt
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pct, on_battery = _get_battery_info()
        assert pct == 45
        assert on_battery is True

    def test_psutil_battery_present_plugged_in(self) -> None:
        """psutil reports battery present, plugged in (AC power)."""
        mock_batt = MagicMock()
        mock_batt.percent = 95
        mock_batt.power_plugged = True
        mock_psutil = MagicMock()
        mock_psutil.sensors_battery.return_value = mock_batt
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pct, on_battery = _get_battery_info()
        assert pct == 95
        assert on_battery is False

    def test_psutil_no_battery(self) -> None:
        """psutil available but sensors_battery() returns None (desktop)."""
        mock_psutil = MagicMock()
        mock_psutil.sensors_battery.return_value = None
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            # With no battery, should fall through to platform fallbacks
            # and eventually return (-1, False) if no other source
            pct, on_battery = _get_battery_info()
            assert isinstance(pct, int)
            assert isinstance(on_battery, bool)

    def test_psutil_not_installed_falls_through(self) -> None:
        """When psutil is not installed, fallback path is used."""
        with patch.dict("sys.modules", {"psutil": None}):
            pct, on_battery = _get_battery_info()
            assert isinstance(pct, int)
            assert isinstance(on_battery, bool)
            # Should be -1 or valid range depending on platform
            assert pct == -1 or 0 <= pct <= 100

    def test_psutil_battery_low_level(self) -> None:
        """Battery at critically low level (1%)."""
        mock_batt = MagicMock()
        mock_batt.percent = 1
        mock_batt.power_plugged = False
        mock_psutil = MagicMock()
        mock_psutil.sensors_battery.return_value = mock_batt
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pct, on_battery = _get_battery_info()
        assert pct == 1
        assert on_battery is True

    def test_psutil_battery_full(self) -> None:
        """Battery at full charge (100%), plugged in."""
        mock_batt = MagicMock()
        mock_batt.percent = 100
        mock_batt.power_plugged = True
        mock_psutil = MagicMock()
        mock_psutil.sensors_battery.return_value = mock_batt
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pct, on_battery = _get_battery_info()
        assert pct == 100
        assert on_battery is False

    def test_psutil_battery_zero(self) -> None:
        """Battery at 0%, on battery (edge case)."""
        mock_batt = MagicMock()
        mock_batt.percent = 0
        mock_batt.power_plugged = False
        mock_psutil = MagicMock()
        mock_psutil.sensors_battery.return_value = mock_batt
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pct, on_battery = _get_battery_info()
        assert pct == 0
        assert on_battery is True
