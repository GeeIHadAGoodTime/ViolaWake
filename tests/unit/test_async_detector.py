"""Unit tests for AsyncWakeDetector.

All tests mock WakeDetector so no ONNX model is required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.async_detector import AsyncWakeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_wake_detector() -> MagicMock:
    """Return a mocked WakeDetector instance."""
    detector = MagicMock()
    detector.detect.return_value = True
    detector.process.return_value = 0.85
    detector.threshold = 0.50
    detector.last_scores = (0.1, 0.5, 0.85)
    detector._policy = MagicMock()
    detector.get_confidence.return_value = MagicMock()
    return detector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncWakeDetectorInit:
    """Tests for basic instantiation."""

    def test_creates_with_mocked_detector(self) -> None:
        """AsyncWakeDetector should instantiate with kwargs forwarded to WakeDetector."""
        with patch("violawake_sdk.async_detector.WakeDetector") as MockWD:
            MockWD.return_value = _mock_wake_detector()
            awd = AsyncWakeDetector(threshold=0.80)
            MockWD.assert_called_once_with(threshold=0.80)
            awd.close()

    def test_executor_starts_none(self) -> None:
        """Executor should be None until first use."""
        with patch("violawake_sdk.async_detector.WakeDetector", return_value=_mock_wake_detector()):
            awd = AsyncWakeDetector()
            assert awd._executor is None
            awd.close()


class TestAsyncContextManager:
    """Tests for __aenter__ / __aexit__."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """AsyncWakeDetector should work as an async context manager."""
        with patch("violawake_sdk.async_detector.WakeDetector", return_value=_mock_wake_detector()):
            async with AsyncWakeDetector() as awd:
                assert isinstance(awd, AsyncWakeDetector)
            # After exit, executor should be shut down
            assert awd._executor is None

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        """__aenter__ should return the same instance."""
        with patch("violawake_sdk.async_detector.WakeDetector", return_value=_mock_wake_detector()):
            awd = AsyncWakeDetector()
            result = await awd.__aenter__()
            assert result is awd
            await awd.__aexit__(None, None, None)


class TestDetect:
    """Tests for detect() delegation."""

    @pytest.mark.asyncio
    async def test_detect_delegates_to_sync(self) -> None:
        """detect() should call the underlying sync detector via executor."""
        mock_wd = _mock_wake_detector()
        mock_wd.detect.return_value = True

        with patch("violawake_sdk.async_detector.WakeDetector", return_value=mock_wd):
            async with AsyncWakeDetector() as awd:
                frame = np.zeros(480, dtype=np.float32)
                result = await awd.detect(frame, is_playing=False)

        assert result is True
        mock_wd.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_delegates_to_sync(self) -> None:
        """process() should return the score from the underlying detector."""
        mock_wd = _mock_wake_detector()
        mock_wd.process.return_value = 0.92

        with patch("violawake_sdk.async_detector.WakeDetector", return_value=mock_wd):
            async with AsyncWakeDetector() as awd:
                frame = np.zeros(480, dtype=np.float32)
                score = await awd.process(frame)

        assert score == 0.92
        mock_wd.process.assert_called_once()


class TestClose:
    """Tests for close() and cleanup."""

    def test_close_is_idempotent(self) -> None:
        """close() should be safe to call multiple times."""
        with patch("violawake_sdk.async_detector.WakeDetector", return_value=_mock_wake_detector()):
            awd = AsyncWakeDetector()
            awd.close()
            awd.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_shuts_down_executor(self) -> None:
        """close() should shut down the executor with wait=True."""
        with patch("violawake_sdk.async_detector.WakeDetector", return_value=_mock_wake_detector()):
            async with AsyncWakeDetector() as awd:
                # Force executor creation
                frame = np.zeros(480, dtype=np.float32)
                await awd.detect(frame)
                assert awd._executor is not None
            # After context exit, executor should be None
            assert awd._executor is None


class TestProperties:
    """Tests for property delegation."""

    def test_threshold_property(self) -> None:
        """threshold should delegate to the underlying detector."""
        mock_wd = _mock_wake_detector()
        mock_wd.threshold = 0.75

        with patch("violawake_sdk.async_detector.WakeDetector", return_value=mock_wd):
            awd = AsyncWakeDetector()
            assert awd.threshold == 0.75
            awd.close()

    def test_last_scores_property(self) -> None:
        """last_scores should delegate to the underlying detector."""
        mock_wd = _mock_wake_detector()
        mock_wd.last_scores = (0.1, 0.5, 0.9)

        with patch("violawake_sdk.async_detector.WakeDetector", return_value=mock_wd):
            awd = AsyncWakeDetector()
            assert awd.last_scores == (0.1, 0.5, 0.9)
            awd.close()
