"""Retention cleanup tests for ViolaWake Console backend."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.config import settings

    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _days_ago(days: int) -> datetime:
    return _utcnow() - timedelta(days=days)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recording(
    rid: int,
    user_id: int = 1,
    created_at: datetime | None = None,
    file_path: str = "",
) -> MagicMock:
    r = MagicMock()
    r.id = rid
    r.user_id = user_id
    r.file_path = file_path or f"recordings/{user_id}/wake/rec_{rid}.wav"
    r.created_at = created_at or _utcnow()
    return r


def _make_model(
    mid: int,
    user_id: int = 1,
    created_at: datetime | None = None,
    file_path: str = "",
) -> MagicMock:
    m = MagicMock()
    m.id = mid
    m.user_id = user_id
    m.file_path = file_path or f"models/{user_id}/model_{mid}.onnx"
    m.created_at = created_at or _utcnow()
    return m


# ---------------------------------------------------------------------------
# Recording cleanup tests
# ---------------------------------------------------------------------------

class TestCleanupExpiredRecordings:

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_disabled_when_retention_days_zero(self):
        """Returns 0 without touching storage when recording_retention_days=0."""
        from app.retention import cleanup_expired_recordings

        with patch.object(settings, "recording_retention_days", 0):
            deleted = self._run(cleanup_expired_recordings())

        assert deleted == 0

    def test_no_expired_recordings_returns_zero(self, tmp_path):
        """Returns 0 when all recordings are within the retention window."""
        from app.retention import cleanup_expired_recordings

        # The DB query filters by created_at < cutoff, so fresh recordings
        # would NOT appear in the result set — mock returns empty list.
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_storage = MagicMock()

        with (
            patch.object(settings, "recording_retention_days", 90),
            patch("app.retention.async_session_factory", return_value=mock_session),
            patch("app.retention.get_storage", return_value=mock_storage),
            patch("app.retention._get_active_recording_ids", new=AsyncMock(return_value=set())),
        ):
            deleted = self._run(cleanup_expired_recordings())

        assert deleted == 0
        mock_storage.delete.assert_not_called()

    def test_expired_recordings_deleted(self):
        """Expired recordings are deleted from storage and removed from the session."""
        from app.retention import cleanup_expired_recordings

        old_recording = _make_recording(42, created_at=_days_ago(200))
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [old_recording]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_storage = MagicMock()
        mock_storage.delete = MagicMock(return_value=True)

        with (
            patch.object(settings, "recording_retention_days", 90),
            patch("app.retention.async_session_factory", return_value=mock_session),
            patch("app.retention.get_storage", return_value=mock_storage),
            patch("app.retention._get_active_recording_ids", new=AsyncMock(return_value=set())),
        ):
            deleted = self._run(cleanup_expired_recordings())

        assert deleted == 1
        mock_storage.delete.assert_called_once_with(old_recording.file_path)
        mock_session.delete.assert_called_once_with(old_recording)
        mock_session.commit.assert_called_once()

    def test_active_job_recording_is_skipped(self):
        """Recordings linked to active training jobs are never deleted."""
        from app.retention import cleanup_expired_recordings

        old_recording = _make_recording(99, created_at=_days_ago(200))
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [old_recording]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_storage = MagicMock()

        # old_recording.id (99) is in the active set
        with (
            patch.object(settings, "recording_retention_days", 90),
            patch("app.retention.async_session_factory", return_value=mock_session),
            patch("app.retention.get_storage", return_value=mock_storage),
            patch("app.retention._get_active_recording_ids", new=AsyncMock(return_value={99})),
        ):
            deleted = self._run(cleanup_expired_recordings())

        assert deleted == 0
        mock_storage.delete.assert_not_called()
        mock_session.delete.assert_not_called()

    def test_storage_error_does_not_abort_cleanup(self):
        """A storage deletion failure logs a warning but proceeds with DB deletion."""
        from app.retention import cleanup_expired_recordings

        old_recording = _make_recording(7, created_at=_days_ago(200))
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [old_recording]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_storage = MagicMock()
        mock_storage.delete = MagicMock(side_effect=OSError("disk full"))

        with (
            patch.object(settings, "recording_retention_days", 90),
            patch("app.retention.async_session_factory", return_value=mock_session),
            patch("app.retention.get_storage", return_value=mock_storage),
            patch("app.retention._get_active_recording_ids", new=AsyncMock(return_value=set())),
        ):
            deleted = self._run(cleanup_expired_recordings())

        # Still counts as deleted from DB perspective even if storage failed
        assert deleted == 1
        mock_session.delete.assert_called_once_with(old_recording)


# ---------------------------------------------------------------------------
# Model cleanup tests
# ---------------------------------------------------------------------------

class TestCleanupExpiredModels:

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_disabled_when_retention_days_zero(self):
        """Returns 0 without touching storage when model_retention_days=0."""
        from app.retention import cleanup_expired_models

        with patch.object(settings, "model_retention_days", 0):
            deleted = self._run(cleanup_expired_models())

        assert deleted == 0

    def test_no_expired_models_returns_zero(self):
        """Returns 0 when all models are within the retention window."""
        from app.retention import cleanup_expired_models

        # The DB query filters by created_at < cutoff, so fresh models
        # would NOT appear in the result set — mock returns empty list.
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_storage = MagicMock()

        with (
            patch.object(settings, "model_retention_days", 365),
            patch("app.retention.async_session_factory", return_value=mock_session),
            patch("app.retention.get_storage", return_value=mock_storage),
        ):
            deleted = self._run(cleanup_expired_models())

        assert deleted == 0
        mock_storage.delete.assert_not_called()

    def test_expired_model_deleted(self):
        """Expired models are deleted from storage and removed from the session."""
        from app.retention import cleanup_expired_models

        old_model = _make_model(5, created_at=_days_ago(400))
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [old_model]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_storage = MagicMock()
        mock_storage.delete = MagicMock(return_value=True)

        with (
            patch.object(settings, "model_retention_days", 365),
            patch("app.retention.async_session_factory", return_value=mock_session),
            patch("app.retention.get_storage", return_value=mock_storage),
        ):
            deleted = self._run(cleanup_expired_models())

        assert deleted == 1
        mock_session.delete.assert_called_once_with(old_model)
        mock_session.commit.assert_called_once()
