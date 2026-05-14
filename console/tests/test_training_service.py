"""Focused tests for backend training progress reporting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.services import training_service

    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")


class _FakeStorage:
    def exists(self, identifier: str) -> bool:
        return True

    def download(self, identifier: str) -> bytes:
        return b"wav"


def _touch_audio_files(directory: Path, count: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for idx in range(count):
        path = directory / f"sample_{idx:03d}.wav"
        path.write_bytes(b"wav")
        files.append(path)
    return files


def test_run_training_job_sync_reports_confusable_progress(monkeypatch, tmp_path: Path) -> None:
    from violawake_sdk.tools import train as train_module

    negatives_dir = tmp_path / "negatives"
    _touch_audio_files(negatives_dir, 5)
    progress_events: list[dict[str, object]] = []

    monkeypatch.setattr(training_service.settings, "tmp_dir", tmp_path)
    monkeypatch.setattr(training_service, "get_storage", lambda: _FakeStorage())

    def _fake_tts_positives(
        wake_word: str,
        output_dir: Path,
        verbose: bool = True,
        *,
        check_cancelled=None,
    ) -> list[Path]:
        return []

    def _fake_confusables(
        wake_word: str,
        output_dir: Path,
        n_confusables: int = 30,
        voices_per_word: int = 10,
        verbose: bool = True,
        *,
        progress_callback=None,
        check_cancelled=None,
    ) -> list[Path]:
        total_samples = 4 if n_confusables == 30 else 2
        generated: list[Path] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(total_samples):
            if check_cancelled is not None:
                check_cancelled()
            sample_path = output_dir / f"confusable_{idx:03d}.wav"
            sample_path.write_bytes(b"wav")
            generated.append(sample_path)
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_word": f"word-{idx + 1}",
                        "word_index": idx + 1,
                        "total_words": total_samples,
                        "voice_index": 1,
                        "total_voices": 1,
                        "completed_samples": idx + 1,
                        "total_samples": total_samples,
                        "generated_files": len(generated),
                    }
                )
        return generated

    def _fake_train_temporal_cnn(
        *,
        output_path: Path,
        progress_callback=None,
        **kwargs,
    ) -> None:
        output_path.write_bytes(b"model")
        if progress_callback is not None:
            progress_callback({"epoch": 1, "total_epochs": 2, "train_loss": 0.25, "val_loss": 0.2})

    monkeypatch.setattr(train_module, "_generate_tts_positives", _fake_tts_positives)
    monkeypatch.setattr(train_module, "_generate_confusable_negatives", _fake_confusables)
    monkeypatch.setattr(train_module, "_train_temporal_cnn", _fake_train_temporal_cnn)

    artifact = training_service.run_training_job_sync(
        job_id=55,
        wake_word="viola",
        recording_identifiers=["r1", "r2", "r3", "r4", "r5"],
        output_path=tmp_path / "model.onnx",
        epochs=2,
        timeout_seconds=120,
        progress_callback=progress_events.append,
        is_cancelled=lambda: False,
        negatives_dir=negatives_dir,
    )

    assert artifact.local_path.exists()

    progress_values = [float(event["progress"]) for event in progress_events]
    assert progress_values[0] == 0.0
    assert 12.0 in progress_values
    assert any(12.0 < value < 22.0 for value in progress_values)
    assert any(22.0 < value < 28.0 for value in progress_values)
    assert 30.0 in progress_values
    assert progress_values[-1] > 30.0

    messages = [str(event["message"]) for event in progress_events]
    assert any("broad confusable negatives" in message for message in messages)
    assert any("tight confusable negatives" in message for message in messages)


def test_run_training_job_sync_wraps_system_exit(monkeypatch, tmp_path: Path) -> None:
    from violawake_sdk.tools import train as train_module

    negatives_dir = tmp_path / "negatives"
    _touch_audio_files(negatives_dir, 5)

    monkeypatch.setattr(training_service.settings, "tmp_dir", tmp_path)
    monkeypatch.setattr(training_service, "get_storage", lambda: _FakeStorage())
    monkeypatch.setattr(train_module, "_generate_tts_positives", lambda *args, **kwargs: [])
    monkeypatch.setattr(train_module, "_generate_confusable_negatives", lambda *args, **kwargs: [])

    def _raise_system_exit(*args, **kwargs) -> None:
        raise SystemExit(9)

    monkeypatch.setattr(train_module, "_train_temporal_cnn", _raise_system_exit)

    with pytest.raises(
        RuntimeError,
        match="Training aborted by fatal control-flow exception: SystemExit: 9",
    ):
        training_service.run_training_job_sync(
            job_id=57,
            wake_word="viola",
            recording_identifiers=["r1", "r2", "r3", "r4", "r5"],
            output_path=tmp_path / "model.onnx",
            epochs=2,
            timeout_seconds=120,
            progress_callback=lambda event: None,
            is_cancelled=lambda: False,
            negatives_dir=negatives_dir,
        )
