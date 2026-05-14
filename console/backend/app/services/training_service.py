"""Training pipeline helpers used by the async job queue."""

from __future__ import annotations

import json
import logging
import random
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.monitoring import log_exception
from app.storage import get_storage

logger = logging.getLogger("violawake.training")

CONFUSABLE_PROGRESS_START = 12.0
CONFUSABLE_ROUND_ONE_END = 22.0
CONFUSABLE_PROGRESS_END = 28.0
CORPUS_READY_PROGRESS = 30.0
TRAINING_PROGRESS_START = CORPUS_READY_PROGRESS
TRAINING_PROGRESS_END = 95.0


class TrainingCancelledError(RuntimeError):
    """Raised when a running training job is cancelled."""


@dataclass(slots=True)
class TrainingArtifact:
    """Artifacts produced by a completed training run."""

    local_path: Path
    config_json: str | None
    config_bytes: bytes | None
    d_prime: float | None
    size_bytes: int


def run_training_job_sync(
    *,
    job_id: int,
    wake_word: str,
    recording_identifiers: list[str],
    output_path: Path,
    epochs: int,
    timeout_seconds: int,
    progress_callback: Callable[[dict[str, Any]], None],
    is_cancelled: Callable[[], bool],
    negatives_dir: Path | None = None,
) -> TrainingArtifact:
    """Run the ViolaWake SDK training pipeline synchronously."""
    positives_dir: Path | None = None
    neg_temp_dir: Path | None = None
    storage = get_storage()

    def _ensure_not_cancelled() -> None:
        if is_cancelled():
            raise TrainingCancelledError("Training cancelled by user")

    try:
        def _publish_running_progress(progress: float, message: str) -> None:
            progress_callback({
                "status": "running",
                "progress": round(progress, 2),
                "epoch": 0,
                "total_epochs": epochs,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "message": message,
                "error": None,
            })

        def _on_confusable_progress(
            stage_label: str,
            progress_start: float,
            progress_end: float,
            info: dict[str, Any],
        ) -> None:
            total_samples = max(int(info.get("total_samples", 0)), 1)
            completed_samples = min(max(int(info.get("completed_samples", 0)), 0), total_samples)
            total_words = max(int(info.get("total_words", 0)), 1)
            word_index = min(max(int(info.get("word_index", 0)), 0), total_words)
            generated_files = max(int(info.get("generated_files", 0)), 0)
            progress = progress_start + (
                (progress_end - progress_start) * (completed_samples / total_samples)
            )
            _publish_running_progress(
                progress,
                (
                    "Corpus: %s positives. Generating %s confusable negatives... "
                    "%s/%s samples, word %s/%s (%s files)"
                )
                % (
                    len(pos_files),
                    stage_label,
                    completed_samples,
                    total_samples,
                    word_index,
                    total_words,
                    generated_files,
                ),
            )

        _ensure_not_cancelled()
        progress_callback({
            "status": "running",
            "progress": 0.0,
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": "Preparing training data...",
            "error": None,
        })

        positives_dir = Path(tempfile.mkdtemp(prefix="violawake_train_", dir=str(settings.tmp_dir)))
        for index, recording_identifier in enumerate(recording_identifiers):
            _ensure_not_cancelled()
            if not storage.exists(recording_identifier):
                logger.warning("Recording %s was missing for training job %s", recording_identifier, job_id)
                continue

            dst = positives_dir / f"sample_{index:04d}.wav"
            dst.write_bytes(storage.download(recording_identifier))

        wav_count = len(list(positives_dir.glob("*.wav")))
        if wav_count < 5:
            raise RuntimeError("Only %s valid WAV files found. Need at least 5." % wav_count)

        pos_files = sorted(positives_dir.glob("*.wav"))

        progress_callback({
            "status": "running",
            "progress": 2.0,
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": "Loaded %s recordings. Generating TTS corpus..." % wav_count,
            "error": None,
        })

        # -- Production pipeline: full auto-corpus (matching CLI train) --
        from violawake_sdk.tools.train import (
            _generate_confusable_negatives,
            _generate_tts_positives,
            _train_temporal_cnn,
        )

        neg_temp_dir = Path(tempfile.mkdtemp(prefix="violawake_neg_", dir=str(settings.tmp_dir)))

        # Auto-generate TTS positives when user has <100 samples (production behavior)
        # Keep track of user-provided files so augmentation targets only real recordings
        user_pos_files = list(pos_files)
        if len(pos_files) < 100:
            tts_pos_dir = neg_temp_dir / "tts_positives"
            try:
                tts_pos_files = _generate_tts_positives(
                    wake_word,
                    tts_pos_dir,
                    verbose=False,
                    check_cancelled=_ensure_not_cancelled,
                )
                if tts_pos_files:
                    pos_files = list(pos_files) + tts_pos_files
                    logger.info(
                        "Generated %s TTS positives for job %s (total: %s)",
                        len(tts_pos_files), job_id, len(pos_files),
                    )
            except Exception as exc:
                logger.error(
                    "TTS positive generation FAILED for job %s: %s — "
                    "model quality will be degraded without TTS diversity",
                    job_id, exc,
                )

        _ensure_not_cancelled()
        _publish_running_progress(
            CONFUSABLE_PROGRESS_START,
            "Corpus: %s positives. Generating broad confusable negatives..." % len(pos_files),
        )
        neg_tag_map: dict[str, list[Path]] = {}

        # Source 1: User/paid-tier corpus negatives
        if negatives_dir and negatives_dir.exists():
            user_neg = sorted(
                list(negatives_dir.rglob("*.wav")) + list(negatives_dir.rglob("*.flac"))
            )
            if user_neg:
                neg_tag_map["neg_user"] = user_neg
                logger.info("Loaded %s corpus negatives for job %s", len(user_neg), job_id)

        _ensure_not_cancelled()

        # Source 2: Auto-generated confusable negatives (phonetically similar)
        # Two rounds matching CLI production pipeline:
        #   Round 1: 30 confusables x 10 voices (broad phonetic coverage)
        #   Round 2: 16 confusables x 10 voices (tight variants for hard negatives)
        confusable_dir_r1 = neg_temp_dir / "confusables_r1"
        try:
            confusable_r1 = _generate_confusable_negatives(
                wake_word,
                confusable_dir_r1,
                n_confusables=30,
                voices_per_word=10,
                verbose=False,
                progress_callback=lambda info: _on_confusable_progress(
                    "broad",
                    CONFUSABLE_PROGRESS_START,
                    CONFUSABLE_ROUND_ONE_END,
                    info,
                ),
                check_cancelled=_ensure_not_cancelled,
            )
            if confusable_r1:
                neg_tag_map["neg_confusable_r1"] = confusable_r1
        except Exception as exc:
            logger.error(
                "Confusable round 1 FAILED for job %s: %s — "
                "model will have higher false positive rate on similar-sounding words",
                job_id, exc,
            )

        _ensure_not_cancelled()

        confusable_dir_r2 = neg_temp_dir / "confusables_r2"
        try:
            confusable_r2 = _generate_confusable_negatives(
                wake_word,
                confusable_dir_r2,
                n_confusables=16,
                voices_per_word=10,
                verbose=False,
                progress_callback=lambda info: _on_confusable_progress(
                    "tight",
                    CONFUSABLE_ROUND_ONE_END,
                    CONFUSABLE_PROGRESS_END,
                    info,
                ),
                check_cancelled=_ensure_not_cancelled,
            )
            if confusable_r2:
                neg_tag_map["neg_confusable_r2"] = confusable_r2
        except Exception as exc:
            logger.error(
                "Confusable round 2 FAILED for job %s: %s",
                job_id, exc,
            )

        _ensure_not_cancelled()
        _publish_running_progress(
            CONFUSABLE_PROGRESS_END,
            "Generated confusables. Loading corpus speech negatives...",
        )

        _ensure_not_cancelled()

        # Source 4: Universal corpus (LibriSpeech, MUSAN) if available
        service_file = Path(__file__).resolve()
        _CORPUS_SEARCH_PATHS = [
            Path("/app/corpus"),
            *([service_file.parents[4] / "corpus"] if len(service_file.parents) > 4 else []),
            Path.home() / ".violawake" / "corpus",
            Path("corpus"),
        ]
        _CORPUS_SUBDIRS: dict[str, tuple[str, ...]] = {
            "neg_librispeech": ("librispeech",),
            "neg_musan_speech": ("musan/musan/speech", "musan/speech"),
            "neg_musan_music": ("musan/musan/music", "musan/music"),
            "neg_musan_noise": ("musan/musan/noise", "musan/noise"),
        }
        _rng = random.Random(42)
        for tag, subdirs in _CORPUS_SUBDIRS.items():
            for corpus_root in _CORPUS_SEARCH_PATHS:
                if not corpus_root.exists():
                    continue
                for subdir in subdirs:
                    candidate = corpus_root / subdir
                    if candidate.exists():
                        corpus_files = sorted(
                            list(candidate.rglob("*.wav")) + list(candidate.rglob("*.flac"))
                        )
                        if corpus_files:
                            if len(corpus_files) > 2000:
                                corpus_files = sorted(_rng.sample(corpus_files, 2000))
                            neg_tag_map[tag] = corpus_files
                            break
                if tag in neg_tag_map:
                    break

        all_neg_files: list[Path] = []
        for files in neg_tag_map.values():
            all_neg_files.extend(files)

        total_neg = len(all_neg_files)
        speech_neg_tags = {
            "neg_user",
            "neg_librispeech",
            "neg_musan_speech",
            "neg_musan_music",
            "neg_musan_noise",
        }
        total_speech_neg = sum(
            len(files) for tag, files in neg_tag_map.items() if tag in speech_neg_tags
        )
        if total_speech_neg < 5 or total_neg < 5:
            raise RuntimeError(
                "No speech negatives available. Mount LibriSpeech + MUSAN corpus at "
                "/app/corpus or run `violawake download-corpus`."
            )

        _publish_running_progress(
            CORPUS_READY_PROGRESS,
            "Corpus ready: %s pos, %s neg. Training TemporalCNN..." % (len(pos_files), total_neg),
        )

        started_at = time.monotonic()

        def _on_epoch(info: dict[str, Any]) -> None:
            _ensure_not_cancelled()
            elapsed = time.monotonic() - started_at
            if elapsed > timeout_seconds:
                raise RuntimeError(
                    "Training job timed out after %ss (%s minutes)"
                    % (timeout_seconds, timeout_seconds // 60)
                )

            epoch = int(info.get("epoch", 0))
            total_epochs = int(info.get("total_epochs", epochs)) or epochs
            train_loss = float(info.get("train_loss", 0.0))
            val_loss = float(info.get("val_loss", 0.0))
            progress = min(
                TRAINING_PROGRESS_START
                + (TRAINING_PROGRESS_END - TRAINING_PROGRESS_START) * (epoch / total_epochs),
                TRAINING_PROGRESS_END,
            )

            progress_callback({
                "status": "running",
                "progress": round(progress, 2),
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "message": "Epoch %s/%s - loss: %.4f" % (epoch, total_epochs, train_loss),
                "error": None,
            })

        _train_temporal_cnn(
            pos_files=pos_files,
            neg_files=all_neg_files,
            output_path=output_path,
            wake_word=wake_word,
            epochs=epochs,
            augment=True,
            eval_dir=None,
            verbose=True,
            progress_callback=_on_epoch,
            neg_tags=neg_tag_map,
            tmp_dir=settings.tmp_dir,
            augment_source_files=user_pos_files,
        )

        _ensure_not_cancelled()
        if not output_path.exists():
            raise RuntimeError("Training completed but no model file was produced")

        config_path = output_path.with_suffix(".config.json")
        config_json: str | None = None
        config_bytes: bytes | None = None
        d_prime_value: float | None = None
        if config_path.exists():
            config_bytes = config_path.read_bytes()
            config_data = json.loads(config_bytes.decode("utf-8"))
            config_json = json.dumps(config_data)
            if isinstance(config_data, dict):
                raw_d_prime = config_data.get("d_prime")
                if isinstance(raw_d_prime, (int, float)):
                    d_prime_value = float(raw_d_prime)

        return TrainingArtifact(
            local_path=output_path,
            config_json=config_json,
            config_bytes=config_bytes,
            d_prime=d_prime_value,
            size_bytes=output_path.stat().st_size,
        )
    except TrainingCancelledError:
        logger.info("Training job %s cancelled", job_id)
        raise
    except (SystemExit, KeyboardInterrupt) as exc:
        detail = str(exc).strip()
        exc_name = type(exc).__name__
        if detail:
            raise RuntimeError(
                "Training aborted by fatal control-flow exception: %s: %s" % (exc_name, detail)
            ) from exc
        raise RuntimeError(
            "Training aborted by fatal control-flow exception: %s" % exc_name
        ) from exc
    except Exception as exc:
        log_exception(
            logger,
            exc,
            message="Training job failed",
            source="training",
            extra={"job_id": job_id, "wake_word": wake_word},
        )
        raise
    finally:
        if positives_dir is not None and positives_dir.exists():
            shutil.rmtree(positives_dir, ignore_errors=True)
        if neg_temp_dir is not None and neg_temp_dir.exists():
            shutil.rmtree(neg_temp_dir, ignore_errors=True)
