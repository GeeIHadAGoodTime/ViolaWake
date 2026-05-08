"""One-shot backfill for trained_models rows that lack d_prime / score arrays.

Pre-2026-05-08 trainings persisted neither d_prime nor the score distributions
because the SDK's quality gate didn't compute them and the training pipeline
only computed d_prime when an external eval_dir was supplied. The frontend
View Performance / View Details panels therefore showed "Unavailable" for every
metric, including d-prime.

This script:
  1. Loads the user's .onnx model directly.
  2. Scores their training-sample recordings (positives).
  3. Scores a slice of LibriSpeech clips and silence (negatives).
  4. Computes d_prime + persists positive_scores / negative_scores into the
     stored .config.json AND the trained_models row.

Run inside the backend container:
    docker exec wakeword-backend-1 python /app/scripts/backfill_dprime.py
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Recording, TrainedModel
from app.storage import build_companion_config_identifier, get_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_dprime")

FRAME_SIZE = 320  # 20ms at 16kHz
NUM_LIBRI_CLIPS = 60
SILENCE_DURATION_S = 5
LIBRI_ROOT = Path("/app/corpus/librispeech/LibriSpeech/dev-clean")


def _load_pcm16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        # crude linear resample — fine for backfill, real pipeline uses librosa
        from numpy import interp

        ratio = 16000 / sr
        new_len = int(round(len(audio) * ratio))
        audio = interp(
            np.linspace(0, len(audio) - 1, new_len), np.arange(len(audio)), audio
        ).astype("float32")
    return audio


def _max_score_for_file(detector, audio: np.ndarray) -> float:
    detector.reset()
    max_score = 0.0
    if len(audio) < FRAME_SIZE:
        audio = np.pad(audio, (0, FRAME_SIZE - len(audio)))
    for start in range(0, len(audio) - FRAME_SIZE + 1, FRAME_SIZE):
        frame = audio[start : start + FRAME_SIZE].astype("float32")
        score = float(detector.process(frame))
        if score > max_score:
            max_score = score
    return max_score


def _gather_libri_paths(n: int) -> list[Path]:
    paths = sorted(LIBRI_ROOT.rglob("*.flac"))
    return paths[:n]


def _score_distribution(detector, audio_paths: list[Path]) -> list[float]:
    scores: list[float] = []
    for p in audio_paths:
        try:
            audio = _load_pcm16k(p)
            scores.append(_max_score_for_file(detector, audio))
        except Exception as exc:
            logger.warning("Skipped %s (%s)", p, exc)
    return scores


def _compute_dprime(pos: list[float], neg: list[float]) -> float | None:
    if len(pos) < 2 or len(neg) < 2:
        return None
    pos_arr = np.asarray(pos, dtype=np.float64)
    neg_arr = np.asarray(neg, dtype=np.float64)
    pooled_var = max(
        (float(pos_arr.var(ddof=1)) + float(neg_arr.var(ddof=1))) / 2.0, 1e-6
    )
    return float((pos_arr.mean() - neg_arr.mean()) / (pooled_var**0.5))


async def _backfill_one(model_row: TrainedModel, session) -> bool:
    from violawake_sdk.wake_detector import WakeDetector

    storage = get_storage()
    if not storage.exists(model_row.file_path):
        logger.warning("Model %s file missing in storage; skipping", model_row.id)
        return False

    # Local path — files in trained_models.file_path are storage-relative;
    # /app/data is the local-storage root inside the backend container.
    local_path = Path("/app/data") / model_row.file_path
    if not local_path.exists():
        logger.warning("Could not locate ONNX file for model %s", model_row.id)
        return False

    detector = WakeDetector(
        model=str(local_path),
        threshold=0.5,
        cooldown_s=0.0,
    )

    rec_result = await session.execute(
        select(Recording).where(
            Recording.user_id == model_row.user_id,
            Recording.wake_word == model_row.wake_word,
        )
    )
    recordings = rec_result.scalars().all()
    pos_paths: list[Path] = []
    for rec in recordings:
        candidate = Path("/app/data") / rec.file_path
        if candidate.exists():
            pos_paths.append(candidate)
    if not pos_paths:
        logger.warning("No recordings on disk for user %s; skipping", model_row.user_id)
        return False

    logger.info(
        "Scoring %d positives + %d LibriSpeech clips for model %s",
        len(pos_paths),
        NUM_LIBRI_CLIPS,
        model_row.id,
    )
    pos_scores = _score_distribution(detector, pos_paths)

    neg_paths = _gather_libri_paths(NUM_LIBRI_CLIPS)
    neg_scores = _score_distribution(detector, neg_paths)

    silence = np.zeros(16000 * SILENCE_DURATION_S, dtype="float32")
    silence_score = _max_score_for_file(detector, silence)
    neg_scores.append(silence_score)

    d_prime = _compute_dprime(pos_scores, neg_scores)
    logger.info(
        "model %s: pos_n=%d pos_mean=%.3f neg_n=%d neg_mean=%.3f d_prime=%s",
        model_row.id,
        len(pos_scores),
        float(np.mean(pos_scores)) if pos_scores else float("nan"),
        len(neg_scores),
        float(np.mean(neg_scores)) if neg_scores else float("nan"),
        f"{d_prime:.3f}" if d_prime is not None else "n/a",
    )

    # Update config_json on disk + storage + DB row.
    config_data = json.loads(model_row.config_json) if model_row.config_json else {}
    if d_prime is not None:
        config_data["d_prime"] = round(d_prime, 2)
    config_data["positive_scores"] = [round(float(s), 6) for s in pos_scores]
    config_data["negative_scores"] = [round(float(s), 6) for s in neg_scores]
    config_data.setdefault("threshold", config_data.get("deployment_threshold", 0.5))
    config_data["far_per_hour"] = round(
        float(np.mean([1.0 if s >= 0.5 else 0.0 for s in neg_scores])) * 60, 4
    )
    config_data["frr"] = round(
        float(np.mean([0.0 if s >= 0.5 else 1.0 for s in pos_scores])), 4
    )

    new_config_bytes = json.dumps(config_data, indent=2).encode("utf-8")
    storage.upload(
        build_companion_config_identifier(model_row.file_path),
        new_config_bytes,
        "application/json",
    )

    model_row.config_json = json.dumps(config_data)
    if d_prime is not None:
        model_row.d_prime = round(d_prime, 4)
    await session.commit()
    return True


async def main() -> None:
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        result = await session.execute(select(TrainedModel))
        models = result.scalars().all()
        logger.info("Found %d trained_models row(s)", len(models))
        for m in models:
            try:
                await _backfill_one(m, session)
            except Exception as exc:
                logger.exception("Backfill failed for model %s: %s", m.id, exc)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
