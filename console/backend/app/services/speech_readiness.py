"""Can training make speech right now?

Every training job needs synthesized speech: TTS positives, confusable negatives,
and the quality gate's own probes. When that dependency dies, the service still
looks healthy: from 2026-10-14 to 2026-11-06 every job failed after about 55
minutes while ``/api/health`` said ok, because edge-tts had started getting 403s
and nothing asked the question this module asks.

The answer is cached, because the check makes a real synthesis call. A failed
answer is cached for less time than a good one, so a recovery is noticed quickly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings

logger = logging.getLogger("violawake.training")

READY_CACHE_SECONDS = 300.0
NOT_READY_CACHE_SECONDS = 60.0
# A check that hangs must not hang the request that asked. edge-tts has no timeout
# of its own on a stalled socket.
PROBE_TIMEOUT_SECONDS = 45.0

TRAINING_UNAVAILABLE_MESSAGE = (
    "Training is down on our side right now. Your recordings are fine and this "
    "attempt was not used. Please try again in a little while."
)


@dataclass(frozen=True, slots=True)
class SpeechReadiness:
    ready: bool
    backend: str | None
    checked_at: float
    detail: str | None = None

    def public_payload(self) -> dict[str, Any]:
        """What anyone may see: no provider errors, no internals."""
        return {
            "ready": self.ready,
            # What the off-box uptime worker reads: it pages on status "down".
            "status": "ok" if self.ready else "down",
            "backend": self.backend,
            "message": None if self.ready else TRAINING_UNAVAILABLE_MESSAGE,
        }


_lock = asyncio.Lock()
_cached: SpeechReadiness | None = None


def reset_speech_readiness_cache() -> None:
    global _cached
    _cached = None


def _probe_blocking() -> SpeechReadiness:
    from violawake_sdk.tools.train import probe_speech_synthesis

    probe = probe_speech_synthesis(tmp_dir=settings.tmp_dir)
    backend = "edge_tts" if probe.edge_tts else "kokoro" if probe.kokoro else None
    return SpeechReadiness(
        ready=probe.available,
        backend=backend,
        checked_at=time.monotonic(),
        detail=probe.edge_tts_error,
    )


def _fresh(cached: SpeechReadiness | None) -> bool:
    if cached is None:
        return False
    max_age = READY_CACHE_SECONDS if cached.ready else NOT_READY_CACHE_SECONDS
    return time.monotonic() - cached.checked_at < max_age


async def get_speech_readiness(*, force: bool = False) -> SpeechReadiness:
    """Return the cached answer, or run the real check when it is stale."""
    global _cached

    if not force and _fresh(_cached):
        return _cached  # type: ignore[return-value]

    async with _lock:
        if not force and _fresh(_cached):
            return _cached  # type: ignore[return-value]
        previous = _cached
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_probe_blocking),
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 -- a failed check IS the answer
            result = SpeechReadiness(
                ready=False,
                backend=None,
                checked_at=time.monotonic(),
                detail=f"{type(exc).__name__}: {exc}",
            )
        if not result.ready and (previous is None or previous.ready):
            logger.error(
                "Training cannot synthesize speech with any backend: %s", result.detail
            )
        elif result.ready and previous is not None and not previous.ready:
            logger.warning("Training can synthesize speech again (%s)", result.backend)
        _cached = result
        return result


__all__ = [
    "SpeechReadiness",
    "TRAINING_UNAVAILABLE_MESSAGE",
    "get_speech_readiness",
    "reset_speech_readiness_cache",
]
