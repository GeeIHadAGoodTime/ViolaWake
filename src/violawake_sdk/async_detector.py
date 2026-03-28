"""Async wrapper for WakeDetector.

Provides ``AsyncWakeDetector`` for use in asyncio-based applications.
All blocking ONNX inference is offloaded to a single-threaded
``ThreadPoolExecutor`` so the event loop stays responsive.

Example::

    import asyncio
    from violawake_sdk.async_detector import AsyncWakeDetector

    async def main():
        detector = AsyncWakeDetector(threshold=0.80)
        # ... feed audio frames ...
        detected = await detector.detect(frame)
        if detected:
            print("Wake word!")

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from violawake_sdk.confidence import ConfidenceResult
from violawake_sdk.wake_detector import (
    WakeDetector,
)


class AsyncWakeDetector:
    """Async wrapper around ``WakeDetector`` for asyncio-based applications.

    All CPU-bound inference is dispatched to a background thread via
    ``loop.run_in_executor``. The wrapper is fully transparent -- it
    accepts the same constructor arguments and exposes the same methods
    as ``WakeDetector``, but with ``async`` signatures.

    Args:
        **kwargs: Forwarded to ``WakeDetector.__init__``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._detector = WakeDetector(**kwargs)
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    async def __aenter__(self) -> AsyncWakeDetector:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Exit async context manager, shutting down the executor."""
        self.close()

    async def detect(
        self,
        audio_frame: bytes | np.ndarray,
        is_playing: bool = False,
    ) -> bool:
        """Async version of ``WakeDetector.detect``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            lambda: self._detector.detect(audio_frame, is_playing),
        )

    async def process(
        self,
        audio_frame: bytes | np.ndarray,
    ) -> float:
        """Async version of ``WakeDetector.process``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            lambda: self._detector.process(audio_frame),
        )

    async def stream(
        self,
        source: AsyncIterator[bytes | np.ndarray],
    ) -> AsyncIterator[bool]:
        """Async generator that yields detection results from an async audio source.

        Usage::

            async for detected in detector.stream(audio_source):
                if detected:
                    print("Wake word!")

        Args:
            source: An async iterator yielding audio frames.

        Yields:
            Boolean detection result for each frame.
        """
        async for frame in source:
            yield await self.detect(frame)

    def reset_cooldown(self) -> None:
        """Reset the cooldown window (delegates to WakeDetector public API)."""
        self._detector.reset_cooldown()

    @property
    def threshold(self) -> float:
        """Current detection threshold."""
        return self._detector.threshold

    def get_confidence(self) -> ConfidenceResult:
        """Return confidence assessment of the current detection state (K2)."""
        return self._detector.get_confidence()

    @property
    def last_scores(self) -> tuple[float, ...]:
        """Return the recent score history (most recent last)."""
        return self._detector.last_scores

    def close(self) -> None:
        """Shut down the background executor and release detector resources.

        Safe to call multiple times.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._detector.close()

    def __del__(self) -> None:
        # Use wait=False in __del__ to avoid blocking the GC/finalizer thread.
        # The explicit close() method still uses wait=True for graceful shutdown.
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
