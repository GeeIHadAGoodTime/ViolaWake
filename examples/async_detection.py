"""Async wake word detection example using AsyncWakeDetector.

Shows the ``async with`` context manager pattern and how to feed
audio frames from an async source.
"""

import asyncio
import numpy as np
from violawake_sdk import AsyncWakeDetector


async def fake_audio_source():
    """Simulate an async audio stream (320 samples = 20ms at 16kHz)."""
    for _ in range(500):
        yield np.random.randn(320).astype(np.float32) * 0.01
        await asyncio.sleep(0.02)


async def main() -> None:
    async with AsyncWakeDetector(model="temporal_cnn", threshold=0.80) as detector:
        print("Listening for 'Viola'...")
        async for frame in fake_audio_source():
            detected = await detector.detect(frame)
            if detected:
                conf = detector.get_confidence()
                print(f"Wake word detected! confidence={conf.confidence}")
                break
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
