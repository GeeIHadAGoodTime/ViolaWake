"""
Playwright-specific configuration for browser E2E tests.

This file is auto-loaded by pytest-playwright. It configures:
  - Chromium with fake audio device (for recording tests)
  - Browser context with microphone permissions
  - Fake WAV file generation for audio input
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def browser_type_launch_args() -> dict:
    """Configure Chromium to use a fake audio capture device.

    This tells Chromium to use a generated audio file instead of
    a real microphone — critical for CI/headless testing.
    """
    # Generate fake audio file
    fake_audio = _generate_fake_audio()

    return {
        "args": [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={fake_audio}",
            "--no-sandbox",
            "--disable-web-security",
            "--allow-file-access-from-files",
        ],
    }


@pytest.fixture(scope="session")
def browser_context_args() -> dict:
    """Configure browser context with microphone permissions."""
    return {
        "permissions": ["microphone"],
        "viewport": {"width": 1280, "height": 720},
    }


def _generate_fake_audio() -> str:
    """Generate a WAV file that Chromium will use as fake microphone input.

    Returns the path to the generated WAV file.
    The file contains a speech-like signal (multiple harmonics + noise)
    that will produce non-silent recordings in the browser.
    """
    tmp_dir = Path.home() / ".violawake" / "test"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / "fake_mic_input.wav"

    if wav_path.exists():
        return str(wav_path)

    sample_rate = 48000  # Browser's default sample rate
    duration = 30.0  # Long enough for 10 recordings
    rng = np.random.default_rng(42)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a speech-like signal with multiple harmonics
    signal = np.zeros_like(t)
    # Fundamental + harmonics (simulates vowel sounds)
    for freq, amp in [(200, 0.3), (400, 0.2), (800, 0.15), (1200, 0.1), (2400, 0.05)]:
        signal += np.sin(2 * np.pi * freq * t) * amp

    # Add natural variation (simulates speaking)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)  # 2Hz modulation
    signal *= envelope

    # Add small noise
    signal += rng.normal(0, 0.02, len(t))

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    pcm = (signal * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return str(wav_path)
