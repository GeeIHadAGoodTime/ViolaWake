"""
Playwright E2E test fixtures for ViolaWake Console.

Manages backend + frontend lifecycle for fully automated testing.
Uses subprocess to start/stop both servers.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import pytest

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "console" / "backend"
FRONTEND_DIR = PROJECT_ROOT / "console" / "frontend"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# Test user credentials
TEST_USER_EMAIL = "test@violawake.dev"
TEST_USER_PASSWORD = "TestPass123!"
TEST_USER_NAME = "Test User"


def _port_in_use(port: int) -> bool:
    """Check if a port is already in use (checks both IPv4 and IPv6)."""
    for family, addr in [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                if s.connect_ex((addr, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if _port_in_use(port):
            return True
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def backend_server() -> Generator[str, None, None]:
    """Start the FastAPI backend server for the test session."""
    if _port_in_use(BACKEND_PORT):
        # Backend already running (dev mode) — use it
        yield BACKEND_URL
        return

    env = os.environ.copy()
    env["VIOLAWAKE_DB_PATH"] = str(BACKEND_DIR / "data" / "test.db")
    env["VIOLAWAKE_SECRET_KEY"] = "test-secret-key-for-e2e"

    proc = subprocess.Popen(
        [sys.executable, "run.py"],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    if not _wait_for_port(BACKEND_PORT, timeout=30):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Backend failed to start within 30s.\n"
            f"stdout: {stdout.decode()[:500]}\n"
            f"stderr: {stderr.decode()[:500]}"
        )

    yield BACKEND_URL

    # Teardown
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()
    proc.wait(timeout=10)

    # Clean up test database
    test_db = BACKEND_DIR / "data" / "test.db"
    if test_db.exists():
        test_db.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def frontend_server(backend_server: str) -> Generator[str, None, None]:
    """Start the Vite dev server for the test session."""
    if _port_in_use(FRONTEND_PORT):
        yield FRONTEND_URL
        return

    env = os.environ.copy()
    env["VITE_API_URL"] = backend_server

    # Use npm/npx to start Vite
    proc = subprocess.Popen(
        ["npx", "vite", "--port", str(FRONTEND_PORT)],
        cwd=str(FRONTEND_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    if not _wait_for_port(FRONTEND_PORT, timeout=60):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Frontend failed to start within 60s.\n"
            f"stdout: {stdout.decode()[:500]}\n"
            f"stderr: {stderr.decode()[:500]}"
        )

    yield FRONTEND_URL

    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def console_servers(backend_server: str, frontend_server: str) -> dict[str, str]:
    """Both servers running — convenience fixture."""
    return {"backend": backend_server, "frontend": frontend_server}


# ── Audio fixtures for fake recording ────────────────────────────────────────

@pytest.fixture
def fake_wake_word_wav(tmp_path: Path) -> Path:
    """Generate a fake 1.5s wake word WAV file (440Hz sine wave)."""
    import wave

    import numpy as np

    sample_rate = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440Hz sine wave with some variation to simulate speech
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    signal += np.sin(2 * np.pi * 220 * t) * 0.3  # harmonic
    signal += np.random.default_rng(42).normal(0, 0.05, len(t))  # noise
    pcm = (signal * 32767).astype(np.int16)

    wav_path = tmp_path / "test_wake_word.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return wav_path


@pytest.fixture
def fake_wake_word_wavs(tmp_path: Path) -> list[Path]:
    """Generate 10 fake wake word WAV files for a complete recording session."""
    import wave

    import numpy as np

    rng = np.random.default_rng(42)
    sample_rate = 16000
    duration = 1.5
    wavs = []

    for i in range(10):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Vary frequency slightly per sample (simulates natural variation)
        freq = 440 + rng.integers(-20, 20)
        signal = np.sin(2 * np.pi * freq * t) * (0.4 + rng.random() * 0.2)
        signal += rng.normal(0, 0.05, len(t))
        pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)

        wav_path = tmp_path / f"sample_{i:02d}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        wavs.append(wav_path)

    return wavs
