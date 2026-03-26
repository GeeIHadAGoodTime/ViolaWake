#!/usr/bin/env python3
"""
Launch the ViolaWake Console (backend + frontend) for local development.

Usage:
    python console/launch.py              # Start both servers
    python console/launch.py --backend    # Backend only (port 8000)
    python console/launch.py --frontend   # Frontend only (port 5173)
    python console/launch.py --install    # Install all dependencies first
    python console/launch.py --test       # Start servers + run E2E tests

This script handles:
  - Installing Python + Node dependencies (--install)
  - Starting FastAPI backend (uvicorn)
  - Starting Vite frontend dev server
  - Clean shutdown on Ctrl+C
  - Health checks before declaring ready
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "console" / "backend"
FRONTEND_DIR = PROJECT_ROOT / "console" / "frontend"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173

processes: list[subprocess.Popen] = []


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def wait_for_port(port: int, name: str, timeout: float = 30.0) -> bool:
    print(f"  Waiting for {name} on port {port}...", end="", flush=True)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if not port_available(port):
            print(f" ready ({time.monotonic() - start:.1f}s)")
            return True
        time.sleep(0.5)
    print(" TIMEOUT")
    return False


def install_deps() -> None:
    """Install all dependencies for backend and frontend."""
    print("=" * 50)
    print("  Installing dependencies")
    print("=" * 50)

    # Backend Python deps
    req_file = BACKEND_DIR / "requirements.txt"
    if req_file.exists():
        print("\n[Backend] Installing Python dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        )
    else:
        print(f"\n[Backend] No requirements.txt at {req_file}")

    # Also install the SDK itself (needed for training)
    print("\n[SDK] Installing ViolaWake SDK...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", f"{PROJECT_ROOT}[training,dev]"],
    )

    # Frontend Node deps
    pkg_json = FRONTEND_DIR / "package.json"
    if pkg_json.exists():
        print("\n[Frontend] Installing Node dependencies...")
        subprocess.check_call(
            ["npm", "install"],
            cwd=str(FRONTEND_DIR),
            shell=True,
        )
    else:
        print(f"\n[Frontend] No package.json at {pkg_json}")

    print("\nAll dependencies installed.")


def start_backend() -> subprocess.Popen | None:
    """Start the FastAPI backend."""
    if not port_available(BACKEND_PORT):
        print(f"[Backend] Port {BACKEND_PORT} already in use — skipping")
        return None

    run_py = BACKEND_DIR / "run.py"
    if not run_py.exists():
        print(f"[Backend] ERROR: {run_py} not found")
        return None

    print(f"[Backend] Starting on port {BACKEND_PORT}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, "run.py"],
        cwd=str(BACKEND_DIR),
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    processes.append(proc)
    return proc


def start_frontend() -> subprocess.Popen | None:
    """Start the Vite dev server."""
    if not port_available(FRONTEND_PORT):
        print(f"[Frontend] Port {FRONTEND_PORT} already in use — skipping")
        return None

    pkg_json = FRONTEND_DIR / "package.json"
    if not pkg_json.exists():
        print(f"[Frontend] ERROR: {pkg_json} not found")
        return None

    print(f"[Frontend] Starting on port {FRONTEND_PORT}...")
    proc = subprocess.Popen(
        ["npx", "vite", "--port", str(FRONTEND_PORT), "--host"],
        cwd=str(FRONTEND_DIR),
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    processes.append(proc)
    return proc


def shutdown() -> None:
    """Shutdown all child processes."""
    print("\nShutting down...")
    for proc in processes:
        if proc.poll() is None:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("All processes stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch ViolaWake Console for development",
    )
    parser.add_argument("--backend", action="store_true", help="Backend only")
    parser.add_argument("--frontend", action="store_true", help="Frontend only")
    parser.add_argument("--install", action="store_true", help="Install deps first")
    parser.add_argument("--test", action="store_true", help="Start + run E2E tests")
    args = parser.parse_args()

    if args.install:
        install_deps()
        if not args.backend and not args.frontend and not args.test:
            return

    start_be = not args.frontend
    start_fe = not args.backend

    print("=" * 50)
    print("  ViolaWake Console")
    print("=" * 50)

    try:
        if start_be:
            start_backend()
            if not wait_for_port(BACKEND_PORT, "backend", timeout=30):
                print("Backend failed to start!")
                shutdown()
                sys.exit(1)

        if start_fe:
            start_frontend()
            if not wait_for_port(FRONTEND_PORT, "frontend", timeout=60):
                print("Frontend failed to start!")
                shutdown()
                sys.exit(1)

        print()
        print("=" * 50)
        if start_be:
            print(f"  Backend:  http://localhost:{BACKEND_PORT}")
            print(f"  API docs: http://localhost:{BACKEND_PORT}/docs")
        if start_fe:
            print(f"  Frontend: http://localhost:{FRONTEND_PORT}")
        print("=" * 50)
        print("\nPress Ctrl+C to stop.\n")

        if args.test:
            # Run E2E tests
            print("Running E2E tests...")
            rc = subprocess.call(
                [sys.executable, "console/run_e2e.py", "--api-only"],
                cwd=str(PROJECT_ROOT),
            )
            shutdown()
            sys.exit(rc)

        # Keep running until Ctrl+C
        while True:
            # Check if any process died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"Process {proc.pid} exited with code {proc.returncode}")
                    shutdown()
                    sys.exit(1)
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
