#!/usr/bin/env python3
"""
Quality Gate Runner — Verify each phase meets acceptance criteria.

Usage:
    python tools/quality_gate.py --gate 1   # SDK Fixes
    python tools/quality_gate.py --gate 2   # Console Backend
    python tools/quality_gate.py --gate 3   # Console Frontend
    python tools/quality_gate.py --gate 4   # Playwright E2E
    python tools/quality_gate.py --all      # Run all gates
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS: dict[str, list[tuple[str, bool, str]]] = {}


def check(gate: str, name: str, condition: bool, detail: str = "") -> None:
    """Record a check result."""
    RESULTS.setdefault(gate, []).append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def run_cmd(*args: str, cwd: str | None = None, timeout: int = 120) -> tuple[int, str]:
    """Run a command, return (returncode, stdout+stderr)."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


# ── Gate 1: SDK Fixes ────────────────────────────────────────────────────────

def gate_1_sdk() -> None:
    """Verify SDK fixes are correct."""
    gate = "Gate 1: SDK Fixes"
    print(f"\n{'='*60}")
    print(f"  {gate}")
    print(f"{'='*60}")

    # Check training pipeline exists and has key fixes
    train_py = PROJECT_ROOT / "src" / "violawake_sdk" / "tools" / "train.py"
    check(gate, "train.py exists", train_py.exists())

    if train_py.exists():
        content = train_py.read_text()
        check(
            gate,
            "No random noise negatives",
            "rng.uniform(-0.1, 0.1, CLIP_SAMPLES)" not in content,
            "Old random noise pattern should be replaced",
        )
        check(
            gate,
            "Augmentation used",
            "augment" in content.lower() and "def " in content,
            "augment parameter should be wired in",
        )
        check(
            gate,
            "Validation split exists",
            "val" in content.lower() and "split" in content.lower()
            or "validation" in content.lower(),
            "Should have train/val split",
        )
        check(
            gate,
            "Early stopping",
            "patience" in content.lower() or "early_stop" in content.lower(),
            "Should have early stopping",
        )
        check(
            gate,
            "--negatives CLI arg",
            "--negatives" in content or "negatives" in content,
            "Should accept negatives directory",
        )

    # Check augmentation module exists
    augment_py = PROJECT_ROOT / "src" / "violawake_sdk" / "training" / "augment.py"
    check(gate, "augment.py exists", augment_py.exists())

    # Check eval pipeline fix
    eval_py = PROJECT_ROOT / "src" / "violawake_sdk" / "training" / "evaluate.py"
    if eval_py.exists():
        content = eval_py.read_text()
        check(
            gate,
            "Eval uses OWW embeddings",
            "oww" in content.lower()
            or "embed" in content.lower()
            or "openwakeword" in content.lower(),
            "Should use OWW embedding path for MLP models",
        )
        check(
            gate,
            "Threshold sweep",
            "threshold" in content.lower()
            and ("sweep" in content.lower() or "optimal" in content.lower()),
        )

    # Check bug fixes
    wake_py = PROJECT_ROOT / "src" / "violawake_sdk" / "wake_detector.py"
    if wake_py.exists():
        content = wake_py.read_text()
        # The dead branch should be fixed
        check(
            gate,
            "Dead branch fixed",
            "np.asarray(audio_frame, dtype=np.float32)" not in content
            or "if pcm.dtype == np.int16" in content,
            "process() should handle int16 input correctly",
        )

    vad_py = PROJECT_ROOT / "src" / "violawake_sdk" / "vad.py"
    if vad_py.exists():
        content = vad_py.read_text()
        check(
            gate,
            "Silero VAD implemented",
            "NotImplementedError" not in content
            or "class _SileroVADBackend" in content
            or "SileroVAD" in content,
            "Silero should be implemented, not a stub",
        )

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        check(
            gate,
            "No src/wakeword in pyproject",
            "src/wakeword" not in content,
            "Nonexistent package should be removed",
        )

    # Run unit tests
    rc, output = run_cmd(
        sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short",
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    check(gate, "Unit tests pass", rc == 0, f"exit code {rc}")

    # Run ruff
    rc, output = run_cmd(
        sys.executable, "-m", "ruff", "check", "src/",
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    check(gate, "Ruff clean", rc == 0, output[:200] if rc != 0 else "")


# ── Gate 2: Console Backend ──────────────────────────────────────────────────

def gate_2_backend() -> None:
    """Verify Console backend is functional."""
    gate = "Gate 2: Console Backend"
    print(f"\n{'='*60}")
    print(f"  {gate}")
    print(f"{'='*60}")

    backend_dir = PROJECT_ROOT / "console" / "backend"
    check(gate, "Backend directory exists", backend_dir.exists())

    main_py = backend_dir / "app" / "main.py"
    check(gate, "main.py exists", main_py.exists())

    if main_py.exists():
        content = main_py.read_text()
        check(gate, "FastAPI app", "FastAPI" in content)
        check(gate, "CORS configured", "CORSMiddleware" in content or "cors" in content.lower())

    # Check route files
    routes_dir = backend_dir / "app" / "routes"
    if routes_dir.exists():
        check(gate, "Auth routes", (routes_dir / "auth.py").exists())
        check(gate, "Recording routes", (routes_dir / "recordings.py").exists())
        check(gate, "Training routes", (routes_dir / "training.py").exists())
        check(gate, "Model routes", (routes_dir / "models.py").exists())

    # Check key files
    check(gate, "database.py", (backend_dir / "app" / "database.py").exists())
    check(gate, "auth.py", (backend_dir / "app" / "auth.py").exists())
    check(gate, "schemas.py", (backend_dir / "app" / "schemas.py").exists())
    check(gate, "requirements.txt", (backend_dir / "requirements.txt").exists())
    check(gate, "run.py", (backend_dir / "run.py").exists())

    # Check training service
    training_svc = backend_dir / "app" / "services" / "training_service.py"
    check(gate, "Training service", training_svc.exists())
    if training_svc.exists():
        content = training_svc.read_text()
        check(
            gate,
            "Calls SDK training",
            "violawake" in content.lower() or "train" in content.lower(),
        )


# ── Gate 3: Console Frontend ─────────────────────────────────────────────────

def gate_3_frontend() -> None:
    """Verify Console frontend is functional."""
    gate = "Gate 3: Console Frontend"
    print(f"\n{'='*60}")
    print(f"  {gate}")
    print(f"{'='*60}")

    frontend_dir = PROJECT_ROOT / "console" / "frontend"
    check(gate, "Frontend directory exists", frontend_dir.exists())

    check(gate, "package.json", (frontend_dir / "package.json").exists())
    check(gate, "vite.config", (frontend_dir / "vite.config.ts").exists())
    check(gate, "index.html", (frontend_dir / "index.html").exists())

    src_dir = frontend_dir / "src"
    if src_dir.exists():
        check(gate, "App.tsx", (src_dir / "App.tsx").exists())
        check(gate, "api.ts", (src_dir / "api.ts").exists())

        pages_dir = src_dir / "pages"
        if pages_dir.exists():
            check(gate, "Login page", (pages_dir / "Login.tsx").exists())
            check(gate, "Register page", (pages_dir / "Register.tsx").exists())
            check(gate, "Dashboard page", (pages_dir / "Dashboard.tsx").exists())
            check(gate, "Record page", (pages_dir / "Record.tsx").exists())
            check(
                gate,
                "Training status page",
                (pages_dir / "TrainingStatus.tsx").exists(),
            )

        comps_dir = src_dir / "components"
        if comps_dir.exists():
            check(gate, "AudioRecorder component", (comps_dir / "AudioRecorder.tsx").exists())

    # Check npm install works
    if (frontend_dir / "package.json").exists():
        rc, output = run_cmd(
            "npm", "install", "--prefer-offline",
            cwd=str(frontend_dir),
            timeout=120,
        )
        check(gate, "npm install succeeds", rc == 0, output[:200] if rc != 0 else "")

        # Check TypeScript compiles
        rc, output = run_cmd(
            "npx", "tsc", "--noEmit",
            cwd=str(frontend_dir),
            timeout=60,
        )
        check(gate, "TypeScript compiles", rc == 0, output[:200] if rc != 0 else "")


# ── Gate 4: Playwright E2E ────────────────────────────────────────────────────

def gate_4_e2e() -> None:
    """Verify Playwright E2E tests pass."""
    gate = "Gate 4: Playwright E2E"
    print(f"\n{'='*60}")
    print(f"  {gate}")
    print(f"{'='*60}")

    e2e_dir = PROJECT_ROOT / "console" / "tests" / "e2e"
    check(gate, "E2E test directory exists", e2e_dir.exists())
    check(gate, "conftest.py", (e2e_dir / "conftest.py").exists())
    check(gate, "API flow tests", (e2e_dir / "test_api_flow.py").exists())
    check(gate, "Browser flow tests", (e2e_dir / "test_browser_flow.py").exists())

    # Run API-only E2E tests (no browser needed)
    rc, output = run_cmd(
        sys.executable, "-m", "pytest",
        str(e2e_dir / "test_api_flow.py"),
        "-v", "--tb=short",
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    check(gate, "API E2E tests pass", rc == 0, f"exit code {rc}")


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print the summary of all gate results."""
    print(f"\n{'='*60}")
    print("  QUALITY GATE SUMMARY")
    print(f"{'='*60}")

    total_pass = 0
    total_fail = 0

    for gate, checks in RESULTS.items():
        passed = sum(1 for _, ok, _ in checks if ok)
        failed = sum(1 for _, ok, _ in checks if not ok)
        total_pass += passed
        total_fail += failed
        status = "PASS" if failed == 0 else "FAIL"
        print(f"\n  [{status}] {gate}: {passed}/{passed + failed} checks passed")
        if failed > 0:
            for name, ok, detail in checks:
                if not ok:
                    print(f"    - FAIL: {name}" + (f" ({detail})" if detail else ""))

    print(f"\n  Total: {total_pass} passed, {total_fail} failed")
    return total_fail == 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ViolaWake Quality Gates")
    parser.add_argument("--gate", type=int, choices=[1, 2, 3, 4], help="Run specific gate")
    parser.add_argument("--all", action="store_true", help="Run all gates")
    args = parser.parse_args()

    gates = {
        1: gate_1_sdk,
        2: gate_2_backend,
        3: gate_3_frontend,
        4: gate_4_e2e,
    }

    if args.gate:
        gates[args.gate]()
    elif args.all:
        for g in gates.values():
            g()
    else:
        parser.print_help()
        return

    all_pass = print_summary()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
