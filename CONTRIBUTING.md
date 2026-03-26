# Contributing to ViolaWake

Thank you for your interest in contributing to ViolaWake. This guide covers the development workflow, code standards, and PR process.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/GeeIHadAGoodTime/ViolaWake
cd ViolaWake

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Optional extras

```bash
# Full install (all features + training)
pip install -e ".[all]"

# Training pipeline only
pip install -e ".[training]"

# TTS only
pip install -e ".[tts]"
```

## Architecture Overview

The project has three main components:

```
src/violawake_sdk/      # The pip-installable SDK
  detector.py           # WakeDetector — core wake word inference
  vad.py                # VADEngine — voice activity detection
  tts.py                # TTSEngine — Kokoro-82M TTS wrapper
  stt.py                # STTEngine — faster-whisper STT wrapper
  pipeline.py           # VoicePipeline — integrated Wake+STT+TTS
  models.py             # Model registry, download, SHA-256 verification
  tools/                # CLI tools (train, evaluate, collect, download)

console/                # Web Console for browser-based training
  backend/              # FastAPI backend (auth, upload, training jobs)
  frontend/             # React+Vite SPA (recording UI, training progress)

tests/                  # Test suites (unit, integration, benchmark)
```

**SDK** is the core library users install via `pip install violawake`. It must stay lean -- no heavy dependencies in the base install.

**Console** is the web application for users who prefer browser-based wake word training over the CLI.

**Tools** are CLI scripts (`violawake-train`, `violawake-eval`, etc.) that ship with the SDK.

## Code Style

### Linting and formatting

We use **ruff** for linting and formatting, and **mypy** (strict mode) for type checking.

```bash
# Lint
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Format
ruff format .

# Type check
mypy src/
```

### Key rules

- **No f-strings in logging** (ruff G004): Use `logger.info("msg %s", var)` not `logger.info(f"msg {var}")`
- **No print() in library code** (ruff T201): Use `logging.getLogger(__name__)`. Print is allowed in `tools/` scripts.
- **Type hints required** on all public functions (mypy strict)
- **Python 3.10+** -- use modern syntax (`X | None` not `Optional[X]`, `list[str]` not `List[str]`)
- Target line length: 100 characters

### Import order

```python
from __future__ import annotations

# 1. Standard library
import logging
from pathlib import Path
from typing import TYPE_CHECKING

# 2. Third-party
import numpy as np
import onnxruntime as ort

# 3. Local
from violawake_sdk.models import get_model_path
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=violawake_sdk --cov-report=term-missing

# Skip tests that require model files
pytest -m "not integration"

# Run benchmarks
pytest -m benchmark --benchmark-only
```

### Test categories

- **Unit tests** -- no external dependencies, fast, always run in CI
- **Integration tests** (marked `@pytest.mark.integration`) -- require model files on disk
- **Benchmark tests** (marked `@pytest.mark.benchmark`) -- performance measurements
- **Hardware tests** (marked `@pytest.mark.hardware`) -- require a microphone, never run in CI

Write tests for new functionality. Aim for coverage on the happy path and key error paths.

## PR Process

1. **Fork** the repo and create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes.** Follow the code style rules above.

3. **Run checks locally** before pushing:
   ```bash
   ruff check .
   ruff format --check .
   mypy src/
   pytest
   ```

4. **Push** and open a pull request against `main`.

5. **PR description** should include:
   - What the change does and why
   - How to test it
   - Any breaking changes

6. **CI** will run lint, type check, and tests automatically. All checks must pass.

7. **Review** -- a maintainer will review your PR. Address feedback, then it gets merged.

### Commit messages

Use conventional-style prefixes:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding or updating tests
- `refactor:` code change that neither fixes a bug nor adds a feature
- `chore:` build process, dependency updates, etc.

## Reporting Issues

Use [GitHub Issues](https://github.com/GeeIHadAGoodTime/ViolaWake/issues). Include:

- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error output

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
