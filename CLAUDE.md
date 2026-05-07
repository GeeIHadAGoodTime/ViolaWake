# CLAUDE.md — ViolaWake SDK

This file provides AI assistants with context, patterns, and rules for the ViolaWake SDK project.

## Project Overview

**ViolaWake SDK** is a standalone Python SDK for on-device voice processing. It extracts the production-hardened voice components from the Viola assistant application into a clean, pip-installable package.

**Core products:**
1. **ViolaWake** — Wake word detection (primary differentiator, TemporalCNN d'=8.577, EER=0.8%)
2. **Kokoro TTS** — On-device sentence-chunked text-to-speech (Apache 2.0 model)
3. **Whisper STT** — Batch speech-to-text via faster-whisper
4. **VAD** — Voice activity detection (WebRTC / Silero / RMS)
5. **VoicePipeline** — Bundled Wake→STT→TTS pipeline class

**Source of truth for competitive context:** `docs/PRD.md` and the three audit docs in the Viola repo:
- `J:\PROJECTS\NOVVIOLA_fixed3_patched\NOVVIOLA\.viola\agents\COMPETITIVE_AUDIT_REPORT.md`
- `J:\PROJECTS\NOVVIOLA_fixed3_patched\NOVVIOLA\.viola\agents\audit_viola_inventory.md`
- `J:\PROJECTS\NOVVIOLA_fixed3_patched\NOVVIOLA\.viola\agents\audit_gap_analysis.md`

## Relationship to NOVVIOLA (Viola assistant)

ViolaWake is **standalone**. NOVVIOLA can use ViolaWake as a Python dependency (the wake-word SDK), but ViolaWake's API, database, infrastructure, and secrets are NOT shared with NOVVIOLA. Concretely:

- ViolaWake runs in its own Docker stack: `wakeword-backend-1` + `wakeword-postgres-1` + `wakeword-tunnel-1` (Cloudflare Tunnel `violawake-api`, UUID `7dbef1da-74e3-4d7f-bba9-aad4a3e72150`)
- ViolaWake env vars use `VIOLAWAKE_*` prefix; NOVVIOLA uses `VIOLA_*`. No crossover.
- ViolaWake's frontend is on Cloudflare Pages (project `violawake`, domain `violawake.com`); NOVVIOLA's UI is separate.
- Cross-project edits should use the SDK as the only contract. **Never hardcode NOVVIOLA URLs, env vars, or DB references into the ViolaWake repo.**

Full architecture and deploy mechanics: `docs/DEPLOYMENT.md`. Current live state and what's verified: `docs/PRODUCTION_STATUS.md`.

## Repository Layout

```
violawake/
├── README.md                   # Main product README, positioning, quick start
├── CLAUDE.md                   # This file — AI context
├── pyproject.toml              # Package metadata, deps, tool config
├── .gitignore
├── docker-compose.production.yml  # Backend + postgres + tunnel stack (production)
├── .env.production             # Production env vars (NOT in git — gitignored)
├── docs/
│   ├── REGISTRY.md             # Doc routing table — consult before reading docs
│   ├── PRD.md                  # Product requirements — source of truth for WHAT we build
│   ├── DEPLOYMENT.md           # How the live deploy actually works (frontend + backend + tunnel)
│   ├── PRODUCTION_STATUS.md    # Current live state, what's verified, known issues
│   ├── TEST_STRATEGY.md        # Testing philosophy, tiers, CI integration
│   └── adr/
│       ├── ADR-001-onnx-runtime.md         # Why ONNX Runtime for inference
│       ├── ADR-002-oww-feature-extractor.md # Why OpenWakeWord embeddings as backbone
│       ├── ADR-003-python-first.md          # Why Python SDK first, not C
│       ├── ADR-004-open-core.md             # Open-core licensing strategy
│       └── ADR-005-packaging.md             # PyPI packaging + model distribution
├── console/                    # SaaS console — NOT the SDK
│   ├── backend/                # FastAPI backend (auth, billing, recordings, training, models, teams)
│   ├── frontend/               # React+Vite frontend (deployed to Cloudflare Pages)
│   ├── docker-compose.yml      # Local-dev compose
│   └── Dockerfile.backend      # Production backend image (referenced from ../docker-compose.production.yml)
├── tests/
│   ├── live/                   # Live smoke + E2E tests against deployed instance — see tests/live/README.md
│   └── ...
├── src/
│   └── violawake_sdk/
│       ├── __init__.py         # Public API surface
│       ├── wake_detector.py    # WakeDetector class
│       ├── vad.py              # VADEngine class
│       ├── tts.py              # TTSEngine (Kokoro) class
│       ├── stt.py              # STTEngine (faster-whisper) class
│       ├── pipeline.py         # VoicePipeline class
│       ├── models.py           # Model download + caching utilities
│       ├── audio.py            # Mic capture + audio utilities
│       └── tools/
│           ├── train.py        # Training CLI
│           ├── evaluate.py     # d' / FAR / FRR evaluation CLI
│           ├── collect_samples.py   # Sample collection tool
│           └── download_model.py    # Model download CLI
├── tests/
│   ├── conftest.py
│   ├── unit/                   # Pure unit tests — no hardware, no models
│   ├── integration/            # Tests needing model files
│   └── benchmarks/             # Performance benchmarks
├── tools/
│   └── dc_sign_measure.py      # DC offset + sign measurement tool (ported from Viola)
└── .github/
    └── workflows/
        ├── ci.yml              # Main CI (lint + unit + integration)
        └── release.yml         # PyPI release automation
```

## Coding Patterns

### Python version
Target: Python 3.10+. Use `match/case` only where it adds clarity. All type hints use PEP 604 (`X | None` not `Optional[X]`).

### Import order
```python
from __future__ import annotations

# stdlib
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

# third-party
import numpy as np
import onnxruntime as ort

# local
from violawake_sdk.audio import chunk_mic_audio
from violawake_sdk.models import get_model_path

if TYPE_CHECKING:
    from violawake_sdk.pipeline import VoicePipeline
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Use % formatting (not f-strings) — compatible with lazy evaluation
logger.info("Processing frame %d, score=%.3f", frame_idx, score)
```

### Public API surface
Everything exported from `src/violawake_sdk/__init__.py` is public and subject to semantic versioning. Internal implementation details (`_helpers.py`, `_feature_extractor.py`) are private and can change without a minor version bump.

### Error handling
```python
class ViolaWakeError(Exception):
    """Base exception for ViolaWake SDK."""

class ModelNotFoundError(ViolaWakeError):
    """Model file not found or not downloaded."""

class AudioCaptureError(ViolaWakeError):
    """Microphone capture failed."""
```

All public API methods should raise specific exceptions from this hierarchy, never bare `Exception`.

### ONNX inference pattern
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],  # GPU opt-in, not default
)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: audio_features})
score = float(output[0][0])
```

## Model files

Models are NOT in git (too large). They live in `~/.violawake/models/` by default (configurable via `VIOLAWAKE_MODEL_DIR` env var). The `ModelCache` class handles download, verification (SHA-256), and caching.

Model registry lives in `src/violawake_sdk/models.py`. Adding a new model requires:
1. Adding a `ModelSpec` entry with name, URL, SHA-256, size
2. Adding it to the manifest table in `docs/PRD.md`
3. Bumping the model registry version in `pyproject.toml`

## Testing Rules

1. **Unit tests MUST NOT require model files or a microphone.** Use `tests/conftest.py` fixtures that generate synthetic audio.
2. **Integration tests** are in `tests/integration/` and require model files. They are skipped in CI if models are not downloaded (marked with `pytest.mark.integration`).
3. **Benchmark tests** require model files and write results to `benchmark-results/`. Run with `pytest tests/benchmarks/ --benchmark-json=benchmark-results/latest.json`.
4. **Never create persistent state in tests.** No files written outside `tmp_path` fixtures.

## Key Numbers (from production Viola)

These must match what's in the PRD. If you change them, update both.

| Metric | Value | Source |
|--------|-------|--------|
| Wake word d' (d-prime) | 8.577 | TemporalCNN, EER=0.8% on production eval set |
| Default threshold | 0.80 | Raised from 0.50 after false-positive flood |
| Frame size | 20ms | 16kHz mono, 320 samples |
| Feature extractor | OpenWakeWord (96-dim embeddings) | OWW audio backbone |
| TTS first audio latency | 0.3–0.8s | Kokoro, sentence-chunked |
| STT latency (base model) | 0.5–2s | faster-whisper, CPU i7 |

## Doc Navigation

All project docs are listed in `docs/REGISTRY.md`. When creating new docs:
1. Add to `docs/REGISTRY.md` first
2. Follow the `<!-- doc-meta -->` frontmatter pattern from existing docs
3. Set appropriate authority level (LIVING / ARCHIVED / ADR)

## Deployment (next-time quick reference)

Both the SaaS frontend and the SaaS backend are **manual deploys** — push to GitHub does not auto-deploy anything. Full details in `docs/DEPLOYMENT.md`.

### Backend (api.violawake.com)

Runs in Docker on the developer machine, exposed to the internet via a Cloudflare Tunnel container in the same stack.

```bash
cd /j/CLAUDE/PROJECTS/Wakeword
docker compose -f docker-compose.production.yml build backend
docker compose -f docker-compose.production.yml up -d backend
# Wait ~30s for healthcheck and tunnel reconnect
curl -sS https://api.violawake.com/api/health   # 200 = live
```

Verify the new code is actually live (e.g., after a security patch):
```bash
curl -sS https://api.violawake.com/openapi.json | python -c "import sys,json; d=json.load(sys.stdin); print('routes:', len(d['paths']))"
```

The stack is three containers (project name `wakeword`):
- `wakeword-backend-1` — uvicorn FastAPI from `console/Dockerfile.backend`
- `wakeword-postgres-1` — Postgres 16
- `wakeword-tunnel-1` — `cloudflare/cloudflared` running tunnel `violawake-api`

NOVVIOLA's containers (`viola-api`, `viola-postgres-local`, etc.) are unrelated — never restart them when deploying ViolaWake.

### Frontend (violawake.com)

Cloudflare Pages, project `violawake`. Built locally with the production API URL baked into the bundle, then deployed via Wrangler CLI.

```bash
cd /j/CLAUDE/PROJECTS/Wakeword/console/frontend
VITE_API_URL=https://api.violawake.com/api npm run build
wrangler pages deploy dist --project-name violawake --branch master --commit-dirty=true
```

`VITE_API_URL` MUST be set at build time — Vite bakes it into the JS bundle. Without it, the bundle defaults to same-origin `/api`, which 405s on Cloudflare Pages because there's no API on that domain. (Confirmed live bug 2026-05-07; fixed by always passing the env var to `npm run build`.)

### SDK (PyPI)

`pip install violawake` installs base SDK. For wake detection, users need:
```bash
pip install "violawake[oww]"
python -c "from openwakeword.utils import download_models; download_models()"
```

The `openwakeword` PyPI wheel does not bundle ONNX backbone files; `download_models()` fetches them on first use. Document this in the README's quickstart so users don't hit `ModelNotFoundError`.

### Living-doc convention

When the deploy state changes (new env var added, new domain, tunnel rerouted, etc.), update **`docs/PRODUCTION_STATUS.md`** with the date and what changed. Don't add it as a one-off note in random docs.

## Agent Coordination

This repo uses the same blackboard protocol as the main Viola project.
Blackboard script: `J:\CLAUDE\update_blackboard.py`

Before editing shared files, check the blackboard. Multiple agents may work on this repo simultaneously.
