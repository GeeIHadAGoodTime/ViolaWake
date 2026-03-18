<!-- doc-meta
scope: Architecture decision — PyPI packaging strategy and model distribution
authority: ADR — immutable once accepted
code-paths: pyproject.toml, src/violawake_sdk/models.py, .github/workflows/release.yml
staleness-signals: PyPI removes binary uploads, model hosting moves to different provider, total model size changes distribution strategy
-->

# ADR-005: PyPI Distribution + Separate Model Hosting

**Status:** Accepted
**Date:** 2026-03-17
**Authors:** ViolaWake team
**Supersedes:** N/A

---

## Context

ViolaWake SDK needs to be distributed to developers via standard Python packaging tooling. The primary mechanism is `pip install violawake`. However, ML models are large (2.1MB for wake word, 330MB for Kokoro TTS) and cannot be bundled into a PyPI wheel.

Two separate distribution problems:
1. **Python code and deps:** How to package and distribute via PyPI
2. **Model files:** How to distribute large binary ONNX files to users

These have different constraints:
- PyPI: SDist + wheels. Free. 100MB per file limit. Automatic via `pip install`.
- Models: 330MB+ for TTS model alone. No PyPI. Requires separate hosting.

---

## Decision

**Package Python code on PyPI via `hatchling`. Distribute model files via GitHub Releases with SHA-256 verification and a download-on-demand model cache.**

---

## Rationale

### PyPI Packaging: `hatchling` vs `setuptools` vs `flit`

We chose `hatchling` (via `hatch`) as the build backend.

**Comparison:**

| Backend | Pros | Cons |
|---------|------|------|
| `setuptools` | Universal, any complexity | Complex config, `setup.py` is legacy technical debt |
| `flit` | Simple, src-layout native | Limited customization, no build hooks |
| `hatchling` | Modern PEP 517/518, src-layout, env management, no setup.py | Newer, smaller ecosystem |
| `poetry` | Bundled dependency management + build | Lock file adds friction for libraries (not apps) |

**Why `hatchling`:**
- Modern `pyproject.toml`-only config (no `setup.py`, no `setup.cfg`)
- Native `src/` layout support (our layout is `src/violawake_sdk/`)
- Build hooks available for future needs (e.g., pre-built Cython extensions)
- `hatch env` for managing dev environments matches Python's direction
- Aligns with where the Python packaging ecosystem is heading (PEP 517/518/621/660)

### Model Distribution: Options Compared

#### Option A: Bundle models in PyPI wheel (rejected)

Include `.onnx` files directly in the Python package.

**Pros:** Simple — `pip install` gets everything.

**Cons:**
- Kokoro TTS model is 330MB — PyPI has a 100MB per-file limit
- Violates PyPI's intended use (code distribution, not ML model hosting)
- 330MB wheel downloads on every `pip install` — unacceptable for CI environments and users who don't need TTS
- Model updates would require a new package version even if code hasn't changed
- Separating wake word detection (2.1MB model) from TTS (330MB model) becomes impossible

**Verdict:** Hard no. PyPI limits and philosophy both reject this.

#### Option B: Download models at install time via post-install script (rejected)

`setup.py install` hooks or pip post-install scripts that download models.

**Pros:** Transparent to user — one `pip install` command.

**Cons:**
- `pip install` in sandboxed environments (Docker builds, CI, restricted networks) would fail non-obviously
- PEP 517 deprecates post-install hooks — not a stable mechanism
- Model download at install time breaks offline installs
- Hard to cache models across pip environments

**Verdict:** Rejected. Fragile, not portable to constrained environments.

#### Option C: Git LFS in the repo (rejected)

Store models in the GitHub repository via Git Large File Storage.

**Pros:** Single source of truth. Model versioning is tied to code versioning.

**Cons:**
- Git LFS has bandwidth limits on free GitHub accounts — 1GB/month free tier
- `git clone` pulls 330MB+ on every fresh checkout — unacceptable for CI
- Doesn't work for models we don't own (e.g., if OWW backbone is downloaded separately)
- LFS bandwidth bills scale with downloads — potential surprise cost at scale

**Verdict:** Rejected. LFS bandwidth limits will bite at even moderate community adoption.

#### Option D: GitHub Releases + `ModelCache` download utility (chosen)

Models are published as binary assets on GitHub Releases. The SDK includes a `ModelCache` class that:
1. Checks `~/.violawake/models/<model_name>.onnx` (or `VIOLAWAKE_MODEL_DIR` override)
2. If missing: downloads from GitHub Releases with progress bar
3. Verifies SHA-256 before use
4. Stores once per machine, reused across all Python environments

```python
from violawake_sdk.models import get_model_path

# Transparently downloads if not cached
model_path = get_model_path("viola_mlp_oww")  # ~/.violawake/models/viola_mlp_oww.onnx
```

**Pros:**
- GitHub Releases: unlimited public release storage (individual files ≤ 2GB)
- No unexpected costs — GitHub Releases bandwidth is free for open-source projects
- Download is lazy (only when user first needs the model, not at install time)
- SHA-256 verification catches corruption or man-in-the-middle
- Model updates independent of code releases
- Works offline after first download
- CI can pre-seed the cache from artifacts

**Cons:**
- First run requires internet access
- GitHub Releases URLs change if the GitHub org/repo renames (mitigated: `models.py` has a single canonical URL config)
- Cannot version-pin a model independently of the SDK version without extra tooling (mitigated: model registry in `models.py` includes SHA-256 — if SHA changes, it's a different model)

**Verdict:** Standard approach for ML model distribution. Used by Whisper, Kokoro, pyannote.audio. We adopt the same pattern.

---

## Model Registry

All models are declared in `src/violawake_sdk/models.py`:

```python
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "viola_mlp_oww": ModelSpec(
        name="viola_mlp_oww",
        url="https://github.com/youorg/violawake/releases/download/v0.1.0/viola_mlp_oww.onnx",
        sha256="abc123...",  # pre-computed SHA-256 of the ONNX file
        size_bytes=2_100_000,  # 2.1 MB
        description="ViolaWake MLP on OWW embeddings — d-prime 15.10 (default, recommended)",
    ),
    "viola_cnn_v4": ModelSpec(
        name="viola_cnn_v4",
        url="...",
        sha256="...",
        size_bytes=1_800_000,
        description="ViolaWake CNN v4 — d-prime 8.2 (lightweight, no OWW dependency)",
    ),
    "kokoro_v1_0": ModelSpec(
        name="kokoro_v1_0",
        url="...",
        sha256="...",
        size_bytes=330_000_000,  # 330 MB
        description="Kokoro-82M TTS model — Apache 2.0",
    ),
    "oww_backbone": ModelSpec(
        name="oww_backbone",
        url="...",
        sha256="...",
        size_bytes=10_000_000,  # 10 MB
        description="OpenWakeWord audio embedding backbone — required for MLP model",
    ),
}
```

---

## Release Automation

GitHub Actions workflow (`.github/workflows/release.yml`) on `git tag v*`:
1. Build wheel and sdist via `hatch build`
2. Run full test suite
3. Upload models to GitHub Release (from `models/` directory, not in git — sourced from secure artifact storage)
4. Publish to PyPI via `twine upload` using PyPI API token (stored in GitHub Secrets)

---

## Optional Extras Architecture

The `[tts]`, `[stt]`, `[vad]`, `[training]` extras align with model size:

| Extra | Installs | Required Models |
|-------|---------|-----------------|
| (core) | onnxruntime, numpy, pyaudio, scipy | `viola_mlp_oww.onnx`, `oww_backbone.onnx` |
| `[tts]` | kokoro-onnx | `kokoro_v1_0.onnx` (330MB, downloaded on demand) |
| `[stt]` | faster-whisper | Whisper models (auto-managed by faster-whisper) |
| `[vad]` | webrtcvad | None (no model file) |
| `[training]` | torch, torchaudio, librosa, sklearn | None (training creates models, not downloads) |

This minimizes unexpected large downloads for users who only need wake detection.

---

## Consequences

**Positive:**
- `pip install violawake` is fast (~35MB for core) — no model bundling
- Models are verified via SHA-256 — security baseline
- Model updates are independent of code releases
- Pattern is familiar to ML ecosystem users (same as Whisper, pyannote)
- GitHub Releases is free at our scale

**Negative:**
- First-run experience requires internet access and download time (330MB for TTS is slow on poor connections)
- Offline-first deployments need to pre-seed `~/.violawake/models/` manually

**Mitigations for offline:**
```bash
# Docker build — pre-seed cache layer
VIOLAWAKE_MODEL_DIR=/app/models violawake-download --model viola_mlp_oww
```

---

## Revisit Criteria

This decision should be revisited if:
- GitHub changes pricing/availability for public release storage
- A model exceeds 2GB (GitHub Release file limit) — would require moving to an object store (S3/GCS)
- `hatchling` proves insufficient for future build requirements (C extensions, etc.)
- PyPI increases file size limits significantly (>500MB) making bundling feasible for wake model only
