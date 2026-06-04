<!-- doc-meta
scope: Architecture decision — ML inference runtime selection (multi-runtime)
authority: ADR — immutable once accepted
code-paths: src/violawake_sdk/backends/onnx_backend.py, src/violawake_sdk/backends/tflite_backend.py, src/violawake_sdk/backends/base.py
supersedes: ADR-001
staleness-signals: A new dominant inference runtime emerges; we drop one of ONNX/TFLite from the backends/ directory
-->

# ADR-006: Multi-runtime Inference (ONNX + TFLite)

**Status:** Accepted
**Date:** 2026-06-04
**Authors:** ViolaWake team
**Supersedes:** [ADR-001](ADR-001-onnx-runtime.md)

---

## Context

ADR-001 (2026-03-17) locked **ONNX Runtime for all model inference**, explicitly stating "no PyTorch, TensorFlow, or JAX at inference time" and that all models are loaded "exclusively via `onnxruntime.InferenceSession`."

Since then the codebase added `src/violawake_sdk/backends/tflite_backend.py` alongside `onnx_backend.py`, both implementing the same `WakeBackend` ABC defined in `backends/base.py`. The TFLite backend exists to support mobile/edge deployment paths where ONNX Runtime is not the practical choice (TFLite has smaller cold-start memory and ships pre-installed in Android Tasks). The wake detector selects the right backend at runtime based on the model file extension (`.onnx` vs `.tflite`).

The 2026-06-03 ADR drift audit (`_diag/2026-06-03/audit_adrs_report.md`) flagged ADR-001 as DRIFT because the code now supports both runtimes but the ADR claims ONNX-only.

---

## Decision

**ViolaWake supports both ONNX Runtime and TensorFlow Lite as inference backends, selected by the model file extension. New backends must implement the `WakeBackend` ABC in `src/violawake_sdk/backends/base.py`.**

ONNX Runtime remains the default backend for desktop/server distribution (the wheels). TFLite is opt-in for edge/mobile targets. There is no PyTorch, TensorFlow, or JAX at inference time — that part of ADR-001 stands.

---

## Rationale

- **ONNX Runtime** stays the desktop default for the reasons in ADR-001: cross-platform, single dependency, sandbox-safe, deterministic on CPU.
- **TFLite** is added for mobile / Android / edge cases where the Python ONNX Runtime wheel is impractical and a small native TFLite interpreter is. Adding it as a parallel backend (not a replacement) preserves ADR-001's distribution choice on desktop while unblocking edge deployment.
- **The ABC pattern keeps callers backend-agnostic.** `wake_detector.py` consumes any `WakeBackend`; adding a backend doesn't change the public API.

---

## Consequences

- The "ONNX-only" guarantee from ADR-001 is replaced with an "ABC-conforming backend" guarantee. Reviewers must enforce the ABC at every new backend.
- The model registry (`src/violawake_sdk/models.py`) needs to track the backend per `ModelSpec` so users can `pip install violawake[tflite]` cleanly when the model only ships in TFLite form.
- Reproducibility: deterministic ONNX runs and TFLite runs may produce slightly different floating-point outputs. Lane 1's per-category FAR bars apply per-backend, not across.

---

## Alternatives considered (and rejected)

- **Re-affirm ONNX-only and remove the TFLite backend.** Rejected: the TFLite backend exists because of a real edge-deployment need; removing it would block that path.
- **Generic Python ML runtime abstraction (e.g., TVM, ONNX Optimum).** Rejected: extra complexity, no current consumer demand.
