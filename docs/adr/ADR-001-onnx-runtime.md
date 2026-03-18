<!-- doc-meta
scope: Architecture decision — ML inference runtime selection
authority: ADR — immutable once accepted
code-paths: src/violawake_sdk/wake_detector.py, src/violawake_sdk/tts.py, src/violawake_sdk/stt.py
staleness-signals: Major ONNX Runtime breaking change, new inference runtime dominates Python ecosystem
-->

# ADR-001: Use ONNX Runtime for All Model Inference

**Status:** Accepted
**Date:** 2026-03-17
**Authors:** ViolaWake team
**Supersedes:** N/A

---

## Context

ViolaWake SDK ships three ML models:
1. The ViolaWake MLP classifier (wake word detection)
2. Kokoro-82M (TTS)
3. OpenWakeWord audio feature extractor (used as fixed backbone for wake detection)

We need to choose how these models are loaded and executed at inference time. The choice affects:
- Cross-platform portability (Linux/Windows/macOS/Pi)
- Package size and dependency complexity
- Inference performance
- Model portability between training and production environments
- Developer ability to contribute models trained in any ML framework

The key decision is whether to use **native framework runtimes** (PyTorch/TensorFlow) or a **dedicated inference runtime** (ONNX Runtime, TensorRT, etc.).

---

## Decision

**Use ONNX Runtime (CPU execution provider, with GPU opt-in) for all model inference in the ViolaWake SDK.**

All models are stored as `.onnx` files and loaded exclusively via `onnxruntime.InferenceSession`. No PyTorch, TensorFlow, or JAX at inference time.

---

## Rationale

### Option A: PyTorch (rejected)

**Pros:**
- Training and inference use same framework
- Rich debugging tools (torchviz, hooks, etc.)
- Familiar to ML engineers

**Cons:**
- torch + torchaudio adds ~2.5GB to the install footprint for inference-only users
- CPU inference is slower than ONNX Runtime (by ~20–30% for our MLP workload)
- PyTorch is a hard dependency even for users who only want wake word detection, not training
- Breaks the design principle: training deps (torch) should be optional extras, not core deps

**Deal-breaker:** Forcing 2.5GB of PyTorch onto a developer who just wants `detector.process(chunk)` is unacceptable for a developer-experience-first SDK.

### Option B: TorchScript / TorchMobile (rejected)

**Pros:**
- Smaller footprint than full PyTorch
- Still PyTorch ecosystem

**Cons:**
- Still requires libtorch (~200MB) at inference
- Less portable than ONNX (platform-specific binaries)
- Kokoro model ships as ONNX — we'd be converting in one direction for some models and another for others
- No WASM target (needed for Phase 2 browser support)

### Option C: ONNX Runtime (chosen)

**Pros:**
- `onnxruntime` CPU wheel: ~30MB. Tiny vs PyTorch.
- True cross-platform: Linux/Windows/macOS/ARM/Pi, same wheel
- WASM build available (`onnxruntime-web`) — critical for Phase 2 browser SDK
- GPU acceleration via `onnxruntime-gpu` with same API
- Faster CPU inference than PyTorch for our model sizes (~20% faster on MLP wake word)
- Framework-agnostic: models trained in PyTorch, TF, JAX all export to ONNX
- This is already what Viola production uses — no new technical risk

**Cons:**
- ONNX format doesn't support every PyTorch operation (dynamic shapes, custom ops)
- Debugging is harder — no gradient hooks, no easy layer inspection
- ONNX Runtime version compatibility can be tricky with complex model architectures

**For our use case the cons are acceptable:**
- Our models are simple (MLP with standard ops) — no custom operator risk
- We're inference-only in the SDK; debugging happens in the training environment (PyTorch)
- ONNX version compatibility is managed by pinning in pyproject.toml (`onnxruntime>=1.17`)

### Option D: TensorRT (rejected)

TensorRT gives better GPU performance but requires NVIDIA GPU and is significantly harder to install. Our CPU inference meets latency targets (8ms/frame). TensorRT is out of scope for Phase 1.

---

## Implementation

```python
import onnxruntime as ort
import numpy as np

class WakeDetector:
    def __init__(self, model_path: str, threshold: float = 0.80) -> None:
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],  # explicit, not default
        )
        self._input_name = self._session.get_inputs()[0].name
        self.threshold = threshold

    def process(self, audio_features: np.ndarray) -> float:
        output = self._session.run(None, {self._input_name: audio_features})
        return float(output[0][0])
```

**GPU opt-in** (not default — users must explicitly enable):
```python
detector = WakeDetector(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

---

## Consequences

**Positive:**
- Core SDK install is ~35MB (onnxruntime + numpy + scipy + pyaudio)
- Training deps (torch, torchaudio, librosa) are `[training]` extras only
- Phase 2 browser SDK can reuse model files directly (same ONNX format, ONNX Runtime Web)
- Model portability: any framework can contribute models as long as they export to ONNX

**Negative:**
- Developers who want to inspect model internals or fine-tune via gradient descent must install PyTorch separately
- We must maintain ONNX export code in the training pipeline (currently exists in `violawake/training/`)

**Neutral:**
- The `[training]` extras group (torch, torchaudio, etc.) is clearly documented as separate from core inference
- The `[gpu]` extras group (onnxruntime-gpu) is clearly documented

---

## Revisit Criteria

This decision should be revisited if:
- ONNX Runtime introduces a breaking API change affecting our model architectures
- A new runtime (e.g., Apache TVM, Triton Inference Server) offers >2x CPU speedup with comparable install size
- Phase 2 browser SDK requirements reveal an incompatibility with onnxruntime-web
