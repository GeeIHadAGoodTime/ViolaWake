<!-- doc-meta
scope: Architecture decision — SDK language and platform strategy
authority: ADR — immutable once accepted
code-paths: src/violawake_sdk/, pyproject.toml
staleness-signals: Python SDK proves insufficient for target user adoption, clear demand for C/embedded binding, Phase 2 WASM work reveals Python-first as wrong foundation
-->

# ADR-003: Python SDK First (Not C Library)

**Status:** Accepted
**Date:** 2026-03-17
**Authors:** ViolaWake team
**Supersedes:** N/A

---

## Context

Picovoice's primary technical advantage is that their core engines (Porcupine, Cheetah, etc.) are written in C and cross-compiled to every platform. The Python, JavaScript, Java, Swift, and Flutter SDKs are thin wrappers around the same C binary. This gives them:
- Sub-millisecond inference on embedded MCUs
- Single implementation tested across all platforms
- Easy bindings for any language

The question is whether ViolaWake SDK should adopt the same architecture (C core, language bindings) or ship a native Python SDK.

**The argument for C first:** Build a real SDK platform, not just a Python library. Target the same market as Picovoice.

**The argument for Python first:** Ship something real in 3 months vs. ship something perfect in 18 months.

---

## Decision

**Ship a Python-native SDK first. Do not build a C library for Phase 1.**

The Phase 1 SDK is Python 3.10+ throughout, with ONNX Runtime as the inference layer. No C extensions beyond what's already in PyPI wheels (onnxruntime, pyaudio, webrtcvad).

---

## Rationale

### Why Not C First?

**1. Timeline mismatch**

Building a production-quality C library requires:
- Designing a stable C API (ABI compatibility, versioning, error handling)
- Cross-compiling for Windows (MSVC), Linux (GCC, clang), macOS (Apple Clang), ARM (GCC cross-compile), WASM (Emscripten)
- Setting up CI matrix for all targets
- Memory safety review (no GC in C — every buffer, every allocation must be explicit)
- Writing Python/JS/Java bindings on top

This is a 6–12 month engineering investment before any Python developer can `pip install` our SDK.

By that point, Picovoice still exists and has a 3-year head start on their C platform. We are not going to out-execute them on C platform engineering.

**2. Our actual users are Python developers**

The gap we identified is: "Python developer who wants an open, affordable alternative to Porcupine with a fine-tunable model."

Python developers:
- `pip install` packages
- Don't want to deal with shared library linking
- Don't need Cortex-M targets
- Value ergonomics and documentation over raw performance

C first optimizes for a different user: embedded engineers building IoT devices. That's a harder market to enter (long sales cycles, hardware certification, specialized toolchain knowledge) and not our first-wave target.

**3. ONNX Runtime already handles the hard part**

The reason Picovoice needed a C core is model inference. We delegate inference to ONNX Runtime, which:
- Is already cross-compiled for all major platforms (including ARM, Android, iOS, WASM)
- Has production-quality performance optimizations we couldn't match in Phase 1
- Has a stable C API underneath its Python wrapper

Our Python code is orchestration on top of ONNX Runtime. The performance-critical path (model inference) is not Python.

**4. Raspberry Pi is our minimum embedded target**

We explicitly ruled out MCU targets (Cortex-M). Pi 4+ runs Python natively. We don't need a C library to target our minimum embedded platform.

**5. Phase 2 WASM is achievable without C**

For Phase 2 browser support:
- `onnxruntime-web` (WASM) handles model inference natively
- Audio capture in browser uses Web Audio API / AudioWorklet (JavaScript, not Python)
- The Python SDK and browser SDK will share model files (ONNX) but have separate language implementations

We don't need a unified C core to achieve cross-platform — we need shared model formats (ONNX) and well-defined interfaces.

---

### Risks of Python First

**Risk 1: Python too slow for some use cases**
*Mitigation:* Our latency targets are 15ms/frame for wake detection. Python overhead for the orchestration code is ~0.1ms. ONNX Runtime handles inference at 8ms. We are within budget. If profiling reveals Python bottleneck, hot paths can be Cython-compiled later.

**Risk 2: "Not a real SDK" perception**
*Mitigation:* Ship excellent docs, publish benchmarks, and point to production use in Viola. "Running in production Python application" is stronger evidence than "theoretically works in C." openWakeWord is Python-first and has significant community traction precisely because of accessibility.

**Risk 3: Python GIL limits multi-threaded use**
*Mitigation:* Our primary use case is a single-threaded audio capture loop. ONNX Runtime releases the GIL during inference. Multi-threaded TTS synthesis is already thread-safe in production Viola (the Kokoro engine uses a lock).

**Risk 4: Can't target mobile natively**
*Accepted risk for Phase 1.* Phase 3 (2027) will address Android/iOS via ONNX Runtime Android/iOS, with the Python SDK serving as the reference implementation and specification for bindings.

---

## Implementation Guidelines

**Python version:** 3.10+ (uses `match/case`, union types)

**Performance profile:** The audio processing loop runs at 50fps (20ms frames). Python can handle this without a C extension — tested in production Viola.

**Type hints everywhere:** The public API is fully typed. `mypy --strict` passes in CI. This compensates for Python's dynamic nature by giving IDEs and type checkers full API documentation at the type level.

**Thread safety:** All inference sessions are created per-instance (not shared). The `VoicePipeline` uses a dedicated thread for mic capture. ONNX Runtime sessions are thread-safe for concurrent `run()` calls.

---

## Consequences

**Positive:**
- Ship Phase 1 MVP in 3 months (not 18 months)
- Lower barrier to contribution (Python contributors >> C contributors)
- ONNX Runtime cross-platform story is handled by a mature project
- Better developer experience for our primary target (Python developers)

**Negative:**
- Cannot target MCU/embedded (Cortex-M, ESP32, etc.) in Phase 1
- No C/C++ SDK for developers who need to embed in non-Python apps
- Cannot easily create Java/Swift/Dart bindings without a C FFI layer

**Accepted trade-offs:**
- MCU targeting is explicitly out of scope
- Non-Python language bindings require reimplementation (not wrapping) — deferred to Phase 3+

---

## Revisit Criteria

This decision should be revisited if:
- Python SDK adoption is significantly blocked by developers requiring C bindings (signal: >20 GitHub issues requesting C API in first 6 months)
- A mobile use case (Android/iOS) becomes a top-priority feature request before Phase 3
- A major contributor wants to tackle the C library and has the cross-compilation expertise
- ONNX Runtime itself drops Python support or becomes hard to distribute
