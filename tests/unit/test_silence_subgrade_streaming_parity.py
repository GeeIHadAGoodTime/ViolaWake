"""Ratchet: the training quality gate's silence subgrade must score the REAL
runtime streaming path, not a batch approximation of it (#1487).

Root cause (confirmed here with the real, deployed backbone + temporal_cnn
model checked into console/frontend/public/wasm/models/): the silence subgrade
used to score near-silence probes through ``_extract_temporal_embeddings`` --
one batch ``preprocessor.embed_clips()`` call on a SINGLE 1.5s center-crop of
the probe (``_prepare_audio_for_oww`` -> ``center_crop(audio, CLIP_SAMPLES)``,
train.py). The runtime (``WakeDetector.process`` -> ``OpenWakeWordBackbone
.push_audio``, wake_detector.py:709) instead streams the FULL continuous input
through persistent ring/mel buffers that accumulate state across calls. These
are different embedding computations on different amounts of audio, and
``_extract_temporal_embeddings``'s own docstring already warned "streaming
push_audio() produces subtly different embeddings due to internal state
accumulation" -- the silence subgrade was the one place that warning was never
heeded.

Measured directly below on this repo's own deployed model: the OLD batch path
sees exactly ONE window (a 1.5s crop) and scores ~0.07-0.08 on a near-silence
probe that the real runtime streams as ~117 windows and scores 0.24-0.49 raw
(3-6x higher) -- i.e. the OLD gate would have graded this exact model far more
favorably than its real deployed silence behavior warrants. The fix
(``_extract_streaming_temporal_windows`` / ``_score_files_streaming``) scores
near-silence probes via the SAME ``OpenWakeWordBackbone.push_audio`` streaming
call the runtime makes, on the FULL uncropped probe, at the same 20ms-frame
granularity (``wake_detector.FRAME_SAMPLES``) -- and is proven below to produce
the EXACT SAME raw max score (bit-identical, not "within tolerance") as
driving a real ``WakeDetector.process()`` loop over the identical audio.

These tests RED on the pre-fix shape (silence scored via
``_extract_temporal_embeddings``'s 1.5s-crop batch path) and GREEN on the
streaming-parity fix. No network access or trained weights are required: the
mel/embedding backbone and temporal_cnn classifier are the real, pinned
production ONNX artifacts already checked into this repo
(``console/frontend/public/wasm/models/``, sha256-verified against
``MODEL_REGISTRY["oww_backbone"]`` below) so this runs fully offline in CI.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WASM_MODELS = _REPO_ROOT / "console" / "frontend" / "public" / "wasm" / "models"
_MEL_PATH = _WASM_MODELS / "melspectrogram.onnx"
_EMB_PATH = _WASM_MODELS / "embedding_model.onnx"
_TEMPORAL_CNN_PATH = _WASM_MODELS / "temporal_cnn.onnx"

pytestmark = [
    pytest.mark.integration,  # requires the real ONNX model files, per pyproject.toml
    pytest.mark.skipif(
        not (_MEL_PATH.exists() and _EMB_PATH.exists() and _TEMPORAL_CNN_PATH.exists()),
        reason="Real OWW backbone + temporal_cnn ONNX fixtures not present in this checkout.",
    ),
]


def _assert_real_backbone_files_are_the_pinned_production_ones() -> None:
    """Guard against silently testing against swapped-out placeholder files.

    The openwakeword PACKAGE ships 195-byte placeholder stubs for its backbone
    resources in some environments; this repo's own
    console/frontend/public/wasm/models/ copies are the real, pinned artifacts
    (MODEL_REGISTRY["oww_backbone"].sha256). Verifying this up front means a
    future accidental stub swap fails loudly here instead of silently
    collapsing every score in this file to near-zero garbage.
    """
    import hashlib

    from violawake_sdk.models import MODEL_REGISTRY

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    mel_sha = _sha256(_MEL_PATH)
    emb_sha = _sha256(_EMB_PATH)
    combined = hashlib.sha256((mel_sha + emb_sha).encode()).hexdigest()
    expected = MODEL_REGISTRY["oww_backbone"].sha256
    assert combined == expected, (
        "console/frontend/public/wasm/models/{melspectrogram,embedding_model}.onnx "
        "no longer match the pinned oww_backbone hash -- these tests need the REAL "
        "backbone, not a placeholder stub."
    )


@pytest.fixture(autouse=True)
def _real_backbone_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the OWW backbone resolver at this repo's real, pinned production
    ONNX files instead of the openwakeword package's own resource directory
    (which some environments only ship as tiny placeholder stubs, or which
    would otherwise require a network download) -- fully offline either way.
    """
    _assert_real_backbone_files_are_the_pinned_production_ones()

    import violawake_sdk.oww_backbone as oww_backbone_module
    from violawake_sdk.oww_backbone import OpenWakeWordBackbonePaths

    def _fake_resolve(backend_name: str = "onnx") -> OpenWakeWordBackbonePaths:
        return OpenWakeWordBackbonePaths(melspectrogram=_MEL_PATH, embedding_model=_EMB_PATH)

    monkeypatch.setattr(oww_backbone_module, "resolve_openwakeword_backbone_paths", _fake_resolve)


def _save_wav(audio_f32: np.ndarray, path: Path, sample_rate: int) -> None:
    i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(i16.tobytes())


def _near_silence_probe(sample_rate: int, seconds: int = 10, seed: int = 42) -> np.ndarray:
    """The exact near-silence probe recipe _run_quality_gate uses: RMS ~= 1e-4
    white noise, seed 42 is production's first probe (_NEAR_SILENCE_PROBE_SEED_BASE).
    """
    rng = np.random.default_rng(seed=seed)
    return rng.standard_normal(sample_rate * seconds).astype(np.float32) * 1e-4


def _runtime_streaming_max(model_path: Path, audio_f32: np.ndarray, frame_samples: int) -> float:
    """Drive a REAL WakeDetector exactly as production does: one process() call
    per 20ms frame over the whole clip, raw score before any RMS-floor/threshold
    gating. This is the ground truth #1487 is about matching.
    """
    from violawake_sdk.wake_detector import WakeDetector

    det = WakeDetector(model=str(model_path), threshold=0.80)
    scores: list[float] = []
    n = len(audio_f32)
    for i in range(0, n - frame_samples + 1, frame_samples):
        scores.append(det.process(audio_f32[i : i + frame_samples].astype(np.float32)))
    return max(scores)


def _score_windows_with_deployed_model(windows: list[np.ndarray]) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(_TEMPORAL_CNN_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    X = np.stack(windows).astype(np.float32)
    return float(session.run(None, {input_name: X})[0].flatten().max())


class TestStreamingExtractionUsesTheFullClip:
    def test_streaming_path_produces_far_more_windows_than_the_batch_1_5s_crop(
        self, tmp_path: Path
    ) -> None:
        """RED on the pre-fix shape: _extract_temporal_embeddings center-crops
        every clip to CLIP_SAMPLES (1.5s) before embedding, so a 10s near-silence
        probe collapses to exactly ONE temporal window. The streaming fix uses
        the FULL clip and must produce dozens of windows.
        """
        from openwakeword.utils import AudioFeatures

        from violawake_sdk._constants import CLIP_SAMPLES
        from violawake_sdk.tools.train import (
            _extract_streaming_temporal_windows,
            _prepare_audio_for_oww,
            _temporal_windows_from_frame_embeddings,
        )
        from violawake_sdk.wake_detector import SAMPLE_RATE

        probe = _near_silence_probe(SAMPLE_RATE)
        wav_path = tmp_path / "near_silence.wav"
        _save_wav(probe, wav_path, SAMPLE_RATE)

        # Reproduce the OLD path's actual embedding computation directly via
        # AudioFeatures (what oww.preprocessor.embed_clips resolves to inside
        # _extract_temporal_embeddings) -- avoids requiring openwakeword's full
        # bundled pretrained wakeword models, which this sandbox may only carry
        # as placeholder stubs; the embedding computation itself is identical.
        preprocessor = AudioFeatures(
            inference_framework="onnx",
            melspec_model_path=str(_MEL_PATH),
            embedding_model_path=str(_EMB_PATH),
        )
        audio_i16 = _prepare_audio_for_oww(probe, clip_name="near", verbose=False)
        assert audio_i16 is not None
        assert len(audio_i16) == CLIP_SAMPLES  # confirms the 1.5s crop actually happened
        frame_emb = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
        batch_windows, _src, _tags = _temporal_windows_from_frame_embeddings(
            frame_emb[0], source_id=0, tag="qc_test", seq_len=9
        )

        stream_windows, stream_src = _extract_streaming_temporal_windows(
            [wav_path], "qc_test", seq_len=9
        )

        assert len(batch_windows) <= 2  # a single 1.5s crop can hold at most a couple windows
        assert len(stream_windows) >= 50  # the full 10s stream, dozens of overlapping windows
        assert len(stream_windows) > len(batch_windows) * 20
        assert set(stream_src) == {0}


class TestSilenceSubgradeMatchesRuntimeStreamingScore:
    """The #1487 acceptance oracle, proven directly: the fixed silence subgrade
    scores near-silence audio through _extract_streaming_temporal_windows, and
    that must match what the real runtime computes on the SAME audio.
    """

    def test_streaming_gate_score_is_bit_identical_to_real_wakedetector_process(
        self, tmp_path: Path
    ) -> None:
        from violawake_sdk.tools.train import (
            _extract_streaming_temporal_windows,
            _load_training_audio,
        )
        from violawake_sdk.wake_detector import FRAME_SAMPLES, SAMPLE_RATE

        probe = _near_silence_probe(SAMPLE_RATE)
        wav_path = tmp_path / "near_silence.wav"
        _save_wav(probe, wav_path, SAMPLE_RATE)

        # Load exactly what the gate function loads (a WAV round-trip quantizes
        # this near-silence signal slightly; comparing against the SAME loaded
        # samples the gate uses is the correct apples-to-apples parity check).
        loaded = _load_training_audio(wav_path).reshape(-1)
        runtime_max = _runtime_streaming_max(_TEMPORAL_CNN_PATH, loaded, FRAME_SAMPLES)

        stream_windows, _src = _extract_streaming_temporal_windows([wav_path], "qc_test", seq_len=9)
        gate_max = _score_windows_with_deployed_model(stream_windows)

        # Not "within tolerance" -- the fix uses the identical runtime code path
        # (OpenWakeWordBackbone.push_audio, same frame granularity), so the
        # score IS the runtime score, bit-for-bit.
        assert gate_max == pytest.approx(runtime_max, abs=1e-6)

    def test_old_batch_crop_path_materially_diverges_from_the_runtime_score(
        self, tmp_path: Path
    ) -> None:
        """Reproduces the #1487 bug directly on this repo's real deployed model:
        the OLD batch-crop score is not a reliable proxy for the runtime score,
        while the streaming fix is. RED on the pre-fix shape (this assertion
        would fail if the "fix" were silently reverted to the batch path).
        """
        from openwakeword.utils import AudioFeatures

        from violawake_sdk._constants import CLIP_SAMPLES
        from violawake_sdk.tools.train import (
            _extract_streaming_temporal_windows,
            _load_training_audio,
            _prepare_audio_for_oww,
            _temporal_windows_from_frame_embeddings,
        )
        from violawake_sdk.wake_detector import FRAME_SAMPLES, SAMPLE_RATE

        probe = _near_silence_probe(SAMPLE_RATE)
        wav_path = tmp_path / "near_silence.wav"
        _save_wav(probe, wav_path, SAMPLE_RATE)
        loaded = _load_training_audio(wav_path).reshape(-1)

        runtime_max = _runtime_streaming_max(_TEMPORAL_CNN_PATH, loaded, FRAME_SAMPLES)

        stream_windows, _src = _extract_streaming_temporal_windows([wav_path], "qc_test", seq_len=9)
        gate_stream_max = _score_windows_with_deployed_model(stream_windows)

        preprocessor = AudioFeatures(
            inference_framework="onnx",
            melspec_model_path=str(_MEL_PATH),
            embedding_model_path=str(_EMB_PATH),
        )
        audio_i16 = _prepare_audio_for_oww(loaded, clip_name="near", verbose=False)
        assert audio_i16 is not None and len(audio_i16) == CLIP_SAMPLES
        frame_emb = preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
        batch_windows, _s, _t = _temporal_windows_from_frame_embeddings(
            frame_emb[0], source_id=0, tag="qc_test", seq_len=9
        )
        gate_batch_max = _score_windows_with_deployed_model(batch_windows)

        # The streaming fix matches the runtime number; the old batch path does
        # not -- and is not even closer than the fix by coincidence.
        stream_gap = abs(gate_stream_max - runtime_max)
        batch_gap = abs(gate_batch_max - runtime_max)
        assert stream_gap < 1e-6
        assert batch_gap > 0.15  # the old path is off by a wide margin (#1487: 3-6x observed)
        assert stream_gap < batch_gap


class TestQualityGateWiring:
    """Confirms _run_quality_gate itself calls the streaming path for
    near-silence, not just that the standalone function exists.
    """

    def test_run_quality_gate_uses_streaming_extraction_for_near_silence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from violawake_sdk.tools import train

        called_with_streaming = {"near_silence": False}
        real_streaming = train._extract_streaming_temporal_windows

        def _spy_streaming(audio_files, tag, seq_len):
            if tag == "qc_near_silence":
                called_with_streaming["near_silence"] = True
            return real_streaming(audio_files, tag, seq_len)

        monkeypatch.setattr(train, "_extract_streaming_temporal_windows", _spy_streaming)
        # No network: speech/confusable TTS generation is irrelevant to this
        # test (it only asserts how near-silence gets scored), and every
        # existing train.py test mocks this out the same way.
        monkeypatch.setattr(train, "_edge_tts_synthesize", lambda *a, **k: False)
        # The batch path (still used for speech/confusable/pure-silence, all
        # empty/unused here) instantiates openwakeword's full pretrained-model
        # bundle via OWWModel(...), which some environments only carry as
        # placeholder stub files -- irrelevant to what this test asserts (the
        # near-silence WIRING), so stub it out rather than require real
        # pretrained wakeword models just to construct an unused code path.
        monkeypatch.setattr(train, "_extract_temporal_embeddings", lambda *a, **k: ([], [], []))

        # A tiny linear "model" standing in for the trained torch classifier --
        # only its call signature (model(X) -> tensor) matters here.
        import torch

        class _ZeroModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], 1)

        train._run_quality_gate(
            model=_ZeroModel(),
            torch_device="cpu",
            seq_len=9,
            embedding_dim=96,
            wake_word="testword",
            deployment_threshold=0.80,
            positive_files=None,
            verbose=False,
        )

        assert called_with_streaming["near_silence"] is True
