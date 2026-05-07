"""Clean-venv installation and SDK behavior probes from PyPI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import CleanVenv, parse_summary_json


pytestmark = pytest.mark.live


def _assert_success(result, label: str) -> None:
    assert result.returncode == 0, f"{label} failed with {result.returncode}\n{result.stdout[-4000:]}"


@pytest.mark.smoke
def test_pip_install_core_smoke(clean_venv: CleanVenv) -> None:
    upgrade = clean_venv.pip_install("--upgrade", "pip", timeout=180)
    _assert_success(upgrade, "pip upgrade")

    install = clean_venv.pip_install("violawake", timeout=300)
    _assert_success(install, "pip install violawake")

    script = r"""
import json
from importlib.metadata import version

import numpy as np
from violawake_sdk import VADEngine, WakeDetector, __version__

def parse(v):
    return tuple(int(part) for part in v.split(".")[:3])

pkg_version = version("violawake")
vad = VADEngine(backend="rms")
silence = np.zeros(320, dtype=np.int16).tobytes()
tone = (np.sin(np.linspace(0, 2 * np.pi, 320)) * 12000).astype(np.int16).tobytes()
summary = {
    "package_version": pkg_version,
    "module_version": __version__,
    "version_ok": parse(pkg_version) >= (0, 2, 2),
    "wake_detector_imported": WakeDetector.__name__ == "WakeDetector",
    "silence_is_speech": vad.is_speech(silence),
    "tone_is_speech": vad.is_speech(tone),
}
print(json.dumps(summary, sort_keys=True))
assert summary["version_ok"]
assert summary["wake_detector_imported"]
assert summary["silence_is_speech"] is False
"""
    probe = clean_venv.run(["-c", script], timeout=120)
    _assert_success(probe, "core import/VAD probe")
    summary = parse_summary_json(probe.stdout)
    assert summary["package_version"] >= "0.2.2"
    assert summary["wake_detector_imported"] is True


def test_pip_install_all_extra(clean_venv: CleanVenv) -> None:
    install = clean_venv.pip_install("violawake[all]", timeout=900)
    _assert_success(install, "pip install violawake[all]")

    script = r"""
import json
from importlib.metadata import version

summary = {"package_version": version("violawake")}
import openwakeword  # noqa: F401
import onnxruntime  # noqa: F401
summary["openwakeword_imported"] = True
summary["onnxruntime_imported"] = True
print(json.dumps(summary, sort_keys=True))
"""
    probe = clean_venv.run(["-c", script], timeout=120)
    _assert_success(probe, "all extra import probe")


def test_readme_wake_detector_example_on_silence(clean_venv: CleanVenv, tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    script = r"""
import json
import os

import numpy as np
from violawake_sdk import WakeDetector

silence = np.zeros(320, dtype=np.int16).tobytes()
detector = WakeDetector(model="temporal_cnn", threshold=0.80)
scores = [detector.process(silence) for _ in range(12)]
print(json.dumps({"scores": scores, "max_score": max(scores)}, sort_keys=True))
assert max(scores) < 0.5
"""
    result = clean_venv.run(
        ["-c", script],
        timeout=300,
        env={"VIOLAWAKE_MODEL_DIR": str(model_dir)},
    )
    _assert_success(result, "WakeDetector README silence probe")


def test_viola_sample_scores_high_when_available(clean_venv: CleanVenv) -> None:
    sample_candidates = [
        Path("tests/corpus/viola.wav"),
        Path("tests/fixtures/viola.wav"),
        Path("examples/viola.wav"),
    ]
    sample = next((path for path in sample_candidates if path.exists()), None)
    if sample is None:
        pytest.xfail(reason="no sample")

    script = f"""
import json
from pathlib import Path

from scipy.io import wavfile
from violawake_sdk import WakeDetector

sr, audio = wavfile.read(Path({str(sample)!r}))
detector = WakeDetector(model="temporal_cnn", threshold=0.80)
scores = []
for start in range(0, len(audio) - 320, 320):
    scores.append(detector.process(audio[start:start + 320]))
score = max(scores) if scores else 0.0
print(json.dumps({{"max_score": score}}, sort_keys=True))
assert score > 0.5
"""
    result = clean_venv.run(["-c", script], timeout=300)
    _assert_success(result, "viola sample score probe")


def test_vad_engines_on_silence_and_sine(clean_venv: CleanVenv) -> None:
    script = r"""
import json
import numpy as np
from violawake_sdk import VADEngine

silence = np.zeros(320, dtype=np.int16).tobytes()
sine = (np.sin(np.linspace(0, 2 * np.pi * 4, 320)) * 12000).astype(np.int16).tobytes()
results = {}
for backend in ("rms", "webrtc", "silero"):
    try:
        vad = VADEngine(backend=backend)
    except ImportError as exc:
        results[backend] = {"available": False, "error": str(exc)}
        continue
    results[backend] = {
        "available": True,
        "silence": vad.is_speech(silence),
        "sine": vad.is_speech(sine, threshold=0.1),
    }
print(json.dumps(results, sort_keys=True))
assert results["rms"]["available"] is True
assert results["rms"]["silence"] is False
assert results["rms"]["sine"] is True
for backend in ("webrtc", "silero"):
    if results[backend]["available"]:
        assert results[backend]["silence"] is False
"""
    result = clean_venv.run(["-c", script], timeout=180)
    _assert_success(result, "VAD engines probe")


def test_kokoro_tts_short_generation_if_installed(clean_venv: CleanVenv, tmp_path: Path) -> None:
    script = r"""
import json
import numpy as np
try:
    from violawake_sdk import TTSEngine
    engine = TTSEngine()
except ImportError as exc:
    print(json.dumps({"skip": str(exc)}))
    raise SystemExit(3)
except Exception as exc:
    print(json.dumps({"setup_error": str(exc)}))
    raise

audio = engine.synthesize("ViolaWake live test.")
print(json.dumps({"samples": int(np.asarray(audio).size)}))
assert np.asarray(audio).size > 0
"""
    result = clean_venv.run(
        ["-c", script],
        timeout=600,
        env={"VIOLAWAKE_MODEL_DIR": str(tmp_path / "models")},
    )
    if result.returncode == 3:
        pytest.skip(parse_summary_json(result.stdout)["skip"])
    _assert_success(result, "Kokoro TTS probe")


def test_stt_engine_known_recording_skip_by_default(clean_venv: CleanVenv) -> None:
    if os.getenv("VIOLAWAKE_LIVE_STT") != "1":
        pytest.skip("Set VIOLAWAKE_LIVE_STT=1 to allow faster-whisper model download.")

    script = r"""
import json
import numpy as np
from violawake_sdk import STTEngine

engine = STTEngine(model="tiny", device="cpu", compute_type="int8", language="en")
text = engine.transcribe(np.zeros(16000, dtype=np.float32))
print(json.dumps({"text": text}))
assert isinstance(text, str)
"""
    result = clean_venv.run(["-c", script], timeout=900)
    _assert_success(result, "STT probe")


def test_voice_pipeline_integration_smoke(clean_venv: CleanVenv, tmp_path: Path) -> None:
    script = r"""
import json
from violawake_sdk import VoicePipeline

pipeline = VoicePipeline(enable_tts=False, vad_backend="rms")
print(json.dumps({"state": pipeline.state, "last_command": pipeline.last_command}))
assert pipeline.state == "idle"
"""
    result = clean_venv.run(
        ["-c", script],
        timeout=300,
        env={"VIOLAWAKE_MODEL_DIR": str(tmp_path / "models")},
    )
    _assert_success(result, "VoicePipeline integration smoke")


def test_sdk_edge_cases_clear_errors(clean_venv: CleanVenv, tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.onnx"
    corrupt_model = tmp_path / "corrupt.onnx"
    corrupt_model.write_bytes(b"not an onnx model")
    script = f"""
import json
import numpy as np
from violawake_sdk import WakeDetector, VADEngine

out = {{}}
try:
    WakeDetector(model={str(missing_model)!r})
except Exception as exc:
    out["missing_model"] = type(exc).__name__ + ": " + str(exc)

try:
    WakeDetector(model={str(corrupt_model)!r})
except Exception as exc:
    out["corrupt_onnx"] = type(exc).__name__ + ": " + str(exc)

vad = VADEngine(backend="rms")
try:
    vad.process_frame(b"")
    out["empty_buffer"] = "no_crash"
except Exception as exc:
    out["empty_buffer"] = type(exc).__name__ + ": " + str(exc)

try:
    vad.process_frame(np.zeros((320, 2), dtype=np.int16))
except Exception as exc:
    out["stereo_input"] = type(exc).__name__ + ": " + str(exc)

try:
    vad.process_frame(np.zeros(160, dtype=np.int16))
    out["wrong_sample_rate_shape"] = "accepted_or_downmixed"
except Exception as exc:
    out["wrong_sample_rate_shape"] = type(exc).__name__ + ": " + str(exc)

print(json.dumps(out, sort_keys=True))
assert "not found" in out["missing_model"].lower()
assert "failed" in out["corrupt_onnx"].lower() or "invalid" in out["corrupt_onnx"].lower()
assert out["empty_buffer"]
assert out["stereo_input"]
assert out["wrong_sample_rate_shape"]
"""
    result = clean_venv.run(["-c", script], timeout=300)
    _assert_success(result, "SDK edge case probe")
