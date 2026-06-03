"""Lane 2 live STT/TTS/VAD/VoicePipeline capability probe.

This script intentionally uses the real STT and TTS engines. Playback is
captured instead of sent to speakers so the probe is safe to run unattended.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from violawake_sdk.pipeline import (
    FRAME_SAMPLES,
    SILENCE_FRAMES_STOP,
    VoicePipeline,
)
from violawake_sdk.stt import STTEngine
from violawake_sdk.tts import TTSEngine
from violawake_sdk.vad import VADEngine

HERE = Path(__file__).resolve().parent
FIXED_WAV = HERE / "lane_02_fixed_kokoro.wav"
TTS_TEXT = "Turn on the kitchen lights. Confirm when ready."
PIPELINE_RESPONSE = "Acknowledged."


def _emit(name: str, **payload: object) -> None:
    print(json.dumps({"probe": name, **payload}, sort_keys=True))


def _to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def _frame_bytes(audio: np.ndarray) -> list[bytes]:
    pcm = _to_int16(audio)
    frames: list[bytes] = []
    for start in range(0, len(pcm), FRAME_SAMPLES):
        frame = pcm[start : start + FRAME_SAMPLES]
        if frame.size < FRAME_SAMPLES:
            padded = np.zeros(FRAME_SAMPLES, dtype=np.int16)
            padded[: frame.size] = frame
            frame = padded
        frames.append(frame.tobytes())
    return frames


def run_tts_probe() -> np.ndarray:
    tts = TTSEngine(voice="af_heart")
    t0 = time.perf_counter()
    tts._get_kokoro()
    _emit("tts_load", load_ms=round((time.perf_counter() - t0) * 1000, 1))

    first_audio_ms: list[float] = []
    synthesized: np.ndarray | None = None
    for i in range(3):
        t0 = time.perf_counter()
        generator = tts.synthesize_chunked(TTS_TEXT)
        first_chunk = next(generator)
        first_ms = (time.perf_counter() - t0) * 1000
        rest = list(generator)
        chunks = [first_chunk, *rest]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        first_audio_ms.append(first_ms)
        if synthesized is None:
            synthesized = audio
        _emit(
            "tts_first_audio",
            iteration=i + 1,
            first_audio_ms=round(first_ms, 1),
            chunks=len(chunks),
            samples=int(audio.size),
            sample_rate=tts.sample_rate,
        )

    assert synthesized is not None and synthesized.size > 0
    wavfile.write(str(FIXED_WAV), tts.sample_rate, _to_int16(synthesized))
    _emit(
        "tts_summary",
        first_audio_min_ms=round(min(first_audio_ms), 1),
        first_audio_max_ms=round(max(first_audio_ms), 1),
        first_audio_avg_ms=round(sum(first_audio_ms) / len(first_audio_ms), 1),
        wav=str(FIXED_WAV),
    )
    return synthesized


def run_stt_probe(audio: np.ndarray) -> str:
    stt = STTEngine(model="base", language="en")
    t0 = time.perf_counter()
    stt.prewarm()
    _emit("stt_load", load_ms=round((time.perf_counter() - t0) * 1000, 1))

    t0 = time.perf_counter()
    result = stt.transcribe_full(audio)
    full_ms = (time.perf_counter() - t0) * 1000
    _emit(
        "stt_full",
        elapsed_ms=round(full_ms, 1),
        text=result.text,
        segment_count=len(result.segments),
        segments=[
            {
                "text": segment.text,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "no_speech_prob": round(segment.no_speech_prob, 3),
            }
            for segment in result.segments
        ],
        language=result.language,
        language_prob=round(result.language_prob, 3),
    )

    t0 = time.perf_counter()
    streaming_segments = list(stt.transcribe_streaming(audio))
    _emit(
        "stt_streaming",
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        segment_count=len(streaming_segments),
        segments=[
            {
                "text": segment.text,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            }
            for segment in streaming_segments
        ],
    )

    if not result.text.strip() or not result.segments:
        raise RuntimeError("STT returned no text or no timestamped segments")
    return result.text


def run_vad_probe(audio: np.ndarray) -> None:
    speech_frame = _frame_bytes(audio)[0]
    silence_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
    for backend in ("rms", "auto", "webrtc", "silero"):
        try:
            vad = VADEngine(backend=backend)
            speech_prob = vad.process_frame(speech_frame)
            silence_prob = vad.process_frame(silence_frame)
            _emit(
                "vad_backend",
                requested=backend,
                active=vad.backend_name,
                speech_prob=round(float(speech_prob), 3),
                silence_prob=round(float(silence_prob), 3),
                speech=bool(vad.is_speech(speech_frame, threshold=0.4)),
                silence=bool(vad.is_speech(silence_frame, threshold=0.4)),
            )
        except Exception as exc:
            _emit(
                "vad_backend_unavailable",
                requested=backend,
                error=f"{type(exc).__name__}: {exc}",
            )


class _FakeWakeDetector:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self._detect_calls = 0
        self.last_scores: list[float] = []

    def stream_mic(self, *, device_index: int | None = None):
        return iter(self._frames)

    def detect(self, frame: bytes, *, is_playing: bool = False) -> bool:
        self._detect_calls += 1
        detected = self._detect_calls == 1
        self.last_scores.append(0.91 if detected else 0.0)
        return detected

    def close(self) -> None:
        pass


def run_pipeline_probe(audio: np.ndarray) -> None:
    frames = [b"\x00" * (FRAME_SAMPLES * 2)]
    frames.extend(_frame_bytes(audio))
    frames.extend([b"\x00" * (FRAME_SAMPLES * 2)] * (SILENCE_FRAMES_STOP + 2))

    fake_detector = _FakeWakeDetector(frames)
    spoken_audio: list[np.ndarray] = []
    events: list[str] = []
    errors: list[str] = []

    def capture_play(tts: TTSEngine, played_audio: np.ndarray, *, blocking: bool = True) -> None:
        spoken_audio.append(np.asarray(played_audio, dtype=np.float32))

    with (
        patch("violawake_sdk.pipeline.WakeDetector", return_value=fake_detector),
        patch("violawake_sdk.tts.TTSEngine.play", capture_play),
    ):
        pipeline = VoicePipeline(
            stt_model="base",
            tts_voice="af_heart",
            vad_backend="rms",
            vad_threshold=0.4,
            enable_tts=True,
        )
        pipeline.on("wake", lambda **_: events.append("wake"))
        pipeline.on("listen_start", lambda **_: events.append("listen_start"))
        pipeline.on("listen_end", lambda **_: events.append("listen_end"))
        pipeline.on("transcribe_end", lambda text, **_: events.append(f"transcribe:{text}"))
        pipeline.on("response", lambda response, **_: events.append(f"response:{response}"))
        pipeline.on("error", lambda error, **_: errors.append(str(error)))

        @pipeline.on_command
        def respond(_text: str) -> str:
            return PIPELINE_RESPONSE

        t0 = time.perf_counter()
        pipeline.run()
        elapsed_ms = (time.perf_counter() - t0) * 1000

    _emit(
        "voice_pipeline",
        elapsed_ms=round(elapsed_ms, 1),
        last_command=pipeline.last_command,
        events=events,
        errors=errors,
        spoken_chunks=len(spoken_audio),
        spoken_samples=sum(int(chunk.size) for chunk in spoken_audio),
    )
    if errors or not spoken_audio or not pipeline.last_command:
        raise RuntimeError("VoicePipeline did not reach spoken response cleanly")


def main() -> None:
    _emit("environment", source_file=__file__)
    audio = run_tts_probe()
    run_stt_probe(audio)
    run_vad_probe(audio)
    run_pipeline_probe(audio)
    _emit("result", verdict="PASS")


if __name__ == "__main__":
    main()
