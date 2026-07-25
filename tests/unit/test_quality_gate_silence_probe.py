"""The silence subgrade's probe must be REAL no-wake audio scored on the RUNTIME path.

Guards the other half of the #2611 root cause (the grading rule itself is guarded by
test_quality_gate_silence_bar.py):

  - the probe must come from real recorded audio at a physically real level, not
    synthetic white noise ~1000x quieter than any microphone produces;
  - the spoken wake word must be excluded from it (otherwise the model correctly
    firing on the wake word would be counted as a false fire);
  - it must be scored through the runtime streaming path over the FULL clip, not a
    1.5s batch center-crop that produced exactly one window.

Reference measurements taken on wakeword-backend-1, 2026-07-24: the retired
synthetic probe sat at int16 RMS 3.29; real recorded room tone (user recordings and
LibriSpeech quiet windows) measured int16 RMS 224-3782.
"""

from __future__ import annotations

import inspect

import numpy as np

from violawake_sdk.tools.train import (
    _extract_room_tone,
    _extract_streaming_temporal_windows,
    _int16_rms,
    _run_quality_gate,
    _RUNTIME_RMS_FLOOR,
)

SR = 16000


def _recording(room_tone_rms_i16: float = 300.0, speech_rms_i16: float = 3000.0):
    """A synthetic stand-in for a user recording: room tone, then the spoken wake
    word, then room tone again -- the shape every real recording has."""
    rng = np.random.default_rng(7)
    quiet_amp = room_tone_rms_i16 / 32767.0
    loud_amp = speech_rms_i16 / 32767.0
    head = rng.standard_normal(SR * 2).astype(np.float32) * quiet_amp
    word = rng.standard_normal(SR).astype(np.float32) * loud_amp
    tail = rng.standard_normal(SR * 2).astype(np.float32) * quiet_amp
    return np.concatenate([head, word, tail])


def test_room_tone_is_extracted_at_a_physically_real_level() -> None:
    """The probe is real room tone, orders of magnitude above the retired synthetic
    probe's int16 RMS of 3.29 and above the runtime's own RMS floor."""
    room_tone = _extract_room_tone(_recording(room_tone_rms_i16=300.0))
    assert room_tone is not None
    rms = _int16_rms(room_tone)
    assert rms > _RUNTIME_RMS_FLOOR, "probe must be audio the runtime would actually score"
    assert rms > 100.0, (
        f"probe int16 RMS {rms:.1f} is near the retired synthetic probe's 3.29; the "
        "silence subgrade must be measured on real-level audio"
    )


def test_room_tone_excludes_the_spoken_wake_word() -> None:
    """Only the quiet parts are kept, so a model firing on the wake word itself is
    never counted as a silence false fire."""
    recording = _recording(room_tone_rms_i16=300.0, speech_rms_i16=3000.0)
    room_tone = _extract_room_tone(recording)
    assert room_tone is not None
    # The kept audio is far quieter than the recording as a whole.
    assert _int16_rms(room_tone) < 0.5 * _int16_rms(recording)
    # And no kept 300ms window reaches speech level.
    win = 4800
    peaks = [
        _int16_rms(room_tone[i : i + win]) for i in range(0, len(room_tone) - win, win)
    ]
    assert peaks and max(peaks) < 1000.0


def test_recording_with_no_room_tone_yields_no_probe() -> None:
    """Wall-to-wall speech yields no probe rather than a bogus one -- the gate then
    reports the silence axis as unmeasurable instead of inventing a number."""
    rng = np.random.default_rng(3)
    all_speech = rng.standard_normal(SR * 4).astype(np.float32) * (3000.0 / 32767.0)
    assert _extract_room_tone(all_speech) is None
    # Sub-RMS-floor digital silence is not a usable probe either: the runtime never
    # scores it (Gate 1 rejects it), so it cannot inform a runtime false-fire rate.
    assert _extract_room_tone(np.zeros(SR * 4, dtype=np.float32)) is None


def test_silence_subgrade_uses_the_streaming_extractor_not_the_batch_crop() -> None:
    """_run_quality_gate must score the silence probes through the runtime streaming
    path. REDs on the pre-fix wiring, which called the batch _score_files helper on a
    center-cropped near-silence wav."""
    src = inspect.getsource(_run_quality_gate)
    assert "_score_windows_streaming(room_tone_clips)" in src
    # The retired synthetic probe must not come back. Compare against CODE only --
    # the comments in that function describe the retired probe on purpose.
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "1e-4" not in code, "the synthetic near-silence probe is retired (#2611)"
    assert "default_rng(seed=42)" not in code, "the fixed-seed probe lottery is retired (#2611)"
    assert "qc_near_silence" not in code


def test_streaming_extractor_windows_the_whole_clip() -> None:
    """The streaming path yields many windows over a full clip, unlike the batch
    center-crop path that produced exactly one (observed silence_window_count == 1 on
    every production job record)."""
    src = inspect.getsource(_extract_streaming_temporal_windows)
    # It drives the same call the runtime makes, frame by frame.
    assert "push_audio" in src
    assert "FRAME_SAMPLES" in src
    # ...and it is fed raw audio arrays, so nothing center-crops it to 1.5s first.
    assert "center_crop" not in src
    assert "_prepare_audio_for_oww" not in src
    params = inspect.signature(_extract_streaming_temporal_windows).parameters
    assert "audio_clips" in params
