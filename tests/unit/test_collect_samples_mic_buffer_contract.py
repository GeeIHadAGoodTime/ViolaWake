"""Ratchet gate: collect-samples-mic-buffer-contract.

Bug class this gate exists to stop — ``violawake-collect`` asked PortAudio for a
whole clip in one buffer::

    n_samples = int(sample_rate * duration)
    stream = pa.open(..., frames_per_buffer=n_samples)
    data = stream.read(n_samples, exception_on_overflow=False)

Measured on macOS/PortAudio, that read blocks for about
``duration * frames_per_buffer / 1024`` seconds (24000-frame buffer = 23x
real time) and returns time-decimated audio with silently dropped frames, so
every saved "1.5 s" training sample is chopped garbage.

The gate asserts the capture boundary reads in bounded 20 ms chunks, keeps
overflow loud, and fails closed if asked for a whole-clip buffer. No
microphone, model file, or network required.
"""

from __future__ import annotations

import array
import ast
import inspect
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from violawake_sdk.tools import collect_samples

# Accessed as module attributes (not imported names) on purpose: against the
# pre-fix implementation each behaviour test fails on its own assertion
# (AttributeError / TypeError) instead of the whole file dying at import time.

# --- shapes the old code used, kept as literals so the probes stay honest ----
OLD_WHOLE_CLIP_BUFFER = 24000  # int(16000 * 1.5) — frames_per_buffer in the bug

OLD_SHAPE_SOURCE = '''
"""Synthetic pre-fix shape: one whole clip per PortAudio buffer."""


def _record_clip(sample_rate=16000, duration=1.5):
    import pyaudio

    pa = pyaudio.PyAudio()
    n_samples = int(sample_rate * duration)
    stream = pa.open(
        format=pa.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=n_samples,
    )
    data = stream.read(n_samples, exception_on_overflow=False)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    return data
'''


def _small_buffer_expr(node: ast.expr) -> bool:
    """True if the expression can only be a bounded small chunk."""
    bound = collect_samples.MAX_MIC_BUFFER_FRAMES
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and 0 < node.value <= bound
    if isinstance(node, ast.Name):
        return node.id in {"chunk_frames", "FRAME_SAMPLES"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"chunk_frames", "FRAME_SAMPLES", "SAMPLES_PER_FRAME"}
    return False


def _mic_buffer_violations(source: str) -> list[str]:
    """Static detector for whole-clip mic buffering around a pyaudio stream."""
    tree = ast.parse(source)
    problems: list[str] = []

    in_loop: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            in_loop.update(id(sub) for sub in ast.walk(node))

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "open":
            for kw in node.keywords:
                if kw.arg == "frames_per_buffer" and not _small_buffer_expr(kw.value):
                    problems.append(
                        f"line {node.lineno}: frames_per_buffer is not a bounded "
                        f"chunk (whole-clip buffers stall the read and drop frames)"
                    )
        elif node.func.attr == "read":
            if id(node) not in in_loop:
                problems.append(
                    f"line {node.lineno}: stream.read() is not inside a chunk loop"
                )
            for kw in node.keywords:
                if (kw.arg == "exception_on_overflow" and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False):
                    problems.append(
                        f"line {node.lineno}: exception_on_overflow=False "
                        "hides dropped frames"
                    )
    return problems


class FakeStream:
    """Deterministic int16 counter source; records every read() request."""

    def __init__(self, *, overflow_on: int | None = None, short_on: int | None = None):
        self.read_calls: list[tuple[int, object]] = []
        self.open_kwargs: dict | None = None
        self.closed = False
        self._counter = 0
        self._overflow_on = overflow_on
        self._short_on = short_on

    def read(self, num_frames: int, exception_on_overflow: object = True) -> bytes:
        call_no = len(self.read_calls) + 1
        self.read_calls.append((num_frames, exception_on_overflow))
        if self._overflow_on == call_no and exception_on_overflow is True:
            raise OSError(-9981, "Insufficient can be read / Overflowed")
        n = num_frames - 1 if self._short_on == call_no else num_frames
        samples = array.array("h", [(self._counter + i) % 32768 for i in range(n)])
        self._counter += n
        return samples.tobytes()

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakePyAudioModule:
    """Stand-in for the pyaudio module: no hardware, no permissions."""

    paInt16 = 8  # real PyAudio constant value

    def __init__(self, stream: FakeStream | None = None):
        self.stream = stream if stream is not None else FakeStream()
        self.instances: list[FakePyAudio] = []

    def PyAudio(self) -> FakePyAudio:
        inst = FakePyAudio(self)
        self.instances.append(inst)
        return inst


class FakePyAudio:
    def __init__(self, module: FakePyAudioModule):
        self._module = module
        self.terminated = False

    def open(self, **kwargs):
        self._module.stream.open_kwargs = kwargs
        return self._module.stream

    def get_device_count(self) -> int:
        return 1

    def get_device_info_by_index(self, index: int) -> dict:
        return {
            "index": index,
            "name": "Fake Mic",
            "maxInputChannels": 1,
            "defaultSampleRate": 16000.0,
        }

    def terminate(self) -> None:
        self.terminated = True


def _record(**kw) -> tuple[bytes | None, FakePyAudioModule]:
    """Run _record_clip against a fake pyaudio module; return (pcm, fake)."""
    fake = FakePyAudioModule()
    kw.setdefault("pa_module", fake)
    return collect_samples._record_clip(**kw), fake


# --------------------------------------------------------------------------
# Negative probes: the gate MUST fail on the pre-fix shape.
# --------------------------------------------------------------------------


def test_detector_flags_the_pre_fix_shape() -> None:
    """The static detector catches the whole-clip buffer bug it was written for."""
    problems = _mic_buffer_violations(OLD_SHAPE_SOURCE)
    assert len(problems) >= 3, problems
    assert any("frames_per_buffer" in p for p in problems)
    assert any("not inside a chunk loop" in p for p in problems)
    assert any("exception_on_overflow=False" in p for p in problems)


def test_detector_is_clean_on_current_module_source() -> None:
    source = Path(inspect.getsourcefile(collect_samples)).read_text(encoding="utf-8")
    assert _mic_buffer_violations(source) == []


def test_record_clip_rejects_whole_clip_buffer() -> None:
    """Fail closed rather than stalling: an oversized chunk is a contract error."""
    with pytest.raises(ValueError, match="mic buffer contract"):
        collect_samples._record_clip(
            sample_rate=16000, duration=1.5, chunk_frames=OLD_WHOLE_CLIP_BUFFER
        )


def test_record_clip_opens_and_reads_in_bounded_chunks() -> None:
    data, fake = _record(sample_rate=16000, duration=1.5)
    assert data is not None
    bound = collect_samples.MAX_MIC_BUFFER_FRAMES
    open_kwargs = fake.stream.open_kwargs
    assert open_kwargs["frames_per_buffer"] <= bound
    assert open_kwargs["frames_per_buffer"] == collect_samples.FRAME_SAMPLES, \
        "20 ms contract frame"
    for num_frames, overflow_flag in fake.stream.read_calls:
        assert num_frames <= bound
        # overflow must stay loud: never explicitly disabled
        assert overflow_flag is not False
    assert len(fake.stream.read_calls) > 1, "a 1.5 s clip must be many chunk reads"


def test_record_clip_never_disables_overflow_exception() -> None:
    _, fake = _record(sample_rate=16000, duration=0.2)
    assert all(flag is not False for _, flag in fake.stream.read_calls)


def test_overflow_aborts_the_clip_instead_of_shortening_it() -> None:
    """A dropped frame must fail the sample, not quietly yield a shorter WAV."""
    fake_mod = FakePyAudioModule(FakeStream(overflow_on=3))
    result = collect_samples._record_clip(
        sample_rate=16000, duration=0.2, pa_module=fake_mod
    )
    assert result is None


def test_short_frame_read_aborts_the_clip() -> None:
    fake_mod = FakePyAudioModule(FakeStream(short_on=2))
    result = collect_samples._record_clip(
        sample_rate=16000, duration=0.2, pa_module=fake_mod
    )
    assert result is None


# --------------------------------------------------------------------------
# Acceptance: continuous, correctly sized audio out of the chunk loop.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration, expected_frames, expected_reads",
    [
        (1.5, 24000, 75),   # 24000 / 320 exactly
        (0.2, 3200, 10),
        (0.01, 160, 1),     # tail read shorter than one frame, never over-read
    ],
)
def test_capture_is_frame_accurate(duration: float, expected_frames: int,
                                   expected_reads: int) -> None:
    data, fake = _record(sample_rate=16000, duration=duration)
    assert data is not None
    assert len(data) // 2 == expected_frames
    assert len(fake.stream.read_calls) == expected_reads
    assert sum(n for n, _ in fake.stream.read_calls) == expected_frames


def test_chunks_are_assembled_in_order_without_gaps() -> None:
    """The int16 counter from the fake stream must survive assembly intact."""
    data, _ = _record(sample_rate=16000, duration=0.1)
    assert data is not None
    samples = array.array("h", data)
    assert list(samples[:20]) == list(range(20))
    assert list(samples[-20:]) == list(range(1600 - 20, 1600))
    # strictly continuous: no dropped or duplicated frames
    assert all(samples[i + 1] - samples[i] == 1 for i in range(len(samples) - 1))


def test_default_device_is_system_default() -> None:
    _, fake = _record(sample_rate=16000, duration=0.05)
    assert fake.stream.open_kwargs["input_device_index"] is None


def test_device_index_is_forwarded() -> None:
    _, fake = _record(sample_rate=16000, duration=0.05, device_index=3)
    assert fake.stream.open_kwargs["input_device_index"] == 3


def test_gain_defaults_to_zero_and_hard_clips_on_request() -> None:
    default_gain, _ = _record(sample_rate=16000, duration=0.01)
    explicit_zero, _ = _record(sample_rate=16000, duration=0.01, gain_db=0.0)
    boosted, _ = _record(sample_rate=16000, duration=0.01, gain_db=6.0)
    assert default_gain is not None and explicit_zero is not None
    assert boosted is not None
    assert default_gain == explicit_zero  # 0 dB is the default: true mic level kept
    assert boosted != default_gain  # a real gain request changes the samples

    # +6.02 dB doubles the signal; a hot sample must clip, never wrap around.
    pcm = array.array("h", [1000, -1000, 30000, -30000]).tobytes()
    out = array.array("h", collect_samples._apply_gain(pcm, 6.0206))
    assert out[0] == 2000
    assert out[1] == -2000
    assert out[2] == 32767, "positive gain must hard-clip"
    assert out[3] == -32768, "negative gain must hard-clip"
    assert collect_samples._apply_gain(pcm, 0.0) == pcm


def test_audio_contract_constants() -> None:
    assert collect_samples.FRAME_SAMPLES == 320, "16 kHz / 20 ms"
    assert collect_samples.MAX_MIC_BUFFER_FRAMES == 1024
    assert OLD_WHOLE_CLIP_BUFFER > collect_samples.MAX_MIC_BUFFER_FRAMES


# --------------------------------------------------------------------------
# CLI wiring: the parsed options actually reach the capture boundary.
# --------------------------------------------------------------------------


def test_cli_forwards_device_and_gain(monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path) -> None:
    seen: list[dict] = []

    def fake_clip(sample_rate: int, duration: float, **kw) -> bytes:
        seen.append({"sample_rate": sample_rate, "duration": duration, **kw})
        return array.array("h", [0] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_record_clip", fake_clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "2", "--duration", "0.02", "--delay", "0",
         "--device", "5", "--gain", "6"],
    )
    collect_samples.main()

    assert len(seen) == 2
    assert all(c["device_index"] == 5 for c in seen)
    assert all(c["gain_db"] == 6.0 for c in seen)


def test_cli_default_gain_is_zero(monkeypatch: pytest.MonkeyPatch,
                                  tmp_path: Path) -> None:
    seen: list[dict] = []

    def fake_clip(sample_rate: int, duration: float, **kw) -> bytes:
        seen.append(kw)
        return array.array("h", [1000] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_record_clip", fake_clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "1", "--duration", "0.02", "--delay", "0"],
    )
    collect_samples.main()
    assert seen[0]["gain_db"] == 0.0
    assert seen[0]["device_index"] is None


def test_cli_list_devices_does_not_require_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["violawake-collect", "--list-devices"])
    monkeypatch.setattr(collect_samples, "_import_pyaudio", lambda: FakePyAudioModule())
    with pytest.raises(SystemExit) as exc:
        collect_samples.main()
    assert exc.value.code == 0


def test_cli_requires_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["violawake-collect", "--word", "jarvis"])
    with pytest.raises(SystemExit) as exc:
        collect_samples.main()
    assert exc.value.code != 0


# --------------------------------------------------------------------------
# Audible pre-roll: the prompt to speak must be heard, not just read.
# --------------------------------------------------------------------------


def _run_cli_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                    extra_args: list[str]) -> list[tuple]:
    """Run main() with cue + capture instrumented; return the ordered event log."""
    events: list[tuple] = []

    def fake_cue(freq_hz: int = collect_samples.CUE_TICK_HZ,
                 duration_ms: int = collect_samples.CUE_MS,
                 *, blocking: bool = True) -> bool:
        events.append(("cue", int(freq_hz), bool(blocking)))
        return True

    def fake_clip(sample_rate: int, duration: float, **kw) -> bytes:
        events.append(("capture", float(duration)))
        return array.array("h", [1000] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_play_cue", fake_cue)
    monkeypatch.setattr(collect_samples, "_record_clip", fake_clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path), *extra_args],
    )
    collect_samples.main()
    return events


def test_start_tone_fires_immediately_before_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The start tone is launched non-blocking, then the mic opens — no gap."""
    events = _run_cli_events(
        monkeypatch, tmp_path,
        ["--count", "1", "--duration", "1.5", "--delay", "0"],
    )
    assert events == [("cue", collect_samples.CUE_START_HZ, False), ("capture", 1.5)]


def test_capture_opens_on_the_same_instant_as_the_start_tone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tone and the mic start together — capture opens within ms of the cue."""
    stamps: list[float] = []

    def cue(freq_hz: int = 700, duration_ms: int = 120, *, blocking: bool = True) -> bool:
        # A blocking tone would sit here for CUE_MS; non-blocking must not.
        if blocking:
            time.sleep(duration_ms / 1000.0)
        stamps.append(time.monotonic())
        return True

    def clip(sample_rate: int, duration: float, **kw) -> bytes:
        stamps.append(time.monotonic())
        return array.array("h", [1000] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_play_cue", cue)
    monkeypatch.setattr(collect_samples, "_record_clip", clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "1", "--duration", "1.5", "--delay", "0"],
    )
    collect_samples.main()

    assert len(stamps) == 2
    assert stamps[1] - stamps[0] < 0.05, (
        f"mic opened {(stamps[1] - stamps[0]) * 1000:.0f} ms after the tone — "
        "the start cue is blocking again and eats the speaker's reaction time"
    )


def test_countdown_ticks_precede_the_start_tone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _run_cli_events(
        monkeypatch, tmp_path,
        ["--count", "1", "--duration", "1.5", "--delay", "1"],
    )
    ticks = [e for e in events if e[1] == collect_samples.CUE_TICK_HZ]
    assert len(ticks) == 1, ticks
    assert all(t[2] is True for t in ticks), "countdown ticks block in sequence"
    start = next(e for e in events if e[1] == collect_samples.CUE_START_HZ)
    assert start[2] is False, "start tone must not block the capture"
    assert events[-1][0] == "capture"


def test_no_cue_flag_silences_prompts_but_still_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _run_cli_events(
        monkeypatch, tmp_path,
        ["--count", "1", "--duration", "1.5", "--delay", "1", "--no-cue"],
    )
    assert not [e for e in events if e[0] == "cue"], events
    assert events == [("capture", 1.5)]


def test_missing_tone_backend_does_not_abort_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """A machine with no playable tone path still records, and says so once."""
    def dead_cue(freq_hz: int = 700, duration_ms: int = 120,
                 *, blocking: bool = True) -> bool:
        return False

    monkeypatch.setattr(collect_samples, "_play_cue", dead_cue)
    monkeypatch.setattr(
        collect_samples, "_record_clip",
        lambda sr, dur, **kw: array.array("h", [1000] * 320).tobytes(),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "1", "--duration", "1.5", "--delay", "0"],
    )
    collect_samples.main()
    captured = capsys.readouterr()
    assert "no system tone backend" in captured.err
    assert (tmp_path / "sample_0001.wav").exists()


def test_play_cue_never_raises_when_backend_is_broken(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """_play_cue must degrade to the terminal bell, never kill a recording run."""
    monkeypatch.setattr(collect_samples.shutil, "which", lambda name: "/usr/bin/afplay")

    def boom(*a, **k):
        raise OSError("afplay exploded")

    monkeypatch.setattr(collect_samples.subprocess, "run", boom)
    assert collect_samples._play_cue(1400, 120) is False

    class _Ok:
        returncode = 0

    monkeypatch.setattr(collect_samples.subprocess, "run", lambda *a, **k: _Ok())

    def boom_popen(*a, **k):
        raise OSError("afplay exploded")

    monkeypatch.setattr(collect_samples.subprocess, "Popen", boom_popen)
    assert collect_samples._play_cue(1400, 120, blocking=False) is False


def test_play_cue_hands_a_real_wav_to_the_backend(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    played: dict = {}

    class Res:
        returncode = 0

    def fake_run(cmd, *a, **k):
        played["cmd"] = list(cmd)
        return Res()

    monkeypatch.setattr(collect_samples.shutil, "which", lambda name: "/usr/bin/afplay")
    monkeypatch.setattr(collect_samples.subprocess, "run", fake_run)
    assert collect_samples._play_cue(1400, 120) is True
    wav = played["cmd"][-1]
    with wave.open(wav) as rf:
        assert rf.getframerate() == 44_100
        assert rf.getnchannels() == 1
        assert abs(rf.getnframes() / rf.getframerate() - 0.12) < 0.01


def test_generated_tone_is_audible_content_at_the_requested_pitch(tmp_path: Path) -> None:
    """Not silence, not a click: a real 1.4 kHz tone with a fade in/out."""
    path = collect_samples._tone_wav(1400, 120)
    with wave.open(str(path)) as rf:
        raw = rf.readframes(rf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    assert np.max(np.abs(samples)) > 32767 * collect_samples.CUE_LEVEL * 0.8
    assert abs(samples[0]) < 500 and abs(samples[-1]) < 500, "click-free envelope"
    spec = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / 44_100)
    assert abs(freqs[int(np.argmax(spec))] - 1400) < 1400 * 0.05


def test_tone_length_does_not_inflate_the_countdown() -> None:
    """Pacing is deadline-based: a 120 ms tone must not add 120 ms per tick."""
    def slow_cue(freq_hz: int = 700, duration_ms: int = 120) -> bool:
        time.sleep(duration_ms / 1000.0)
        return True

    original = collect_samples._play_cue
    collect_samples._play_cue = slow_cue
    try:
        start = time.monotonic()
        assert collect_samples._countdown(2.0, cue=True) is True
        elapsed = time.monotonic() - start
    finally:
        collect_samples._play_cue = original
    assert 1.9 <= elapsed < 2.15, elapsed


def test_cue_constants_are_distinct() -> None:
    assert collect_samples.CUE_START_HZ > collect_samples.CUE_TICK_HZ
    assert 0 < collect_samples.CUE_LEVEL < 1
    assert collect_samples.CUE_TRIM_MS > collect_samples.CUE_MS, \
        "trim must cover the tone plus its room decay"


# --------------------------------------------------------------------------
# Cue lead-in: capture opens ON the tone, and that window is captured then
# thrown away so the beep never becomes part of the training corpus.
# --------------------------------------------------------------------------


def test_cue_lead_in_is_captured_then_discarded() -> None:
    """Saved clip must be the full duration, starting after the dropped head."""
    fake_mod = FakePyAudioModule()
    data = collect_samples._record_clip(
        sample_rate=16000, duration=1.5, skip_frames=3200, pa_module=fake_mod
    )
    assert data is not None
    samples = array.array("h", data)
    assert len(samples) == 24000, "trim must not shorten the saved clip"
    assert samples[0] == 3200, "the cue window is dropped from the HEAD, not the tail"
    assert list(samples[:5]) == list(range(3200, 3205))
    assert list(samples[-3:]) == [27197, 27198, 27199]
    # total mic time consumed = lead-in + clip
    assert sum(n for n, _ in fake_mod.stream.read_calls) == 3200 + 24000


@pytest.mark.parametrize("skip", [0, 1, 3199, 3200, 3201, 24000])
def test_lead_in_reads_stay_inside_the_chunk_bound(skip: int) -> None:
    """The ratchet covers the discard loop too, not just the saved-clip loop."""
    _, fake = _record(sample_rate=16000, duration=0.05, skip_frames=skip)
    assert fake.stream.read_calls, "lead-in must be read in chunks, never one gulp"
    for num_frames, _ in fake.stream.read_calls:
        assert num_frames <= collect_samples.MAX_MIC_BUFFER_FRAMES


def test_lead_in_rejects_negative_skip() -> None:
    with pytest.raises(ValueError, match="skip_frames"):
        collect_samples._record_clip(sample_rate=16000, duration=0.05, skip_frames=-1)


def test_cli_trims_cue_window_by_default(monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    seen: list[dict] = []

    def clip(sample_rate: int, duration: float, **kw) -> bytes:
        seen.append(kw)
        return array.array("h", [1000] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_play_cue", lambda *a, **k: True)
    monkeypatch.setattr(collect_samples, "_record_clip", clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "1", "--duration", "1.5", "--delay", "0"],
    )
    collect_samples.main()
    expected = int(round(16000 * collect_samples.CUE_TRIM_MS / 1000.0))
    assert seen[0]["skip_frames"] == expected


def test_cli_no_cue_trim_keeps_cue_in_sample(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    seen: list[dict] = []

    def clip(sample_rate: int, duration: float, **kw) -> bytes:
        seen.append(kw)
        return array.array("h", [1000] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_play_cue", lambda *a, **k: True)
    monkeypatch.setattr(collect_samples, "_record_clip", clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "1", "--duration", "1.5", "--delay", "0", "--no-cue-trim"],
    )
    collect_samples.main()
    assert seen[0]["skip_frames"] == 0


def test_cli_without_cue_does_not_trim(monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path) -> None:
    seen: list[dict] = []

    def clip(sample_rate: int, duration: float, **kw) -> bytes:
        seen.append(kw)
        return array.array("h", [1000] * 320).tobytes()

    monkeypatch.setattr(collect_samples, "_record_clip", clip)
    monkeypatch.setattr(
        sys, "argv",
        ["violawake-collect", "--word", "jarvis", "--output", str(tmp_path),
         "--count", "1", "--duration", "1.5", "--delay", "0", "--no-cue"],
    )
    collect_samples.main()
    assert seen[0]["skip_frames"] == 0


def test_reap_cue_procs_drops_finished_players_only() -> None:
    class Proc:
        def __init__(self, rc):
            self._rc = rc

        def poll(self):
            return self._rc

    done, running = Proc(0), Proc(None)
    collect_samples.CUE_PROCS[:] = [done, running]
    collect_samples._reap_cue_procs()
    assert [running] == collect_samples.CUE_PROCS
    collect_samples.CUE_PROCS.clear()


def test_non_blocking_cue_launches_the_real_player(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """blocking=False must hand the tone to a background player, not wait."""
    launched: list[list[str]] = []

    class Proc:
        def poll(self):
            return None

    def fake_popen(cmd, *a, **k):
        launched.append(list(cmd))
        return Proc()

    monkeypatch.setattr(collect_samples.shutil, "which", lambda name: "/usr/bin/afplay")
    monkeypatch.setattr(collect_samples.subprocess, "Popen", fake_popen)
    start = time.monotonic()
    assert collect_samples._play_cue(1400, 120, blocking=False) is True
    assert time.monotonic() - start < 0.05, "non-blocking cue waited for playback"
    assert len(launched) == 1
    collect_samples.CUE_PROCS.clear()


def test_cue_trim_covers_the_measured_player_onset_latency() -> None:
    """The trim window must cover when the tone is *audible*, not its length.

    Measured on macOS: launching the tone process returns in ~2 ms, the tone is
    audible ~210-230 ms later (the player opens the audio device first), the
    tone itself is CUE_MS long, and the player process lives ~1.0 s — which is
    exactly why "wait for the player to exit" is the wrong signal: it would
    discard a second of live speech.
    """
    trim_frames = collect_samples.CUE_TRIM_MS / 1000.0
    tone_s = collect_samples.CUE_MS / 1000.0
    onset_s = 0.230  # measured audible onset
    assert trim_frames >= onset_s + tone_s, (
        f"trim {collect_samples.CUE_TRIM_MS} ms does not cover onset {onset_s * 1000:.0f}"
        f" ms + tone {collect_samples.CUE_MS} ms — the cue would leak into samples"
    )
    assert trim_frames <= 0.6, (
        "trim longer than 600 ms starts eating the speaker's own reaction window"
    )


def test_trimmed_head_still_yields_the_full_clip_length() -> None:
    """Discarding the cue must not shorten the saved sample by that amount."""
    trim = int(round(16000 * collect_samples.CUE_TRIM_MS / 1000.0))
    fake_mod = FakePyAudioModule()
    stats: dict = {}
    data = collect_samples._record_clip(
        sample_rate=16000, duration=1.5, skip_frames=trim, stats=stats,
        pa_module=fake_mod,
    )
    assert data is not None
    assert len(data) // 2 == 24000
    assert stats["discarded_frames"] == trim
    # the mic ran trim longer, and what it heard first is gone
    assert sum(n for n, _ in fake_mod.stream.read_calls) == trim + 24000
    samples = array.array("h", data)
    assert samples[0] == trim % 32768
