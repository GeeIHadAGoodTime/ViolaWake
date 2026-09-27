"""
violawake-collect CLI — Record positive wake word samples for model training.

Entry point: ``violawake-collect`` (declared in pyproject.toml).

Requires: ``pip install "violawake[audio]"`` and a working microphone.

Usage::

    violawake-collect --word "jarvis" --output data/jarvis/positives/ --count 50

    Records 50 1.5-second audio clips of the wake word "jarvis" to the given
    output directory. Each clip is saved as positives/sample_001.wav, etc.

    The recording loop displays a countdown timer, so you can pace your
    recordings: each clip will record automatically when the timer reaches 0.

    violawake-collect --list-devices          # show capture devices
    violawake-collect --word "jarvis" --device 0 --gain 6 ...   # pick a mic, boost it

    Before every clip the tool plays an audible pre-roll: one low tick for each
    countdown second, then a high tone that starts on the *same instant* the
    microphone opens, so your reaction time falls inside the recording. That
    cue window is captured and discarded, keeping the beep out of the training
    samples. Pass ``--no-cue`` for text-only prompts, or ``--no-cue-trim`` to
    keep the cue inside the file.

Audio contract (see ``violawake_sdk.audio_source``): 16 kHz mono, 16-bit, read
in 20 ms (320-sample) frames. A microphone clip MUST NOT be requested as one
giant PortAudio buffer. With ``frames_per_buffer`` set to the whole clip, the
blocking read takes roughly ``duration * frames_per_buffer / 1024`` seconds and
hands back time-decimated audio with dropped frames — a "1.5 second" sample is
actually ~36 seconds of chopped-up sound. ``exception_on_overflow`` is also
left at its default so a dropped frame becomes a visible error, not silence.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from violawake_sdk._constants import SAMPLE_RATE
from violawake_sdk.audio_source import FRAME_SAMPLES

# Largest PortAudio buffer (in frames) the mic stream may be opened with, or
# read from, in one call. Kept above the 20 ms contract frame so non-contract
# chunk sizes stay possible, but far below "one whole clip per read".
MAX_MIC_BUFFER_FRAMES = 1024

# Int16 peak below which a clip is treated as (near) silence — the mic is muted,
# the wrong device was picked, or the speaker was too quiet to be useful.
NEAR_SILENT_PEAK = 500

# --- audible pre-roll cue -------------------------------------------------
# Each countdown second plays CUE_TICK_HZ; the exact moment capture opens
# plays CUE_START_HZ. Hands-free pacing must be heard, not read.
CUE_TICK_HZ = 700
CUE_START_HZ = 1400
CUE_MS = 120
CUE_LEVEL = 0.28  # amplitude of the generated tone, fraction of full scale

# The capture opens *with* the start tone, so the tone is picked up by the mic.
# That window is captured and thrown away — the saved clip starts once the tone
# is genuinely over, so no 1.4 kHz cue leaks into training samples (a cue in
# every positive clip is a feature the wake model would happily learn).
#
# Why 450 ms and not "tone length": measured on macOS, the tone is not audible
# until ~210-230 ms after the player is launched (afplay opens the CoreAudio
# device first; the process itself lives ~1.0 s for a 120 ms tone, which is why
# waiting on the player to exit is the wrong signal). 450 = ~230 onset + 120
# tone + ~100 room-decay guard. Nothing is lost at the tail: the capture window
# is extended by exactly the discarded amount.
CUE_TRIM_MS = 450

_CUE_DIR: Path | None = None
_CUE_TONES: dict[tuple[int, int], Path] = {}
CUE_PROCS: list[object] = []  # non-blocking tone players, reaped after each clip


def _tone_wav(freq_hz: int, duration_ms: int) -> Path:
    """Generate (once) and return a short sine WAV for the cue backend to play."""
    global _CUE_DIR
    key = (freq_hz, duration_ms)
    cached = _CUE_TONES.get(key)
    if cached is not None and cached.exists():
        return cached
    if _CUE_DIR is None or not _CUE_DIR.exists():
        _CUE_DIR = Path(tempfile.mkdtemp(prefix="violawake-cue-"))
        _CUE_TONES.clear()
    sr = 44_100
    n = max(1, int(sr * duration_ms / 1000))
    fade = max(1, int(sr * 0.005))  # 5 ms envelope, keeps the tone click-free
    path = _CUE_DIR / f"cue_{freq_hz}_{duration_ms}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            env = min(1.0, i / fade, (n - i) / fade)
            value = int(32767 * CUE_LEVEL * env * math.sin(2 * math.pi * freq_hz * i / sr))
            frames += value.to_bytes(2, "little", signed=True)
        wf.writeframes(bytes(frames))
    _CUE_TONES[key] = path
    return path


def _play_cue(freq_hz: int = CUE_TICK_HZ, duration_ms: int = CUE_MS,
              *, blocking: bool = True) -> bool:
    """Play a best-effort audible tone. Never raises — a cue must not kill a session.

    Args:
        blocking: When False the tone is started in the background so the
            caller can open the microphone on the same instant the tone sounds.

    Returns True when a real tone backend was launched. False means the only
    fallback was the terminal bell, which the terminal may have muted.
    """
    try:
        if sys.platform == "win32":
            import winsound

            if blocking:
                winsound.Beep(int(freq_hz), int(duration_ms))
            else:
                threading.Thread(
                    target=winsound.Beep, args=(int(freq_hz), int(duration_ms)),
                    daemon=True,
                ).start()
            return True

        wav = _tone_wav(int(freq_hz), int(duration_ms))
        for exe, args in (("afplay", [str(wav)]),
                          ("aplay", ["-q", "-N", str(wav)]),
                          ("paplay", [str(wav)])):
            found = shutil.which(exe)
            if found is None:
                continue
            if blocking:
                done = subprocess.run([found, *args], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, timeout=2.0, check=False)
                if done.returncode == 0:
                    return True
                continue
            CUE_PROCS.append(subprocess.Popen([found, *args], stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL))
            return True
    except Exception as e:  # noqa: BLE001 — cue failure must never abort recording
        print(f"note: audio cue failed ({e}); using terminal bell instead", file=sys.stderr)

    with contextlib.suppress(Exception):
        sys.stdout.write("\a")
        sys.stdout.flush()
    return False


def _reap_cue_procs() -> None:
    """Reap background tone players so a long session does not leak processes."""
    for proc in list(CUE_PROCS):
        with contextlib.suppress(Exception):
            if proc.poll() is not None:
                CUE_PROCS.remove(proc)


def _cleanup_cue_files() -> None:
    """Remove the generated cue tone cache (best effort)."""
    if _CUE_DIR is None:
        return
    shutil.rmtree(_CUE_DIR, ignore_errors=True)


def _import_pyaudio() -> object | None:
    """Import pyaudio, or print the install hint and return None."""
    try:
        import pyaudio
    except ImportError:
        print(
            "ERROR: pyaudio is required for microphone features. "
            "Install with: pip install violawake[audio]",
            file=sys.stderr,
        )
        return None
    return pyaudio


def _apply_gain(raw: bytes, gain_db: float) -> bytes:
    """Scale int16 PCM by ``gain_db`` and hard-clip to the int16 range."""
    if gain_db == 0.0:
        return raw
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    samples *= 10.0 ** (gain_db / 20.0)
    return np.clip(samples, -32768.0, 32767.0).astype(np.int16).tobytes()


def _peak(raw: bytes) -> int:
    """Max abs sample value of an int16 PCM buffer (0 for empty input)."""
    if not raw:
        return 0
    samples = np.frombuffer(raw, dtype=np.int16)
    return int(np.max(np.abs(samples))) if samples.size else 0


def _record_clip(
    sample_rate: int = SAMPLE_RATE,
    duration: float = 1.5,
    *,
    device_index: int | None = None,
    gain_db: float = 0.0,
    chunk_frames: int = FRAME_SAMPLES,
    skip_frames: int = 0,
    stats: dict | None = None,
    pa_module: object | None = None,
) -> bytes | None:
    """Record a single audio clip from a microphone, in 20 ms frames.

    Args:
        sample_rate: Capture rate in Hz (16000 = SDK audio contract).
        duration: Clip length in seconds.
        device_index: PyAudio input device index, or None for the system default.
        gain_db: Post-capture gain in dB. 0.0 keeps the true mic level.
        chunk_frames: Frames requested per ``stream.read()``; bounded by
            :data:`MAX_MIC_BUFFER_FRAMES`.
        skip_frames: Frames captured and discarded at the head of the clip —
            the start-tone window, so the cue never lands in the saved sample.
            The clip is still ``sample_rate * duration`` frames long; the
            capture simply runs ``skip_frames`` longer at the head.
        stats: Optional dict filled with capture diagnostics
            (``discarded_frames``, ``captured_frames``).
        pa_module: Injection point for the pyaudio module (tests use a fake).

    Returns:
        Raw little-endian int16 mono PCM of ``sample_rate * duration`` frames,
        or None if the capture failed.

    Raises:
        ValueError: If the requested chunk violates the mic buffer contract.
    """
    if chunk_frames > MAX_MIC_BUFFER_FRAMES:
        raise ValueError(
            "collect_samples violates the ViolaWake mic buffer contract: "
            f"chunk_frames={chunk_frames} exceeds {MAX_MIC_BUFFER_FRAMES}. "
            "Reading a whole clip in one blocking stream.read() stalls for "
            "seconds and returns dropped-frame audio."
        )
    if sample_rate <= 0 or duration <= 0:
        raise ValueError(f"bad capture request: rate={sample_rate} duration={duration}")
    if skip_frames < 0:
        raise ValueError(f"bad skip_frames: {skip_frames}")

    total_frames = int(round(sample_rate * duration))
    pa = pa_module if pa_module is not None else _import_pyaudio()
    if pa is None:
        return None

    instance = pa.PyAudio()  # type: ignore[attr-defined]
    stream = None
    try:
        stream = instance.open(
            format=pa.paInt16,  # type: ignore[attr-defined]
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_frames,
            input_device_index=device_index,
        )
        parts: list[bytes] = []
        captured = 0

        # Lead-in: the mic is already open while the start tone plays, so this
        # window is read and dropped rather than saved into the training clip.
        discarded = 0
        while discarded < skip_frames:
            n_frames = min(chunk_frames, skip_frames - discarded)
            dropped = stream.read(n_frames)
            if len(dropped) // 2 != n_frames:
                print(
                    f"ERROR: microphone returned a short frame during cue "
                    f"lead-in ({len(dropped) // 2} of {n_frames}); clip aborted",
                    file=sys.stderr,
                )
                return None
            discarded += n_frames

        while captured < total_frames:
            n_frames = min(chunk_frames, total_frames - captured)
            # exception_on_overflow stays at its default (True): a dropped
            # frame must abort the clip, not silently shorten it.
            data = stream.read(n_frames)
            if len(data) // 2 != n_frames:
                print(
                    f"ERROR: microphone returned a short frame "
                    f"({len(data) // 2} of {n_frames} samples); clip aborted",
                    file=sys.stderr,
                )
                return None
            parts.append(data)
            captured += n_frames

        raw = b"".join(parts)
        if stats is not None:
            stats["discarded_frames"] = discarded
            stats["captured_frames"] = captured
    except Exception as e:
        print(f"ERROR: Could not record from microphone: {e}", file=sys.stderr)
        return None
    finally:
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.stop_stream()
                stream.close()
        with contextlib.suppress(Exception):
            instance.terminate()

    return _apply_gain(raw, gain_db)


def _save_wav(data: bytes, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Save raw PCM bytes as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def _list_input_devices(pa_module: object) -> int:
    """Print input-capable devices; returns the number printed."""
    instance = pa_module.PyAudio()  # type: ignore[attr-defined]
    count = 0
    try:
        for i in range(instance.get_device_count()):
            info = instance.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                count += 1
                print(
                    f"  [{info['index']}] {info['name']} — "
                    f"{info['maxInputChannels']} input channel(s), "
                    f"default {info['defaultSampleRate']:.0f}Hz"
                )
    finally:
        with contextlib.suppress(Exception):
            instance.terminate()
    if count == 0:
        print("No input devices found. Check the microphone connection.")
    return count


def _warn_cue_backend() -> None:
    """Tell the user once that the audible cue fell back to the terminal bell."""
    print(
        "\nnote: no system tone backend — prompts fell back to the terminal bell."
        " Enable the terminal bell, or use --no-cue for text-only prompts.",
        file=sys.stderr,
    )


def _countdown(seconds: float, *, cue: bool = True) -> bool:
    """Count down on screen (and by tone) so sub-second delays still work.

    Pacing is deadline-based: the tone itself takes ~120 ms, so sleeping a flat
    1.0 s per tick would drift the countdown longer than --delay asks for.

    Returns True when every tick was an audible tone (or cues are off), False
    when the terminal-bell fallback had to be used.
    """
    end = time.monotonic() + seconds
    tick = math.ceil(seconds)
    audible = True
    while time.monotonic() < end:
        if cue:
            audible = _play_cue(CUE_TICK_HZ) and audible
        print(f"{tick}... ", end="", flush=True)
        time.sleep(max(0.0, end - (tick - 1) - time.monotonic()))
        tick -= 1
    return audible


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-collect",
        description="Record positive wake word samples for model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--word",
        required=False,
        default="",
        metavar="WORD",
        help="The wake word you are recording (used for display only)",
    )
    parser.add_argument(
        "--output",
        required=False,
        metavar="DIR",
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        metavar="N",
        help="Number of samples to record (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        metavar="SEC",
        help="Duration of each recording in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Pause between recordings in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        metavar="HZ",
        help=f"Sample rate in Hz (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Input device index (default: system default device)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.0,
        metavar="DB",
        help="Gain in dB applied after capture (default: 0 = keep true mic level)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit",
    )
    parser.add_argument(
        "--no-cue",
        action="store_true",
        help="Disable the audible countdown/start tones (text prompts only)",
    )
    parser.add_argument(
        "--no-cue-trim",
        action="store_true",
        help=f"Keep the start tone inside the recorded sample (default: the "
             f"{CUE_TRIM_MS} ms cue window is captured, then discarded)",
    )
    args = parser.parse_args()

    if args.list_devices:
        pa = _import_pyaudio()
        if pa is None:
            sys.exit(1)
        _list_input_devices(pa)
        sys.exit(0)

    if not args.output:
        parser.error("--output is required unless --list-devices is used")
    if args.count < 1:
        parser.error("--count must be at least 1")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing samples to continue numbering
    existing = sorted(output_dir.glob("sample_*.wav"))
    start_idx = len(existing) + 1
    last_idx = start_idx + args.count - 1

    word = args.word or "(wake word)"
    device_desc = "system default" if args.device is None else f"index {args.device}"
    cue = not args.no_cue
    # Capture opens on the start tone itself, so the cue window is read and
    # thrown away unless the user asks to keep it.
    cue_skip = 0 if (not cue or args.no_cue_trim) else int(
        round(args.sample_rate * CUE_TRIM_MS / 1000.0)
    )
    print(f"Recording '{word}' wake word samples")
    print(f"Output: {output_dir}")
    print(f"Count: {args.count} | Duration: {args.duration}s | Delay: {args.delay}s")
    print(f"Device: {device_desc} | Gain: {args.gain:+.1f}dB | "
          f"Frame: {FRAME_SAMPLES} samples @ {args.sample_rate}Hz")
    if cue:
        print(f"Cue: {CUE_TICK_HZ}Hz ticks + {CUE_START_HZ}Hz start tone "
              f"(lead-in trimmed: {cue_skip} frames)")
    print()
    print("Press Ctrl+C to stop early.")
    print()

    recorded = 0
    quiet_warned = False
    cue_fallback_warned = False
    try:
        for i in range(start_idx, last_idx + 1):
            path = output_dir / f"sample_{i:04d}.wav"

            if args.delay > 0:
                print(f"[{i}/{last_idx}] Ready in ", end="", flush=True)
                if not _countdown(args.delay, cue=cue) and not cue_fallback_warned:
                    _warn_cue_backend()
                    cue_fallback_warned = True
                print()

            print(f"SAY '{word}' NOW!", end=" ", flush=True)
            if cue and not _play_cue(CUE_START_HZ, blocking=False) \
                    and not cue_fallback_warned:
                _warn_cue_backend()
                cue_fallback_warned = True
            clip_stats: dict = {}
            data = _record_clip(
                args.sample_rate,
                args.duration,
                device_index=args.device,
                gain_db=args.gain,
                skip_frames=cue_skip,
                stats=clip_stats,
            )
            _reap_cue_procs()

            if data is not None:
                _save_wav(data, path, args.sample_rate)
                peak = _peak(data)
                lead_in = int(clip_stats.get("discarded_frames", 0))
                note = ""
                if peak < NEAR_SILENT_PEAK and not quiet_warned:
                    note = "  (!) near-silent — wrong device, muted, or too quiet"
                    quiet_warned = True
                print(f"OK {path.name} (peak {peak}, cue lead-in {lead_in} frames){note}")
                recorded += 1
            else:
                print("FAILED (recording failed)")

    except KeyboardInterrupt:
        print()
        print("Recording stopped early.")

    _cleanup_cue_files()

    print()
    print(f"Recorded {recorded} samples to {output_dir}")

    if recorded < 20:
        print()
        print("TIP: For good model accuracy, collect at least 50 samples.")
        print("     Use different speaking styles, distances, and room positions.")

    if recorded == 0 and args.count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
