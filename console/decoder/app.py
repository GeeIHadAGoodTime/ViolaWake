"""Isolated audio decoder sidecar for ViolaWake uploads."""

from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from scipy import signal

app = FastAPI(title="ViolaWake Decoder", version="1.0.0")

TARGET_SAMPLE_RATE = 16_000
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_DURATION_S = 30.0
MIN_DURATION_S = 0.3
MIN_RMS_FLOAT = 10.0 / 32767.0


def _raise_bad_audio(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/decode")
async def decode(request: Request) -> Response:
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file is larger than 5 MB.",
        )

    raw = await request.body()
    if not raw:
        _raise_bad_audio("Audio file is empty.")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file is larger than 5 MB.",
        )

    try:
        data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Could not decode audio as WAV or FLAC.",
        ) from exc

    samples = np.asarray(data, dtype=np.float32)
    if samples.size == 0:
        _raise_bad_audio("Audio file contains no samples.")
    if sample_rate <= 0:
        _raise_bad_audio("Audio file sample rate is invalid.")
    if not np.all(np.isfinite(samples)):
        _raise_bad_audio("Audio file contains invalid samples.")

    mono = samples.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        mono = signal.resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)

    duration_s = float(len(mono) / TARGET_SAMPLE_RATE)
    if duration_s < MIN_DURATION_S:
        _raise_bad_audio("Recording is too short. Minimum duration is 0.3 seconds.")
    if duration_s > MAX_DURATION_S:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Recording is too long. Maximum duration is 30 seconds.",
        )

    rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
    if rms < MIN_RMS_FLOAT:
        _raise_bad_audio("Recording appears to be silent. Please check the audio and try again.")

    out = io.BytesIO()
    sf.write(
        out,
        np.clip(mono, -1.0, 1.0),
        TARGET_SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    return Response(
        content=out.getvalue(),
        media_type="audio/wav",
        headers={"X-Duration-Seconds": f"{duration_s:.6f}"},
    )
