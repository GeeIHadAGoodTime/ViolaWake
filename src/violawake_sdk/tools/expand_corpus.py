"""Download and prepare negative evaluation corpora.

Supports:
- LibriSpeech test-clean (5.4h, clean read speech)
- LibriSpeech test-other (5.1h, noisier speech)
- Common Voice clips (configurable size)
- MUSAN noise corpus (music, speech, noise)

Usage::

    # Download LibriSpeech test-clean for FAPH evaluation
    python -m violawake_sdk.tools.expand_corpus \\
        --corpus librispeech-test-clean --output data/negative_corpora/

    # Prepare streaming corpus for continuous FAPH measurement
    python -m violawake_sdk.tools.expand_corpus \\
        --corpus librispeech-test-clean --output data/negative_corpora/ \\
        --prepare-streaming --chunk-seconds 30
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Corpus registry: name -> (URL, description, estimated size)
CORPUS_REGISTRY = {
    "librispeech-test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "description": "LibriSpeech test-clean (5.4h, clean read speech)",
        "size_mb": 346,
    },
    "librispeech-test-other": {
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "description": "LibriSpeech test-other (5.1h, noisier speech)",
        "size_mb": 328,
    },
    "musan-speech": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "description": "MUSAN corpus (music, speech, noise — 42h total)",
        "size_mb": 10_300,
    },
}


def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    """Extract tar safely, preventing path traversal on all Python versions.

    Python 3.12+ supports ``filter="data"`` which blocks traversal natively.
    On 3.10-3.11 the filter kwarg is silently ignored, so we manually reject
    any member whose resolved path escapes *dest*.
    """
    dest_resolved = os.path.realpath(dest)
    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(dest, member.name))
        if not member_path.startswith(dest_resolved + os.sep) and member_path != dest_resolved:
            raise ValueError(
                f"Refusing to extract {member.name!r}: path traversal detected"
            )
    # Safe — all members resolve inside dest
    try:
        tar.extractall(path=dest, filter="data")
    except TypeError:
        # Python <3.12: filter kwarg not supported, but we already validated
        tar.extractall(path=dest)


def _stream_download_and_extract(
    url: str,
    out_path: Path,
    *,
    verbose: bool = True,
    timeout: int = 600,
) -> None:
    """Download a tar.gz to a temp file, then extract — avoids buffering in RAM."""
    try:
        import requests
    except ImportError:
        print("ERROR: requests required. pip install requests", file=sys.stderr)
        sys.exit(1)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        for chunk in response.iter_content(chunk_size=65536):
            tmp.write(chunk)
            downloaded += len(chunk)
            if verbose and total > 0 and downloaded % (10 * 1024 * 1024) < 65536:
                pct = downloaded * 100 // total
                print(f"  Downloaded {downloaded // (1024 * 1024)} MB / {total // (1024 * 1024)} MB ({pct}%)")

    try:
        if verbose:
            print(f"Extracting to {out_path}...")
        with tarfile.open(tmp_path, mode="r:gz") as tar:
            _safe_extract(tar, str(out_path))
    finally:
        os.unlink(tmp_path)


def download_librispeech_test(
    output_dir: str,
    subset: str = "test-clean",
    verbose: bool = True,
) -> Path:
    """Download LibriSpeech test set for FAPH evaluation.

    Args:
        output_dir: Directory to save extracted audio files.
        subset: "test-clean" or "test-other".
        verbose: Print progress messages.

    Returns:
        Path to the directory containing extracted audio files.
    """
    corpus_key = f"librispeech-{subset}"
    if corpus_key not in CORPUS_REGISTRY:
        raise ValueError(
            f"Unknown subset {subset!r}. Available: "
            + ", ".join(k for k in CORPUS_REGISTRY if k.startswith("librispeech"))
        )

    info = CORPUS_REGISTRY[corpus_key]
    url = info["url"]
    out_path = Path(output_dir) / corpus_key
    out_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_flacs = list(out_path.rglob("*.flac"))
    if existing_flacs:
        if verbose:
            print(f"Already downloaded: {len(existing_flacs)} FLAC files in {out_path}")
        return out_path

    if verbose:
        print(f"Downloading {info['description']} (~{info['size_mb']} MB)...")
        print(f"URL: {url}")

    _stream_download_and_extract(url, out_path, verbose=verbose, timeout=300)

    extracted_flacs = list(out_path.rglob("*.flac"))
    if verbose:
        print(f"Extracted {len(extracted_flacs)} FLAC files to {out_path}")

    return out_path


def download_corpus(
    corpus_name: str,
    output_dir: str,
    verbose: bool = True,
) -> Path:
    """Download a corpus by name.

    Args:
        corpus_name: Name from CORPUS_REGISTRY.
        output_dir: Directory to save extracted files.
        verbose: Print progress.

    Returns:
        Path to extracted corpus directory.
    """
    if corpus_name.startswith("librispeech"):
        subset = corpus_name.replace("librispeech-", "")
        return download_librispeech_test(output_dir, subset, verbose)

    if corpus_name not in CORPUS_REGISTRY:
        available = ", ".join(CORPUS_REGISTRY.keys())
        raise ValueError(f"Unknown corpus {corpus_name!r}. Available: {available}")

    # Generic tar.gz download for MUSAN and others
    info = CORPUS_REGISTRY[corpus_name]
    out_path = Path(output_dir) / corpus_name
    out_path.mkdir(parents=True, exist_ok=True)

    existing = list(out_path.rglob("*.wav")) + list(out_path.rglob("*.flac"))
    if existing:
        if verbose:
            print(f"Already downloaded: {len(existing)} audio files in {out_path}")
        return out_path

    if verbose:
        print(f"Downloading {info['description']} (~{info['size_mb']} MB)...")

    _stream_download_and_extract(info["url"], out_path, verbose=verbose, timeout=600)

    if verbose:
        extracted = list(out_path.rglob("*.wav")) + list(out_path.rglob("*.flac"))
        print(f"Extracted {len(extracted)} audio files to {out_path}")

    return out_path


def prepare_streaming_corpus(
    audio_dir: str,
    output_path: str,
    chunk_seconds: float = 30.0,
    sample_rate: int = 16000,
    verbose: bool = True,
) -> Path:
    """Concatenate clips into long streaming files for FAPH measurement.

    Reads all audio files in a directory, concatenates them with small gaps,
    and saves as WAV files. Each output file is approximately chunk_seconds long.

    Args:
        audio_dir: Directory containing audio files (WAV, FLAC).
        output_path: Output directory for streaming WAV files.
        chunk_seconds: Target duration per output file in seconds.
        sample_rate: Target sample rate (default 16000).
        verbose: Print progress.

    Returns:
        Path to the output directory containing streaming WAV files.
    """
    import numpy as np

    from violawake_sdk.audio import load_audio

    audio_files = sorted(
        list(Path(audio_dir).rglob("*.wav"))
        + list(Path(audio_dir).rglob("*.flac"))
    )

    if not audio_files:
        print(f"ERROR: No audio files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading {len(audio_files)} audio files...")

    # Load and concatenate audio with 0.1s silence gaps
    gap_samples = int(sample_rate * 0.1)
    gap = np.zeros(gap_samples, dtype=np.float32)
    chunk_samples = int(chunk_seconds * sample_rate)

    current_chunk: list[np.ndarray] = []
    current_len = 0
    chunk_idx = 0

    def _save_chunk(chunk_data: list[np.ndarray], idx: int) -> None:
        audio = np.concatenate(chunk_data)
        # Convert to int16 for WAV
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wav_path = out_dir / f"stream_{idx:04d}.wav"
        _write_wav(wav_path, audio_int16, sample_rate)
        if verbose:
            duration = len(audio) / sample_rate
            print(f"  Saved {wav_path.name} ({duration:.1f}s)")

    for f in audio_files:
        audio = load_audio(f, target_sr=sample_rate)
        if audio is None:
            continue

        current_chunk.append(audio)
        current_chunk.append(gap)
        current_len += len(audio) + gap_samples

        if current_len >= chunk_samples:
            _save_chunk(current_chunk, chunk_idx)
            chunk_idx += 1
            current_chunk = []
            current_len = 0

    # Save remaining audio
    if current_chunk:
        _save_chunk(current_chunk, chunk_idx)
        chunk_idx += 1

    if verbose:
        total_files = chunk_idx
        print(f"Created {total_files} streaming files in {out_dir}")

    return out_dir


def _write_wav(path: Path, audio_int16: np.ndarray, sample_rate: int) -> None:
    """Write a mono int16 WAV file without external dependencies."""
    n_samples = len(audio_int16)
    data_bytes = audio_int16.tobytes()
    # WAV header (44 bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data_bytes),  # file size - 8
        b"WAVE",
        b"fmt ",
        16,      # chunk size
        1,       # PCM format
        1,       # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,       # block align
        16,      # bits per sample
        b"data",
        len(data_bytes),
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(data_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-expand-corpus",
        description="Download and prepare negative evaluation corpora.",
    )
    parser.add_argument(
        "--corpus",
        choices=list(CORPUS_REGISTRY.keys()),
        help="Corpus to download",
    )
    parser.add_argument(
        "--output",
        help="Output directory for downloaded/prepared files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available corpora and exit",
    )
    parser.add_argument(
        "--prepare-streaming",
        action="store_true",
        help="Prepare streaming corpus from already-downloaded audio",
    )
    parser.add_argument(
        "--audio-dir",
        help="Audio directory for --prepare-streaming (default: auto from --corpus)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk duration for streaming files (default: 30s)",
    )

    args = parser.parse_args()

    if args.list:
        print("Available corpora:")
        for name, info in CORPUS_REGISTRY.items():
            print(f"  {name}: {info['description']} (~{info['size_mb']} MB)")
        return

    if not args.output:
        print("ERROR: --output is required (or use --list)", file=sys.stderr)
        sys.exit(1)

    if args.prepare_streaming:
        audio_dir = args.audio_dir or args.output
        streaming_out = Path(args.output) / "streaming"
        prepare_streaming_corpus(
            audio_dir, str(streaming_out), args.chunk_seconds,
        )
        return

    if args.corpus is None:
        print("ERROR: --corpus is required (or use --list)", file=sys.stderr)
        sys.exit(1)

    corpus_path = download_corpus(args.corpus, args.output)
    print(f"\nCorpus ready at: {corpus_path}")


if __name__ == "__main__":
    main()
