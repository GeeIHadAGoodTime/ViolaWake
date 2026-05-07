"""Smoke tests for the corpus downloader CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

from violawake_sdk.tools import download_corpus


def _write_librispeech_archive(archive_path: Path, tmp_path: Path) -> None:
    sample_dir = tmp_path / "archive_src" / "LibriSpeech" / "dev-clean" / "84" / "121123"
    sample_dir.mkdir(parents=True)
    (sample_dir / "84-121123-0000.flac").write_bytes(b"fake flac")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(tmp_path / "archive_src" / "LibriSpeech", arcname="LibriSpeech")


def test_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "violawake_sdk.tools.download_corpus", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "violawake-download-corpus" in result.stdout
    assert "--target-dir" in result.stdout


def test_download_librispeech_uses_mocked_urlretrieve(tmp_path: Path) -> None:
    archive_path = tmp_path / "dev-clean.tar.gz"
    _write_librispeech_archive(archive_path, tmp_path)
    target_dir = tmp_path / "corpus"

    def fake_urlretrieve(url: str, filename: str | Path):
        assert url == download_corpus.LIBRISPEECH_DEV_CLEAN_URL
        shutil.copyfile(archive_path, filename)
        return str(filename), None

    with patch(
        "violawake_sdk.tools.download_corpus.urlretrieve",
        side_effect=fake_urlretrieve,
    ) as mocked_urlretrieve:
        destination = download_corpus.download_librispeech_dev_clean(
            target_dir,
            quiet=True,
        )

    assert destination == target_dir / "librispeech" / "dev-clean"
    assert (destination / "84" / "121123" / "84-121123-0000.flac").exists()
    mocked_urlretrieve.assert_called_once()
