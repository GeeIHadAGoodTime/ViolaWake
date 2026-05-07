"""Download the shared speech-negative corpus for ViolaWake training.

Entry point: ``violawake-download-corpus`` (declared in pyproject.toml).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

LIBRISPEECH_DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DEFAULT_CORPUS_DIR = Path.home() / ".violawake" / "corpus"


def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return directory == target or directory in target.parents


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a tar archive after rejecting path traversal entries."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing to extract link from corpus archive: {member.name}")
            member_path = destination / member.name
            if not _is_within_directory(destination, member_path):
                raise RuntimeError(f"Refusing unsafe corpus archive path: {member.name}")
        tar.extractall(destination)


def _find_dev_clean(extract_dir: Path) -> Path:
    expected = extract_dir / "LibriSpeech" / "dev-clean"
    if expected.exists():
        return expected

    for path in extract_dir.rglob("dev-clean"):
        if path.is_dir():
            return path

    raise RuntimeError("Downloaded LibriSpeech archive did not contain dev-clean/")


def download_librispeech_dev_clean(
    target_dir: Path = DEFAULT_CORPUS_DIR,
    *,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Download and extract LibriSpeech dev-clean under ``target_dir/librispeech``."""
    corpus_dir = target_dir.expanduser().resolve()
    librispeech_dir = corpus_dir / "librispeech"
    destination = librispeech_dir / "dev-clean"

    if destination.exists() and any(destination.rglob("*.flac")) and not force:
        if not quiet:
            print(f"LibriSpeech dev-clean already exists at {destination}")
        return destination

    corpus_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="violawake_corpus_"))
    archive_path = temp_dir / "dev-clean.tar.gz"
    extract_dir = temp_dir / "extract"
    extract_dir.mkdir()

    try:
        if not quiet:
            print(f"Downloading LibriSpeech dev-clean from {LIBRISPEECH_DEV_CLEAN_URL}")
            print(f"Target corpus directory: {corpus_dir}")
        urlretrieve(LIBRISPEECH_DEV_CLEAN_URL, archive_path)

        if not quiet:
            print("Extracting LibriSpeech dev-clean...")
        _safe_extract_tar(archive_path, extract_dir)
        extracted_dev_clean = _find_dev_clean(extract_dir)

        if destination.exists():
            shutil.rmtree(destination)
        librispeech_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_dev_clean), str(destination))

        if not quiet:
            flac_count = sum(1 for _ in destination.rglob("*.flac"))
            print(f"Installed LibriSpeech dev-clean at {destination} ({flac_count} FLAC files)")
        return destination
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="violawake-download-corpus",
        description=(
            "Download the shared speech-negative corpus used by ViolaWake training. "
            "Currently installs LibriSpeech dev-clean."
        ),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Corpus install directory (default: ~/.violawake/corpus).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing LibriSpeech dev-clean download.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        destination = download_librispeech_dev_clean(
            args.target_dir,
            force=args.force,
            quiet=args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: failed to download corpus: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Corpus ready.")
        print(f"Mount this directory as /app/corpus in production: {destination.parent.parent}")


if __name__ == "__main__":
    main()
