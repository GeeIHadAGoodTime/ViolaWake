"""The backend must start with a folder in its data volume that root cannot read.

2026-11-07: the container runs with every capability dropped except CHOWN, SETUID and
SETGID. Root therefore cannot read a folder the app locked to 700: ONNX Runtime's cache,
and every temp folder from mkdtemp. ``chown -R`` skips it and exits 1, and under ``set -e``
that ended the boot, on the old image as well as the new one, so the deploy's rollback
could not help. The running container had only ever started before the folder existed.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO / "console" / "backend" / "entrypoint.sh"
sys.path.insert(0, str(REPO / "console" / "backend"))

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="runs the real POSIX entrypoint")


def _stub(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_entrypoint(tmp_path: Path, *, chown_exit: int, writable: bool = True, env: dict | None = None):
    """The real entrypoint, as root, with the system tools replaced by stubs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    _stub(bin_dir, "id", 'echo 0')
    _stub(bin_dir, "alembic", 'exit 0')
    _stub(bin_dir, "chown", f'echo "chown: cannot read directory" >&2; exit {chown_exit}')
    _stub(bin_dir, "python", 'echo "SERVER STARTED $*"; exit 0')
    _stub(
        bin_dir,
        "gosu",
        'shift; if [ "$1" = test ]; then exit %d; fi; exec "$@"' % (0 if writable else 1),
    )
    return subprocess.run(
        ["sh", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "VIOLAWAKE_ENTRYPOINT_DATA_DIR": str(data),
            **(env or {}),
        },
    )


@posix_only
def test_a_failed_chown_does_not_stop_startup(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, chown_exit=1)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SERVER STARTED run.py" in result.stdout
    assert "WARNING" in result.stdout


@posix_only
def test_a_data_folder_the_app_cannot_write_still_stops_startup(tmp_path: Path) -> None:
    """Tolerating chown must not turn a real permission fault into a server that fails later."""
    result = _run_entrypoint(tmp_path, chown_exit=1, writable=False)

    assert result.returncode == 1
    assert "FATAL" in result.stdout
    assert "SERVER STARTED" not in result.stdout


@posix_only
def test_the_boot_check_proves_the_step_and_starts_nothing(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, chown_exit=1, env={"VIOLAWAKE_BOOT_CHECK": "1"})

    assert result.returncode == 0
    assert "boot check passed" in result.stdout
    assert "SERVER STARTED" not in result.stdout
    assert "alembic" not in result.stdout, "a preflight must not migrate the live database"


def _docker_works() -> bool:
    if sys.platform == "win32" or shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False).returncode == 0


@pytest.mark.skipif(not _docker_works(), reason="needs docker: CI's Linux runners, never a developer PC")
def test_boots_with_a_locked_cache_and_temp_folder_under_production_capabilities(tmp_path: Path) -> None:
    """The real thing: real root, the production capability set, real 700 folders owned by app.

    This is what stubs cannot show: that root really is locked out, and the boot survives it.
    """
    data = tmp_path / "data"
    stubs = tmp_path / "stubs"
    data.mkdir()
    stubs.mkdir()
    data.chmod(0o777)
    _stub(stubs, "alembic", "exit 0")
    _stub(stubs, "python", 'echo "SERVER STARTED $*"; exit 0')
    _stub(stubs, "gosu", "")
    (stubs / "gosu").write_text(
        '#!/bin/sh\nshift\nexec setpriv --reuid=100 --regid=101 --clear-groups "$@"\n', encoding="utf-8"
    )
    image = "python:3.11-slim"
    mounts = ["-v", f"{data}:/app/data", "-v", f"{stubs}:/stubs:ro", "-v", f"{ENTRYPOINT}:/entrypoint.sh:ro"]

    # A folder of each kind, made with full capabilities, owned by the app user, locked.
    setup = (
        "mkdir -p /app/data/cache/Microsoft /app/data/tmp/violawake_train_abc /app/data/recordings /app/data/models"
        " && chown -R 100:101 /app/data"
        " && chmod 700 /app/data/cache /app/data/cache/Microsoft /app/data/tmp/violawake_train_abc"
    )
    subprocess.run(["docker", "run", "--rm", *mounts, image, "sh", "-c", setup], check=True, timeout=300)

    boot = (
        "echo 'app:x:100:101::/app:/bin/sh' >> /etc/passwd && echo 'app:x:101:' >> /etc/group"
        " && export PATH=/stubs:$PATH"
        " && if ls /app/data/cache >/dev/null 2>&1; then echo 'ROOT CAN READ IT'; fi"
        " && exec sh /entrypoint.sh"
    )
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--security-opt", "no-new-privileges:true",
            *mounts, image, "sh", "-c", boot,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    subprocess.run(["docker", "run", "--rm", *mounts, image, "chmod", "-R", "777", "/app/data"], check=False, timeout=120)

    output = result.stdout + result.stderr
    assert "ROOT CAN READ IT" not in output, "the test no longer reproduces the lockout"
    assert result.returncode == 0, output
    assert "SERVER STARTED run.py" in output


def test_a_restart_clears_what_killed_jobs_left_behind(tmp_path: Path) -> None:
    from app.services.tmp_cleanup import remove_stale_training_dirs

    for name in ("violawake_train_abc", "violawake_neg_x", "violawake_job_42_y"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"x")
        if sys.platform != "win32":
            folder.chmod(0o700)
    (tmp_path / "something_else").mkdir()

    assert remove_stale_training_dirs(tmp_path) == 3

    assert [path.name for path in tmp_path.iterdir()] == ["something_else"]


def test_cleanup_cannot_stop_startup(tmp_path: Path) -> None:
    from app.services.tmp_cleanup import remove_stale_training_dirs

    assert remove_stale_training_dirs(tmp_path / "missing") == 0


def test_startup_clears_them_before_the_queue_can_start_a_job() -> None:
    source = (REPO / "console" / "backend" / "app" / "main.py").read_text(encoding="utf-8")
    assert source.index("remove_stale_training_dirs()") < source.index("await init_job_queue()")
