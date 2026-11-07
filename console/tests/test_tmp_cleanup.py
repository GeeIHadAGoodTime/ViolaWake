"""A killed training job must not leave locked folders behind for the next boot."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


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
