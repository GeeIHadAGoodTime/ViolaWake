"""Remove what a killed training job left in the temp folder.

A finished job cleans up after itself. A job killed by a restart or a deploy cannot, and
mkdtemp makes its folders 700, so root cannot read them and they sit there for good.
Nothing can be using them at boot: the queue runs in this process and has not started.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import settings

logger = logging.getLogger("violawake.training")

# Every temp folder training makes: job_queue, training_service, and the SDK.
_PREFIX = "violawake_"


def remove_stale_training_dirs(tmp_dir: Path | None = None) -> int:
    """Delete leftover training folders. Returns how many went. Never stops startup."""
    root = tmp_dir or settings.tmp_dir
    removed = 0
    try:
        leftovers = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(_PREFIX)]
    except OSError as exc:
        logger.warning("Could not read the training temp folder %s: %s", root, exc)
        return 0
    for path in leftovers:
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1
    if removed:
        logger.info("Removed %s leftover training folders from %s", removed, root)
    return removed


__all__ = ["remove_stale_training_dirs"]
