"""Pre-commit gate: a commit touching CLAUDE.md must carry an explicit approval.

Rare-but-high-blast-radius problem: one unauthorized CLAUDE.md edit propagates
to every future session's context. The 2026-05-23 incident: Claude added
sections Jay didn't approve; he asked "why would you add things to Claude.md
that I didn't approve?" (`feedback_claudemd_requires_approval.md`).

What this check enforces:
    A commit whose diff touches CLAUDE.md must satisfy BOTH:
      1. The commit message contains a `CLAUDEMD-Approved: <session-or-date>`
         trailer.
      2. A companion approval file exists at
         `_diag/<date>/claudemd_approval_*.md`
         (any date directory under _diag/ matches; the file's contents are
         the human-approved wording, kept in the repo for the audit trail.)

Bypass: this check honors the standing `Ratchet-Exempt: docs-only` pattern.
A revert of an unauthorized CLAUDE.md change does NOT need a new approval if
the commit message contains `CLAUDEMD-Revert: <sha>`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        # Old Python / non-TextIOWrapper stream / already-detached stream.
        pass

CLAUDE_MD_PATTERN = re.compile(r"^CLAUDE\.md$|/CLAUDE\.md$")
APPROVED_TRAILER = re.compile(r"^CLAUDEMD-Approved:\s*\S", re.MULTILINE)
REVERT_TRAILER = re.compile(r"^CLAUDEMD-Revert:\s*[0-9a-f]{6,}", re.MULTILINE)
APPROVAL_FILE_GLOB = "_diag/*/claudemd_approval_*.md"


def _staged_files() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _commit_message() -> str:
    """Read commit message from .git/COMMIT_EDITMSG if present (pre-commit hook context)."""
    for candidate in (Path(".git/COMMIT_EDITMSG"), Path(".git/MERGE_MSG")):
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return ""


def _approval_file_exists(repo_root: Path) -> bool:
    for path in repo_root.glob(APPROVAL_FILE_GLOB):
        if path.is_file():
            return True
    return False


def main() -> int:
    files = _staged_files()
    touches_claudemd = any(CLAUDE_MD_PATTERN.search(f) for f in files)
    if not touches_claudemd:
        return 0
    msg = _commit_message()
    if REVERT_TRAILER.search(msg):
        return 0
    if not APPROVED_TRAILER.search(msg):
        sys.stderr.write(
            "BLOCKED: commit touches CLAUDE.md without explicit approval.\n"
            "\n"
            "CLAUDE.md is loaded into every future Claude session — an unauthorized\n"
            "edit propagates indefinitely. Per feedback-claudemd-requires-approval,\n"
            "Jay must approve the exact wording before commit.\n"
            "\n"
            "To unblock this commit (do BOTH):\n"
            "  1. Get Jay to approve the exact wording, then write the approved text to:\n"
            "       _diag/<today's date>/claudemd_approval_<short-slug>.md\n"
            "     and `git add` that file.\n"
            "  2. Add the trailer to the commit message:\n"
            "       CLAUDEMD-Approved: <session-id-or-date>\n"
            "\n"
            "If this commit REVERTS a prior unauthorized CLAUDE.md change, add:\n"
            "  CLAUDEMD-Revert: <reverted-sha>\n"
        )
        return 1
    repo_root = Path.cwd()
    if not _approval_file_exists(repo_root):
        sys.stderr.write(
            "BLOCKED: commit touches CLAUDE.md and has CLAUDEMD-Approved trailer,\n"
            "but no approval companion file found under _diag/*/claudemd_approval_*.md.\n"
            "\n"
            "Add a file at `_diag/<today>/claudemd_approval_<short-slug>.md` containing\n"
            "the approved wording, then `git add` it and re-commit.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
