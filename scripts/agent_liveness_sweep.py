"""agent_liveness_sweep — ground-truth check before any "in flight / done / asleep" claim.

Reads the authoritative sources (not the task tracker) for every recent codex
dispatch and worktree:
  - dispatch *.out files in _diag/ from the last 24h
  - matching *.out.pid sidecar (PID of the codex bash wrapper)
  - matching ~/.codex/sessions/<...>/rollout-*.jsonl (live activity)
  - git worktree list + each worktree's HEAD vs main + last-commit age

Emits per-agent state ∈ {RUNNING, DONE_UNPROCESSED, STALLED_WITH_EVIDENCE,
DEAD_NEEDS_REDISPATCH, UNKNOWN_NEEDS_RAW_JSONL_READ} so the caller can stop
reporting "still running" for an agent that finished hours ago.

Run from the project root:
  python scripts/agent_liveness_sweep.py
  python scripts/agent_liveness_sweep.py --json    # machine-readable
  python scripts/agent_liveness_sweep.py --diag _diag/2026-06-07
  python scripts/agent_liveness_sweep.py --window-hours 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        # Old Python / non-TextIOWrapper stream / already-detached stream.
        pass

DEFAULT_DIAG = Path("_diag")
DONE_MARKER = re.compile(r"\[dispatch_codex\]\s+DONE\s+rc=(-?\d+)")


@dataclass
class AgentReport:
    out_file: str
    state: str
    rc: int | None
    pid: int | None
    pid_alive: bool | None
    out_mtime_age_s: float
    last_lines: list[str]
    rollout_path: str | None
    rollout_mtime_age_s: float | None
    note: str


@dataclass
class WorktreeReport:
    path: str
    branch: str
    head_sha: str
    ahead_main: int | None
    last_commit_age_s: float | None
    has_uncommitted: bool | None
    note: str


def _process_alive(pid: int) -> bool:
    """Return True if a process with given PID is alive (cross-platform)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return str(pid) in out
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _tail_lines(path: Path, n: int = 8) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    return lines[-n:]


def _parse_done_rc(lines: list[str]) -> int | None:
    for line in reversed(lines):
        m = DONE_MARKER.search(line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _find_rollout_for_cwd(project_root: Path, after_epoch: float) -> tuple[Path | None, float | None]:
    """Find the most recent codex rollout JSONL that mentions the project root."""
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return None, None
    project_str = str(project_root).replace("\\", "/").lower()
    best: tuple[Path, float] | None = None
    # Walk in mtime-descending order by listing recent dates first
    for path in sorted(sessions_root.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime < after_epoch:
            break
        # Sniff first 4KB for cwd
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                head = fh.read(4096).lower()
        except OSError:
            continue
        if project_str in head:
            best = (path, mtime)
            break
    if best is None:
        return None, None
    return best[0], time.time() - best[1]


def sweep_agents(
    diag_root: Path,
    window_seconds: float,
    project_root: Path,
) -> list[AgentReport]:
    if not diag_root.exists():
        return []
    cutoff = time.time() - window_seconds
    out_files: list[Path] = []
    for path in diag_root.rglob("*.out"):
        try:
            if path.stat().st_mtime >= cutoff:
                out_files.append(path)
        except OSError:
            continue
    # Add *.out.txt and codex_*_out.txt patterns too — dispatchers vary
    for pattern in ("*_out.txt", "*.out.txt"):
        for path in diag_root.rglob(pattern):
            try:
                if path.stat().st_mtime >= cutoff and path not in out_files:
                    out_files.append(path)
            except OSError:
                continue

    reports: list[AgentReport] = []
    now = time.time()
    for out in sorted(out_files, key=lambda p: p.stat().st_mtime, reverse=True):
        out_mtime = out.stat().st_mtime
        age = now - out_mtime
        lines = _tail_lines(out)
        rc = _parse_done_rc(lines)
        pid_file = out.with_suffix(out.suffix + ".pid")
        pid: int | None = None
        alive: bool | None = None
        if pid_file.exists():
            try:
                pid_text = pid_file.read_text(encoding="utf-8", errors="replace").strip()
                pid = int(pid_text)
                alive = _process_alive(pid)
            except (OSError, ValueError):
                pid = None
                alive = None

        rollout_path, rollout_age = _find_rollout_for_cwd(project_root, after_epoch=out_mtime - 60)
        note = ""
        if rc is not None:
            state = "DONE_UNPROCESSED" if rc == 0 else "DEAD_NEEDS_REDISPATCH"
            note = f"completed with rc={rc}"
        elif alive is True:
            # Process alive and no DONE marker -> still running
            state = "RUNNING"
            note = f"pid {pid} alive"
        elif alive is False and pid is not None:
            # Process is gone but no DONE marker -> stalled / died mid-flight
            state = "STALLED_WITH_EVIDENCE"
            note = f"pid {pid} dead, no DONE marker"
        elif rollout_age is not None and rollout_age < 60:
            # No pid info but rollout is actively writing
            state = "RUNNING"
            note = f"rollout updated {rollout_age:.0f}s ago"
        elif age > 600 and rc is None:
            state = "DEAD_NEEDS_REDISPATCH"
            note = "no DONE marker, file idle >10min, no live rollout"
        else:
            state = "UNKNOWN_NEEDS_RAW_JSONL_READ"
            note = "ambiguous — read the rollout JSONL by hand"
        reports.append(
            AgentReport(
                out_file=str(out),
                state=state,
                rc=rc,
                pid=pid,
                pid_alive=alive,
                out_mtime_age_s=age,
                last_lines=lines,
                rollout_path=str(rollout_path) if rollout_path else None,
                rollout_mtime_age_s=rollout_age,
                note=note,
            )
        )
    return reports


def _git_one(cwd: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def sweep_worktrees(project_root: Path) -> list[WorktreeReport]:
    raw = _git_one(project_root, ["worktree", "list", "--porcelain"])
    if not raw:
        return []
    blocks: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = {}
            continue
        if line.startswith("worktree "):
            if cur:
                blocks.append(cur)
                cur = {}
            cur["path"] = line.split(" ", 1)[1]
        elif line.startswith("HEAD "):
            cur["head"] = line.split(" ", 1)[1]
        elif line.startswith("branch "):
            cur["branch"] = line.split(" ", 1)[1]
    if cur:
        blocks.append(cur)

    reports: list[WorktreeReport] = []
    main_sha = _git_one(project_root, ["rev-parse", "main"])
    now = time.time()
    for b in blocks:
        path = b.get("path", "")
        head = b.get("head", "")
        branch = b.get("branch", "(detached)")
        wt = Path(path)
        if not wt.exists():
            reports.append(WorktreeReport(path, branch, head, None, None, None, "worktree path missing"))
            continue
        ahead: int | None = None
        if main_sha and head:
            try:
                count_raw = _git_one(wt, ["rev-list", "--count", f"main..{head}"])
                ahead = int(count_raw) if count_raw.isdigit() else None
            except (ValueError, OSError, subprocess.SubprocessError):
                ahead = None
        last_commit_ts_raw = _git_one(wt, ["log", "-1", "--format=%ct"])
        last_commit_age: float | None
        try:
            last_commit_age = now - int(last_commit_ts_raw) if last_commit_ts_raw.isdigit() else None
        except (ValueError, TypeError):
            last_commit_age = None
        status_raw = _git_one(wt, ["status", "--porcelain"])
        has_uncommitted = bool(status_raw.strip())
        note_bits = []
        if ahead is not None:
            note_bits.append(f"{ahead} ahead of main")
        if last_commit_age is not None:
            note_bits.append(f"last commit {last_commit_age / 3600:.1f}h ago")
        if has_uncommitted:
            note_bits.append("UNCOMMITTED CHANGES")
        reports.append(
            WorktreeReport(
                path=path,
                branch=branch,
                head_sha=head[:12],
                ahead_main=ahead,
                last_commit_age_s=last_commit_age,
                has_uncommitted=has_uncommitted,
                note=", ".join(note_bits) if note_bits else "clean",
            )
        )
    return reports


def emit_text(agents: list[AgentReport], worktrees: list[WorktreeReport]) -> None:
    print("=== AGENT LIVENESS ===")
    if not agents:
        print("  (no agent .out files in window)")
    for a in agents:
        age_h = a.out_mtime_age_s / 3600
        print(f"  {a.state:30s} {a.out_file}")
        print(f"      age={age_h:.1f}h  pid={a.pid} alive={a.pid_alive}  rc={a.rc}  {a.note}")
        if a.rollout_path:
            assert a.rollout_mtime_age_s is not None
            print(f"      rollout={a.rollout_path}  ({a.rollout_mtime_age_s/60:.1f}min ago)")
        if a.last_lines:
            print(f"      tail: {a.last_lines[-1][:140]}")
    print()
    print("=== WORKTREES ===")
    if not worktrees:
        print("  (no worktrees beyond main)")
    for w in worktrees:
        print(f"  {w.branch:40s} {w.head_sha}  {w.path}")
        print(f"      {w.note}")


def emit_json(agents: list[AgentReport], worktrees: list[WorktreeReport]) -> None:
    print(
        json.dumps(
            {
                "agents": [asdict(a) for a in agents],
                "worktrees": [asdict(w) for w in worktrees],
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", type=Path, default=DEFAULT_DIAG, help="Root to scan for *.out (default: _diag/)")
    parser.add_argument("--window-hours", type=float, default=24.0, help="Look-back window (default 24h)")
    parser.add_argument("--json", action="store_true", help="Emit JSON for machine consumers")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root (default cwd)")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    diag_root = args.diag if args.diag.is_absolute() else (project_root / args.diag)

    agents = sweep_agents(diag_root, args.window_hours * 3600, project_root)
    worktrees = sweep_worktrees(project_root)
    if args.json:
        emit_json(agents, worktrees)
    else:
        emit_text(agents, worktrees)
    return 0


if __name__ == "__main__":
    sys.exit(main())
