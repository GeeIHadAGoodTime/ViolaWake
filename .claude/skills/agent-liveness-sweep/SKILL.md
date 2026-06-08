---
name: agent-liveness-sweep
description: Ground-truth check before any "in flight / still running / done / asleep / stalled" claim about codex dispatches or worktrees. Reads the authoritative sources (.pid sidecar, DONE marker, rollout JSONL, worktree HEAD) and emits per-agent state. Trust the worktree, not the task tracker. Triggers on phrases like "is X still running", "what's in flight", "status on the lanes", "are the agents done", "did codex finish".
---

# Agent Liveness Sweep

**Use BEFORE reporting "still running" / "in flight" / "done" / "asleep" / "stalled" for any codex agent or worktree.**

Per Jay's standing rule (`feedback_sweep_worktrees_before_status.md`):
> Before reporting "in flight" / "still running" for any codex lane, sweep
> `git worktree list` + per-worktree REPORT.md mtime + branch HEAD vs master.
> Trust the worktree, not the task tracker.

The bash background-task "completed" notification is unreliable for detached
codex processes (`feedback_codex_completion_signals.md`). Use this skill.

## Run

```bash
python scripts/agent_liveness_sweep.py
python scripts/agent_liveness_sweep.py --json          # machine-readable
python scripts/agent_liveness_sweep.py --diag _diag/2026-06-07
python scripts/agent_liveness_sweep.py --window-hours 6
```

## What it reports per agent

Each `*.out` file in `_diag/` modified in the look-back window (default 24h) is
classified as ONE of:

- **RUNNING** — pid alive OR rollout JSONL updated in last 60s
- **DONE_UNPROCESSED** — file has `[dispatch_codex] DONE rc=0`; pick up the
  result and continue the workflow
- **DEAD_NEEDS_REDISPATCH** — file has `DONE rc=<nonzero>` OR no DONE marker,
  file idle >10min, no live rollout
- **STALLED_WITH_EVIDENCE** — pid is dead but no DONE marker was written
  (codex died mid-flight without the wrapper writing its sentinel)
- **UNKNOWN_NEEDS_RAW_JSONL_READ** — ambiguous; open the rollout JSONL by hand

## What it reports per worktree

For each `git worktree list` entry: branch, head SHA, ahead-of-main count,
last-commit age, uncommitted-changes flag.

## Decision rules

- Any `RUNNING` lane is genuinely in flight. Do not redispatch.
- Any `DONE_UNPROCESSED` lane needs your attention NOW — read the file and
  continue the workflow. Reporting "still running" is wrong.
- Any `DEAD_NEEDS_REDISPATCH` lane has finished badly. Read the file for
  the rc and rationale, then either redispatch or escalate.
- `STALLED_WITH_EVIDENCE` and `UNKNOWN_NEEDS_RAW_JSONL_READ` need an actual
  human/raw-jsonl read; do not guess the state from the task tracker.

## Trust order

1. `.pid` sidecar + process alive check (authoritative for "is the process running")
2. `[dispatch_codex] DONE rc=<n>` in the `*.out` file (authoritative for "did it finish")
3. Rollout JSONL mtime in `~/.codex/sessions/<date>/` (live activity proof)
4. `git worktree list` + HEAD vs main (authoritative for "what did it actually land")
5. NEVER: the task tracker, blackboard, or your own memory of what you dispatched

## Do not skip

This is the standing rule. If you find yourself about to say "lane X is still
running" without having run this sweep, stop and run it. The 2026-06-07 incident
that motivated this skill: 4 FewerJobs lanes (Z4/Z6/Z7/Z9) were reported "in
flight" when all had finished 7h earlier.
