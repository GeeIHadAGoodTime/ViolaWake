---
name: ground-truth-refresh
description: Compact "truth board" snapshot before any status report / merge / deploy / lane-done claim. Reads git status, worktree list, branch ahead/behind, current main SHA, deployed SHA, and recent doc mtimes from the authoritative sources — never your own previous summary. Triggers on phrases like "what's the current state", "are we ready to merge", "status before deploy", "ground truth", "current main".
---

# Ground-Truth Refresh

**Use BEFORE any of these claims:**
- "main is green / clean / current"
- "ready to merge / ready to deploy"
- "X is deployed / live in production"
- "all lanes are healthy"
- "this matches what's on main"

Per CLAUDE.md `Verify on ground truth, not your own diagnostics`:
> "Delivered" is not "received." Your own sweep output is not proof an agent
> acted. To know what's true, read the authoritative source: jsonl mtime, git
> log/status, the live service.

## Run

The skill is a curated sequence of commands. Run them in this order and READ
the actual output before drawing any conclusion. Do NOT rely on cached memory
of "what main was last time you checked."

### 1. Local working tree

```bash
git status --short
git worktree list
git stash list
```

### 2. Branch state vs main

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git rev-parse main
git rev-list --left-right --count main...HEAD     # diverged commits
git log -5 --oneline                              # last 5 here
git log -5 --oneline main                         # last 5 on main
```

### 3. Remote state

```bash
git fetch --quiet origin
git rev-parse origin/main
git rev-list --left-right --count main...origin/main
```

### 4. Deployed SHA (cloud)

```bash
curl -s https://api.useviola.com/health 2>&1 | head -3
# or, for the version endpoint if /health is brief:
curl -s https://api.useviola.com/v1/version 2>&1 | head -3
```

### 5. Recent doc activity (last 24h)

```bash
find docs/ -name '*.md' -mtime -1 -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort
find _diag/ -maxdepth 2 -newer docs/REGISTRY.md -name '*.md' 2>&1 | head -10
```

### 6. Agent dispatch state (use the liveness sweep)

```bash
python scripts/agent_liveness_sweep.py --window-hours 6
```

## Emit a truth board

After running, emit a compact table BEFORE the status claim:

```
TRUTH BOARD (refreshed <ISO timestamp>)
  Branch:       <name>  @<sha7>
  Main:         <sha7>  (HEAD is +<ahead>/-<behind> vs main)
  Origin/main:  <sha7>  (local is +<ahead>/-<behind> vs origin)
  Deployed:     <sha7 or unknown>  (from /health or /version)
  Uncommitted:  <count of git status --short>
  Worktrees:    <N> + main
  Stashes:      <N>
  Recent docs:  <list of *.md modified in last 24h>
  Agents:       <RUNNING / DONE_UNPROCESSED / DEAD counts from sweep>
```

## Rule

If the truth board contradicts the claim you were about to make, the truth
board wins. Update the claim. Never the other way around.
