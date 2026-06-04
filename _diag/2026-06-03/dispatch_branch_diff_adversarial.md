# ADVERSARIAL REVIEW — All lane branches, cross-cutting

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-branch-diff-adversarial
Worktree off `master`. Don't touch master, don't push, don't merge.

## Mission
The lane audits were both auditor and fixer (same codex agent per
lane). Per CLAUDE.md memory's standing rule, before declaring lanes
done, an independent adversarial reviewer reads the diffs and looks
for the failure modes the in-lane agent has incentive NOT to catch.

You are that adversarial reviewer. Recommendations-only — NO code
edits, NO branch modifications.

## Branches to review

All branches under `audit-2026-06-03/*` and `codex/adr-audit-2026-06-03`.
Discover them with `git -C "J:/CLAUDE/PROJECTS/Wakeword" branch --all
| grep -E "audit-2026-06-03|codex/adr"`. For each, walk the commits
(`git log master..<branch>`) and inspect the diffs
(`git diff master..<branch>`).

## What to look for (open lens — examples, not a checklist)

- **Test changed to match buggy code** (the canonical "fixed the
  test, not the system" failure).
- **Assertions weakened** (greater-than-or-equal where there was
  strict; broader except clauses; mock too permissive).
- **Coverage deleted** rather than fixed.
- **Negative probe constructed but its assertion is wrong** (probe
  passes by accident, not by catching the bug).
- **Silent fallback hiding the fix** (fix avoids the bug by skipping
  the broken path rather than fixing it).
- **Fix in worktree A relies on code in worktree B that isn't
  guaranteed to land** (load-bearing dependency the merge order
  doesn't enforce).
- **Ratchet gate claims to detect class X but actually only detects
  one instance** (the gate is itself single-instance).
- **The fix changed the documented contract instead of meeting it**
  (e.g., relaxed an SLA to make a measurement pass).

Don't enumerate these as a checklist for codex; find every gap, at
any layer, at any severity. Zero is the bar.

## What is NOT a finding

- Stylistic nits, alternative implementation suggestions, "could be
  cleaner."
- Future improvements not relevant to whether this fix is real.
- Anything that requires you to change another lane's scope.

## Output
Write `_diag/2026-06-03/adversarial_review_report.md` with:
- Per branch: PASS or list of specific findings with branch + commit
  SHA + file:line.
- Aggregate: which branches you'd block from merge vs which are
  ready.
- Mandatory five-bullet self-audit gate.

Commit on `post-audit-2026-06-03/branch-diff-adversarial`. Don't
edit any other artifact. Don't push, don't merge.
