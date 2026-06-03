# Audit Gates Bootstrap Report

Verdict: PASS

Source checkout: `J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap`
Branch: `audit-2026-06-03/gates-bootstrap`
Source HEAD: `56cad9ef3abb3438daa12eedd1f68aea5953487c`

The required correction note was not present in this worktree, but was read from
the main checkout at
`J:\CLAUDE\PROJECTS\Wakeword\_diag\2026-06-03\SC_AUDIT_ROUND_1_CORRECTIONS.md`.
Section A2 says ordinary audit lanes must not create `quality/gates.yaml`; this
bootstrap lane is the explicit exception in the dispatch prompt.

## Artifacts

All required artifacts exist.

- `quality/gates.yaml:1` documents the registry header; `quality/gates.yaml:3`
  documents the schema; `quality/gates.yaml:14` starts the registry empty with
  `gates: []`.
- `scripts/check_no_direct_main_commits.py:41` implements the direct-main
  checker. `scripts/check_no_direct_main_commits.py:52` allows `MERGE_HEAD`,
  `scripts/check_no_direct_main_commits.py:55` rejects the main checkout, and
  `scripts/check_no_direct_main_commits.py:62` allows linked worktrees.
- `scripts/check_ratchet_rule.py:15` defines fix-like commits,
  `scripts/check_ratchet_rule.py:19` defines the closed exemption enum,
  `scripts/check_ratchet_rule.py:26` defines production surfaces,
  `scripts/check_ratchet_rule.py:90` validates the registry schema, and
  `scripts/check_ratchet_rule.py:240` enforces the trailer rule.
- `.githooks/pre-commit:4` runs the direct-main checker and
  `.githooks/pre-commit:5` runs the ratchet checker. `.githooks/commit-msg:4`
  also runs the ratchet checker with the actual commit message because Git
  `pre-commit` does not receive the final message.
- `.github/workflows/gates.yml:17` runs the direct-main checker on PRs and
  `.github/workflows/gates.yml:21` runs the ratchet checker across the PR
  commit range. `README.md:1235` documents
  `git config core.hooksPath .githooks`.

## Probe Outputs

Probe base:
`C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338`

### (a) Direct main checkout rejects non-merge commit

Temp repo seed SHA:
`d3299c23962810632aef58e630eba8ade0e3f480`

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_no_direct_main_commits.py --repo C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\direct-main
FAIL: refusing non-merge commit from the main checkout. Create a linked worktree off master and commit there.
exit: 1
```

Result: PASS, because the checker rejected the non-merge main-checkout probe.

### (b) Linked worktree allows non-merge commit

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_no_direct_main_commits.py --repo C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\linked-worktree
PASS: non-merge commit allowed from linked worktree C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\linked-worktree.
exit: 0
```

Result: PASS, because the checker allowed the linked-worktree probe.

### (c) Main checkout allows merge commit

Merge setup SHAs:

- feature side: `0571d5e7f2c32597a0b9ea6ae860a0ef2ce6d264`
- master side: `cf6c7ad2bbac41a740c2174064e9f82f23e27daf`

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_no_direct_main_commits.py --repo C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\direct-main
PASS: merge commit allowed from C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\direct-main because MERGE_HEAD is present.
exit: 0
```

Result: PASS, because the checker allowed the main-checkout merge state with
`MERGE_HEAD` present.

### (d) Ratchet rejects fix-like production commit with no trailer

Probe commit SHA:
`2036fea91f12808b24ff227c97867944e1375743`

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_ratchet_rule.py --repo C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\ratchet --commit-range 2036fea91f12808b24ff227c97867944e1375743^..2036fea91f12808b24ff227c97867944e1375743
FAIL: 2036fea91f12: fix-like production commit requires Ratchet: <gate-id> or Ratchet-Exempt: <closed-enum-reason>.
exit: 1
```

Result: PASS, because the checker rejected a `fix:` commit touching `src/app.py`
with no ratchet trailer.

### (e) Ratchet accepts closed-enum exemption

Probe commit SHA:
`0a8bdb5985c0c0eef9c045722ade872221462eb5`

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_ratchet_rule.py --repo C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\ratchet --commit-range 0a8bdb5985c0c0eef9c045722ade872221462eb5^..0a8bdb5985c0c0eef9c045722ade872221462eb5
PASS: 0a8bdb5985c0: accepted Ratchet-Exempt: docs-only.
exit: 0
```

Result: PASS, because `Ratchet-Exempt: docs-only` is in the closed enum.

### (f) Ratchet rejects free-text exemption

Probe commit SHA:
`5598de522494d21860cb7df6be24ebc79cbbeece`

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_ratchet_rule.py --repo C:\Users\jihad\AppData\Local\Temp\wakeword-gates-probes-20260603151338\ratchet --commit-range 5598de522494d21860cb7df6be24ebc79cbbeece^..5598de522494d21860cb7df6be24ebc79cbbeece
FAIL: 5598de522494: invalid Ratchet-Exempt reason(s): because I said so. Allowed: docs-only, external-dep-bump, revert-related, single-instance-data.
exit: 1
```

Result: PASS, because the checker enforced the closed enum.

### (g) Gate registry validates against its documented schema

```text
cwd: J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap
cmd: python J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\scripts\check_ratchet_rule.py --repo J:/CLAUDE/PROJECTS/Wakeword-gates-bootstrap --validate-gates
PASS: J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap\quality\gates.yaml validates; registered gates: 0.
exit: 0
```

Result: PASS, because the bootstrap registry matches its documented schema and
starts empty.

## Additional Verification

```text
cmd: bash .githooks/pre-commit
PASS: non-merge commit allowed from linked worktree J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap.
PASS: ratchet pre-commit validation ran; commit message is not available to pre-commit, so trailer enforcement is deferred to commit-message-aware invocation or CI.
exit: 0
```

```text
cmd: python -m py_compile scripts/check_no_direct_main_commits.py scripts/check_ratchet_rule.py
exit: 0
```

```text
cmd: git diff --check
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
exit: 0
```

## Mandatory Self-Audit Gate

- I did not exhaustively test every possible YAML scalar form because the
  documented registry schema is intentionally narrow; I tested the shipped
  empty registry and the parser enforces only that supported shape.
- I did not test real GitHub Actions execution because this branch must not be
  pushed; I validated the workflow command paths locally and kept the workflow
  dependency-free beyond `actions/checkout`.
- I did not test a registered `Ratchet: <gate-id>` accepting path because the
  bootstrap registry must start empty; the code path is present and will become
  testable when the orchestrator integration commit adds a gate entry.
- I did not enforce the semantic truth of `docs-only` against changed files
  because the dispatch probe explicitly requires accepting the same production
  change with `Ratchet-Exempt: docs-only`; this implementation enforces the
  closed enum exactly.
- I did not install hooks globally or modify user-level Git config; the README
  documents `git config core.hooksPath .githooks`, and the hook itself was run
  directly from this worktree.

Conclusion: PASS. No MUST-FIX remains for the bootstrap framework.
