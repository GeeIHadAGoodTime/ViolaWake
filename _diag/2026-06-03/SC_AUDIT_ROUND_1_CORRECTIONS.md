# SC-audit-round-1 corrections — BINDING

These are the corrections from the orchestrator's SC self-audit before
dispatch. **They override anything in your dispatch prompt they
contradict.** Read this file IN FULL before starting.

The SC audit applied this binary bar to your prompt: (1) catches
plausibly broken implementations, (2) probes are realistic broken
shapes, (3) baseline runnable with documented resources, (4) lane file
ownership disjoint, (5) SC doesn't force other-lane work, (6) reviewer's
binary question not trivially gameable.

Where your prompt failed that bar, the correction is here.

---

## A. Common corrections (apply to ALL audit lanes)

### A1. Construct AND run the negative probes — don't just review them

Your dispatch prompt's "Investigate" or "Sources" section may reference
the lane's oracle SC negative probes in `docs/LANE_LEDGER.md`. Those
probes are **not a reading exercise.** For each named probe shape:

1. **Construct a minimal broken variant** of the implementation
   matching that probe's failure shape (in a scratch script, a
   temporary git stash, or a sibling branch — do NOT corrupt the lane's
   real source). Examples per common shape:
   - "threshold lowered to 0.50" → flip the constant, run the test.
   - "audio fed at 8/22/48 kHz" → write a 22 kHz WAV, feed it through
     the SDK entry point.
   - "model URL goes 404" → point a ModelSpec at a known-404 URL in a
     fixture, run the downloader.
2. **Run the lane's verification path against that broken variant.**
3. **The oracle PASSES this probe iff the verification path catches it.**
   If the verification path lets the broken variant through, that is
   itself a MUST-FIX — the oracle is broken, even if the trunk is fine.

This is the difference between "I audited" and "I established." A
review without ran-probes is theatre.

### A2. Do NOT touch `quality/gates.yaml` in this audit

Multiple lanes are running concurrently in their own worktrees. If each
creates `quality/gates.yaml` independently, they will conflict at merge
time.

**Instead:** for every class-level fix that needs a `Ratchet:` gate,
write the planned gate spec in your final report in this exact shape:

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: <kebab-case-id>
contract: <one-sentence: what shape this gate detects>
detector: <path to detector script you wrote in this worktree, OR
           "TBD — orchestrator will write" if blocked>
own_tests:
  - <path to a test that proves the gate triggers on the broken shape>
  - <path to a test that proves the gate is quiet on the fixed shape>
```

The orchestrator will land one integration commit that creates
`quality/gates.yaml` with ALL lanes' planned gates merged.

### A3. No production-destructive actions

Read-only probes of production are fine. **You may NOT:**

- Deploy a new image to production (no `docker compose up -d` against
  the production stack)
- Restart, stop, or modify any production container (`wakeword-*`,
  `viola-*` — the NOVVIOLA-bridge containers are explicitly off-limits)
- Charge a real card or trigger real billing
- Replay a real customer's request
- Push a git tag, push to remote, or merge to master
- Write to a production database
- Rotate tunnels, DNS, or any external secret

If your SC seems to require one of these, **surface it in the report
as `BLOCKED — requires founder authorization`** and STOP that
probe — do not improvise around it. Continue with whatever else in
the lane you can audit without prod-destruction.

### A4. Cite evidence inline (file:line + command output)

A finding without an inline excerpt is theatre. For every PASS
assertion and every MUST-FIX:

- The verifying command, exactly as you ran it
- The actual stdout/stderr (excerpt OK; mark with `...truncated`)
- `file.py:LN` where the relevant code lives

The self-audit gate at the end of your report — **five bullets, what
you did NOT exhaustively probe, and why each was out of scope** — is
non-negotiable.

---

## B. Lane-specific corrections

### B1. Lane 1 (Wake Detection) — PASS clause stronger

Replace your dispatch prompt's "Success criteria — binary verdict" PASS
clause ("the lane's success criteria and oracle SC hold on the current
trunk") with:

> **PASS** = ALL of the following hold, with cited evidence:
>
> 1. The four audio-contract assertions (16 kHz, 20 ms frames,
>    320-sample stride, 96-dim OWW embeddings) are tested at the SDK
>    entry boundary. Show the test + output.
> 2. The four negative probes from § 1 oracle SC of the ledger
>    (threshold 0.50, wrong sample rates, swapped backbone, removed
>    decision-policy gate) have each been constructed and run; each
>    fires a detectable failure in the lane's verification path. Show
>    each probe + its caught failure.
> 3. Per-category FAR on the documented confusables set (`alexa`,
>    `hey siri`, music speech, ...) stays under its documented bar.
>    If the documented bar isn't in the repo, **report this absence as
>    a MUST-FIX of its own** ("documented bar missing").
> 4. The 4-gate decision policy is exercised end-to-end in a single
>    test (write the test if absent). Show the test + output.

Also: **remove the "a public number that doesn't reproduce" example from
your MUST-FIX list — that overlaps Lane 5.** Public-claim reproducibility
is Lane 5's scope.

### B2. Lane 7 (Public API & Distribution) — pre-publish handling

Replace your dispatch prompt's "Investigate" step "Pull the latest PyPI
version" with:

> **If `violawake` has been published to PyPI:** `pip download
> violawake==<latest> --no-deps -d /tmp/wheel` and validate the
> published wheel.
>
> **If `violawake` has NEVER been published:** mark this audit as
> `PRE-PUBLISH` in the report. Build the wheel locally
> (`python -m build` in this worktree), then apply ALL the same
> checks to that locally-built wheel — install in a clean venv,
> import, surface diff, ModelSpec resolution, CHANGELOG presence.
> A pre-publish audit is still a real audit; only the artifact source
> changes.

This prevents a codex from incorrectly flagging "pip install fails" as a
MUST-FIX when the package hasn't shipped yet.

### B3. Lane 8 (SaaS Console — Backend) — test-account fallback

Replace your dispatch prompt's "Use test accounts; do not burn real
billing" with:

> **If a test account or staging environment already exists**, use it
> for the full sign-up → training → download flow probe.
>
> **If no test account or staging exists**, you may NOT create
> billing transactions against the production system to test billing
> auth. Audit what you can WITHOUT live billing — read-only API
> shape audit, auth surface review against the OpenAPI, the
> inbound-email Worker against fixture inputs. **Mark the live
> billing-flow probe as `BLOCKED — needs founder to provision test
> account` in the report.**

This prevents the codex from improvising real billing transactions to
satisfy the SC.

### B4. Lane 10 (Infrastructure & DevOps) — no live deploy

Replace your dispatch prompt's PASS condition (a) ("the documented
deploy steps, run today from a clean shell, land the expected image
SHA on the live URL") with:

> **PASS** (a) = the documented deploy steps reproducibly BUILD the
> expected image on this machine (you may run
> `docker compose -f docker-compose.production.yml build`; you may NOT
> run `up -d` against production). Verify the built image SHA is
> stable across two consecutive builds at the same source SHA.
> Verify the cloudflared tunnel config + DNS routes match what's
> documented in `docs/DEPLOYMENT.md` by **read-only inspection** of
> the live config. Do NOT rotate the tunnel, restart the production
> container, or otherwise touch live production state.

For PASS condition (b), backup-restore-drill into a scratch container
is fine and required (it's read-of-backup + write-to-scratch; no
production write).

---

## C. Reminders that aren't new but get violated often

- One commit per fix. Each commit message names the gap and the fix.
- `Ratchet:` for class-level fixes (gate spec to report per § A2).
- `Ratchet-Exempt: <enum>` for single-instance fixes (enum values in
  CLAUDE.md → "Ratchet Rule").
- Do NOT push to remote. Do NOT merge to master. Do NOT modify
  `CLAUDE.md` or `docs/LANE_LEDGER.md`. Do NOT touch files outside
  your lane's "Owns" glob in the ledger.
- The self-audit gate at the end of your report is mandatory:
  five bullets, what you did NOT exhaustively probe, and why each
  was out of scope.

---

**If anything in this file is unclear or seems to contradict your
dispatch prompt in a way you can't resolve**, surface it in the report
as `SC-CONFLICT: <description>` and STOP that branch. Don't guess at
the orchestrator's intent.
