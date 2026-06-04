# AUDIT — Lane 2: Companion Engines & VoicePipeline

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l2-companions
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies; especially § A1 — construct and RUN the negative probes
> (don't just review them).

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 2):
*"Does the SDK ship working STT / TTS / VAD engines, and a
`VoicePipeline` composition that wires Wake → STT → TTS correctly?"*

## Success criteria — binary verdict
PASS = lane SC + oracle SC from the ledger hold on the trunk.

MUST-FIX = a real user import / instantiate / run path already fails;
the composition deadlocks or silently emits empty output; the
documented first-audio latency (Kokoro 0.3–0.8 s) does not hold on the
reference hardware.

NOT MUST-FIX: stylistic, hypothetical edge cases without evidence,
"could be cleaner," missing future features.

## Sources
- `docs/LANE_LEDGER.md` § 2
- `CLAUDE.md` (Investigation Discipline, Ratchet Rule, AGENT DISPATCH
  PRINCIPLES)
- Files owned by this lane (see ledger § 2 "Owns")

## Investigate
Drive the actual capability — instantiate the engines, run the pipeline
on real audio, measure the latency. Don't conclude from reading code.
Find every gap, at any layer. Zero is the bar. Default fallbacks that
hide bugs are exactly the bugs.

## Decide & implement
One topic branch, one commit per fix. `Ratchet:` for class-level fixes
(add to `quality/gates.yaml`, create if missing). `Ratchet-Exempt:
<enum>` for single-instance.

Do NOT push, do NOT merge to master, do NOT modify `CLAUDE.md` or
`docs/LANE_LEDGER.md`, do NOT touch other lanes' files.

## Prove it
Command output + file:line for each fix. Show, don't claim.

## Report
`_diag/2026-06-03/audit_lane_02_report.md`:
- Binary verdict
- Per fix: gap, file:line, evidence, commit SHA
- MANDATORY self-audit gate (five bullets, why-not-probed).

## Scaffolding
- TTS engine: Kokoro (Apache 2.0), sentence-chunked, first-audio
  latency budget 0.3–0.8 s.
- STT engine: faster-whisper, batch with segments.
- VAD: WebRTC / Silero / RMS, interchangeable.
- VoicePipeline is the Wake → STT → TTS composition; Lane 1's wake
  detector is consumed via the public API, not a private hook.
