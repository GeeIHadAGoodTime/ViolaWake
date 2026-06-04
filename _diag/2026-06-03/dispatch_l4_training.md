# AUDIT — Lane 4: Training & Augmentation

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l4-training
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. Lane 4 additional binding: **you do NOT need to run a
> full from-scratch retrain.** The lane SC is pipeline integrity, not
> retrain reproducibility against d'/EER (that's a milestone oracle,
> not the fast oracle). Verify the augmentation pipeline produces the
> documented transforms; the contamination check actually detects
> contamination; the data loaders honor the audio contract; loss
> functions compute as specified. A full retrain is welcome if the
> codex agent has time and corpus, but absence is NOT a MUST-FIX.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 4):
*"Given labeled audio, does the training pipeline produce a model that
passes Evaluation & Benchmarking's bars — reproducibly?"*

## Success criteria — binary verdict
PASS = (a) the augmentation pipeline (`src/violawake_sdk/training/augment.py`,
+ RIR, SpecAugment, noise mix) produces the documented transforms,
verifiable by running it on a known input and checking the output;
(b) `tests/integration/test_training_e2e.py` passes;
(c) the contamination check (`src/violawake_sdk/tools/contamination_check.py`)
DETECTS contamination when given a deliberately contaminated corpus
(construct + run this probe per § A1);
(d) the data loaders honor the audio contract (16 kHz mono, 20 ms
frames) — feed mis-rated audio and assert it fails fast.

MUST-FIX = augmentation silently bypassed; contamination undetectable;
audio contract drifts inside the loader; the documented recipe (`docs/
PROVEN_TRAINING_RECIPE.md`) references files or steps that no longer
exist.

NOT MUST-FIX: missing future loss functions, augmentation ideas not
yet shipped, training speed.

## Sources
- `docs/LANE_LEDGER.md` § 4
- `CLAUDE.md`
- Files owned by this lane (ledger § 4 "Owns")
- `docs/PROVEN_TRAINING_RECIPE.md`, `TRAINING_PIPELINE_AUDIT_2026-05-07.md`

## Investigate
Run the pipeline pieces. Construct the contamination probe (synthesize a
corpus where train set rows leak into eval set). Construct the
audio-contract probe (a 22 kHz WAV fed through the loader).

## Decide, prove, report
One topic branch, one commit per fix. Report at
`_diag/2026-06-03/audit_lane_04_report.md` with verdict + fixes +
mandatory five-bullet self-audit gate. Gate spec per § A2.
