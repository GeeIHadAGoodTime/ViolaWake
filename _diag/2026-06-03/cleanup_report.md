# Lane 12 Cleanup Report - 2026-06-03

Worktree: `J:\CLAUDE\PROJECTS\Wakeword-cleanup`
Branch: `audit-2026-06-03/cleanup`

Master was not touched. Nothing was pushed.

## Baseline And Count Check

- Baseline `git ls-files` count before cleanup: 556.
- Count after committed cleanup dispositions, before this report file is added
  to git: 555.
- Expected decrease: 1 tracked file.
- Actual decrease: 1 tracked file.
- Reason: all `git mv` operations are count-neutral; the empty top-level
  `python` file is the only committed deletion; SECURITY was left unchanged
  because the two files differ.

## Dispositions

### Old TTS Sweep Artifacts

Disposition requested:

- Move root `test_*.mp3`, `test_long.mp3`, `test_short.mp3`,
  `test_model.config.json`, and `test_model.onnx` into
  `_diag/2026-06-03/cleanup/old_tts_sweep/`.

Result:

- Files moved: 0.
- Commit SHA: none.
- Surprise: these files were not present in this checkout as tracked files,
  untracked root files, or files discoverable by `rg --files` with the listed
  names. No empty destination directory was committed because git does not track
  empty directories and there was no evidence to preserve.

### Scratch Scripts

Disposition requested:

- Move `_write_wake_detector.py` and `diagnostic_embedding_analysis.py` into
  `_diag/2026-06-03/cleanup/scratch/`.

Result:

- Files moved: 2.
- Commit SHA: `c952c11`.
- Paths:
  - `_diag/2026-06-03/cleanup/scratch/_write_wake_detector.py`
  - `_diag/2026-06-03/cleanup/scratch/diagnostic_embedding_analysis.py`
- Surprise: the commit command wrapper timed out after the commit had already
  landed; `git log` and clean status confirmed the commit was valid.

### Empty Top-Level `python`

Disposition requested:

- Delete the empty top-level `python` file with `git rm`.

Result:

- Files deleted: 1.
- Commit SHA: `e4743e4`.
- Surprise: none. The file was zero bytes.

### Top-Level Docs

Disposition requested:

- Move the listed top-level docs into `docs/`.

Result:

- Files moved: 7.
- Commit SHA: `e97faad`.
- Paths:
  - `docs/ACCURACY_MISSION.md`
  - `docs/ADVERSARY_AUDIT.md`
  - `docs/BUILD_VS_BUY_AUDIT.md`
  - `docs/E2E_READINESS.md`
  - `docs/FUNCTIONAL_GAP_ANALYSIS.md`
  - `docs/LAUNCH_READINESS.md`
  - `docs/PROGRESS.md`
- Link/reference pass: no allowed-scope Markdown link required an update.
  Remaining hits were locked ledger text, same-directory doc references,
  historical prose, or non-link references in other lane-owned files.
- Surprise: none.

### Top-Level `SECURITY.md`

Disposition requested:

- If top-level `SECURITY.md` is identical to `docs/SECURITY.md`, remove the
  duplicate and replace it with a one-line pointer.
- If contents differ, surface the diff in this report and do not delete.

Result:

- Files changed: 0.
- Commit SHA: none.
- Top-level `SECURITY.md` hash:
  `13E308D213AD2EF6761D0E1771996EA3ABF49251FC2A38117993BD759EB8B54A`.
- `docs/SECURITY.md` hash:
  `06786C556F61A02990C9FC4036C17DA6DCFB402316B8CF345A65BE37C51BC670`.
- Diff stat: `SECURITY.md => docs/SECURITY.md | 138 ++++++++++++++++++++++++++++++----------`; 106 insertions, 32 deletions.
- Summary of difference:
  - Top-level `SECURITY.md` is a general security policy. It contains
    vulnerability reporting, security defaults, model integrity, and console
    security controls.
  - `docs/SECURITY.md` is a narrower security notes document. It focuses on
    recording upload hardening, Tier 3 container hardening, and Cloudflare WAF
    rules.
- Action: left both files untouched, per instruction.

## New Or Out-Of-Scope Cruft Noticed

- `.viola-nova-ww10-output.txt` remains as a tracked top-level file. It was not
  listed in the Cruft section and was not classified here.
- `pyproject.toml` still contains stale-looking references to old root paths
  such as `_write_wake_detector.py`, `diagnostic_embedding_analysis.py`,
  `ACCURACY_MISSION.md`, `PROGRESS.md`, and `test_model.*`. I did not edit it
  because `pyproject.toml` is Lane 7 owned and these are not internal Markdown
  links required by the doc move instruction.
- `experiments/confusable_mining.py` contains a prose comment referencing
  `ACCURACY_MISSION.md`. I did not edit it because `experiments/` is Lane 4
  owned and this was not a move-broken Markdown link.
- `docs/LANE_LEDGER.md` still lists the cruft block because this task
  explicitly said not to touch the ledger.
- Top-level `SECURITY.md` remains unresolved because its content differs from
  `docs/SECURITY.md`; merging authority between them needs a separate Lane 10
  or founder decision.

## Self-Audit: Not Exhaustively Probed

1. `.gitignore` quality for future generated MP3/ONNX artifacts was not probed.
   This pass was limited to executing the ledger dispositions, not changing
   prevention policy.
2. Gitignored cruft under `dist/`, `logs/`, corpora, and benchmark output dirs
   was not scanned. Those areas belong to other lanes or generated-output
   policy, not this mechanical root cleanup.
3. Packaging impact of stale `pyproject.toml` exclude entries was not tested.
   `pyproject.toml` is Lane 7 owned, and changing package build behavior would
   exceed Lane 12 housekeeping scope.
4. The content accuracy of the moved docs was not audited. The requested work
   was relocation only, not a historical-doc correctness review.
5. SECURITY authority was not reconciled beyond the required diff check.
   The files differ materially, so the safe action was to preserve both and
   report the divergence.
