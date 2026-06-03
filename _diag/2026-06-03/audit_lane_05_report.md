# Lane 5 Audit Report -- Evaluation & Benchmarking

## Verdict: MUST-FIX

Lane 5 does not pass the dispatch bar today.

What now works: Benchmark v2 headline metrics are reproducible from checked-in
score artifacts with a pinned `temporal_cnn` `ModelSpec` SHA via
`benchmark_v2/reproduce_claims.py`, and per-category FAR/FRR is now published in
`benchmark_v2/BENCHMARK_REPORT_v2.md`.

What still fails the binary bar: the raw audio corpora are not checked into this
worktree, the full scorer cannot run from this worktree, the `d'=8.577 / EER
0.8%` production-reference claim still appears on public copy surfaces without a
Lane 5 reproducer/corpus, and `benchmark_regression_check.py` exits cleanly only
because there is no benchmark-results directory to compare.

## Fixes landed in this branch

1. Added `benchmark_v2/reproduce_claims.py` as the cheap public-claim
   reproducer. It validates score-row label/path consistency, verifies the
   pinned registry SHA, recomputes EER/FAR/FRR/AUC from checked-in score CSVs,
   and can fail if raw audio files are required
   (`benchmark_v2/reproduce_claims.py:102`,
   `benchmark_v2/reproduce_claims.py:238`,
   `benchmark_v2/reproduce_claims.py:382`).
2. Made the full benchmark script worktree-relative and registry-backed instead
   of hard-coding the master checkout and experiment model path
   (`benchmark_v2/run_benchmark.py:34`,
   `benchmark_v2/run_benchmark.py:39`,
   `benchmark_v2/run_benchmark.py:259`,
   `benchmark_v2/run_benchmark.py:270`).
3. Added the pinned `temporal_cnn` SHA to the benchmark JSON
   (`benchmark_v2/benchmark_results_v2.json:9`) and regenerated the public
   benchmark report with the reproducer command, SHA, corrected OWW ROC AUC
   from the checked-in CSV, corrected OWW FRR-at-FAR values, and per-category
   FAR/FRR (`benchmark_v2/BENCHMARK_REPORT_v2.md:5`,
   `benchmark_v2/BENCHMARK_REPORT_v2.md:22`,
   `benchmark_v2/BENCHMARK_REPORT_v2.md:33`).
4. Removed unreproducible exact "operator" benchmark numbers from the Lane
   5-owned operator report. It now explicitly says it is not current public
   evidence until a model/corpus/reproducer exists
   (`benchmark_v2/OPERATOR_BENCHMARK.md:1`,
   `benchmark_v2/OPERATOR_BENCHMARK.md:4`,
   `benchmark_v2/OPERATOR_BENCHMARK.md:21`).

## Claim verification evidence

Verified from checked-in score artifacts:

```text
Command:
python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2 --report benchmark_v2/BENCHMARK_REPORT_v2.md

Output excerpt:
| EER | 5.49% | 8.24% |
| ROC AUC | 0.9877 | 0.9574 |
| FAR @ FRR=5% | 5.43% | 8.86% |
...
OK: Benchmark v2 public claims reproduced from checked-in score artifacts.
```

Published per-category FAR/FRR:

```text
Command:
python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2 --report benchmark_v2/BENCHMARK_REPORT_v2.md

Output excerpt:
| ViolaWake | adversarial_viola | 105 | 7.62% | 4.76% |
| ViolaWake | speech | 200 | 13.50% | 10.50% |
| OpenWakeWord | adversarial_alexa | 105 | 56.19% | 53.33% |
| OpenWakeWord | noise | 20 | 10.00% | 5.00% |
```

Full raw benchmark still fails from this worktree because the audio corpus is
missing:

```text
Command:
python benchmark_v2/run_benchmark.py

Output:
ERROR: Corpus not found at J:\CLAUDE\PROJECTS\Wakeword-l5-eval\benchmark_v2\corpus
Run build_corpus.py first!
```

Raw-corpus requirement fails even for the new reproducer when strict audio
existence is required:

```text
Command:
python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2 --require-audio-files

Output excerpt:
ERROR: ViolaWake score-corpus validation failed:
line 2: audio file missing in this worktree: benchmark_v2\corpus\positives\viola\en-au-natashaneural_hey_viola.wav
... 860 more
```

The raw assets are intentionally untracked/ignored, so the worktree cannot meet
the "checked into this repo" corpus bar:

```text
Command:
git ls-files benchmark_v2/corpus eval_clean/negatives experiments/models/j5_temporal/temporal_cnn.onnx models src/violawake_sdk/models.py

Output:
src/violawake_sdk/models.py
```

```text
Command:
rg -n "corpus/|models/|experiments/models" .gitignore

Output:
66:models/
70:corpus/
71:_training_corpus/
```

Production-reference cached eval artifacts do not support the public
`d'=8.577 / EER=0.8%` claim:

```text
Files:
eval_clean/results_meanpool.json:2  "d_prime": 2.067363770655967
eval_clean/results_meanpool.json:15 "eer_approx": 0.1560435818862785
eval_clean/results_maxpool.json:2   "d_prime": 1.6219748218323782
eval_clean/results_maxpool.json:15  "eer_approx": 0.1977698331630916
```

The Python production-eval script also does not run cleanly on this machine:

```text
Command:
python eval_clean/analyze_final.py

Output excerpt:
Version v4.0.30319 of the .NET Framework is not installed and it is required to run version 3 of Windows PowerShell.
```

`benchmark_regression_check.py` exits 0, but skips because no benchmark results
exist:

```text
Command:
python tools/benchmark_regression_check.py

Output:
No benchmark results directory found at benchmark-results
Run benchmarks first: pytest tests/benchmarks/ --benchmark-json=benchmark-results/latency.json
```

## Negative probes

Wrong model SHA is caught by the new claim reproducer:

```text
Probe shape:
copy benchmark_results_v2.json to a temp benchmark dir and replace metadata.model.sha256 with 64 zeros.

Command:
python benchmark_v2\reproduce_claims.py --benchmark-dir <temp-sha-probe>

Output:
ERROR: metadata.model.sha256 mismatch: results='0000000000000000000000000000000000000000000000000000000000000000' registry='9c0b12c68593cfdb3d320a3b34667913b18d63e89eb01247d6332d7839ac9efe'
```

Mislabeled corpus row is caught by the new score-corpus validator:

```text
Probe shape:
insert benchmark_v2/corpus/negatives/music/song.wav,positive,0.990000,positive_viola into a temp ViolaWake score CSV.

Command:
python benchmark_v2\reproduce_claims.py --benchmark-dir <temp-label-probe>

Output:
ERROR: ViolaWake score-corpus validation failed:
line 2: positive 'viola' row does not point under corpus/positives/viola: benchmark_v2/corpus/negatives/music/song.wav
```

The old contamination checker does not catch that mislabeled-audio class:

```text
Probe shape:
train/positives/viola/real_viola.wav and eval/positives/viola/music_labeled_viola.wav, with no filename/hash overlap.

Command:
python -m violawake_sdk.tools.contamination_check --train <temp>\train --eval <temp>\eval --method filename

Output:
{
  "method": "filename",
  "train_files": 1,
  "eval_files": 1,
  "overlap_count": 0,
  "overlapping_files": [],
  "contamination_rate": 0.0
}

No contamination detected.
```

## Remaining MUST-FIX items

1. Raw Benchmark v2 audio corpus is absent from this worktree and ignored by
   git. The public raw-run bar cannot pass until the project either checks in a
   small canonical audio corpus via an approved storage strategy or changes the
   public claim contract to score-artifact reproduction only.
2. `d'=8.577 / EER=0.8% / AUC=0.9993` remains public on non-Lane-5 surfaces:
   `console/frontend/src/pages/Landing.tsx:94`,
   `console/frontend/src/pages/Landing.tsx:223`,
   `docs/SHOW_HN_DRAFT.md:22`,
   `docs/S1.3_REQUIREMENTS_SYNTHESIS.md:10`,
   `docs/S1.3_REQUIREMENTS_SYNTHESIS.md:83`, and
   `ACCURACY_MISSION.md:5`. I did not edit those public-copy/governance files
   in this Lane 5 branch.
3. README still repeats the removed operator benchmark numbers
   (`README.md:773`, `README.md:777`, `README.md:778`, `README.md:779`). Lane 5
   removed them from `benchmark_v2/OPERATOR_BENCHMARK.md`, but README is Lane 11
   copy and needs the same correction.
4. `benchmark_regression_check.py` is runnable but not meaningful without
   checked-in or CI-produced `benchmark-results/latency-*.json` files.

## Suggested public-copy diffs (do not edit in Lane 5)

README operator section:

```diff
-### Proof: "Operator" Custom Wake Word (89 seconds, EER 7.2%)
-| | ViolaWake "viola" | ViolaWake "operator" | OWW "alexa" (pre-trained) |
-| **EER** | **5.49%** | **7.2%** | 8.24% |
-| **ROC AUC** | 0.988 | 0.984 | 0.956 |
-| **Training time** | ~48s | **89s** | N/A (pre-trained) |
-Full methodology: [`benchmark_v2/OPERATOR_BENCHMARK.md`](benchmark_v2/OPERATOR_BENCHMARK.md)
+### Custom Wake Word Benchmarks
+The previous "operator" benchmark is not current public evidence because its model,
+eval corpus, score CSVs, and reproducer are not checked in. Use
+[`benchmark_v2/BENCHMARK_REPORT_v2.md`](benchmark_v2/BENCHMARK_REPORT_v2.md)
+for current reproducible public accuracy claims.
```

Landing page production-reference claim:

```diff
-0.8% EER on production reference model; user-trained accuracy varies
+Benchmark v2: 5.49% EER on synthetic/TTS audio; user-trained accuracy varies
```

```diff
-Production reference model: 0.8% EER and d-prime 8.58 on a curated benchmark.
+Benchmark v2 reference: 5.49% EER on synthetic/TTS audio with the pinned temporal_cnn model.
```

Show HN / requirements copy:

```diff
-TemporalCNN d'=8.577, EER 0.8%, AUC 0.9993
+Benchmark v2 temporal_cnn EER 5.49% on synthetic/TTS audio; production d'/EER claim pending a checked-in raw-corpus reproducer
```

## Planned gate

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: public-accuracy-claim-reproducer
contract: Public benchmark claims must reproduce from pinned score artifacts, match the registered model SHA, and fail on label/category mismatches.
detector: benchmark_v2/reproduce_claims.py
own_tests:
  - tests/unit/test_benchmark_reproduce_claims.py::test_validate_model_metadata_rejects_wrong_sha
  - tests/unit/test_benchmark_reproduce_claims.py::test_score_validation_rejects_positive_row_under_negative_path
  - tests/unit/test_benchmark_reproduce_claims.py::test_reproducer_current_artifacts_pass
```

## Verification commands

```text
python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2 --report benchmark_v2/BENCHMARK_REPORT_v2.md
python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2 --require-audio-files
python benchmark_v2/run_benchmark.py
python tools/benchmark_regression_check.py
pytest -o addopts='' tests/unit/test_benchmark_reproduce_claims.py -q
python -m py_compile benchmark_v2/reproduce_claims.py benchmark_v2/run_benchmark.py tests/unit/test_benchmark_reproduce_claims.py
git diff --check
```

## Self-audit gate: not exhaustively probed

1. I did not regenerate the raw Benchmark v2 audio corpus with Edge TTS because
   the binary success criterion requires a corpus checked into the repo, not a
   transient local regeneration, and committing large generated audio naked is
   blocked by project rules.
2. I did not run the full OpenWakeWord/ViolaWake scorer after copying artifacts
   from the master checkout because that would not prove this worktree is
   reproducible from checked-in files and would risk depending on untracked
   master-local state.
3. I did not edit README, Landing, Show HN, or requirements copy because those
   files route to public-copy/frontend/governance lanes; suggested diffs are
   recorded above.
4. I did not audit every historical experiment document under `experiments/` or
   `docs/archive/`; this audit focused on current public claim surfaces named in
   the dispatch plus active benchmark lane artifacts.
5. I did not validate competitor external claims against current vendor pages;
   Lane 5's bar here is reproducibility from this repo, and external vendor-copy
   freshness belongs to public copy/marketing review.
