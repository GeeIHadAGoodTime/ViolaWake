# Benchmark v2 — reproducibility layering

This directory holds the reproducibility evidence for every public
accuracy claim ViolaWake makes (`d'`, EER, FAR/FRR, per-category bars).
It is structured in three layers, each fully reproducible from the one
below it, so the public claim can be verified at whichever level you
have resources for.

## Layer 1 — public-claim reproducer (in repo, always runnable)

`reproduce_claims.py` + `violawake_scores_v2.csv` + `oww_scores_v2.csv`.

The CSVs are the per-sample, per-model raw scores from the full
benchmark. The reproducer recomputes every headline number (EER, FAR
per category, FRR, d', ROC-AUC) from those score files and checks they
match the values published in `BENCHMARK_REPORT_v2.md`.

This layer requires **no audio, no models, no GPU**, only Python + the
SDK deps. It's what the Lane 5 Ratchet gate runs on every commit.

Run it:

```bash
python benchmark_v2/reproduce_claims.py \
    --benchmark-dir benchmark_v2 \
    --report benchmark_v2/BENCHMARK_REPORT_v2.md
```

If the CSVs drift or the registered model SHA changes without the
report being updated in the same commit, this fails. Both negative
probes (wrong SHA, mislabeled corpus row) are part of the gate.

## Layer 2 — re-score from the corpus (corpus must exist locally)

`run_benchmark.py` + `corpus/positives/` + `corpus/negatives/`.

The corpus contains the actual 16 kHz mono WAVs that produced the
score CSVs in layer 1. Re-running the benchmark on the corpus
regenerates the CSVs and is what you'd do to re-evaluate a new model
candidate against the same fixed corpus.

The corpus directory is **not tracked in git** because the raw audio is
large (>1 GB) and trivially regenerable from layer 3. To populate it,
run layer 3 once.

Run it (after layer 3 has populated `corpus/`):

```bash
python benchmark_v2/run_benchmark.py
```

This emits new `violawake_scores_v2.csv` / `oww_scores_v2.csv`. Layer 1
will then verify the published numbers against the new CSVs.

## Layer 3 — rebuild the corpus from scratch (regenerable any time)

`build_corpus.py`.

The corpus is generated from Edge-TTS (20 voices × 3 phrases × 3
augmentations per system) plus public-domain speech/noise corpora.
Both positives and negatives are produced identically for ViolaWake and
OpenWakeWord so the comparison is methodologically clean (no "alexa"
contamination in OWW's negative set, both systems see the same
adversarial confusables).

Run it:

```bash
python benchmark_v2/build_corpus.py
```

Output: populated `corpus/positives/{viola,alexa}/` and
`corpus/negatives/{speech,noise,adversarial_viola,adversarial_alexa}/`.

This step needs `edge-tts` installed and a network connection; runtime
is ~20–25 minutes on a typical desktop.

## How the layers connect

| Layer | Inputs | Outputs | In git? | Network? |
|---|---|---|---|---|
| 1 | score CSVs + model SHA | verified report numbers | Yes | No |
| 2 | corpus WAVs | score CSVs | No (regenerated) | No |
| 3 | TTS + public speech | corpus WAVs | No (regenerated) | Yes |

Public claims (the README's headline numbers, the
`compare/picovoice` page, `docs/COMPETITIVE_ANALYSIS.md`) are tied to
**layer 1** — if the gate passes, the claim reproduces. Layer 2 and
layer 3 are how you'd evaluate a model that didn't exist when the
public score CSVs were generated.

## What's NOT included

- Speaker recordings of real humans. The benchmark is TTS+public-speech
  only by design (controlled, reproducible). Real-speaker evaluation is
  expected at the customer's deployment site and is documented under
  Lane 5's success criteria as the next required layer.
- Hardware-pinned latency numbers. The Lane 2 audit measured Kokoro
  first-audio at ~1.8 s warm on the test rig; the older 0.3–0.8 s budget
  was a cold-start-excluded target that did not hold under the
  documented run. Latency targets are tracked in the Companion Engines
  lane, not here.
