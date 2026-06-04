<!-- doc-meta
scope: Architecture decision — wake word detection head architecture
authority: ADR — immutable once accepted
code-paths: src/violawake_sdk/wake_detector.py, src/violawake_sdk/training/temporal_model.py, src/violawake_sdk/models.py
supersedes: ADR-002
staleness-signals: A new dominant audio embedding backbone; temporal_cnn underperforms a successor on the production eval set; we revert to MLP
-->

# ADR-007: TemporalCNN Wake Head on Frozen OWW Backbone

**Status:** Accepted
**Date:** 2026-06-04
**Authors:** ViolaWake team
**Supersedes:** [ADR-002](ADR-002-oww-feature-extractor.md)

---

## Context

ADR-002 (2026-03-17) locked "**Use OpenWakeWord (OWW) as the fixed audio feature extractor backbone** plus a separate ViolaWake **MLP** wake head." The decision explicitly compared MLP-on-OWW (`viola_mlp_oww.onnx`) against a custom CNN (`viola_v1–v4.onnx`) and chose MLP-on-OWW for its higher d-prime.

ADR-002's own body acknowledges that "the shipped default wake head later moved to `temporal_cnn`" — an inline note that the head is no longer the MLP — but no superseding ADR was written. The current registry's default model is `temporal_cnn.onnx`, which is a **temporal convolutional network** (not the original MLP and not the original v1–v4 CNN). It produces the d' = 8.577 / EER = 0.8% headline cited in the README on the production eval set.

The 2026-06-03 ADR drift audit (`_diag/2026-06-03/audit_adrs_report.md`) flagged ADR-002 as DRIFT for this reason. The frozen-OWW-backbone half of ADR-002 still holds; only the wake-head architecture changed.

---

## Decision

**ViolaWake's default wake head is the temporal CNN (`temporal_cnn.onnx`), trained on frozen OpenWakeWord (OWW) 96-dim audio embeddings.**

The decision splits cleanly into two halves carried forward from ADR-002:

1. **Backbone (unchanged from ADR-002):** OWW is the fixed audio feature extractor. The 96-dim embedding contract is preserved. Implementation: `src/violawake_sdk/oww_backbone.py`.
2. **Head (new from ADR-002's MLP):** the temporal CNN replaces the MLP. The temporal head consumes a sliding window of OWW embeddings (configurable; current production: 9 frames) and emits a single wake-word probability. Implementation: `src/violawake_sdk/training/temporal_model.py` + the ONNX export consumed by `wake_detector.py`.

---

## Rationale

- **Real-world separability over synthetic-d'.** ADR-002 chose MLP-on-OWW based on a synthetic-negative benchmark d' (15.10 vs CNN 3.07). On the production eval set with real speech negatives the temporal CNN reaches d' = 8.577 / EER = 0.8% — substantially better in the regime that matters for shipped users.
- **Temporal context matters more than head depth.** A sliding-window CNN that sees N consecutive OWW embeddings captures phoneme-sequence structure that a 1-frame MLP cannot, regardless of MLP depth.
- **OWW backbone still wins.** The frozen-backbone-plus-trained-head decomposition from ADR-002 is preserved as the right partition: it isolates the in-house contribution (the head) from a stable upstream contract (the OWW embedding) so OWW updates can be evaluated independently.

---

## Consequences

- **Lane 1 oracle bars are temporal-CNN-specific.** Per-category FAR bars in `benchmark_v2` and the production eval set are pinned to `temporal_cnn.onnx` at its registered SHA. A new head architecture would require its own bars + a new ADR.
- **Compatibility:** the MLP-on-OWW model artifacts remain reachable as historical (`viola_mlp_oww.onnx`) but are not the registry default. `WakeDetector` accepts either; production ships only `temporal_cnn`.
- **The training pipeline** (`src/violawake_sdk/tools/train.py`) produces both heads from OWW embeddings depending on `--architecture {temporal,mlp}`; the temporal path is the default and the only path the production recipe (`docs/PROVEN_TRAINING_RECIPE.md`) covers.

---

## Alternatives considered (and rejected)

- **Stay on MLP-on-OWW.** Rejected: real-world eval shows the temporal CNN is materially better.
- **Drop the OWW backbone for a custom from-scratch embedding.** Rejected: ADR-002's reasoning still holds — OWW gives us a strong free backbone, and head-only training is computationally cheap.
- **Train end-to-end (jointly fine-tune the OWW backbone with the temporal head).** Rejected for now: requires far more training compute and risks overfitting the backbone to our negatives; can be revisited with a separate ADR if real-eval gains warrant it.
