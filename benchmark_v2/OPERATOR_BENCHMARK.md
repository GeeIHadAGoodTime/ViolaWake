# Custom Wake Word Benchmark: "Operator" -- Not Current Public Evidence

This document used to publish exact "operator" benchmark numbers, including
EER, ROC AUC, training time, and confusion-matrix counts.

Those numbers are not reproducible from this repository today. The referenced
`models/operator_v2.onnx` artifact, `eval_operator` corpus, score CSVs, and a
claim-reproduction script are not checked in. Do not cite this page as public
accuracy evidence until the operator benchmark has the same reproducibility
contract as `benchmark_v2/BENCHMARK_REPORT_v2.md`:

```bash
python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2
```

The current reproducible public comparison is Benchmark v2:

- `temporal_cnn` registry version: `0.1.0`
- `temporal_cnn` SHA-256:
  `9c0b12c68593cfdb3d320a3b34667913b18d63e89eb01247d6332d7839ac9efe`
- Reproducer: `benchmark_v2/reproduce_claims.py`
- Report: `benchmark_v2/BENCHMARK_REPORT_v2.md`

To reinstate an operator claim, add the operator model registry entry, checked-in
score corpus or checked-in raw corpus, and a reproducer that fails on stale
scores, wrong model SHA, and label/category mismatches.
