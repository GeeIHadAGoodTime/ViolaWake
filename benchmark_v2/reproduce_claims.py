#!/usr/bin/env python3
"""Reproduce and validate Benchmark v2 public accuracy claims.

This script is the cheap, CI-friendly claim instrument for the checked-in
Benchmark v2 artifacts. The full scorer in run_benchmark.py still requires
audio files plus OpenWakeWord runtime models; this reproducer validates the
published numbers from the checked-in score corpus and benchmark JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TARGET_FRRS = (0.01, 0.03, 0.05, 0.10)
TARGET_FARS = (0.001, 0.005, 0.01, 0.05)
NEGATIVE_CATEGORIES = {
    "adversarial_viola",
    "adversarial_alexa",
    "speech",
    "speech_existing",
    "noise",
}


@dataclass
class ScoreRow:
    file: str
    label: str
    score: float
    category: str


@dataclass
class ScoreSet:
    system_name: str
    wake_word: str
    rows: list[ScoreRow]

    @property
    def positives(self) -> list[ScoreRow]:
        return [row for row in self.rows if row.label == "positive"]

    @property
    def negatives(self) -> list[ScoreRow]:
        return [row for row in self.rows if row.label == "negative"]

    @property
    def pos_scores(self) -> np.ndarray:
        return np.array([row.score for row in self.positives], dtype=np.float64)

    @property
    def neg_scores(self) -> np.ndarray:
        return np.array([row.score for row in self.negatives], dtype=np.float64)


def _path_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").lower().split("/") if part]


def _resolve_corpus_path(file_value: str, benchmark_dir: Path) -> Path:
    parts = _path_parts(file_value)
    if "corpus" in parts:
        after_corpus = parts[parts.index("corpus") + 1 :]
        return benchmark_dir / "corpus" / Path(*after_corpus)
    return Path(file_value)


def load_scores(path: Path, system_name: str, wake_word: str) -> ScoreSet:
    rows: list[ScoreRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "label", "score", "category"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for line_no, raw in enumerate(reader, start=2):
            try:
                score = float(raw["score"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} invalid score {raw['score']!r}") from exc
            rows.append(
                ScoreRow(
                    file=raw["file"].strip(),
                    label=raw["label"].strip().lower(),
                    score=score,
                    category=raw["category"].strip().lower(),
                )
            )
    return ScoreSet(system_name=system_name, wake_word=wake_word, rows=rows)


def validate_score_rows(score_set: ScoreSet, benchmark_dir: Path, require_audio_files: bool = False) -> None:
    errors: list[str] = []
    expected_positive_category = f"positive_{score_set.wake_word}"

    for idx, row in enumerate(score_set.rows, start=2):
        parts = _path_parts(row.file)
        if row.label not in {"positive", "negative"}:
            errors.append(f"line {idx}: invalid label {row.label!r}")
            continue

        if row.label == "positive":
            if "positives" not in parts or score_set.wake_word not in parts:
                errors.append(
                    f"line {idx}: positive {score_set.wake_word!r} row does not point under "
                    f"corpus/positives/{score_set.wake_word}: {row.file}"
                )
            if row.category != expected_positive_category:
                errors.append(
                    f"line {idx}: positive row category {row.category!r} != "
                    f"{expected_positive_category!r}"
                )
        else:
            if "negatives" not in parts:
                errors.append(f"line {idx}: negative row does not point under corpus/negatives: {row.file}")
            if row.category not in NEGATIVE_CATEGORIES:
                errors.append(f"line {idx}: unknown negative category {row.category!r}")

        if not 0.0 <= row.score <= 1.0:
            errors.append(f"line {idx}: score outside [0, 1]: {row.score}")

        if require_audio_files:
            resolved = _resolve_corpus_path(row.file, benchmark_dir)
            if not resolved.exists():
                errors.append(f"line {idx}: audio file missing in this worktree: {resolved}")

    if errors:
        preview = "\n".join(errors[:20])
        more = "" if len(errors) <= 20 else f"\n... {len(errors) - 20} more"
        raise ValueError(f"{score_set.system_name} score-corpus validation failed:\n{preview}{more}")


def _threshold_grid() -> np.ndarray:
    return np.linspace(0.0, 1.0, 10001)


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    thresholds = np.linspace(0.0, 1.0, 2001)
    fpr = np.array([np.mean(neg >= t) for t in thresholds])
    tpr = np.array([np.mean(pos >= t) for t in thresholds])
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    best_diff = float("inf")
    best_eer = 0.5
    best_threshold = 0.0
    for threshold in _threshold_grid():
        frr = float(np.mean(pos < threshold))
        far = float(np.mean(neg >= threshold))
        diff = abs(frr - far)
        if diff < best_diff:
            best_diff = diff
            best_eer = (frr + far) / 2
            best_threshold = float(threshold)
    return best_eer, best_threshold


def far_at_frr(pos: np.ndarray, neg: np.ndarray, target_frr: float) -> tuple[float, float]:
    best_diff = float("inf")
    best_far = 1.0
    best_threshold = 0.0
    for threshold in _threshold_grid():
        frr = float(np.mean(pos < threshold))
        far = float(np.mean(neg >= threshold))
        diff = abs(frr - target_frr)
        if diff < best_diff:
            best_diff = diff
            best_far = far
            best_threshold = float(threshold)
    return best_far, best_threshold


def frr_at_far(pos: np.ndarray, neg: np.ndarray, target_far: float) -> tuple[float, float]:
    best_diff = float("inf")
    best_frr = 1.0
    best_threshold = 0.0
    for threshold in _threshold_grid():
        frr = float(np.mean(pos < threshold))
        far = float(np.mean(neg >= threshold))
        diff = abs(far - target_far)
        if diff < best_diff:
            best_diff = diff
            best_frr = frr
            best_threshold = float(threshold)
    return best_frr, best_threshold


def metrics(score_set: ScoreSet) -> dict[str, Any]:
    pos = score_set.pos_scores
    neg = score_set.neg_scores
    metric_eer, eer_threshold = eer(pos, neg)
    return {
        "eer": metric_eer,
        "eer_threshold": eer_threshold,
        "roc_auc": roc_auc(pos, neg),
        "far_at_frr": {f"frr_{int(target * 100)}pct": far_at_frr(pos, neg, target)[0] for target in TARGET_FRRS},
        "frr_at_far": {
            f"far_{target * 100:.1f}pct".replace(".", "p"): frr_at_far(pos, neg, target)[0]
            for target in TARGET_FARS
        },
    }


def category_far_table(score_set: ScoreSet, frr_targets: tuple[float, ...] = (0.05, 0.10)) -> list[dict[str, Any]]:
    pos = score_set.pos_scores
    neg_by_category: dict[str, list[float]] = {}
    for row in score_set.negatives:
        neg_by_category.setdefault(row.category, []).append(row.score)

    rows: list[dict[str, Any]] = []
    for category, scores in sorted(neg_by_category.items()):
        arr = np.array(scores, dtype=np.float64)
        row: dict[str, Any] = {"category": category, "n": len(arr)}
        for target in frr_targets:
            _, threshold = far_at_frr(pos, score_set.neg_scores, target)
            row[f"far_at_frr_{int(target * 100)}"] = float(np.mean(arr >= threshold))
        rows.append(row)
    return rows


def load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_model_metadata(results: dict[str, Any]) -> None:
    from violawake_sdk.models import MODEL_REGISTRY

    metadata = results.get("metadata", {})
    model = metadata.get("model")
    if not isinstance(model, dict):
        raise ValueError("benchmark_results_v2.json missing metadata.model with pinned ModelSpec SHA")

    spec = MODEL_REGISTRY["temporal_cnn"]
    expected = {
        "name": spec.name,
        "version": spec.version,
        "sha256": spec.sha256,
        "size_bytes": spec.size_bytes,
    }
    for key, expected_value in expected.items():
        if model.get(key) != expected_value:
            raise ValueError(
                f"metadata.model.{key} mismatch: results={model.get(key)!r} registry={expected_value!r}"
            )


def validate_counts(results: dict[str, Any], vw: ScoreSet, oww: ScoreSet) -> None:
    metadata = results["metadata"]
    if len(vw.positives) != metadata["n_viola_pos"]:
        raise ValueError(f"ViolaWake positive count mismatch: CSV={len(vw.positives)} JSON={metadata['n_viola_pos']}")
    if len(oww.positives) != metadata["n_alexa_pos"]:
        raise ValueError(f"OWW positive count mismatch: CSV={len(oww.positives)} JSON={metadata['n_alexa_pos']}")
    if len(vw.negatives) != metadata["n_negatives"] or len(oww.negatives) != metadata["n_negatives"]:
        raise ValueError("negative count mismatch between CSVs and JSON metadata")

    actual_categories: dict[str, int] = {}
    for row in vw.negatives:
        actual_categories[row.category] = actual_categories.get(row.category, 0) + 1
    if actual_categories != metadata["neg_categories"]:
        raise ValueError(f"negative category mismatch: CSV={actual_categories} JSON={metadata['neg_categories']}")


def _assert_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{label} mismatch: computed={actual:.8f} expected={expected:.8f} tolerance={tolerance}")


def validate_metrics(results: dict[str, Any], vw_metrics: dict[str, Any], oww_metrics: dict[str, Any]) -> None:
    # CSV scores are rounded to six decimals, so AUC has a looser tolerance than
    # EER/FAR/FRR. This still catches stale score files and swapped systems.
    tolerances = {"eer": 0.0002, "roc_auc": 0.005}
    for system_key, computed in (("violawake", vw_metrics), ("oww", oww_metrics)):
        expected = results[system_key]
        _assert_close(f"{system_key}.eer", computed["eer"], expected["eer"], tolerances["eer"])
        _assert_close(f"{system_key}.roc_auc", computed["roc_auc"], expected["roc_auc"], tolerances["roc_auc"])
        for key, value in computed["far_at_frr"].items():
            _assert_close(f"{system_key}.far_at_frr.{key}", value, expected["far_at_frr"][key], 0.0002)
        for key, value in computed["frr_at_far"].items():
            _assert_close(f"{system_key}.frr_at_far.{key}", value, expected["frr_at_far"][key], 0.0002)


def render_report(results: dict[str, Any], vw: ScoreSet, oww: ScoreSet) -> str:
    metadata = results["metadata"]
    model = metadata["model"]
    lines = [
        "## ViolaWake vs OpenWakeWord -- Corrected Benchmark v2",
        "",
        "### Reproduction",
        "",
        "- Script: `python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2`",
        f"- Model: `{model['name']}` version `{model['version']}`, SHA-256 `{model['sha256']}`",
        f"- Shared negative score corpus: {metadata['n_negatives']} files",
    ]
    for category, count in sorted(metadata["neg_categories"].items()):
        lines.append(f"  - {category}: {count} files")
    lines.extend(
        [
            f"- Matched positives: {metadata['n_viola_pos']} viola, {metadata['n_alexa_pos']} alexa",
            "- Same 20 Edge TTS voices, same 3 augmentations (clean, noisy, reverb)",
            f"- Streaming inference: {metadata['chunk_samples']}-sample chunks (80ms at 16kHz), max-score per file",
            "- Primary metrics: EER, FAR@FRR",
            "",
            "### Results",
            "",
            "| Metric | ViolaWake (viola) | OWW (alexa) |",
            "|--------|-------------------|-------------|",
            f"| EER | {results['violawake']['eer'] * 100:.2f}% | {results['oww']['eer'] * 100:.2f}% |",
            f"| ROC AUC | {results['violawake']['roc_auc']:.4f} | {results['oww']['roc_auc']:.4f} |",
        ]
    )

    for target in TARGET_FRRS:
        key = f"frr_{int(target * 100)}pct"
        lines.append(
            f"| FAR @ FRR={target * 100:.0f}% | "
            f"{results['violawake']['far_at_frr'][key] * 100:.2f}% | "
            f"{results['oww']['far_at_frr'][key] * 100:.2f}% |"
        )
    for target in TARGET_FARS:
        key = f"far_{target * 100:.1f}pct".replace(".", "p")
        lines.append(
            f"| FRR @ FAR={target * 100:.1f}% | "
            f"{results['violawake']['frr_at_far'][key] * 100:.2f}% | "
            f"{results['oww']['frr_at_far'][key] * 100:.2f}% |"
        )

    lines.extend(["", "### Per-Category FAR/FRR", ""])
    lines.append("Per-category FAR is computed at the global threshold selected for the target FRR.")
    lines.append("")
    lines.append("| System | Negative category | N | FAR @ FRR=5% | FAR @ FRR=10% |")
    lines.append("|--------|-------------------|---:|-------------:|--------------:|")
    for score_set in (vw, oww):
        for row in category_far_table(score_set):
            lines.append(
                f"| {score_set.system_name} | {row['category']} | {row['n']} | "
                f"{row['far_at_frr_5'] * 100:.2f}% | {row['far_at_frr_10'] * 100:.2f}% |"
            )

    lines.extend(
        [
            "",
            "### Analysis",
            "",
            f"**ViolaWake has lower EER** ({results['violawake']['eer'] * 100:.2f}% vs "
            f"{results['oww']['eer'] * 100:.2f}%), indicating better overall discrimination.",
            "",
            f"ViolaWake has higher AUC ({results['violawake']['roc_auc']:.4f} vs "
            f"{results['oww']['roc_auc']:.4f}).",
            "",
            "### Context",
            "",
            "- OWW's `alexa` model: pre-trained by David Scripka on a larger real-speech corpus",
            "- ViolaWake's `viola` model: temporal CNN on OWW embeddings, TTS-trained",
            "- Both evaluated on TTS audio only (no real recordings in this benchmark)",
            "- Adversarial negatives included for both systems",
            "- Negatives do not contain either actual wake word",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> str:
    benchmark_dir = args.benchmark_dir
    results = load_results(benchmark_dir / "benchmark_results_v2.json")
    vw = load_scores(benchmark_dir / "violawake_scores_v2.csv", "ViolaWake", "viola")
    oww = load_scores(benchmark_dir / "oww_scores_v2.csv", "OpenWakeWord", "alexa")

    validate_model_metadata(results)
    validate_score_rows(vw, benchmark_dir, require_audio_files=args.require_audio_files)
    validate_score_rows(oww, benchmark_dir, require_audio_files=args.require_audio_files)
    validate_counts(results, vw, oww)

    vw_metrics = metrics(vw)
    oww_metrics = metrics(oww)
    validate_metrics(results, vw_metrics, oww_metrics)

    report = render_report(results, vw, oww)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path, help="Write regenerated markdown report to this path")
    parser.add_argument(
        "--require-audio-files",
        action="store_true",
        help="Require score CSV file paths to exist under benchmark-dir/corpus",
    )
    args = parser.parse_args()

    try:
        report = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(report)
    print("OK: Benchmark v2 public claims reproduced from checked-in score artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
