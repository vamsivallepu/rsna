#!/usr/bin/env python3
"""Audit the frozen P8 fold-safe noisy-student teacher on expert OOF rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ID = "StudyInstanceUID"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
FOLD_SEED = 20260831
REPORT_WEIGHT = 0.50
IMAGE_WEIGHT = 0.50
IMAGE_COMPONENT_WEIGHT = 1.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consensus",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-report-weak-supervision/consensus_weak_labels.csv"
        ),
    )
    parser.add_argument(
        "--p2-direct",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-p2-coatnet-rmlp2-v2/"
            "p2_direct_expert_predictions.csv"
        ),
    )
    parser.add_argument(
        "--p4-oof",
        type=Path,
        default=Path(
            "outputs/experiments/P4-coatnet-e2e-oof/p4_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--p6-direct",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-p6-p2-transfer-v1/"
            "p2_direct_expert_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/results/P8-A-TEACHER-AUDIT.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    assert len(positive) and len(negative)
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    repeats: int = 5_000,
    seed: int = FOLD_SEED + 800_000,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), len(labels))
        target_deltas: list[float] = []
        for target_index in range(len(TARGETS)):
            sampled_labels = labels[sampled, target_index]
            if np.unique(sampled_labels).size != 2:
                break
            target_deltas.append(
                binary_auc(sampled_labels, candidate[sampled, target_index])
                - binary_auc(sampled_labels, baseline[sampled, target_index])
            )
        if len(target_deltas) == len(TARGETS):
            deltas.append(float(np.mean(target_deltas)))
    values = np.asarray(deltas)
    assert len(values) == repeats
    return {
        "repeats": repeats,
        "mean_delta": float(values.mean()),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "probability_positive": float((values > 0).mean()),
    }


def load_prediction(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={ID: str})
    assert frame[ID].nunique() == len(frame) == 58
    labels = frame[[f"label__{target}" for target in TARGETS]].to_numpy(np.float64)
    predictions = frame[[f"pred__{target}" for target in TARGETS]].to_numpy(
        np.float64
    )
    assert np.isfinite(predictions).all()
    return frame, labels, predictions


def main() -> int:
    args = parse_args()
    p4, labels, p4_predictions = load_prediction(args.p4_oof)
    p2, p2_labels, p2_predictions = load_prediction(args.p2_direct)
    p6, p6_labels, p6_predictions = load_prediction(args.p6_direct)
    p2 = p2.set_index(ID).loc[p4[ID]].reset_index()
    p6 = p6.set_index(ID).loc[p4[ID]].reset_index()
    p2_labels = p2[[f"label__{target}" for target in TARGETS]].to_numpy(np.float64)
    p6_labels = p6[[f"label__{target}" for target in TARGETS]].to_numpy(np.float64)
    p2_predictions = p2[[f"pred__{target}" for target in TARGETS]].to_numpy(
        np.float64
    )
    p6_predictions = p6[[f"pred__{target}" for target in TARGETS]].to_numpy(
        np.float64
    )
    assert np.array_equal(labels, p2_labels) and np.array_equal(labels, p6_labels)
    assert sorted(p4["fold"].unique().tolist()) == [0, 1, 2, 3, 4]

    consensus = pd.read_csv(args.consensus, dtype={ID: str}).set_index(ID).loc[p4[ID]]
    report_predictions = consensus[
        [f"mean__{target}" for target in TARGETS]
    ].to_numpy(np.float64)
    assert np.isfinite(report_predictions).all()
    image_predictions = IMAGE_COMPONENT_WEIGHT * (
        p2_predictions + p4_predictions + p6_predictions
    )
    relabel_predictions = (
        REPORT_WEIGHT * report_predictions + IMAGE_WEIGHT * image_predictions
    )

    report_metrics = metrics(labels, report_predictions)
    image_metrics = metrics(labels, image_predictions)
    relabel_metrics = metrics(labels, relabel_predictions)
    target_deltas = {
        target: relabel_metrics[target] - report_metrics[target] for target in TARGETS
    }
    bootstrap = paired_bootstrap(labels, report_predictions, relabel_predictions)
    gates = {
        "macro_at_least_0_90": relabel_metrics["MACRO"] >= 0.90,
        "macro_above_report_teacher": (
            relabel_metrics["MACRO"] > report_metrics["MACRO"]
        ),
        "nonnegative_targets": sum(value >= 0 for value in target_deltas.values()) >= 8,
        "worst_target": min(target_deltas.values()) > -0.05,
        "bootstrap_probability": bootstrap["probability_positive"] >= 0.80,
    }
    result = {
        "schema_version": 1,
        "experiment_id": "P8-A-TEACHER-AUDIT",
        "status": "validated",
        "date": "2026-09-06",
        "formula": {
            "report_weight": REPORT_WEIGHT,
            "image_weight": IMAGE_WEIGHT,
            "image_components": ["P2 direct", "P4 fold-safe OOF", "P6 direct"],
            "image_component_weight": IMAGE_COMPONENT_WEIGHT,
            "target_specific_adjustments": False,
        },
        "source_sha256": {
            "consensus": sha256(args.consensus),
            "p2_direct": sha256(args.p2_direct),
            "p4_oof": sha256(args.p4_oof),
            "p6_direct": sha256(args.p6_direct),
        },
        "expert_rows": 58,
        "report_teacher": report_metrics,
        "image_teacher": image_metrics,
        "relabel_teacher": relabel_metrics,
        "relabel_minus_report_target_deltas": target_deltas,
        "nonnegative_target_deltas": sum(
            value >= 0 for value in target_deltas.values()
        ),
        "worst_target_delta": min(target_deltas.values()),
        "paired_bootstrap": bootstrap,
        "gates": gates,
        "advance_gate_passed": all(gates.values()),
        "decision": (
            "Advance to fold-specific P8-A GPU teacher generation."
            if all(gates.values())
            else "Stop P8 without changing the frozen teacher formula."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
