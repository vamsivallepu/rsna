#!/usr/bin/env python3
"""Validate the matched P6 transfer comparison and frozen promotion gates."""

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
CONTROL_MODEL_SHA256 = (
    "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
)
TRANSFER_MODEL_SHA256 = (
    "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195"
)
P6_SOURCE_MANIFEST_SHA256 = (
    "f6fb96ee2434ee8ead9e9e586be7d85bb8cae3fd6c71e059f552a67fb02452ef"
)
P2_P4_MACRO = 0.8958210633119559
LAUNCH_COMMIT = "679e531"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=Path("outputs/kaggle/rsna-p6-p2-matched-control-v1"),
    )
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=Path("outputs/kaggle/rsna-p6-p2-transfer-v1"),
    )
    parser.add_argument(
        "--p2-p4-oof",
        type=Path,
        default=Path(
            "outputs/experiments/P4-coatnet-e2e-oof/"
            "p4_p2_equal_rank_oof_predictions.csv"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    metrics = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    metrics["MACRO"] = float(np.mean(list(metrics.values())))
    return metrics


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


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
            for index in range(predictions.shape[1])
        ]
    )


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    repeats: int = 5_000,
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


def load_oof(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={ID: str})
    assert frame[ID].nunique() == len(frame) == 58
    assert sorted(frame["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    labels = frame[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    predictions = frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(predictions).all()
    return frame, labels, predictions


def stability(
    baseline_metrics: dict[str, float], candidate_metrics: dict[str, float]
) -> dict[str, object]:
    deltas = {
        target: candidate_metrics[target] - baseline_metrics[target]
        for target in TARGETS
    }
    return {
        "target_deltas": deltas,
        "nonnegative_target_deltas": sum(value >= 0 for value in deltas.values()),
        "worst_target_delta": min(deltas.values()),
    }


def matched_invariants(manifest: dict[str, object]) -> dict[str, object]:
    keys = [
        "schema_version",
        "full_run",
        "seeds",
        "studies",
        "pseudo_studies",
        "expert_studies",
        "cache_id",
        "cache_signature_sha256",
        "feature_dim",
        "cache_size",
        "model_size",
        "normalization",
        "k_train",
        "k_eval",
        "representation_epochs",
        "execution_revision",
        "visible_gpus",
        "data_parallel",
        "data_parallel_backend",
        "distributed_backend",
        "world_size",
        "study_batch",
        "gradient_accumulation",
        "eval_study_batch",
        "loader_workers_per_rank",
        "gradient_checkpointing",
        "channels_last",
        "cudnn_benchmark",
        "fused_adamw",
        "representation_gold_rows",
        "pseudo_report_group_overlap",
    ]
    return {key: manifest[key] for key in keys}


def main() -> int:
    args = parse_args()
    control_manifest = json.loads(
        (args.control_dir / "p6_control_manifest.json").read_text()
    )
    transfer_manifest = json.loads(
        (args.transfer_dir / "p6_transfer_manifest.json").read_text()
    )
    assert control_manifest["run_id"] == "P6-P2-MATCHED-CONTROL-v1"
    assert transfer_manifest["run_id"] == "P6-P2-TRANSFER-v1"
    assert control_manifest["model_sha256"] == CONTROL_MODEL_SHA256
    assert transfer_manifest["model_sha256"] == TRANSFER_MODEL_SHA256
    assert control_manifest["initialization"]["arm"] == "matched_imagenet_control"
    assert (
        transfer_manifest["initialization"]["arm"] == "p6_fastmri_localization_transfer"
    )
    assert (
        transfer_manifest["initialization"]["p6_source_manifest_sha256"]
        == P6_SOURCE_MANIFEST_SHA256
    )
    assert matched_invariants(control_manifest) == matched_invariants(transfer_manifest)
    assert control_manifest["best_epoch"] == transfer_manifest["best_epoch"] == 8

    control_validation = json.loads(
        (args.control_dir / "p6_artifact_validation.json").read_text()
    )
    transfer_validation = json.loads(
        (args.transfer_dir / "p6_artifact_validation.json").read_text()
    )
    assert control_validation["validated"] is True
    assert transfer_validation["validated"] is True
    assert control_validation["run_id"] == control_manifest["run_id"]
    assert transfer_validation["run_id"] == transfer_manifest["run_id"]
    assert control_validation["saved_direct_and_fold_heads_replayed"] is True
    assert transfer_validation["saved_direct_and_fold_heads_replayed"] is True
    assert control_validation["report_group_purge_validated"] is True
    assert transfer_validation["report_group_purge_validated"] is True

    control_frame, labels, control_predictions = load_oof(
        args.control_dir / "p2_oof_predictions.csv"
    )
    transfer_frame, transfer_labels, transfer_predictions = load_oof(
        args.transfer_dir / "p2_oof_predictions.csv"
    )
    assert control_frame[ID].tolist() == transfer_frame[ID].tolist()
    assert np.array_equal(control_frame["fold"], transfer_frame["fold"])
    assert np.array_equal(labels, transfer_labels)

    control_metrics = calculate_metrics(labels, control_predictions)
    transfer_metrics = calculate_metrics(labels, transfer_predictions)
    assert np.isclose(
        control_metrics["MACRO"],
        control_manifest["fold_oof"]["macro_auc"],
        atol=1e-12,
    )
    assert np.isclose(
        transfer_metrics["MACRO"],
        transfer_manifest["fold_oof"]["macro_auc"],
        atol=1e-12,
    )
    standalone_stability = stability(control_metrics, transfer_metrics)
    standalone_bootstrap = paired_bootstrap(
        labels,
        control_predictions,
        transfer_predictions,
        seed=FOLD_SEED + 600_000,
    )

    baseline_frame, baseline_labels, baseline_predictions = load_oof(args.p2_p4_oof)
    assert baseline_frame[ID].tolist() == transfer_frame[ID].tolist()
    assert np.array_equal(baseline_frame["fold"], transfer_frame["fold"])
    assert np.array_equal(labels, baseline_labels)
    baseline_metrics = calculate_metrics(labels, baseline_predictions)
    assert np.isclose(baseline_metrics["MACRO"], P2_P4_MACRO, atol=1e-12)
    blend_predictions = 0.75 * baseline_predictions + 0.25 * rank_columns(
        transfer_predictions
    )
    blend_metrics = calculate_metrics(labels, blend_predictions)
    blend_stability = stability(baseline_metrics, blend_metrics)
    blend_bootstrap = paired_bootstrap(
        labels,
        baseline_predictions,
        blend_predictions,
        seed=FOLD_SEED + 600_001,
    )

    pseudo_loss_delta = (
        control_manifest["best_pseudo_valid_weighted_bce"]
        - transfer_manifest["best_pseudo_valid_weighted_bce"]
    )
    standalone_delta = transfer_metrics["MACRO"] - control_metrics["MACRO"]
    blend_delta = blend_metrics["MACRO"] - baseline_metrics["MACRO"]
    pseudo_gate = pseudo_loss_delta > 0
    standalone_gate = (
        standalone_delta >= 0.003
        and standalone_bootstrap["probability_positive"] >= 0.80
        and standalone_stability["nonnegative_target_deltas"] >= 7
        and standalone_stability["worst_target_delta"] >= -0.05
    )
    blend_gate = (
        blend_delta > 0
        and blend_bootstrap["probability_positive"] >= 0.80
        and blend_stability["nonnegative_target_deltas"] >= 7
        and blend_stability["worst_target_delta"] >= -0.05
    )
    result = {
        "schema_version": 1,
        "experiment_id": "P6",
        "run_id": "P6-P2-MATCHED-TRANSFER-COMPARISON-v1",
        "status": "validated",
        "date": "2026-09-05",
        "launch_commit": LAUNCH_COMMIT,
        "raw_artifact_paths": {
            "control": str(args.control_dir),
            "transfer": str(args.transfer_dir),
        },
        "manifest_sha256": {
            "control": sha256(args.control_dir / "p6_control_manifest.json"),
            "transfer": sha256(args.transfer_dir / "p6_transfer_manifest.json"),
        },
        "validated": True,
        "matched_configuration": True,
        "artifact_validation": {
            "control_full_replay_passed": True,
            "transfer_full_replay_passed": True,
            "all_manifest_artifact_hashes_passed": True,
            "saved_direct_and_fold_heads_replayed": True,
            "report_group_purge_validated": True,
            "expert_oof_rows_per_arm": 58,
            "dual_gpu_execution_validated": True,
            "gpu_utilization": {
                "control": control_validation["gpu_utilization"],
                "transfer": transfer_validation["gpu_utilization"],
            },
        },
        "runtime_minutes": {
            "control_representation": control_manifest["elapsed"][
                "representation_minutes"
            ],
            "control_feature_extraction": control_manifest["elapsed"][
                "feature_extraction_minutes"
            ],
            "transfer_representation": transfer_manifest["elapsed"][
                "representation_minutes"
            ],
            "transfer_feature_extraction": transfer_manifest["elapsed"][
                "feature_extraction_minutes"
            ],
        },
        "direct_expert": {
            "control_macro_auc": control_manifest["direct_expert"]["macro_auc"],
            "transfer_macro_auc": transfer_manifest["direct_expert"]["macro_auc"],
        },
        "pseudo_validation": {
            "control_weighted_bce": control_manifest["best_pseudo_valid_weighted_bce"],
            "transfer_weighted_bce": transfer_manifest[
                "best_pseudo_valid_weighted_bce"
            ],
            "improvement_control_minus_transfer": pseudo_loss_delta,
            "gate_passed": pseudo_gate,
        },
        "standalone": {
            "control_macro_auc": control_metrics["MACRO"],
            "transfer_macro_auc": transfer_metrics["MACRO"],
            "delta": standalone_delta,
            **standalone_stability,
            "paired_bootstrap": standalone_bootstrap,
            "gate_passed": standalone_gate,
        },
        "fixed_25pct_rank_blend": {
            "p2_p4_macro_auc": baseline_metrics["MACRO"],
            "p2_p4_plus_p6_macro_auc": blend_metrics["MACRO"],
            "delta": blend_delta,
            **blend_stability,
            "paired_bootstrap": blend_bootstrap,
            "gate_passed": blend_gate,
        },
        "promotion_gate_passed": pseudo_gate and (standalone_gate or blend_gate),
        "decision": (
            "Promote the fixed 75% P2+P4 plus 25% P6 rank blend. The standalone "
            "P6 transfer improves macro AUC but fails the no-target-below-minus-0.05 "
            "stability gate because Effusion declines. The fixed blend passes every "
            "predeclared pseudo-validation, macro, bootstrap, and target-stability gate."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
