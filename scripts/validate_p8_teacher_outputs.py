#!/usr/bin/env python3
"""Independently validate completed P8 fold-safe teacher-label artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
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
N_FOLDS = 5
FOLD_SEED = 20260831
REPORT_WEIGHT = 0.5
IMAGE_WEIGHT = 0.5
IMAGE_COMPONENT_WEIGHT = 1.0 / 3.0
EXPECTED_MACROS = {
    "report": 0.8948112944772516,
    "image": 0.9034874698692111,
    "relabel": 0.9209204490507412,
}
EXPECTED_SOURCE_HASHES = {
    "p2_manifest_sha256": "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508",
    "p6_manifest_sha256": "885b58c5a9af1bffaac1f52c9703db3538d25f8d9660b036b563b351b1067bce",
    "weak_manifest_sha256": "561a8f6a8d645a05ef6f193a2f1dd93db3f146a7ccaa4d16dd109e24b6ec2816",
    "training_targets_sha256": "ca63cfff863c87286e07a8059fee88f86ff8b3f4b640a98078d62113f6b93f2a",
    "consensus_sha256": "c152263597c8588894b6b70ef54c4bd3fd517b610ae7177e4f78443b8ed0dfc7",
    "p4_manifest_sha256": {
        "0": "6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932",
        "1": "9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5",
        "2": "93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac",
        "3": "9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c",
        "4": "e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228",
    },
}
P2_ROOT = ROOT / "outputs/kaggle/rsna-p2-coatnet-rmlp2-v2"
P6_ROOT = ROOT / "outputs/kaggle/rsna-p6-p2-transfer-v1"
P4_ROOT = ROOT / "outputs/kaggle/rsna-p4-coatnet-e2e-v1"
WEAK_ROOT = ROOT / "outputs/kaggle/rsna-report-weak-supervision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p8-teacher-labels-v1",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "experiments/results/P8-A-TEACHER-LABELS.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def prediction_columns(prefix: str) -> list[str]:
    return [f"{prefix}__{target}" for target in TARGETS]


def main() -> int:
    args = parse_args()
    manifest_path = args.root / "p8_teacher_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert manifest["studies"] == 4_407
    assert manifest["pseudo_studies"] == 4_349
    assert manifest["expert_studies"] == 58
    assert manifest["folds"] == N_FOLDS
    assert manifest["formula"] == {
        "report_weight": REPORT_WEIGHT,
        "image_weight": IMAGE_WEIGHT,
        "image_component_weight": IMAGE_COMPONENT_WEIGHT,
        "image_components": ["P2 direct", "P4 fold", "P6 direct"],
        "target_specific_adjustments": False,
        "pseudo_weights_changed": False,
        "expert_targets_changed": False,
    }
    assert manifest["fold_safety"] == {
        "p2_direct_expert_gradient_rows": 0,
        "p6_direct_expert_gradient_rows": 0,
        "p4_teacher_matches_student_fold": True,
        "expert_validation_teacher_gradient_rows": 0,
        "expert_fold_seed": FOLD_SEED,
    }
    replay_preflight = manifest["replay_preflight"]
    assert replay_preflight["passed"] is True
    assert replay_preflight["studies"] == 58
    assert replay_preflight["inference_batch_per_rank"] == 2
    assert replay_preflight["canonical_expert_values_used_for_audit"] is True
    assert replay_preflight["direct_maximum_absolute_tolerance"] == 1e-5
    assert replay_preflight["p4_maximum_absolute_tolerance"] == 5e-3
    assert replay_preflight["p4_mean_absolute_tolerance"] == 5e-4
    for statistics in replay_preflight["p4_fold_differences"].values():
        assert statistics["maximum_absolute_error"] <= 5e-3
        assert statistics["mean_absolute_error"] <= 5e-4
    for statistics in replay_preflight["direct_differences"].values():
        assert statistics["maximum_absolute_error"] <= 1e-5
    assert manifest["source_hashes"] | {
        "cache_manifest_sha256": None
    } == EXPECTED_SOURCE_HASHES | {"cache_manifest_sha256": None}
    assert manifest["execution"]["world_size"] == 2
    assert manifest["execution"]["visible_gpus"] == 2
    assert manifest["execution"]["backend"] == "nccl"
    assert len(manifest["execution"]["peak_allocated_mib_by_rank"]) == 2
    assert all(
        value > 1_000
        for value in manifest["execution"]["peak_allocated_mib_by_rank"]
    )
    generated = ROOT / "kaggle/rsna-p8-teacher-labels/rsna-p8-teacher-labels.py"
    assert manifest["script_sha256"] == sha256(generated)
    for artifact, expected in manifest["artifacts_sha256"].items():
        assert sha256(args.root / artifact) == expected, artifact

    targets = pd.read_csv(WEAK_ROOT / "training_targets.csv", dtype={ID: str})
    consensus = pd.read_csv(
        WEAK_ROOT / "consensus_weak_labels.csv", dtype={ID: str}
    )
    studies = pd.read_csv(args.root / "p8_teacher_studies.csv", dtype={ID: str})
    assert studies[ID].tolist() == targets[ID].tolist() == consensus[ID].tolist()
    labels = targets[TARGETS].to_numpy(np.float64)
    report = consensus[prediction_columns("mean")].to_numpy(np.float64)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]

    p2 = np.load(args.root / "p8_p2_direct_predictions.npy")
    p6 = np.load(args.root / "p8_p6_direct_predictions.npy")
    p4 = np.load(args.root / "p8_p4_fold_predictions.npy")
    expert_fold = np.load(args.root / "p8_expert_fold.npy")
    assert p2.shape == p6.shape == (4_407, len(TARGETS))
    assert p4.shape == (N_FOLDS, 4_407, len(TARGETS))
    assert expert_fold.shape == (4_407,)
    assert np.isfinite(p2).all() and np.isfinite(p6).all() and np.isfinite(p4).all()

    expected_fold = np.full(4_407, -1, dtype=np.int8)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, valid_position) in enumerate(splitter.split(gold_indices)):
        valid_indices = gold_indices[valid_position]
        expected_fold[valid_indices] = fold
        p4_saved = pd.read_csv(
            P4_ROOT / f"fold-{fold}/p4_fold_{fold}_oof_predictions.csv",
            dtype={ID: str},
        ).set_index(ID).loc[studies.iloc[valid_indices][ID]]
        assert np.allclose(
            p4[fold, valid_indices],
            p4_saved[prediction_columns("pred")].to_numpy(np.float32),
            atol=1e-5,
            rtol=0.0,
        )
    assert np.array_equal(expert_fold, expected_fold)

    for source_root, computed in ((P2_ROOT, p2), (P6_ROOT, p6)):
        saved = pd.read_csv(
            source_root / "p2_direct_expert_predictions.csv", dtype={ID: str}
        ).set_index(ID).loc[studies.iloc[gold_indices][ID]]
        assert np.allclose(
            computed[gold_indices],
            saved[prediction_columns("pred")].to_numpy(np.float32),
            atol=1e-5,
            rtol=0.0,
        )

    expert_p4 = np.stack(
        [p4[expert_fold[index], index] for index in gold_indices]
    )
    expert_image = IMAGE_COMPONENT_WEIGHT * (
        p2[gold_indices] + expert_p4 + p6[gold_indices]
    )
    expert_relabel = (
        REPORT_WEIGHT * report[gold_indices] + IMAGE_WEIGHT * expert_image
    )
    calculated_metrics = {
        "report": metrics(labels[gold_indices], report[gold_indices]),
        "image": metrics(labels[gold_indices], expert_image),
        "relabel": metrics(labels[gold_indices], expert_relabel),
    }
    for name, expected in EXPECTED_MACROS.items():
        assert np.isclose(
            calculated_metrics[name]["MACRO"], expected, atol=1e-12, rtol=0.0
        )
        assert np.isclose(
            manifest["expert_audit_macro_auc"][name],
            expected,
            atol=1e-12,
            rtol=0.0,
        )

    audit = pd.read_csv(args.root / "p8_teacher_expert_audit.csv", dtype={ID: str})
    assert audit[ID].tolist() == studies.iloc[gold_indices][ID].tolist()
    assert np.array_equal(audit["fold"].to_numpy(np.int8), expert_fold[gold_indices])
    assert np.allclose(
        audit[prediction_columns("relabel")].to_numpy(np.float64),
        expert_relabel,
        atol=1e-8,
        rtol=0.0,
    )

    non_target_columns = [column for column in targets if column not in TARGETS]
    for fold in range(N_FOLDS):
        generated_targets = pd.read_csv(
            args.root / f"p8_fold_{fold}_training_targets.csv", dtype={ID: str}
        )
        assert generated_targets[ID].tolist() == studies[ID].tolist()
        pd.testing.assert_frame_equal(
            generated_targets[non_target_columns],
            targets[non_target_columns],
            check_dtype=False,
            check_exact=True,
        )
        expected_image = IMAGE_COMPONENT_WEIGHT * (p2 + p4[fold] + p6)
        expected_relabel = REPORT_WEIGHT * report + IMAGE_WEIGHT * expected_image
        assert np.allclose(
            generated_targets.loc[pseudo_indices, TARGETS].to_numpy(np.float64),
            expected_relabel[pseudo_indices],
            atol=1e-12,
            rtol=0.0,
        )
        assert np.array_equal(
            generated_targets.loc[gold_indices, TARGETS].to_numpy(np.float64),
            labels[gold_indices],
        )

    telemetry = pd.read_csv(args.root / "p8_teacher_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    assert len(telemetry) >= 4 and telemetry.notna().all().all()
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] > 0).all()

    result = {
        "schema_version": 1,
        "experiment_id": "P8-A-TEACHER-LABELS",
        "status": "validated",
        "manifest_sha256": sha256(manifest_path),
        "teacher_formula_replayed": True,
        "fold_specific_targets_replayed": True,
        "expert_validation_teacher_gradient_rows": 0,
        "expert_audit_macro_auc": {
            name: values["MACRO"] for name, values in calculated_metrics.items()
        },
        "execution": manifest["execution"],
        "gpu_utilization": {
            str(index): {key: float(value) for key, value in row.items()}
            for index, row in utilization.to_dict(orient="index").items()
        },
        "advance_gate_passed": True,
        "decision": "Advance to five P8 student folds.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
