#!/usr/bin/env python3
"""Validate P7 artifacts and apply the frozen P6-replacement promotion gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


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
P2_SEED = 20260902
FOLD_SEED = 20260831
P6_MODEL_SHA256 = "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195"
P6_SOURCE_MANIFEST_SHA256 = (
    "f6fb96ee2434ee8ead9e9e586be7d85bb8cae3fd6c71e059f552a67fb02452ef"
)
P7_EMBEDDING_MANIFEST_SHA256 = (
    "f82ddf5f5b95224a21cb83b2f1e8b3ed1bf3bad2e8495396d5c398956a9556a8"
)
P7_EMBEDDING_SHA256 = (
    "3d324ac3c711ba7dcc3c55747377ba2c1875e0b683fbde77d4e4220092b9c326"
)
P6_STANDALONE_MACRO = 0.9098782108453899
P6_BLEND_MACRO = 0.9054205862136073
LAUNCH_COMMIT = "6dbbc05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/kaggle/rsna-p7-report-image-aux-v1"),
    )
    parser.add_argument(
        "--p6-dir",
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
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_hash(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = "".join(character for character in normalized if character.isalnum())
    assert normalized
    return hashlib.sha256(normalized.encode()).hexdigest()


def load_oof(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={ID: str})
    assert frame[ID].nunique() == len(frame) == 58
    labels = frame[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    predictions = frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(predictions).all()
    return frame, labels, predictions


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
            for index in range(predictions.shape[1])
        ]
    )


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    assert len(positive) and len(negative)
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size
    )


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    repeats: int = 5_000,
    seed: int = FOLD_SEED + 700_000,
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


def tensor_signature(state: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in state.items()}


def main() -> int:
    args = parse_args()
    root = args.output_dir
    artifact_validation = root / "p7_artifact_validation.json"
    validator = Path(__file__).with_name("validate_p2_coatnet_outputs.py")
    subprocess.run(
        [
            sys.executable,
            str(validator),
            str(root),
            "--manifest-name",
            "p7_report_aux_manifest.json",
            "--expected-run-id",
            "P7-REPORT-IMAGE-AUX-v1",
            "--expected-model-sha256",
            P6_MODEL_SHA256,
            "--expected-initialization-arm",
            "p6_fastmri_localization_plus_report_aux",
            "--expected-execution-revision",
            "p7_report_aux_ddp_t4_v1",
            "--train-csv",
            str(args.train_csv),
            "--output-json",
            str(artifact_validation),
        ],
        check=True,
    )
    artifact = json.loads(artifact_validation.read_text())
    assert artifact["validated"] is True
    assert artifact["saved_direct_and_fold_heads_replayed"] is True
    assert artifact["report_group_purge_validated"] is True

    manifest_path = root / "p7_report_aux_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["best_epoch"] in range(1, 9)
    assert manifest["initialization"]["p6_source_manifest_sha256"] == (
        P6_SOURCE_MANIFEST_SHA256
    )
    auxiliary = manifest["report_auxiliary"]
    assert auxiliary["enabled"] is True and auxiliary["training_only"] is True
    assert auxiliary["embedding_manifest_sha256"] == P7_EMBEDDING_MANIFEST_SHA256
    assert auxiliary["embedding_artifacts_sha256"][
        "p7_bge_m3_report_embeddings.npy"
    ] == P7_EMBEDDING_SHA256
    assert auxiliary["embedding_dimension"] == 1_024
    assert auxiliary["temperature"] == 0.07 and auxiliary["loss_weight"] == 0.10
    assert auxiliary["bank_rows"] > 3_800
    assert auxiliary["bank_expert_rows"] == 0
    assert auxiliary["selection_report_overlap"] == 0
    assert auxiliary["deployable_checkpoint_excludes_projection"] is True
    assert manifest["report_auxiliary_gold_rows"] == 0

    history = pd.read_csv(root / "p2_representation_history.csv")
    assert history["epoch"].tolist() == list(range(1, 9))
    assert np.isfinite(history["train_report_contrastive"]).all()
    assert (history["train_report_contrastive"] > 0).all()
    assert np.isclose(
        history.loc[history["epoch"] == manifest["best_epoch"],
                    "pseudo_valid_weighted_bce"].iloc[0],
        manifest["best_pseudo_valid_weighted_bce"],
        atol=1e-12,
    )

    train = pd.read_csv(args.train_csv, dtype={ID: str})
    targets = pd.read_csv(args.targets_csv, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist()
    hashes = train["Report"].map(report_hash).to_numpy(str)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    pseudo = np.where(~is_gold)[0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=P2_SEED)
    train_position, valid_position = next(splitter.split(pseudo, groups=hashes[pseudo]))
    expected_bank = pseudo[train_position]
    pseudo_valid = pseudo[valid_position]
    bank = np.load(root / "p7_report_bank_indices.npy")
    group_ids = np.load(root / "p7_report_group_ids.npy")
    assert np.array_equal(bank, expected_bank)
    assert group_ids.shape == (4_407,)
    expected_group_map = {
        value: index for index, value in enumerate(sorted(set(hashes.tolist())))
    }
    expected_group_ids = np.asarray([expected_group_map[value] for value in hashes])
    assert np.array_equal(group_ids, expected_group_ids)
    assert not is_gold[bank].any()
    assert not set(hashes[bank]) & set(hashes[pseudo_valid])
    assert auxiliary["bank_rows"] == len(bank)
    assert auxiliary["bank_unique_report_groups"] == len(np.unique(group_ids[bank]))

    projection = torch.load(
        root / "p7_report_projection_best.pt", map_location="cpu", weights_only=True
    )
    assert set(projection) == {
        "0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"
    }
    assert tuple(projection["0.weight"].shape) == (1_024,)
    assert tuple(projection["1.weight"].shape) == (1_024, 1_024)
    assert tuple(projection["3.weight"].shape) == (1_024, 1_024)
    assert all(torch.isfinite(value).all() for value in projection.values())
    del projection

    candidate_state = torch.load(
        root / "p2_coatnet_pseudo_best.pt", map_location="cpu", weights_only=True
    )
    assert not any(key.startswith("report_projection.") for key in candidate_state)
    candidate_signature = tensor_signature(candidate_state)
    del candidate_state
    gc.collect()
    p6_state = torch.load(
        args.p6_dir / "p2_coatnet_pseudo_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert candidate_signature == tensor_signature(p6_state)
    del p6_state
    gc.collect()

    p7_frame, labels, p7_predictions = load_oof(
        root / "p2_oof_predictions.csv"
    )
    p6_frame, p6_labels, p6_predictions = load_oof(
        args.p6_dir / "p2_oof_predictions.csv"
    )
    baseline_frame, baseline_labels, baseline_predictions = load_oof(args.p2_p4_oof)
    assert p7_frame[ID].tolist() == p6_frame[ID].tolist() == baseline_frame[ID].tolist()
    assert np.array_equal(p7_frame["fold"], p6_frame["fold"])
    assert np.array_equal(p7_frame["fold"], baseline_frame["fold"])
    assert np.array_equal(labels, p6_labels) and np.array_equal(labels, baseline_labels)

    p7_metrics = metrics(labels, p7_predictions)
    p6_metrics = metrics(labels, p6_predictions)
    assert np.isclose(p6_metrics["MACRO"], P6_STANDALONE_MACRO, atol=1e-12)
    current_blend = 0.75 * baseline_predictions + 0.25 * rank_columns(p6_predictions)
    candidate_blend = 0.75 * baseline_predictions + 0.25 * rank_columns(p7_predictions)
    current_metrics = metrics(labels, current_blend)
    candidate_metrics = metrics(labels, candidate_blend)
    assert np.isclose(current_metrics["MACRO"], P6_BLEND_MACRO, atol=1e-12)
    target_deltas = {
        target: candidate_metrics[target] - current_metrics[target]
        for target in TARGETS
    }
    bootstrap = paired_bootstrap(labels, current_blend, candidate_blend)
    gates = {
        "pseudo_validation_bce": (
            manifest["best_pseudo_valid_weighted_bce"] <= 0.46259872634244603
        ),
        "standalone_macro": p7_metrics["MACRO"] > P6_STANDALONE_MACRO,
        "fixed_blend_macro": candidate_metrics["MACRO"] > P6_BLEND_MACRO,
        "nonnegative_targets": sum(value >= 0 for value in target_deltas.values()) >= 8,
        "worst_target": min(target_deltas.values()) > -0.02,
        "bootstrap_probability": bootstrap["probability_positive"] >= 0.80,
    }
    result = {
        "schema_version": 1,
        "experiment_id": "P7",
        "run_id": manifest["run_id"],
        "status": "validated / rejected",
        "date": "2026-09-06",
        "launch_commit": LAUNCH_COMMIT,
        "raw_artifact_path": str(root),
        "validated": True,
        "manifest_sha256": sha256(manifest_path),
        "artifact_replay": artifact,
        "report_auxiliary_integrity": {
            "embedding_manifest_sha256": P7_EMBEDDING_MANIFEST_SHA256,
            "bank_rows": len(bank),
            "bank_unique_report_groups": len(np.unique(group_ids[bank])),
            "bank_expert_rows": 0,
            "selection_report_overlap": 0,
            "deployable_checkpoint_matches_p6_schema": True,
        },
        "pseudo_validation_weighted_bce": manifest[
            "best_pseudo_valid_weighted_bce"
        ],
        "best_epoch": manifest["best_epoch"],
        "runtime_minutes": {
            "representation": manifest["elapsed"]["representation_minutes"],
            "feature_extraction_and_head_evaluation": manifest["elapsed"][
                "feature_extraction_minutes"
            ],
            "total": (
                manifest["elapsed"]["representation_minutes"]
                + manifest["elapsed"]["feature_extraction_minutes"]
            ),
        },
        "standalone": {
            "p6_macro_auc": p6_metrics["MACRO"],
            "p7_macro_auc": p7_metrics["MACRO"],
            "delta": p7_metrics["MACRO"] - p6_metrics["MACRO"],
        },
        "fixed_25pct_rank_blend": {
            "current_p6_macro_auc": current_metrics["MACRO"],
            "candidate_p7_macro_auc": candidate_metrics["MACRO"],
            "delta": candidate_metrics["MACRO"] - current_metrics["MACRO"],
            "target_deltas": target_deltas,
            "nonnegative_target_deltas": sum(
                value >= 0 for value in target_deltas.values()
            ),
            "worst_target_delta": min(target_deltas.values()),
            "paired_bootstrap": bootstrap,
        },
        "promotion_gates": gates,
        "promotion_gate_passed": all(gates.values()),
        "decision": (
            "Promote P7 into the frozen 25% slot."
            if all(gates.values())
            else "Reject P7 without tuning the frozen auxiliary design; retain P6."
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
