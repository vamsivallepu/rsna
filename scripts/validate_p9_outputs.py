#!/usr/bin/env python3
"""Independently validate P9 artifacts and evaluate its frozen 15% rank blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from p9_s3d_specialist import TargetSlotAttentionHead


ROOT = Path(__file__).resolve().parents[1]
ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
FOLD_SEED = 20260831
EXPECTED_MANIFEST_SHA256 = (
    "c0e9b4b7f6a8ddcf44968451c116523994d8038a0367b047919635a20c6d77f5"
)
EXPECTED_TEACHER_SHA256 = (
    "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
)
EXPECTED_WEIGHT_SHA256 = (
    "d76dad2f4b2a6701ad7f64e85b294b2d1989f1ea0d3f8b240446ed2152850bbb"
)
EXPECTED_CURRENT_MACRO = 0.9097982866758175


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p9-root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p9-s3d-specialist-v1",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p8-teacher-labels-v2",
    )
    parser.add_argument(
        "--current-oof",
        type=Path,
        default=(
            ROOT
            / "outputs/experiments/P8-noisy-student-oof/"
            "p8_fixed_blend_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/experiments/P9-s3d-oof",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=ROOT / "experiments/results/P9.json",
    )
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
    if not normalized:
        raise ValueError("empty normalized report")
    return hashlib.sha256(normalized.encode()).hexdigest()


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
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
    seed: int = FOLD_SEED + 900_001,
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
                roc_auc_score(sampled_labels, candidate[sampled, target_index])
                - roc_auc_score(sampled_labels, baseline[sampled, target_index])
            )
        if len(target_deltas) == len(TARGETS):
            deltas.append(float(np.mean(target_deltas)))
    values = np.asarray(deltas)
    assert len(values) == repeats
    return {
        "repeats": repeats,
        "mean_delta": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float((values > 0).mean()),
    }


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index])
            .rank(method="average", pct=True)
            .to_numpy()
            for index in range(predictions.shape[1])
        ]
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    root = args.p9_root
    manifest_path = root / "p9_manifest.json"
    assert sha256(manifest_path) == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P9-S3D-SPECIALIST-v1"
    assert manifest["status"] == "complete"
    assert manifest["studies"] == 4_407
    assert manifest["pseudo_studies"] == 4_349
    assert manifest["expert_studies"] == 58
    assert manifest["model"]["checkpoint_sha256"] == EXPECTED_WEIGHT_SHA256
    assert manifest["teacher_manifest_sha256"] == EXPECTED_TEACHER_SHA256
    assert manifest["pseudo_target_formula"] == (
        "0.50 report + 0.25 P2 direct + 0.25 P6 direct"
    )
    assert manifest["p4_teacher_excluded"] is True
    assert manifest["representation_expert_gradient_rows"] == 0
    assert manifest["pseudo_report_group_overlap"] == 0
    assert manifest["representation_epochs"] == 6
    assert manifest["execution"]["world_size"] == 2
    assert manifest["execution"]["distributed_backend"] == "nccl"
    generated_source = (
        ROOT / "kaggle/rsna-p9-s3d-specialist/rsna-p9-s3d-specialist.py"
    )
    assert sha256(generated_source) == manifest["script_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file() and sha256(path) == expected, name

    train = pd.read_csv(ROOT / "data/raw/train.csv", dtype={ID: str})
    original = pd.read_csv(
        ROOT / "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv",
        dtype={ID: str},
    )
    safe = pd.read_csv(root / "p9_safe_training_targets.csv", dtype={ID: str})
    studies = pd.read_csv(root / "p9_s3d_feature_studies.csv", dtype={ID: str})
    assert train[ID].tolist() == original[ID].tolist() == safe[ID].tolist()
    assert studies[ID].tolist() == train[ID].tolist()
    is_gold = original["is_gold"].astype(bool).to_numpy()
    gold = np.where(is_gold)[0]
    pseudo = np.where(~is_gold)[0]
    assert len(gold) == 58 and len(pseudo) == 4_349

    teacher_manifest_path = args.teacher_root / "p8_teacher_manifest.json"
    assert sha256(teacher_manifest_path) == EXPECTED_TEACHER_SHA256
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    teacher_studies = pd.read_csv(
        args.teacher_root / "p8_teacher_studies.csv", dtype={ID: str}
    )
    assert teacher_studies[ID].tolist() == train[ID].tolist()
    p2 = np.load(args.teacher_root / "p8_p2_direct_predictions.npy")
    p6 = np.load(args.teacher_root / "p8_p6_direct_predictions.npy")
    for name, array in (
        ("p8_p2_direct_predictions.npy", p2),
        ("p8_p6_direct_predictions.npy", p6),
    ):
        assert sha256(args.teacher_root / name) == teacher_manifest["artifacts_sha256"][name]
        assert array.shape == (4_407, 12)
    report_targets = original[TARGETS].to_numpy(np.float32)
    expected_safe = report_targets.copy()
    expected_safe[pseudo] = (
        0.50 * report_targets[pseudo]
        + 0.25 * p2[pseudo]
        + 0.25 * p6[pseudo]
    )
    observed_safe = safe[TARGETS].to_numpy(np.float32)
    assert np.array_equal(observed_safe[gold], report_targets[gold])
    assert np.allclose(observed_safe[pseudo], expected_safe[pseudo], atol=1e-7)
    non_targets = [column for column in original if column not in TARGETS]
    pd.testing.assert_frame_equal(
        safe[non_targets], original[non_targets], check_dtype=False, check_exact=True
    )

    features = np.load(root / "p9_s3d_features.npy", mmap_mode="r")
    masks = np.load(root / "p9_s3d_feature_mask.npy").astype(bool)
    assert features.shape == (4_407, 5, 1_024) and features.dtype == np.float16
    assert masks.shape == (4_407, 5) and masks.any(axis=1).all()
    assert np.isfinite(features).all()

    history = pd.read_csv(root / "p9_representation_history.csv")
    assert history["epoch"].tolist() == list(range(1, 7))
    assert history[
        ["train_weighted_bce", "pseudo_valid_weighted_bce"]
    ].notna().all().all()
    selected = history.loc[history["pseudo_valid_weighted_bce"].idxmin()]
    assert manifest["best_epoch"] == int(selected["epoch"]) == 5
    assert np.isclose(
        manifest["best_pseudo_valid_weighted_bce"],
        selected["pseudo_valid_weighted_bce"],
        atol=1e-12,
    )

    hashes = train["Report"].map(report_hash).to_numpy(str)
    fold_summary = pd.read_csv(root / "p9_fold_summary.csv")
    oof = pd.read_csv(root / "p9_oof_predictions.csv", dtype={ID: str})
    assert oof[ID].nunique() == len(oof) == 58
    gold_ids = train.iloc[gold][ID].tolist()
    assert oof[ID].tolist() == gold_ids
    labels = oof[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    expected_labels = original.loc[is_gold, TARGETS].to_numpy(np.float32)
    assert np.array_equal(labels, expected_labels)
    predictions = oof[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(predictions).all()

    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    mask_tensor = torch.from_numpy(masks)
    splitter = KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    replay = np.full_like(predictions, np.nan)
    for fold, (train_position, valid_position) in enumerate(splitter.split(gold)):
        valid_index = gold[valid_position]
        valid_hashes = set(hashes[valid_index])
        keep_pseudo = np.asarray(
            [index for index in pseudo if hashes[index] not in valid_hashes], dtype=int
        )
        expected_train_rows = len(keep_pseudo) + len(train_position)
        row = fold_summary.loc[fold_summary["fold"] == fold].iloc[0]
        assert int(row["train_rows"]) == expected_train_rows
        assert int(row["pseudo_rows"]) == len(keep_pseudo)
        assert int(row["expert_train_rows"]) == len(train_position)
        assert int(row["expert_valid_rows"]) == len(valid_position)
        assert int(row["excluded_pseudo_rows"]) == len(pseudo) - len(keep_pseudo)
        assert int(row["report_hash_overlap"]) == 0
        assert not (set(hashes[np.concatenate([keep_pseudo, gold[train_position]])]) & valid_hashes)
        assert np.array_equal(oof.iloc[valid_position]["fold"].to_numpy(), np.full(len(valid_position), fold))
        head = TargetSlotAttentionHead()
        state = torch.load(root / f"p9_fold_{fold}_head.pt", map_location="cpu", weights_only=True)
        incompatible = head.load_state_dict(state, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        head.eval()
        with torch.inference_mode():
            replay[valid_position] = torch.sigmoid(
                head(feature_tensor[valid_index], mask_tensor[valid_index])
            ).numpy()
    assert np.isfinite(replay).all()
    assert np.allclose(replay, predictions, atol=2e-7)

    representation_state = torch.load(
        root / "p9_s3d_representation.pt", map_location="cpu", weights_only=True
    )
    direct_head = TargetSlotAttentionHead()
    direct_state = {
        key.removeprefix("head."): value
        for key, value in representation_state.items()
        if key.startswith("head.")
    }
    direct_head.load_state_dict(direct_state, strict=True)
    direct_head.eval()
    with torch.inference_mode():
        replay_direct = torch.sigmoid(
            direct_head(feature_tensor[gold], mask_tensor[gold])
        ).numpy()
    direct = pd.read_csv(root / "p9_direct_expert_predictions.csv", dtype={ID: str})
    assert direct[ID].tolist() == gold_ids
    direct_predictions = direct[
        [f"pred__{target}" for target in TARGETS]
    ].to_numpy(np.float32)
    assert np.allclose(replay_direct, direct_predictions, atol=2e-7)

    p9_metrics = calculate_metrics(labels, predictions)
    direct_metrics = calculate_metrics(labels, direct_predictions)
    saved_oof_metrics = pd.read_csv(root / "p9_oof_metrics.csv").set_index("target")["auc"]
    saved_direct_metrics = pd.read_csv(root / "p9_direct_expert_metrics.csv").set_index("target")["auc"]
    for target in [*TARGETS, "MACRO"]:
        assert np.isclose(p9_metrics[target], saved_oof_metrics[target], atol=1e-12)
        assert np.isclose(direct_metrics[target], saved_direct_metrics[target], atol=1e-12)
    assert np.isclose(p9_metrics["MACRO"], manifest["oof_macro_auc"], atol=1e-12)
    assert np.isclose(
        direct_metrics["MACRO"], manifest["direct_expert_macro_auc"], atol=1e-12
    )

    current = pd.read_csv(args.current_oof, dtype={ID: str}).set_index(ID).loc[gold_ids].reset_index()
    current_labels = current[
        [f"label__{target}" for target in TARGETS]
    ].to_numpy(np.float32)
    assert np.array_equal(current_labels, labels)
    current_predictions = current[
        [f"pred__{target}" for target in TARGETS]
    ].to_numpy(np.float64)
    candidate_predictions = 0.85 * current_predictions + 0.15 * rank_columns(predictions)
    current_metrics = calculate_metrics(labels, current_predictions)
    candidate_metrics = calculate_metrics(labels, candidate_predictions)
    assert np.isclose(current_metrics["MACRO"], EXPECTED_CURRENT_MACRO, atol=1e-12)
    target_deltas = {
        target: candidate_metrics[target] - current_metrics[target]
        for target in TARGETS
    }
    delta_values = np.asarray(list(target_deltas.values()))
    bootstrap = paired_bootstrap(labels, current_predictions, candidate_predictions)
    score_gate = candidate_metrics["MACRO"] > current_metrics["MACRO"]
    nonnegative_gate = int((delta_values >= 0).sum()) >= 8
    worst_gate = float(delta_values.min()) > -0.02
    bootstrap_gate = bootstrap["probability_positive"] >= 0.80
    promotion_gate = all((score_gate, nonnegative_gate, worst_gate, bootstrap_gate))

    telemetry = pd.read_csv(root / "p9_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] == 100).all()
    log_path = root / "rsna-knee-p9-s3d-specialist.log"
    log = log_path.read_text(errors="replace")
    assert "P9_FULL_RUN_COMPLETE" in log
    assert "'rank': 0" in log and "'rank': 1" in log
    assert "Traceback" not in log and "CUDA out of memory" not in log

    target_comparison = pd.DataFrame({"target": TARGETS})
    target_comparison["p9_auc"] = [p9_metrics[target] for target in TARGETS]
    target_comparison["current_auc"] = [current_metrics[target] for target in TARGETS]
    target_comparison["candidate_auc"] = [candidate_metrics[target] for target in TARGETS]
    target_comparison["candidate_minus_current"] = [target_deltas[target] for target in TARGETS]
    target_comparison.to_csv(args.output_dir / "p9_target_comparison.csv", index=False)
    candidate_frame = pd.DataFrame({ID: gold_ids, "fold": oof["fold"]})
    for index, target in enumerate(TARGETS):
        candidate_frame[f"label__{target}"] = labels[:, index]
        candidate_frame[f"pred__{target}"] = candidate_predictions[:, index]
    candidate_frame.to_csv(args.output_dir / "p9_fixed_blend_oof_predictions.csv", index=False)

    result = {
        "schema_version": 1,
        "experiment_id": "P9",
        "run_id": "P9-S3D-SPECIALIST-v1",
        "status": "validated / promoted" if promotion_gate else "validated / rejected",
        "date": "2026-09-08",
        "validated": True,
        "kernel": {
            "ref": "vallepuvamsikrishna/rsna-knee-p9-s3d-specialist",
            "version": 1,
            "status": "KernelWorkerStatus.COMPLETE",
            "code_commit": "98d10af",
        },
        "source": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "script_sha256": manifest["script_sha256"],
            "checkpoint_sha256": EXPECTED_WEIGHT_SHA256,
            "teacher_manifest_sha256": EXPECTED_TEACHER_SHA256,
        },
        "integrity": {
            "artifact_hashes_passed": True,
            "safe_pseudo_targets_reconstructed": True,
            "expert_targets_unchanged": True,
            "p4_teacher_excluded": True,
            "representation_expert_gradient_rows": 0,
            "feature_shape": list(features.shape),
            "five_saved_heads_replayed": True,
            "direct_head_replayed": True,
            "expert_oof_rows": 58,
            "report_hash_overlap": 0,
            "traceback_absent": True,
            "cuda_oom_absent": True,
        },
        "training": {
            "best_epoch": manifest["best_epoch"],
            "best_pseudo_valid_weighted_bce": manifest[
                "best_pseudo_valid_weighted_bce"
            ],
            "representation_minutes": manifest["elapsed"]["representation_minutes"],
            "feature_and_head_minutes": manifest["elapsed"]["feature_and_head_minutes"],
            "total_minutes": manifest["elapsed"]["total_minutes"],
            "gpu_utilization_mean_percent": {
                str(index): float(utilization.loc[index, "mean"]) for index in (0, 1)
            },
            "gpu_utilization_max_percent": {
                str(index): int(utilization.loc[index, "max"]) for index in (0, 1)
            },
            "maximum_observed_memory_mib": int(telemetry["memory_used_mib"].max()),
        },
        "standalone": {
            "direct_macro_auc": direct_metrics["MACRO"],
            "oof_macro_auc": p9_metrics["MACRO"],
            "target_auc": {target: p9_metrics[target] for target in TARGETS},
        },
        "fixed_blend": {
            "formula": "0.85 current_P8_blend + 0.15 rank(P9)",
            "current_macro_auc": current_metrics["MACRO"],
            "candidate_macro_auc": candidate_metrics["MACRO"],
            "delta": candidate_metrics["MACRO"] - current_metrics["MACRO"],
            "target_deltas": target_deltas,
            "nonnegative_target_deltas": int((delta_values >= 0).sum()),
            "worst_target_delta": float(delta_values.min()),
            "paired_bootstrap": bootstrap,
        },
        "promotion_gates": {
            "candidate_exceeds_current": score_gate,
            "at_least_8_nonnegative_targets": nonnegative_gate,
            "worst_target_above_minus_0_02": worst_gate,
            "bootstrap_probability_at_least_0_80": bootstrap_gate,
            "all_passed": promotion_gate,
        },
        "decision": (
            "Promote P9 to hidden-test runtime validation."
            if promotion_gate
            else "Reject P9 without blend-weight or target-specific tuning; retain the promoted P8 submission system."
        ),
    }
    result_path = args.output_dir / "p9_comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    args.result_json.write_text(json.dumps(result, indent=2) + "\n")
    aggregate_manifest = {
        "schema_version": 1,
        "run_id": "P9-S3D-OOF-VALIDATION-v1",
        "validated": True,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifacts_sha256": {
            "p9_target_comparison.csv": sha256(
                args.output_dir / "p9_target_comparison.csv"
            ),
            "p9_fixed_blend_oof_predictions.csv": sha256(
                args.output_dir / "p9_fixed_blend_oof_predictions.csv"
            ),
            "p9_comparison.json": sha256(result_path),
        },
    }
    aggregate_path = args.output_dir / "p9_aggregate_manifest.json"
    aggregate_path.write_text(json.dumps(aggregate_manifest, indent=2) + "\n")
    print(json.dumps({
        "P9_VALIDATION_COMPLETE": True,
        "standalone_oof_macro_auc": p9_metrics["MACRO"],
        "current_blend_macro_auc": current_metrics["MACRO"],
        "candidate_blend_macro_auc": candidate_metrics["MACRO"],
        "blend_delta": candidate_metrics["MACRO"] - current_metrics["MACRO"],
        "nonnegative_target_deltas": int((delta_values >= 0).sum()),
        "worst_target_delta": float(delta_values.min()),
        "bootstrap_probability_positive": bootstrap["probability_positive"],
        "promotion_gate_passed": promotion_gate,
        "aggregate_manifest_sha256": sha256(aggregate_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
