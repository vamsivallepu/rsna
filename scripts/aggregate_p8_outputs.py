#!/usr/bin/env python3
"""Validate five P8 folds and evaluate the frozen P8 promotion contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from aggregate_p4_outputs import (
    CACHE_SIGNATURE,
    FOLD_SEED,
    ID,
    N_FOLDS,
    PSEUDO_SPLIT_SEED,
    TARGETS,
    expected_contract,
    paired_bootstrap,
    rank_columns,
    report_hash,
    sha256,
    stable_group_split,
)


ROOT = Path(__file__).resolve().parents[1]
P6_MODEL_SHA256 = "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195"
TEACHER_MANIFEST_SHA256 = (
    "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
)
EXPECTED_P4_MACRO = 0.89116089660126
EXPECTED_CURRENT_BLEND_MACRO = 0.9054205862136073


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p8-noisy-student-v1",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p8-teacher-labels-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/experiments/P8-noisy-student-oof",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=ROOT / "experiments/results/P8.json",
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/train.csv")
    parser.add_argument(
        "--targets",
        type=Path,
        default=(
            ROOT
            / "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"
        ),
    )
    parser.add_argument(
        "--p2-oof",
        type=Path,
        default=(
            ROOT
            / "outputs/kaggle/rsna-p2-coatnet-rmlp2-v2/p2_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--p4-oof",
        type=Path,
        default=(
            ROOT
            / "outputs/experiments/P4-coatnet-e2e-oof/p4_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--current-p2-p4-oof",
        type=Path,
        default=(
            ROOT
            / "outputs/experiments/P4-coatnet-e2e-oof/"
            "p4_p2_equal_rank_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--p6-oof",
        type=Path,
        default=(
            ROOT
            / "outputs/kaggle/rsna-p6-p2-transfer-v1/p2_oof_predictions.csv"
        ),
    )
    return parser.parse_args()


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def load_oof(
    path: Path, gold_ids: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={ID: str})
    assert frame[ID].nunique() == len(frame) == 58
    frame = frame.set_index(ID).loc[gold_ids].reset_index()
    assert sorted(frame["fold"].unique().tolist()) == list(range(N_FOLDS))
    labels = frame[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    predictions = frame[
        [f"pred__{target}" for target in TARGETS]
    ].to_numpy(np.float32)
    assert np.isfinite(predictions).all()
    return frame, labels, predictions


def stability(
    baseline_metrics: dict[str, float], candidate_metrics: dict[str, float]
) -> dict[str, object]:
    target_deltas = {
        target: candidate_metrics[target] - baseline_metrics[target]
        for target in TARGETS
    }
    values = np.asarray(list(target_deltas.values()), dtype=float)
    return {
        "target_deltas": target_deltas,
        "nonnegative_target_deltas": int((values >= 0).sum()),
        "worst_target_delta": float(values.min()),
    }


def rank_columns_float64(predictions: np.ndarray) -> np.ndarray:
    """Match the canonical P6 blend's rank precision exactly."""
    return np.column_stack(
        [
            pd.Series(predictions[:, index])
            .rank(method="average", pct=True)
            .to_numpy()
            for index in range(predictions.shape[1])
        ]
    )


def validate_fold(
    fold: int,
    *,
    args: argparse.Namespace,
    train: pd.DataFrame,
    original_targets: pd.DataFrame,
    hashes: np.ndarray,
    is_gold: np.ndarray,
    base_pseudo_train: np.ndarray,
    base_pseudo_valid: np.ndarray,
    expert_train: np.ndarray,
    expert_valid: np.ndarray,
    teacher_manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    roles, representation, pseudo_valid = expected_contract(
        fold,
        train=train,
        hashes=hashes,
        is_gold=is_gold,
        base_pseudo_train=base_pseudo_train,
        base_pseudo_valid=base_pseudo_valid,
        expert_train=expert_train,
        expert_valid=expert_valid,
    )
    root = args.fold_root / f"fold-{fold}"
    manifest_path = root / f"p8_fold_{fold}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    teacher_target_path = args.teacher_root / f"p8_fold_{fold}_training_targets.csv"
    teacher_targets = pd.read_csv(teacher_target_path, dtype={ID: str})
    assert teacher_targets[ID].tolist() == train[ID].tolist()
    assert np.array_equal(
        teacher_targets.loc[is_gold, TARGETS].to_numpy(np.float32),
        original_targets.loc[is_gold, TARGETS].to_numpy(np.float32),
    )
    teacher_target_sha256 = sha256(teacher_target_path)
    assert (
        teacher_target_sha256
        == teacher_manifest["artifacts_sha256"][teacher_target_path.name]
    )

    assert manifest["run_id"] == f"P8-NOISY-STUDENT-FOLD-{fold}-v1"
    assert manifest["full_run"] is True and manifest["speed_smoke"] is False
    assert manifest["fold"] == fold
    assert manifest["seeds"] == {
        "pseudo_split": PSEUDO_SPLIT_SEED,
        "expert_folds": FOLD_SEED,
        "representation": 20260908 + fold,
    }
    assert manifest["model_sha256"] == P6_MODEL_SHA256
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    assert manifest["teacher_manifest_sha256"] == TEACHER_MANIFEST_SHA256
    assert manifest["teacher_target_sha256"] == teacher_target_sha256
    assert manifest["teacher_target_file"] == teacher_target_path.name
    assert manifest["pseudo_target_formula"] == (
        "0.50 report + 0.50 mean(P2 direct, P4 fold, P6 direct)"
    )
    assert manifest["representation_rows"] == len(representation)
    assert manifest["representation_gold_rows"] == len(expert_train)
    assert manifest["expert_train_rows"] == len(expert_train)
    assert manifest["expert_valid_rows"] == len(expert_valid)
    assert manifest["pseudo_valid_rows"] == len(pseudo_valid)
    assert manifest["checkpoint_selection_gold_rows"] == 0
    assert manifest["expert_valid_gradient_rows"] == 0
    assert manifest["expert_valid_report_overlap"] == 0
    assert manifest["selection_report_overlap"] == 0
    assert manifest["execution_revision"] == "p8_e2e_fold_ddp_t4_v1"
    assert manifest["visible_gpus"] == 2 and manifest["world_size"] == 2
    assert manifest["data_parallel_backend"] == "DistributedDataParallel"
    assert manifest["distributed_backend"] == "nccl"
    assert manifest["representation_epochs"] == 8
    generated_source = (
        ROOT
        / f"kaggle/rsna-p8-student-folds/fold-{fold}/"
        f"rsna-p8-noisy-student-fold-{fold}.py"
    )
    assert sha256(generated_source) == manifest["script_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        artifact = root / name
        assert artifact.is_file() and sha256(artifact) == expected, name

    split = pd.read_csv(root / f"p8_fold_{fold}_split.csv", dtype={ID: str})
    assert split[ID].tolist() == train[ID].tolist()
    assert split["report_hash"].tolist() == hashes.tolist()
    assert np.array_equal(split["is_gold"].astype(bool).to_numpy(), is_gold)
    assert split["role"].tolist() == roles.tolist()

    history = pd.read_csv(root / f"p8_fold_{fold}_history.csv")
    assert history["epoch"].tolist() == list(range(1, 9))
    assert history[["train_loss", "pseudo_valid_weighted_bce"]].notna().all().all()
    selected = history.loc[history["pseudo_valid_weighted_bce"].idxmin()]
    assert manifest["best_epoch"] == int(selected["epoch"])
    assert np.isclose(
        manifest["best_pseudo_valid_weighted_bce"],
        selected["pseudo_valid_weighted_bce"],
        atol=1e-12,
    )

    predictions = pd.read_csv(
        root / f"p8_fold_{fold}_oof_predictions.csv", dtype={ID: str}
    )
    assert predictions[ID].tolist() == train.iloc[expert_valid][ID].tolist()
    assert predictions[ID].nunique() == len(predictions) == len(expert_valid)
    assert (predictions["fold"] == fold).all()
    for target in TARGETS:
        assert np.array_equal(
            predictions[f"label__{target}"].to_numpy(np.float32),
            original_targets.iloc[expert_valid][target].to_numpy(np.float32),
        )
        values = predictions[f"pred__{target}"].to_numpy(float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all()

    telemetry = pd.read_csv(root / f"p8_fold_{fold}_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] >= 90).all()
    summary = {
        "fold": fold,
        "representation_rows": len(representation),
        "pseudo_valid_rows": len(pseudo_valid),
        "expert_train_rows": len(expert_train),
        "expert_valid_rows": len(expert_valid),
        "best_epoch": manifest["best_epoch"],
        "best_pseudo_valid_weighted_bce": manifest[
            "best_pseudo_valid_weighted_bce"
        ],
        "representation_minutes": manifest["elapsed"]["representation_minutes"],
        "fold_inference_minutes": manifest["elapsed"]["fold_inference_minutes"],
        "gpu_0_mean": float(utilization.loc[0, "mean"]),
        "gpu_1_mean": float(utilization.loc[1, "mean"]),
        "gpu_0_max": int(utilization.loc[0, "max"]),
        "gpu_1_max": int(utilization.loc[1, "max"]),
    }
    sources = {
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest["artifacts_sha256"][f"p8_fold_{fold}_best.pt"],
        "teacher_target_sha256": teacher_target_sha256,
        "script_sha256": manifest["script_sha256"],
    }
    return predictions, summary, sources


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train, dtype={ID: str})
    original_targets = pd.read_csv(args.targets, dtype={ID: str})
    assert train[ID].tolist() == original_targets[ID].tolist() and len(train) == 4_407
    teacher_manifest_path = args.teacher_root / "p8_teacher_manifest.json"
    assert sha256(teacher_manifest_path) == TEACHER_MANIFEST_SHA256
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    assert teacher_manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert teacher_manifest["expert_audit_macro_auc"]["relabel"] == (
        0.9209204490507412
    )

    hashes = train["Report"].map(report_hash).to_numpy(str)
    is_gold = original_targets["is_gold"].astype(bool).to_numpy()
    gold = np.where(is_gold)[0]
    pseudo = np.where(~is_gold)[0]
    assert len(gold) == 58 and len(pseudo) == 4_349
    gold_ids = train.iloc[gold][ID].tolist()
    base_pseudo_train, base_pseudo_valid = stable_group_split(
        pseudo,
        hashes[pseudo],
        valid_fraction=0.10,
        seed=PSEUDO_SPLIT_SEED,
    )

    prediction_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    fold_sources: dict[str, dict[str, str]] = {}
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (expert_train_position, expert_valid_position) in enumerate(
        splitter.split(gold)
    ):
        predictions, summary, sources = validate_fold(
            fold,
            args=args,
            train=train,
            original_targets=original_targets,
            hashes=hashes,
            is_gold=is_gold,
            base_pseudo_train=base_pseudo_train,
            base_pseudo_valid=base_pseudo_valid,
            expert_train=gold[expert_train_position],
            expert_valid=gold[expert_valid_position],
            teacher_manifest=teacher_manifest,
        )
        prediction_frames.append(predictions)
        fold_summaries.append(summary)
        fold_sources[str(fold)] = sources

    combined = pd.concat(prediction_frames, ignore_index=True)
    assert combined[ID].nunique() == len(combined) == 58
    combined = combined.set_index(ID).loc[gold_ids].reset_index()
    labels = combined[
        [f"label__{target}" for target in TARGETS]
    ].to_numpy(np.float32)
    expected_labels = original_targets.loc[
        is_gold, TARGETS
    ].to_numpy(np.float32)
    assert np.array_equal(labels, expected_labels)
    p8_predictions = combined[
        [f"pred__{target}" for target in TARGETS]
    ].to_numpy(np.float32)

    p2_frame, p2_labels, p2_predictions = load_oof(args.p2_oof, gold_ids)
    p4_frame, p4_labels, p4_predictions = load_oof(args.p4_oof, gold_ids)
    p6_frame, p6_labels, p6_predictions = load_oof(args.p6_oof, gold_ids)
    for frame, source_labels in (
        (p2_frame, p2_labels),
        (p4_frame, p4_labels),
        (p6_frame, p6_labels),
    ):
        assert frame[ID].tolist() == gold_ids
        assert np.array_equal(frame["fold"].to_numpy(), combined["fold"].to_numpy())
        assert np.array_equal(labels, source_labels)

    p4_metrics = calculate_metrics(labels, p4_predictions)
    p8_metrics = calculate_metrics(labels, p8_predictions)
    assert np.isclose(p4_metrics["MACRO"], EXPECTED_P4_MACRO, atol=1e-12)
    standalone_stability = stability(p4_metrics, p8_metrics)

    current_base_frame, current_base_labels, current_base_predictions = load_oof(
        args.current_p2_p4_oof, gold_ids
    )
    assert current_base_frame[ID].tolist() == gold_ids
    assert np.array_equal(current_base_labels, labels)
    current_base_metrics = calculate_metrics(labels, current_base_predictions)
    assert np.isclose(current_base_metrics["MACRO"], 0.8958210633119559, atol=1e-12)

    # Reproduce the promoted P6 calculation exactly: a saved FP32 two-model rank
    # baseline is mixed with FP64 P6 ranks. Build the P2+P8 replacement with the
    # same FP32 baseline path so only the P4-to-P8 model substitution changes.
    p2_rank = rank_columns(p2_predictions)
    p8_rank = rank_columns(p8_predictions)
    candidate_base_predictions = (p2_rank + p8_rank) / 2.0
    p6_rank = rank_columns_float64(p6_predictions)
    current_predictions = 0.75 * current_base_predictions + 0.25 * p6_rank
    candidate_predictions = 0.75 * candidate_base_predictions + 0.25 * p6_rank
    current_metrics = calculate_metrics(labels, current_predictions)
    candidate_metrics = calculate_metrics(labels, candidate_predictions)
    assert np.isclose(
        current_metrics["MACRO"], EXPECTED_CURRENT_BLEND_MACRO, atol=1e-12
    )
    standalone_bootstrap = paired_bootstrap(
        labels,
        p4_predictions,
        p8_predictions,
        seed=FOLD_SEED + 800_000,
    )
    blend_stability = stability(current_metrics, candidate_metrics)
    blend_bootstrap = paired_bootstrap(
        labels,
        current_predictions,
        candidate_predictions,
        seed=FOLD_SEED + 800_001,
    )

    standalone_gate = p8_metrics["MACRO"] > p4_metrics["MACRO"]
    blend_score_gate = candidate_metrics["MACRO"] > current_metrics["MACRO"]
    nonnegative_gate = blend_stability["nonnegative_target_deltas"] >= 8
    worst_target_gate = blend_stability["worst_target_delta"] > -0.02
    bootstrap_gate = blend_bootstrap["probability_positive"] >= 0.80
    oof_gate = all(
        (
            standalone_gate,
            blend_score_gate,
            nonnegative_gate,
            worst_target_gate,
            bootstrap_gate,
        )
    )

    standalone_comparison = pd.DataFrame({"target": [*TARGETS, "MACRO"]})
    standalone_comparison["p4_auc"] = [
        p4_metrics[name] for name in standalone_comparison["target"]
    ]
    standalone_comparison["p8_auc"] = [
        p8_metrics[name] for name in standalone_comparison["target"]
    ]
    standalone_comparison["p8_minus_p4"] = (
        standalone_comparison["p8_auc"] - standalone_comparison["p4_auc"]
    )
    blend_comparison = pd.DataFrame({"target": [*TARGETS, "MACRO"]})
    blend_comparison["current_auc"] = [
        current_metrics[name] for name in blend_comparison["target"]
    ]
    blend_comparison["candidate_auc"] = [
        candidate_metrics[name] for name in blend_comparison["target"]
    ]
    blend_comparison["candidate_minus_current"] = (
        blend_comparison["candidate_auc"] - blend_comparison["current_auc"]
    )

    candidate_frame = pd.DataFrame({ID: gold_ids, "fold": combined["fold"]})
    for index, target in enumerate(TARGETS):
        candidate_frame[f"label__{target}"] = labels[:, index]
        candidate_frame[f"pred__{target}"] = candidate_predictions[:, index]

    result = {
        "schema_version": 1,
        "experiment_id": "P8",
        "run_id": "P8-NOISY-STUDENT-OOF-v1",
        "status": "validated / canary required" if oof_gate else "validated / rejected",
        "date": "2026-09-07",
        "validated": True,
        "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
        "expert_oof_rows": 58,
        "folds": N_FOLDS,
        "fold_sources": fold_sources,
        "fold_summary": fold_summaries,
        "standalone": {
            "p4_macro_auc": p4_metrics["MACRO"],
            "p8_macro_auc": p8_metrics["MACRO"],
            "delta": p8_metrics["MACRO"] - p4_metrics["MACRO"],
            **standalone_stability,
            "paired_bootstrap": standalone_bootstrap,
            "gate_passed": standalone_gate,
        },
        "fixed_seven_model_blend": {
            "formula_current": "0.375 rank(P2) + 0.375 rank(P4) + 0.25 rank(P6)",
            "formula_candidate": "0.375 rank(P2) + 0.375 rank(P8) + 0.25 rank(P6)",
            "current_p6_macro_auc": current_metrics["MACRO"],
            "candidate_p8_macro_auc": candidate_metrics["MACRO"],
            "delta": candidate_metrics["MACRO"] - current_metrics["MACRO"],
            **blend_stability,
            "paired_bootstrap": blend_bootstrap,
        },
        "promotion_gates": {
            "standalone_exceeds_p4": standalone_gate,
            "fixed_blend_exceeds_current": blend_score_gate,
            "at_least_8_nonnegative_targets": nonnegative_gate,
            "worst_target_above_minus_0_02": worst_target_gate,
            "bootstrap_probability_at_least_0_80": bootstrap_gate,
            "oof_gate_passed": oof_gate,
            "hidden_runtime_canary_pending": oof_gate,
        },
        "promotion_gate_passed": False,
        "decision": (
            "Advance the frozen P8 seven-model blend to a hidden-test runtime canary; "
            "promotion remains blocked until the canary preserves at least 0.9 hours "
            "of Kaggle headroom."
            if oof_gate
            else "Reject P8 without tuning the frozen teacher or blend; retain P6 and "
            "advance to P9."
        ),
    }

    combined.to_csv(args.output_dir / "p8_oof_predictions.csv", index=False)
    pd.DataFrame(
        [{"target": target, "auc": p8_metrics[target]} for target in [*TARGETS, "MACRO"]]
    ).to_csv(args.output_dir / "p8_oof_metrics.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(
        args.output_dir / "p8_fold_summary.csv", index=False
    )
    standalone_comparison.to_csv(
        args.output_dir / "p8_vs_p4_target_comparison.csv", index=False
    )
    candidate_frame.to_csv(
        args.output_dir / "p8_fixed_blend_oof_predictions.csv", index=False
    )
    pd.DataFrame(
        [
            {"target": target, "auc": candidate_metrics[target]}
            for target in [*TARGETS, "MACRO"]
        ]
    ).to_csv(args.output_dir / "p8_fixed_blend_metrics.csv", index=False)
    blend_comparison.to_csv(
        args.output_dir / "p8_vs_current_blend_target_comparison.csv", index=False
    )
    comparison_path = args.output_dir / "p8_comparison.json"
    comparison_path.write_text(json.dumps(result, indent=2) + "\n")
    args.result_json.write_text(json.dumps(result, indent=2) + "\n")

    artifact_names = [
        "p8_oof_predictions.csv",
        "p8_oof_metrics.csv",
        "p8_fold_summary.csv",
        "p8_vs_p4_target_comparison.csv",
        "p8_fixed_blend_oof_predictions.csv",
        "p8_fixed_blend_metrics.csv",
        "p8_vs_current_blend_target_comparison.csv",
        "p8_comparison.json",
    ]
    aggregate_manifest = {
        "schema_version": 1,
        "run_id": "P8-NOISY-STUDENT-OOF-v1",
        "validated_fold_outputs": True,
        "expert_oof_rows": 58,
        "folds": N_FOLDS,
        "fold_sources": fold_sources,
        "result": result,
        "artifacts_sha256": {
            name: sha256(args.output_dir / name) for name in artifact_names
        },
    }
    manifest_path = args.output_dir / "p8_aggregate_manifest.json"
    manifest_path.write_text(json.dumps(aggregate_manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "P8_AGGREGATE_COMPLETE": True,
                "p8_standalone_macro_auc": p8_metrics["MACRO"],
                "candidate_blend_macro_auc": candidate_metrics["MACRO"],
                "current_blend_macro_auc": current_metrics["MACRO"],
                "oof_gate_passed": oof_gate,
                "manifest_sha256": sha256(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
