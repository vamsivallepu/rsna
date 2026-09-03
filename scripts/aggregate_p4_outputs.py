#!/usr/bin/env python3
"""Validate five P4 fold outputs and build the canonical aggregate OOF result."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PSEUDO_SPLIT_SEED = 20260902
FOLD_SEED = 20260831
N_FOLDS = 5
MODEL_SHA256 = "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/kaggle/rsna-p4-coatnet-e2e-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/P4-coatnet-e2e-oof"),
    )
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"
        ),
    )
    parser.add_argument(
        "--p2-oof",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-p2-coatnet-rmlp2-v2/p2_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("kaggle/rsna-p4-coatnet-folds"),
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


def stable_group_split(
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups, dtype=str)
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_by_group = dict(zip(unique_groups.tolist(), counts.tolist(), strict=True))
    ordered_groups = sorted(
        unique_groups.tolist(),
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    target_rows = round(len(indices) * valid_fraction)
    valid_groups: set[str] = set()
    valid_rows = 0
    for group in ordered_groups:
        candidate_rows = valid_rows + count_by_group[group]
        if abs(candidate_rows - target_rows) <= abs(valid_rows - target_rows):
            valid_groups.add(group)
            valid_rows = candidate_rows
        if valid_rows == target_rows:
            break
    valid_mask = np.asarray([group in valid_groups for group in groups], dtype=bool)
    assert valid_mask.any() and (~valid_mask).any()
    assert not set(groups[valid_mask]) & set(groups[~valid_mask])
    return indices[~valid_mask], indices[valid_mask]


def metrics(labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    rows = [
        {
            "target": target,
            "auc": float(roc_auc_score(labels[:, target_index], predictions[:, target_index])),
        }
        for target_index, target in enumerate(TARGETS)
    ]
    rows.append({"target": "MACRO", "auc": float(np.mean([row["auc"] for row in rows]))})
    return pd.DataFrame(rows)


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
    return np.column_stack([
        pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
        for index in range(predictions.shape[1])
    ]).astype(np.float32)


def expected_contract(
    fold: int,
    *,
    train: pd.DataFrame,
    hashes: np.ndarray,
    is_gold: np.ndarray,
    base_pseudo_train: np.ndarray,
    base_pseudo_valid: np.ndarray,
    expert_train: np.ndarray,
    expert_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pseudo = np.where(~is_gold)[0]
    valid_hashes = set(hashes[expert_valid])
    train_hashes = set(hashes[expert_train])
    pseudo_train = np.asarray(
        [index for index in base_pseudo_train if hashes[index] not in valid_hashes],
        dtype=int,
    )
    pseudo_valid = np.asarray(
        [
            index
            for index in base_pseudo_valid
            if hashes[index] not in valid_hashes and hashes[index] not in train_hashes
        ],
        dtype=int,
    )
    roles = np.full(len(train), "", dtype=object)
    roles[base_pseudo_train] = "pseudo_train"
    roles[base_pseudo_valid] = "pseudo_valid"
    roles[expert_train] = "expert_train"
    roles[expert_valid] = "expert_valid"
    base_valid_set = set(base_pseudo_valid.tolist())
    for index in pseudo:
        if hashes[index] in valid_hashes:
            roles[index] = "excluded_expert_valid_report_twin"
        elif index in base_valid_set and hashes[index] in train_hashes:
            roles[index] = "excluded_pseudo_valid_expert_train_twin"
    representation = np.concatenate([pseudo_train, expert_train]).astype(int)
    if len(representation) % 2:
        pseudo_positions = np.where(~is_gold[representation])[0]
        assert len(pseudo_positions) > 0
        drop_position = int(pseudo_positions[-1])
        roles[representation[drop_position]] = "pseudo_ddp_balance_drop"
        representation = np.delete(representation, drop_position)
    assert (roles != "").all()
    assert len(representation) % 2 == 0
    assert not set(hashes[representation]) & valid_hashes
    assert not set(hashes[pseudo_valid]) & set(hashes[representation])
    assert not set(hashes[pseudo_valid]) & valid_hashes
    return roles, representation, pseudo_valid


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train, dtype={ID: str})
    targets = pd.read_csv(args.targets, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist() and len(train) == 4_407
    hashes = train["Report"].map(report_hash).to_numpy(str)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold = np.where(is_gold)[0]
    pseudo = np.where(~is_gold)[0]
    assert len(gold) == 58 and len(pseudo) == 4_349
    base_pseudo_train, base_pseudo_valid = stable_group_split(
        pseudo,
        hashes[pseudo],
        valid_fraction=0.10,
        seed=PSEUDO_SPLIT_SEED,
    )

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    expert_splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (expert_train_pos, expert_valid_pos) in enumerate(
        expert_splitter.split(gold)
    ):
        expert_train = gold[expert_train_pos]
        expert_valid = gold[expert_valid_pos]
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
        manifest_path = root / f"p4_fold_{fold}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["run_id"] == f"P4-COATNET-E2E-FOLD-{fold}-v1"
        assert manifest["full_run"] is True and manifest["speed_smoke"] is False
        assert manifest["fold"] == fold
        assert manifest["model_sha256"] == MODEL_SHA256
        assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
        assert manifest["seeds"] == {
            "pseudo_split": PSEUDO_SPLIT_SEED,
            "expert_folds": FOLD_SEED,
            "representation": 20260904 + fold,
        }
        assert manifest["expert_valid_rows"] == len(expert_valid)
        assert manifest["expert_train_rows"] == len(expert_train)
        assert manifest["representation_rows"] == len(representation)
        assert manifest["representation_gold_rows"] == len(expert_train)
        assert manifest["pseudo_valid_rows"] == len(pseudo_valid)
        assert manifest["checkpoint_selection_gold_rows"] == 0
        assert manifest["expert_valid_gradient_rows"] == 0
        assert manifest["expert_valid_report_overlap"] == 0
        assert manifest["selection_report_overlap"] == 0
        assert manifest["execution_revision"] == "p4_e2e_fold_ddp_t4_v1"
        assert manifest["visible_gpus"] == 2 and manifest["world_size"] == 2
        assert manifest["data_parallel_backend"] == "DistributedDataParallel"
        assert manifest["distributed_backend"] == "nccl"
        assert manifest["representation_epochs"] == 8
        generated_source = (
            args.bundle_root
            / f"fold-{fold}"
            / f"rsna-p4-coatnet-e2e-fold-{fold}.py"
        )
        assert sha256(generated_source) == manifest["script_sha256"]
        for name, expected in manifest["artifacts_sha256"].items():
            path = root / name
            assert path.is_file() and sha256(path) == expected, name

        split = pd.read_csv(root / f"p4_fold_{fold}_split.csv", dtype={ID: str})
        assert split[ID].tolist() == train[ID].tolist()
        assert split["report_hash"].tolist() == hashes.tolist()
        assert np.array_equal(split["is_gold"].astype(bool).to_numpy(), is_gold)
        assert split["role"].tolist() == roles.tolist()

        history = pd.read_csv(root / f"p4_fold_{fold}_history.csv")
        assert history["epoch"].tolist() == list(range(1, 9))
        assert history[["train_loss", "pseudo_valid_weighted_bce"]].notna().all().all()
        selected_row = history.loc[history["pseudo_valid_weighted_bce"].idxmin()]
        assert manifest["best_epoch"] == int(selected_row["epoch"])
        assert np.isclose(
            manifest["best_pseudo_valid_weighted_bce"],
            selected_row["pseudo_valid_weighted_bce"],
            atol=1e-12,
        )

        predictions = pd.read_csv(
            root / f"p4_fold_{fold}_oof_predictions.csv", dtype={ID: str}
        )
        expected_ids = train.iloc[expert_valid][ID].tolist()
        assert predictions[ID].tolist() == expected_ids
        assert predictions[ID].nunique() == len(predictions) == len(expert_valid)
        assert (predictions["fold"] == fold).all()
        for target in TARGETS:
            assert np.array_equal(
                predictions[f"label__{target}"].to_numpy(np.float32),
                targets.iloc[expert_valid][target].to_numpy(np.float32),
            )
            values = predictions[f"pred__{target}"].to_numpy(float)
            assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all()
        prediction_frames.append(predictions)

        telemetry = pd.read_csv(root / f"p4_fold_{fold}_gpu_telemetry.csv")
        assert set(telemetry["index"].unique()) == {0, 1}
        utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
            ["count", "mean", "median", "max"]
        )
        assert (utilization["max"] >= 90).all()
        fold_rows.append({
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
            "gpu_0_median": float(utilization.loc[0, "median"]),
            "gpu_1_median": float(utilization.loc[1, "median"]),
            "gpu_0_max": int(utilization.loc[0, "max"]),
            "gpu_1_max": int(utilization.loc[1, "max"]),
        })
        source_hashes[str(fold)] = {
            "manifest_sha256": sha256(manifest_path),
            "checkpoint_sha256": manifest["artifacts_sha256"][
                f"p4_fold_{fold}_best.pt"
            ],
            "script_sha256": manifest["script_sha256"],
        }

    combined = pd.concat(prediction_frames, ignore_index=True)
    assert combined[ID].nunique() == len(combined) == 58
    gold_ids = train.iloc[gold][ID].tolist()
    combined = combined.set_index(ID).loc[gold_ids].reset_index()
    labels = combined[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    p4_predictions = combined[
        [f"pred__{target}" for target in TARGETS]
    ].to_numpy(np.float32)
    p4_metrics = metrics(labels, p4_predictions)
    p4_macro = float(p4_metrics.loc[p4_metrics["target"] == "MACRO", "auc"].iloc[0])

    p2 = pd.read_csv(args.p2_oof, dtype={ID: str}).set_index(ID).loc[gold_ids].reset_index()
    p2_labels = p2[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    p2_predictions = p2[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.array_equal(labels, p2_labels)
    p2_metrics = metrics(labels, p2_predictions)
    p2_macro = float(p2_metrics.loc[p2_metrics["target"] == "MACRO", "auc"].iloc[0])
    assert np.isclose(p2_macro, 0.8911253543643616, atol=1e-12)

    rank_predictions = (
        rank_columns(p2_predictions) + rank_columns(p4_predictions)
    ) / 2.0
    rank_metrics = metrics(labels, rank_predictions)
    rank_macro = float(
        rank_metrics.loc[rank_metrics["target"] == "MACRO", "auc"].iloc[0]
    )
    comparison = pd.DataFrame({"target": [*TARGETS, "MACRO"]})
    comparison["p2_auc"] = p2_metrics["auc"]
    comparison["p4_auc"] = p4_metrics["auc"]
    comparison["equal_rank_auc"] = rank_metrics["auc"]
    comparison["p4_minus_p2"] = comparison["p4_auc"] - comparison["p2_auc"]
    comparison["rank_minus_p2"] = comparison["equal_rank_auc"] - comparison["p2_auc"]
    p4_target_deltas = comparison.loc[
        comparison["target"] != "MACRO", "p4_minus_p2"
    ].to_numpy(float)
    rank_target_deltas = comparison.loc[
        comparison["target"] != "MACRO", "rank_minus_p2"
    ].to_numpy(float)
    p4_bootstrap = paired_bootstrap(
        labels,
        p2_predictions,
        p4_predictions,
        seed=FOLD_SEED + 400_000,
    )
    rank_bootstrap = paired_bootstrap(
        labels,
        p2_predictions,
        rank_predictions,
        seed=FOLD_SEED + 400_001,
    )
    result = {
        "p2_macro_auc": p2_macro,
        "p4_macro_auc": p4_macro,
        "p4_minus_p2": p4_macro - p2_macro,
        "p4_nonnegative_targets": int((p4_target_deltas >= 0).sum()),
        "p4_worst_target_delta": float(p4_target_deltas.min()),
        "p4_vs_p2_bootstrap": p4_bootstrap,
        "equal_rank_macro_auc": rank_macro,
        "equal_rank_minus_p2": rank_macro - p2_macro,
        "equal_rank_nonnegative_targets": int((rank_target_deltas >= 0).sum()),
        "equal_rank_worst_target_delta": float(rank_target_deltas.min()),
        "equal_rank_vs_p2_bootstrap": rank_bootstrap,
    }

    combined.to_csv(args.output_dir / "p4_oof_predictions.csv", index=False)
    p4_metrics.to_csv(args.output_dir / "p4_oof_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "p4_fold_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "p4_p2_target_comparison.csv", index=False)
    rank_metrics.to_csv(args.output_dir / "p4_p2_equal_rank_metrics.csv", index=False)
    rank_frame = pd.DataFrame({ID: gold_ids, "fold": combined["fold"]})
    for target_index, target in enumerate(TARGETS):
        rank_frame[f"label__{target}"] = labels[:, target_index]
        rank_frame[f"pred__{target}"] = rank_predictions[:, target_index]
    rank_frame.to_csv(
        args.output_dir / "p4_p2_equal_rank_oof_predictions.csv", index=False
    )
    (args.output_dir / "p4_p2_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    artifact_names = [
        "p4_oof_predictions.csv",
        "p4_oof_metrics.csv",
        "p4_fold_summary.csv",
        "p4_p2_target_comparison.csv",
        "p4_p2_equal_rank_metrics.csv",
        "p4_p2_equal_rank_oof_predictions.csv",
        "p4_p2_comparison.json",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": "P4-COATNET-E2E-OOF-v1",
        "validated_fold_outputs": True,
        "studies": 4_407,
        "expert_oof_rows": 58,
        "folds": N_FOLDS,
        "fold_sources": source_hashes,
        "result": result,
        "artifacts_sha256": {
            name: sha256(args.output_dir / name) for name in artifact_names
        },
    }
    manifest_path = args.output_dir / "p4_aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "P4_AGGREGATE_COMPLETE": True,
        **result,
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
