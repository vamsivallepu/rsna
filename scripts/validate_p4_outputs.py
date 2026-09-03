#!/usr/bin/env python3
"""Independently validate P4 fold sources, aggregate OOF metrics, and P2 comparison."""

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
MODEL_SHA256 = "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/kaggle/rsna-p4-coatnet-e2e-v1"),
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


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack([
        pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
        for index in range(predictions.shape[1])
    ]).astype(np.float32)


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


def assert_bootstrap_equal(
    observed: dict[str, object], expected: dict[str, object]
) -> None:
    assert observed["repeats"] == expected["repeats"] == 5_000
    assert np.isclose(observed["mean_delta"], expected["mean_delta"], atol=1e-12)
    assert np.allclose(observed["ci95"], expected["ci95"], atol=1e-12)
    assert np.isclose(
        observed["probability_positive"],
        expected["probability_positive"],
        atol=1e-12,
    )


def main() -> int:
    args = parse_args()
    root = args.aggregate_dir
    manifest_path = root / "p4_aggregate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P4-COATNET-E2E-OOF-v1"
    assert manifest["validated_fold_outputs"] is True
    assert manifest["studies"] == 4_407
    assert manifest["expert_oof_rows"] == 58 and manifest["folds"] == 5
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file() and sha256(path) == expected, name

    train = pd.read_csv(args.train, dtype={ID: str})
    targets = pd.read_csv(args.targets, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist() and len(train) == 4_407
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold = np.where(is_gold)[0]
    pseudo = np.where(~is_gold)[0]
    assert len(gold) == 58 and len(pseudo) == 4_349
    hashes = train["Report"].map(report_hash).to_numpy(str)
    base_pseudo_train, base_pseudo_valid = stable_group_split(
        pseudo,
        hashes[pseudo],
        valid_fraction=0.10,
        seed=PSEUDO_SPLIT_SEED,
    )
    assert len(base_pseudo_train) == 3_914 and len(base_pseudo_valid) == 435

    oof = pd.read_csv(root / "p4_oof_predictions.csv", dtype={ID: str})
    gold_ids = train.iloc[gold][ID].tolist()
    assert oof[ID].tolist() == gold_ids
    assert oof[ID].nunique() == len(oof) == 58
    expected_folds = np.full(58, -1, dtype=int)
    expert_splitter = KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for fold, (expert_train_pos, expert_valid_pos) in enumerate(
        expert_splitter.split(gold)
    ):
        expected_folds[expert_valid_pos] = fold
        expert_train = gold[expert_train_pos]
        expert_valid = gold[expert_valid_pos]
        fold_root = args.fold_root / f"fold-{fold}"
        fold_manifest_path = fold_root / f"p4_fold_{fold}_manifest.json"
        fold_manifest = json.loads(fold_manifest_path.read_text())
        source = manifest["fold_sources"][str(fold)]
        assert sha256(fold_manifest_path) == source["manifest_sha256"]
        assert fold_manifest["artifacts_sha256"][f"p4_fold_{fold}_best.pt"] == (
            source["checkpoint_sha256"]
        )
        assert fold_manifest["script_sha256"] == source["script_sha256"]
        assert fold_manifest["model_sha256"] == MODEL_SHA256
        assert fold_manifest["cache_signature_sha256"] == CACHE_SIGNATURE
        for name, expected in fold_manifest["artifacts_sha256"].items():
            path = fold_root / name
            assert path.is_file() and sha256(path) == expected, name

        split = pd.read_csv(
            fold_root / f"p4_fold_{fold}_split.csv", dtype={ID: str}
        )
        assert split[ID].tolist() == train[ID].tolist()
        assert split["report_hash"].tolist() == hashes.tolist()
        assert split.loc[split["role"] == "expert_valid", ID].tolist() == (
            train.iloc[expert_valid][ID].tolist()
        )
        assert split.loc[split["role"] == "expert_train", ID].tolist() == (
            train.iloc[expert_train][ID].tolist()
        )
        training_roles = ["pseudo_train", "expert_train"]
        training_hashes = set(
            split.loc[split["role"].isin(training_roles), "report_hash"]
        )
        selection_hashes = set(
            split.loc[split["role"] == "pseudo_valid", "report_hash"]
        )
        validation_hashes = set(
            split.loc[split["role"] == "expert_valid", "report_hash"]
        )
        assert not training_hashes & selection_hashes
        assert not training_hashes & validation_hashes
        assert not selection_hashes & validation_hashes
        assert len(selection_hashes) <= len(base_pseudo_valid)

        fold_predictions = pd.read_csv(
            fold_root / f"p4_fold_{fold}_oof_predictions.csv", dtype={ID: str}
        )
        aggregate_fold = oof.loc[oof["fold"] == fold].reset_index(drop=True)
        assert fold_predictions.equals(aggregate_fold)
    assert np.array_equal(oof["fold"].to_numpy(int), expected_folds)

    labels = oof[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    p4_predictions = oof[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    expected_labels = targets.iloc[gold][TARGETS].to_numpy(np.float32)
    assert np.array_equal(labels, expected_labels)
    assert np.isfinite(p4_predictions).all()
    p4_metrics = calculate_metrics(labels, p4_predictions)
    stored_p4_metrics = pd.read_csv(root / "p4_oof_metrics.csv").set_index("target")["auc"]
    for target in [*TARGETS, "MACRO"]:
        assert np.isclose(p4_metrics[target], stored_p4_metrics.loc[target], atol=1e-12)

    p2 = pd.read_csv(args.p2_oof, dtype={ID: str}).set_index(ID).loc[gold_ids].reset_index()
    p2_labels = p2[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    p2_predictions = p2[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.array_equal(labels, p2_labels)
    p2_metrics = calculate_metrics(labels, p2_predictions)
    assert np.isclose(p2_metrics["MACRO"], 0.8911253543643616, atol=1e-12)
    rank_predictions = (
        rank_columns(p2_predictions) + rank_columns(p4_predictions)
    ) / 2.0
    rank_metrics = calculate_metrics(labels, rank_predictions)

    stored_comparison = json.loads((root / "p4_p2_comparison.json").read_text())
    result = manifest["result"]
    assert result == stored_comparison
    assert np.isclose(result["p2_macro_auc"], p2_metrics["MACRO"], atol=1e-12)
    assert np.isclose(result["p4_macro_auc"], p4_metrics["MACRO"], atol=1e-12)
    assert np.isclose(
        result["equal_rank_macro_auc"], rank_metrics["MACRO"], atol=1e-12
    )
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
    assert_bootstrap_equal(result["p4_vs_p2_bootstrap"], p4_bootstrap)
    assert_bootstrap_equal(result["equal_rank_vs_p2_bootstrap"], rank_bootstrap)

    print(json.dumps({
        "validated": True,
        "expert_oof_rows": len(oof),
        "p2_macro_auc": p2_metrics["MACRO"],
        "p4_macro_auc": p4_metrics["MACRO"],
        "p4_minus_p2": p4_metrics["MACRO"] - p2_metrics["MACRO"],
        "equal_rank_macro_auc": rank_metrics["MACRO"],
        "equal_rank_minus_p2": rank_metrics["MACRO"] - p2_metrics["MACRO"],
        "p4_vs_p2_bootstrap": p4_bootstrap,
        "equal_rank_vs_p2_bootstrap": rank_bootstrap,
        "fold_artifacts_and_report_purges_validated": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
