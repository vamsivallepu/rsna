#!/usr/bin/env python3
"""Independently validate P2 CoAtNet artifacts, OOF leakage, and P0 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
FEATURE_DIM = 1024
K_EVAL = 32
FOLD_SEED = 20260831
MODEL_SHA256 = "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"


class TargetAttentionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.attention = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256), nn.Tanh(), nn.Dropout(0.20),
            nn.Linear(256, len(TARGETS)),
        )
        self.target_weight = nn.Parameter(torch.empty(len(TARGETS), FEATURE_DIM))
        self.target_bias = nn.Parameter(torch.empty(len(TARGETS)))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(features)
        scores = self.attention(hidden).transpose(1, 2)
        attention = scores.masked_fill(~mask[:, None, :].bool(), -1e4).softmax(dim=-1)
        pooled = torch.einsum("btk,bkf->btf", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--p0-dir",
        type=Path,
        default=Path("outputs/experiments/P0-report-group-purged-dino-oof"),
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
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


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def replay(head: TargetAttentionHead, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 64):
            feature = torch.from_numpy(np.asarray(features[start:start + 64]).astype(np.float32))
            valid = torch.from_numpy(np.asarray(mask[start:start + 64]).astype(bool))
            output.append(torch.sigmoid(head(feature, valid)).numpy())
    return np.concatenate(output)


def paired_bootstrap(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    repeats: int = 5_000, seed: int = FOLD_SEED + 100_000,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), len(labels))
        target_deltas = []
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


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest = json.loads((root / "p2_manifest.json").read_text())
    assert manifest["full_run"] is True
    assert manifest["run_id"] == "P2-COATNET-v1"
    assert manifest["studies"] == 4_407
    assert manifest["pseudo_studies"] == 4_349 and manifest["expert_studies"] == 58
    assert manifest["representation_gold_rows"] == 0
    assert manifest["pseudo_report_group_overlap"] == 0
    assert manifest["model_sha256"] == MODEL_SHA256
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    assert manifest["feature_dim"] == FEATURE_DIM and manifest["k_eval"] == K_EVAL
    assert manifest["execution_revision"] == "ddp_t4_v3"
    assert manifest["visible_gpus"] == 2 and manifest["data_parallel"] is True
    assert manifest["data_parallel_backend"] == "DistributedDataParallel"
    assert manifest["distributed_backend"] == "nccl" and manifest["world_size"] == 2
    assert manifest["study_batch"] == 2 and manifest["eval_study_batch"] == 4
    assert manifest["loader_workers_per_rank"] == 2
    assert manifest["gradient_checkpointing"] is False
    assert manifest["channels_last"] is True and manifest["cudnn_benchmark"] is True
    assert manifest["fused_adamw"] is True
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file() and sha256(path) == expected, name

    telemetry = pd.read_csv(root / "p2_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    assert len(telemetry) >= 20 and telemetry.notna().all().all()
    gpu_utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )

    studies = pd.read_csv(root / "p2_coatnet_feature_studies.csv", dtype={ID: str})
    train = pd.read_csv(args.train_csv, dtype={ID: str})
    assert studies[ID].tolist() == train[ID].tolist()
    features = np.load(root / "p2_coatnet_features.npy", mmap_mode="r")
    mask = np.load(root / "p2_coatnet_feature_mask.npy", mmap_mode="r").astype(bool)
    positions = np.load(root / "p2_coatnet_feature_positions.npy", mmap_mode="r")
    assert features.shape == (4_407, K_EVAL, FEATURE_DIM)
    assert mask.shape == positions.shape == (4_407, K_EVAL)
    assert np.isfinite(features).all() and mask.any(axis=1).all()
    assert (positions[mask] >= 1).all() and (positions[~mask] == -1).all()

    oof = pd.read_csv(root / "p2_oof_predictions.csv", dtype={ID: str})
    direct = pd.read_csv(root / "p2_direct_expert_predictions.csv", dtype={ID: str})
    folds = pd.read_csv(root / "p2_fold_summary.csv")
    assert oof[ID].nunique() == len(oof) == 58
    assert direct[ID].tolist() == oof[ID].tolist()
    assert folds.shape[0] == 5 and folds["expert_valid_rows"].sum() == 58
    assert (folds["report_hash_overlap"] == 0).all()
    labels = oof[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    oof_predictions = oof[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    direct_predictions = direct[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(oof_predictions).all() and np.isfinite(direct_predictions).all()

    stored_oof = pd.read_csv(root / "p2_oof_metrics.csv").set_index("target")["auc"]
    stored_direct = pd.read_csv(root / "p2_direct_expert_metrics.csv").set_index("target")["auc"]
    oof_metrics = metrics(labels, oof_predictions)
    direct_metrics = metrics(labels, direct_predictions)
    for target in [*TARGETS, "MACRO"]:
        assert np.isclose(oof_metrics[target], stored_oof.loc[target], atol=1e-12)
        assert np.isclose(direct_metrics[target], stored_direct.loc[target], atol=1e-12)
    assert np.isclose(oof_metrics["MACRO"], manifest["fold_oof"]["macro_auc"], atol=1e-12)
    assert np.isclose(direct_metrics["MACRO"], manifest["direct_expert"]["macro_auc"], atol=1e-12)

    id_to_index = {uid: index for index, uid in enumerate(studies[ID])}
    feature_gold = features[[id_to_index[uid] for uid in oof[ID]]]
    mask_gold = mask[[id_to_index[uid] for uid in oof[ID]]]
    full_state = torch.load(root / "p2_coatnet_pseudo_best.pt", map_location="cpu", weights_only=True)
    direct_state = {
        key.removeprefix("head."): value for key, value in full_state.items() if key.startswith("head.")
    }
    direct_head = TargetAttentionHead()
    direct_head.load_state_dict(direct_state, strict=True)
    assert np.allclose(replay(direct_head, feature_gold, mask_gold), direct_predictions, atol=2e-6)

    replayed_oof = np.full_like(oof_predictions, np.nan)
    hashes = train["Report"].map(report_hash).to_numpy(str)
    gold_indices = np.asarray([id_to_index[uid] for uid in oof[ID]], dtype=int)
    pseudo_indices = np.setdiff1d(np.arange(len(train)), gold_indices)
    expected_fold = np.full(58, -1, dtype=int)
    splitter = KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, valid_pos) in enumerate(splitter.split(gold_indices)):
        expected_fold[valid_pos] = fold
        valid_hashes = set(hashes[gold_indices[valid_pos]])
        expected_excluded = sum(hashes[index] in valid_hashes for index in pseudo_indices)
        row = folds.loc[folds["fold"] == fold].iloc[0]
        assert int(row["excluded_pseudo_rows"]) == expected_excluded
        state = torch.load(root / f"p2_fold_{fold}_head.pt", map_location="cpu", weights_only=True)
        head = TargetAttentionHead()
        head.load_state_dict(state, strict=True)
        replayed_oof[valid_pos] = replay(head, feature_gold[valid_pos], mask_gold[valid_pos])
    assert np.array_equal(expected_fold, oof["fold"].to_numpy(int))
    assert np.allclose(replayed_oof, oof_predictions, atol=2e-6)

    p0 = pd.read_csv(args.p0_dir / "p0_b_oof_predictions.csv", dtype={ID: str}).set_index(ID).loc[oof[ID]]
    p0_predictions = p0[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    p0_labels = p0[TARGETS].to_numpy(np.float32)
    assert np.array_equal(p0_labels, labels)
    p0_metrics = metrics(labels, p0_predictions)
    bootstrap = paired_bootstrap(labels, p0_predictions, oof_predictions)
    target_deltas = {
        target: oof_metrics[target] - p0_metrics[target] for target in TARGETS
    }
    summary = {
        "validated": True,
        "direct_macro_auc": direct_metrics["MACRO"],
        "p2_oof_macro_auc": oof_metrics["MACRO"],
        "p0_b_macro_auc": p0_metrics["MACRO"],
        "delta__p2_minus_p0_b": oof_metrics["MACRO"] - p0_metrics["MACRO"],
        "nonnegative_target_deltas": sum(value >= 0 for value in target_deltas.values()),
        "worst_target_delta": min(target_deltas.values()),
        "paired_bootstrap": bootstrap,
        "saved_direct_and_fold_heads_replayed": True,
        "report_group_purge_validated": True,
        "gpu_utilization": gpu_utilization.to_dict("index"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
