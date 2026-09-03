#!/usr/bin/env python3
"""Independently validate P3 artifacts, OOF purges, replay, and P2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
N_TARGETS = len(TARGETS)
N_SLOTS = 5
N_REGIONS = 5
BACKBONE_DIM = 384
HEAD_DIM = 256
K_EVAL = 32
FOLD_SEED = 20260831
CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
MODEL_SHA256 = "1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050"
CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"


class HierarchicalDiagnosisHead(nn.Module):
    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(BACKBONE_DIM)
        self.projection = nn.Sequential(
            nn.Linear(BACKBONE_DIM, HEAD_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.region_embedding = nn.Parameter(torch.empty(N_REGIONS, HEAD_DIM))
        self.slot_embedding = nn.Parameter(torch.empty(N_SLOTS, HEAD_DIM))
        self.position_mlp = nn.Sequential(
            nn.Linear(1, HEAD_DIM),
            nn.GELU(),
            nn.Linear(HEAD_DIM, HEAD_DIM),
        )
        self.region_query = nn.Parameter(torch.empty(N_TARGETS, HEAD_DIM))
        self.window_query = nn.Parameter(torch.empty(N_TARGETS, HEAD_DIM))
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, HEAD_DIM))
        self.target_bias = nn.Parameter(torch.empty(N_TARGETS))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        slot_ids: torch.Tensor,
        relative_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.projection(self.input_norm(features))
        hidden = hidden + self.region_embedding[None, None]
        region_scores = torch.einsum("bkrh,th->bkrt", hidden, self.region_query)
        region_attention = (region_scores / math.sqrt(HEAD_DIM)).softmax(dim=2)
        local = torch.einsum("bkrt,bkrh->bkth", region_attention, hidden)
        context = self.slot_embedding[slot_ids.clamp(0, N_SLOTS - 1)]
        context = context + self.position_mlp(relative_positions[..., None])
        local = local + context[:, :, None, :]
        window_scores = torch.einsum("bkth,th->bkt", torch.tanh(local), self.window_query)
        window_scores = (window_scores / math.sqrt(HEAD_DIM)).masked_fill(
            ~mask[:, :, None].bool(), -1e4
        )
        pooled = torch.einsum("bkt,bkth->bth", window_scores.softmax(dim=1), local)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--p2-dir",
        type=Path,
        default=Path("outputs/kaggle/rsna-p2-coatnet-rmlp2-v2"),
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
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


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack([
        pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
        for index in range(predictions.shape[1])
    ]).astype(np.float32)


def replay(
    head: HierarchicalDiagnosisHead,
    features: np.ndarray,
    mask: np.ndarray,
    slots: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    output = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 16):
            stop = start + 16
            logits = head(
                torch.from_numpy(np.asarray(features[start:stop]).astype(np.float32)),
                torch.from_numpy(np.asarray(mask[start:stop]).astype(bool)),
                torch.from_numpy(np.asarray(slots[start:stop]).astype(np.int64)),
                torch.from_numpy(np.asarray(positions[start:stop]).astype(np.float32)),
            )
            output.append(torch.sigmoid(logits).numpy())
    return np.concatenate(output)


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repeats: int = 5_000,
    seed: int = FOLD_SEED + 300_000,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), len(labels))
        target_deltas = []
        for target_index in range(labels.shape[1]):
            sampled_labels = labels[sampled, target_index]
            if np.unique(sampled_labels).size < 2:
                break
            target_deltas.append(
                roc_auc_score(sampled_labels, candidate[sampled, target_index])
                - roc_auc_score(sampled_labels, baseline[sampled, target_index])
            )
        if len(target_deltas) == labels.shape[1]:
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
    manifest = json.loads((root / "p3_manifest.json").read_text())
    assert manifest["full_run"] is True
    assert manifest["run_id"] == "P3-DINOV2-HMIL-v1"
    assert manifest["studies"] == 4_407
    assert manifest["pseudo_studies"] == 4_349 and manifest["expert_studies"] == 58
    assert manifest["representation_gold_rows"] == 0
    assert manifest["pseudo_report_group_overlap"] == 0
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    assert manifest["model_sha256"] == MODEL_SHA256
    assert manifest["config_sha256"] == CONFIG_SHA256
    assert manifest["model_license"] == "Apache-2.0"
    assert manifest["cache_size"] == 336 and manifest["patch_grid"] == 24
    assert manifest["backbone_dim"] == BACKBONE_DIM
    assert manifest["region_grid"] == 2 and manifest["region_tokens"] == N_REGIONS
    assert manifest["unfrozen_blocks"] == 2
    assert manifest["k_eval"] == K_EVAL
    assert manifest["execution_revision"] == "p3_ddp_t4_v1"
    assert manifest["visible_gpus"] == 2 and manifest["data_parallel"] is True
    assert manifest["data_parallel_backend"] == "DistributedDataParallel"
    assert manifest["distributed_backend"] == "nccl" and manifest["world_size"] == 2
    assert manifest["study_batch"] == 2 and manifest["gradient_accumulation"] == 4
    assert manifest["loader_workers_per_rank"] == 2
    assert 3_000_000 < manifest["trainable_parameters"] < 6_000_000
    assert len(manifest["dropped_for_ddp"]) <= 1
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file() and sha256(path) == expected, name

    telemetry = pd.read_csv(root / "p3_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    assert telemetry.notna().all().all()
    gpu_utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (gpu_utilization["max"] >= 80.0).all()

    studies = pd.read_csv(root / "p3_feature_studies.csv", dtype={ID: str})
    train = pd.read_csv(args.train_csv, dtype={ID: str})
    assert studies[ID].tolist() == train[ID].tolist()
    features = np.load(root / "p3_dinov2_region_features.npy", mmap_mode="r")
    mask = np.load(root / "p3_feature_mask.npy", mmap_mode="r").astype(bool)
    slots = np.load(root / "p3_feature_slots.npy", mmap_mode="r")
    positions = np.load(root / "p3_feature_relative_positions.npy", mmap_mode="r")
    assert features.shape == (4_407, K_EVAL, N_REGIONS, BACKBONE_DIM)
    assert mask.shape == slots.shape == positions.shape == (4_407, K_EVAL)
    assert np.isfinite(features).all() and np.isfinite(positions).all()
    assert mask.any(axis=1).all()
    assert ((slots[mask] >= 0) & (slots[mask] < N_SLOTS)).all()
    assert (slots[~mask] == -1).all()
    assert ((positions[mask] >= -1.001) & (positions[mask] <= 1.001)).all()

    oof = pd.read_csv(root / "p3_oof_predictions.csv", dtype={ID: str})
    direct = pd.read_csv(root / "p3_direct_expert_predictions.csv", dtype={ID: str})
    folds = pd.read_csv(root / "p3_fold_summary.csv")
    assert oof[ID].nunique() == len(oof) == 58
    assert direct[ID].tolist() == oof[ID].tolist()
    assert folds.shape[0] == 5 and folds["expert_valid_rows"].sum() == 58
    assert (folds["report_hash_overlap"] == 0).all()
    labels = oof[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    oof_predictions = oof[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    direct_predictions = direct[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(oof_predictions).all() and np.isfinite(direct_predictions).all()

    stored_oof = pd.read_csv(root / "p3_oof_metrics.csv").set_index("target")["auc"]
    stored_direct = pd.read_csv(root / "p3_direct_expert_metrics.csv").set_index("target")["auc"]
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
    slot_gold = slots[[id_to_index[uid] for uid in oof[ID]]]
    position_gold = positions[[id_to_index[uid] for uid in oof[ID]]]
    full_state = torch.load(root / "p3_dinov2_pseudo_best.pt", map_location="cpu", weights_only=True)
    direct_state = {
        key.removeprefix("head."): value
        for key, value in full_state.items()
        if key.startswith("head.")
    }
    direct_head = HierarchicalDiagnosisHead()
    direct_head.load_state_dict(direct_state, strict=True)
    assert np.allclose(
        replay(direct_head, feature_gold, mask_gold, slot_gold, position_gold),
        direct_predictions,
        atol=2e-6,
    )

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
        state = torch.load(root / f"p3_fold_{fold}_head.pt", map_location="cpu", weights_only=True)
        head = HierarchicalDiagnosisHead()
        head.load_state_dict(state, strict=True)
        replayed_oof[valid_pos] = replay(
            head,
            feature_gold[valid_pos],
            mask_gold[valid_pos],
            slot_gold[valid_pos],
            position_gold[valid_pos],
        )
    assert np.array_equal(expected_fold, oof["fold"].to_numpy(int))
    assert np.allclose(replayed_oof, oof_predictions, atol=2e-6)

    p2 = pd.read_csv(args.p2_dir / "p2_oof_predictions.csv", dtype={ID: str})
    p2 = p2.set_index(ID).loc[oof[ID]].reset_index()
    p2_labels = p2[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
    p2_predictions = p2[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.array_equal(p2_labels, labels)
    p2_metrics = metrics(labels, p2_predictions)
    rank_predictions = (rank_columns(p2_predictions) + rank_columns(oof_predictions)) / 2.0
    rank_metrics = metrics(labels, rank_predictions)
    stored_comparison = json.loads((root / "p3_p2_comparison.json").read_text())
    p3_bootstrap = paired_bootstrap(labels, p2_predictions, oof_predictions)
    rank_bootstrap = paired_bootstrap(
        labels, p2_predictions, rank_predictions, seed=FOLD_SEED + 300_001
    )
    assert np.isclose(stored_comparison["p2_macro_auc"], p2_metrics["MACRO"], atol=1e-12)
    assert np.isclose(stored_comparison["p3_macro_auc"], oof_metrics["MACRO"], atol=1e-12)
    assert np.isclose(stored_comparison["equal_rank_macro_auc"], rank_metrics["MACRO"], atol=1e-12)
    assert stored_comparison["p3_vs_p2_bootstrap"] == p3_bootstrap
    assert stored_comparison["equal_rank_vs_p2_bootstrap"] == rank_bootstrap
    assert manifest["p2_comparison"] == stored_comparison

    summary = {
        "validated": True,
        "direct_macro_auc": direct_metrics["MACRO"],
        "p3_oof_macro_auc": oof_metrics["MACRO"],
        "p2_oof_macro_auc": p2_metrics["MACRO"],
        "p3_minus_p2": oof_metrics["MACRO"] - p2_metrics["MACRO"],
        "equal_rank_macro_auc": rank_metrics["MACRO"],
        "equal_rank_minus_p2": rank_metrics["MACRO"] - p2_metrics["MACRO"],
        "p3_vs_p2_bootstrap": p3_bootstrap,
        "equal_rank_vs_p2_bootstrap": rank_bootstrap,
        "saved_direct_and_fold_heads_replayed": True,
        "report_group_purge_validated": True,
        "gpu_utilization": gpu_utilization.to_dict("index"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
