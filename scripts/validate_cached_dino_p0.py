#!/usr/bin/env python3
"""Independently validate P0 cached-DINO report-purge artifacts and saved heads."""

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
N_SLOTS = 6
N_POS = 3
TOKEN_DIM = 768
HIDDEN = 384


class TokenHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM),
            nn.Linear(TOKEN_DIM, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.slot_embedding = nn.Parameter(torch.empty(N_SLOTS, HIDDEN))
        self.position_embedding = nn.Parameter(torch.empty(N_POS, HIDDEN))
        self.query = nn.Parameter(torch.empty(len(TARGETS), HIDDEN))
        self.target_weight = nn.Parameter(torch.empty(len(TARGETS), HIDDEN))
        self.target_bias = nn.Parameter(torch.empty(len(TARGETS)))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[:, None, :] + self.position_embedding[None, :, :]
        hidden = hidden.reshape(batch, N_SLOTS * N_POS, HIDDEN)
        flat_mask = mask.reshape(batch, N_SLOTS * N_POS).bool()
        scores = torch.einsum("bkh,th->btk", hidden, self.query) / (HIDDEN**0.5)
        scores = scores.masked_fill(~flat_mask[:, None, :], -1e4)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("btk,bkh->bth", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/kaggle/rsna-dinov2-pseudo-adaptation"),
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    return parser.parse_args()


def normalized_hash(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = "".join(character for character in normalized if character.isalnum())
    if not normalized:
        raise ValueError("empty normalized report")
    return hashlib.sha256(normalized.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    values = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    values["MACRO"] = float(np.mean(list(values.values())))
    return values


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["experiment_id"] == "P0"
    config = manifest["config"]

    studies = pd.read_csv(args.cache_dir / "dino_feature_studies.csv", dtype={ID: str})
    targets = pd.read_csv(args.cache_dir / "training_targets.csv", dtype={ID: str})
    metadata = pd.read_csv(args.train_csv, dtype={ID: str})
    study_ids = studies[ID].to_numpy(str)
    targets = targets.set_index(ID).loc[study_ids].reset_index()
    metadata = metadata.set_index(ID).loc[study_ids].reset_index()
    hashes = metadata["Report"].map(normalized_hash).to_numpy(str)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    assert len(study_ids) == 4_407 and len(gold_indices) == 58 and len(pseudo_indices) == 4_349

    counts = pd.Series(hashes).value_counts()
    duplicate_hashes = set(counts[counts > 1].index)
    assert len(duplicate_hashes) == manifest["data"]["duplicate_report_groups"] == 52
    assert int(counts[counts > 1].sum()) == manifest["data"]["studies_in_duplicate_groups"] == 223
    assert manifest["data"]["cross_expert_pseudo_groups"] == 1
    assert manifest["data"]["expert_expert_duplicate_groups"] == 0

    fold_summary = pd.read_csv(args.output_dir / "fold_summary.csv")
    assert len(fold_summary) == 2 * config["folds"]
    assert fold_summary.groupby("arm")["expert_valid_rows"].sum().to_dict() == {
        "P0-A": 58,
        "P0-B": 58,
    }
    splitter = KFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    expected_exclusions = 0
    expected_folds = np.full(len(gold_indices), -1, dtype=int)
    expected_valid_indices: dict[int, np.ndarray] = {}
    for fold, (_, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[gold_valid_pos]
        expected_valid_indices[fold] = valid_index
        expected_folds[gold_valid_pos] = fold
        valid_hashes = set(hashes[valid_index])
        excluded = [index for index in pseudo_indices if hashes[index] in valid_hashes]
        expected_exclusions += len(excluded)
        row = fold_summary[(fold_summary["arm"] == "P0-B") & (fold_summary["fold"] == fold)].iloc[0]
        assert int(row["excluded_pseudo_rows"]) == len(excluded)
        assert int(row["report_hash_overlap"]) == 0
        expected_ids = sorted(study_ids[excluded])
        observed_ids = [] if pd.isna(row["excluded_pseudo_ids"]) else sorted(str(row["excluded_pseudo_ids"]).split(";"))
        assert observed_ids == expected_ids
    assert expected_exclusions == manifest["data"]["excluded_pseudo_studies_total_across_folds"]

    features = np.load(args.cache_dir / "dino_token_features.npy", mmap_mode="r")
    mask = np.load(args.cache_dir / "dino_window_mask.npy", mmap_mode="r").astype(bool)
    assert list(features.shape) == manifest["data"]["token_shape"] == [4_407, 6, 3, 768]
    assert mask.shape == (4_407, 6, 3) and np.isfinite(features).all() and mask.all()

    labels = targets.loc[is_gold, TARGETS].to_numpy(np.float32)
    observed_predictions: dict[str, np.ndarray] = {}
    for arm in ("P0-A", "P0-B"):
        slug = arm.lower().replace("-", "_")
        prediction_frame = pd.read_csv(args.output_dir / f"{slug}_oof_predictions.csv", dtype={ID: str})
        assert prediction_frame[ID].tolist() == study_ids[gold_indices].tolist()
        assert prediction_frame["fold"].to_numpy(int).tolist() == expected_folds.tolist()
        assert np.array_equal(prediction_frame[TARGETS].to_numpy(np.float32), labels)
        predictions = prediction_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
        assert np.isfinite(predictions).all()
        observed_predictions[arm] = predictions
        recomputed = metrics(labels, predictions)
        recorded = pd.read_csv(args.output_dir / f"{slug}_metrics.csv").set_index("target")["auc"]
        for target, value in recomputed.items():
            assert np.isclose(value, recorded.loc[target], atol=1e-12)
        assert np.isclose(recomputed["MACRO"], manifest["results"][arm]["macro_auc"], atol=1e-12)

    gold_feature_tensor = torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32))
    gold_mask_tensor = torch.from_numpy(mask[gold_indices])
    for arm in ("P0-A", "P0-B"):
        reconstructed = np.full_like(observed_predictions[arm], np.nan)
        for fold, valid_index in expected_valid_indices.items():
            row = fold_summary[(fold_summary["arm"] == arm) & (fold_summary["fold"] == fold)].iloc[0]
            state_path = args.output_dir / str(row["head_file"])
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            assert all(torch.isfinite(value).all() for value in state.values())
            head = TokenHead().eval()
            head.load_state_dict(state, strict=True)
            gold_positions = np.where(expected_folds == fold)[0]
            with torch.inference_mode():
                reconstructed[gold_positions] = torch.sigmoid(
                    head(gold_feature_tensor[gold_positions], gold_mask_tensor[gold_positions])
                ).numpy()
        assert np.allclose(reconstructed, observed_predictions[arm], atol=1e-7, rtol=1e-6)

    for name, expected in manifest["source_sha256"].items():
        path = args.train_csv if name == "train.csv" else args.cache_dir / name
        assert sha256(path) == expected
    for name, expected in manifest["output_sha256_excluding_manifest"].items():
        assert sha256(args.output_dir / name) == expected

    delta = metrics(labels, observed_predictions["P0-B"])["MACRO"] - metrics(
        labels, observed_predictions["P0-A"]
    )["MACRO"]
    assert np.isclose(delta, manifest["macro_delta__p0_b_minus_p0_a"], atol=1e-12)
    summary = {
        "validated": True,
        "studies": len(study_ids),
        "expert_oof_rows_per_arm": len(gold_indices),
        "duplicate_report_groups": len(duplicate_hashes),
        "studies_in_duplicate_groups": int(counts[counts > 1].sum()),
        "excluded_pseudo_studies": expected_exclusions,
        "p0_a_macro_auc": manifest["results"]["P0-A"]["macro_auc"],
        "p0_b_macro_auc": manifest["results"]["P0-B"]["macro_auc"],
        "macro_delta": delta,
        "saved_heads_reproduced": True,
        "source_and_output_hashes_verified": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
