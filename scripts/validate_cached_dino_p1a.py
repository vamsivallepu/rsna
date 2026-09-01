#!/usr/bin/env python3
"""Independently validate P1-A raw not-addressed weight experiment artifacts."""

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
    parser.add_argument(
        "--p0-dir",
        type=Path,
        default=Path("outputs/experiments/P0-report-group-purged-dino-oof"),
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--raw-labels",
        type=Path,
        default=Path("data/sample/public_weak_labels/llm_labels_full.csv"),
    )
    return parser.parse_args()


def report_hash(value: object) -> str:
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


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    deltas = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), size=len(labels))
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
    if len(values) != repeats:
        raise RuntimeError("could not reproduce requested paired bootstrap")
    return {
        "requested_repeats": repeats,
        "valid_repeats": len(values),
        "attempts": attempts,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci_2_5": float(np.quantile(values, 0.025)),
        "ci_97_5": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float((values > 0).mean()),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["experiment_id"] == "P1-A"
    config = manifest["config"]

    studies = pd.read_csv(args.cache_dir / "dino_feature_studies.csv", dtype={ID: str})
    targets = pd.read_csv(args.cache_dir / "training_targets.csv", dtype={ID: str})
    metadata = pd.read_csv(args.train_csv, dtype={ID: str})
    raw = pd.read_csv(args.raw_labels, dtype={ID: str})
    study_ids = studies[ID].to_numpy(str)
    targets = targets.set_index(ID).loc[study_ids].reset_index()
    metadata = metadata.set_index(ID).loc[study_ids].reset_index()
    raw = raw.set_index(ID).loc[study_ids].reset_index()
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    assert len(study_ids) == 4_407 and len(gold_indices) == 58 and len(pseudo_indices) == 4_349

    raw_values = raw[TARGETS].to_numpy(np.float32)
    raw_confidence = 2.0 * np.abs(raw_values - 0.5)
    reconstructed_weights = raw_confidence.copy()
    reconstructed_weights[gold_indices] = 2.0
    assert np.isfinite(reconstructed_weights).all()
    expected_zero_fraction = float((reconstructed_weights[pseudo_indices] == 0).mean())
    assert np.isclose(expected_zero_fraction, manifest["data"]["zero_weight_pseudo_cell_fraction"])
    weight_audit = pd.read_csv(args.output_dir / "weight_audit.csv").set_index("target")
    current_weights = targets[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
    for target_index, target in enumerate(TARGETS):
        pseudo_raw = raw_values[pseudo_indices, target_index]
        pseudo_weight = raw_confidence[pseudo_indices, target_index]
        assert np.isclose(
            weight_audit.loc[target, "not_addressed_fraction"],
            np.isclose(pseudo_raw, 0.5).mean(),
        )
        assert np.isclose(weight_audit.loc[target, "raw_confidence_mean"], pseudo_weight.mean())
        assert np.isclose(
            weight_audit.loc[target, "current_agreement_weight_mean"],
            current_weights[pseudo_indices, target_index].mean(),
        )

    hashes = metadata["Report"].map(report_hash).to_numpy(str)
    splitter = KFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    fold_summary = pd.read_csv(args.output_dir / "fold_summary.csv")
    assert len(fold_summary) == config["folds"] and fold_summary["expert_valid_rows"].sum() == 58
    expected_folds = np.full(len(gold_indices), -1, dtype=int)
    expected_valid_positions = {}
    exclusions = 0
    for fold, (_, valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[valid_pos]
        valid_hashes = set(hashes[valid_index])
        excluded = [index for index in pseudo_indices if hashes[index] in valid_hashes]
        keep_pseudo = np.setdiff1d(pseudo_indices, excluded, assume_unique=True)
        train_gold = np.setdiff1d(gold_indices, valid_index, assume_unique=True)
        train_index = np.concatenate([keep_pseudo, train_gold])
        assert not (set(hashes[train_index]) & valid_hashes)
        row = fold_summary[fold_summary["fold"] == fold].iloc[0]
        assert int(row["excluded_pseudo_rows"]) == len(excluded)
        assert int(row["report_hash_overlap"]) == 0
        assert np.isclose(
            row["zero_weight_cell_fraction"],
            (reconstructed_weights[train_index] == 0).mean(),
        )
        expected_folds[valid_pos] = fold
        expected_valid_positions[fold] = valid_pos
        exclusions += len(excluded)
    assert exclusions == manifest["data"]["excluded_pseudo_studies_total_across_folds"] == 4

    features = np.load(args.cache_dir / "dino_token_features.npy", mmap_mode="r")
    mask = np.load(args.cache_dir / "dino_window_mask.npy", mmap_mode="r").astype(bool)
    assert list(features.shape) == manifest["data"]["token_shape"] == [4_407, 6, 3, 768]
    assert mask.shape == (4_407, 6, 3) and np.isfinite(features).all() and mask.all()
    labels = targets.loc[is_gold, TARGETS].to_numpy(np.float32)

    p1_frame = pd.read_csv(args.output_dir / "p1_a_oof_predictions.csv", dtype={ID: str})
    p0_frame = pd.read_csv(args.p0_dir / "p0_b_oof_predictions.csv", dtype={ID: str})
    for frame in (p1_frame, p0_frame):
        assert frame[ID].tolist() == study_ids[gold_indices].tolist()
        assert np.array_equal(frame[TARGETS].to_numpy(np.float32), labels)
    assert p1_frame["fold"].to_numpy(int).tolist() == expected_folds.tolist()
    p1_predictions = p1_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    p0_predictions = p0_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(p1_predictions).all() and np.isfinite(p0_predictions).all()
    p1_scores = metrics(labels, p1_predictions)
    p0_scores = metrics(labels, p0_predictions)
    recorded_metrics = pd.read_csv(args.output_dir / "p1_a_metrics.csv").set_index("target")["auc"]
    comparison = pd.read_csv(args.output_dir / "target_comparison.csv").set_index("target")
    for target in [*TARGETS, "MACRO"]:
        assert np.isclose(p1_scores[target], recorded_metrics.loc[target], atol=1e-12)
        assert np.isclose(p1_scores[target], comparison.loc[target, "auc__p1_a"], atol=1e-12)
        assert np.isclose(p0_scores[target], comparison.loc[target, "auc__p0_b"], atol=1e-12)
    assert np.isclose(p1_scores["MACRO"], manifest["results"]["P1-A"]["macro_auc"])
    assert np.isclose(p0_scores["MACRO"], manifest["results"]["P0-B"]["macro_auc"])

    gold_features = torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32))
    gold_mask = torch.from_numpy(mask[gold_indices])
    replayed = np.full_like(p1_predictions, np.nan)
    for fold, valid_pos in expected_valid_positions.items():
        row = fold_summary[fold_summary["fold"] == fold].iloc[0]
        state = torch.load(
            args.output_dir / str(row["head_file"]), map_location="cpu", weights_only=True
        )
        assert all(torch.isfinite(value).all() for value in state.values())
        head = TokenHead().eval()
        head.load_state_dict(state, strict=True)
        with torch.inference_mode():
            replayed[valid_pos] = torch.sigmoid(
                head(gold_features[valid_pos], gold_mask[valid_pos])
            ).numpy()
    assert np.allclose(replayed, p1_predictions, atol=1e-7, rtol=1e-6)

    recomputed_bootstrap = paired_bootstrap(
        labels,
        p0_predictions,
        p1_predictions,
        config["bootstrap_repeats"],
        config["seed"] + 200_000,
    )
    for key, value in recomputed_bootstrap.items():
        assert np.isclose(value, manifest["paired_bootstrap"][key], atol=1e-12)

    source_lookup = {
        "dino_token_features.npy": args.cache_dir / "dino_token_features.npy",
        "dino_window_mask.npy": args.cache_dir / "dino_window_mask.npy",
        "training_targets.csv": args.cache_dir / "training_targets.csv",
        "dino_pseudo_adapted_model.pt": args.cache_dir / "dino_pseudo_adapted_model.pt",
        "train.csv": args.train_csv,
        "llm_labels_full.csv": args.raw_labels,
        "P0_manifest.json": args.p0_dir / "manifest.json",
    }
    for name, expected in manifest["source_sha256"].items():
        assert sha256(source_lookup[name]) == expected
    for name, expected in manifest["output_sha256_excluding_manifest"].items():
        assert sha256(args.output_dir / name) == expected

    delta = p1_scores["MACRO"] - p0_scores["MACRO"]
    assert np.isclose(delta, manifest["macro_delta__p1_a_minus_p0_b"], atol=1e-12)
    summary = {
        "validated": True,
        "expert_oof_rows": len(gold_indices),
        "excluded_pseudo_studies": exclusions,
        "zero_weight_pseudo_cell_fraction": expected_zero_fraction,
        "p0_b_macro_auc": p0_scores["MACRO"],
        "p1_a_macro_auc": p1_scores["MACRO"],
        "macro_delta": delta,
        "bootstrap_ci": [recomputed_bootstrap["ci_2_5"], recomputed_bootstrap["ci_97_5"]],
        "saved_heads_reproduced": True,
        "bootstrap_recomputed": True,
        "source_and_output_hashes_verified": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
