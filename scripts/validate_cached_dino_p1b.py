#!/usr/bin/env python3
"""Independently validate P1-B pooling-control predictions, heads, and bootstraps."""

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
EXPECTED_VARIANTS = ("shared_attention", "mean_pool", "max_pool")


class HeadCore(nn.Module):
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
        self.target_weight = nn.Parameter(torch.empty(len(TARGETS), HIDDEN))
        self.target_bias = nn.Parameter(torch.empty(len(TARGETS)))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[:, None, :] + self.position_embedding[None, :, :]
        return hidden.reshape(features.shape[0], N_SLOTS * N_POS, HIDDEN)

    def output(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bh,th->bt", pooled, self.target_weight) + self.target_bias


class SharedAttentionHead(HeadCore):
    def __init__(self) -> None:
        super().__init__()
        self.shared_query = nn.Parameter(torch.empty(HIDDEN))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        flat_mask = mask.reshape(features.shape[0], N_SLOTS * N_POS).bool()
        scores = torch.einsum("bkh,h->bk", hidden, self.shared_query) / (HIDDEN**0.5)
        attention = scores.masked_fill(~flat_mask, -1e4).softmax(dim=-1)
        return self.output(torch.einsum("bk,bkh->bh", attention, hidden))


class PoolingHead(HeadCore):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        flat_mask = mask.reshape(features.shape[0], N_SLOTS * N_POS).bool()
        if self.mode == "mean":
            pooled = (hidden * flat_mask[:, :, None]).sum(dim=1)
            pooled = pooled / flat_mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            pooled = hidden.masked_fill(~flat_mask[:, :, None], -1e4).amax(dim=1)
        return self.output(pooled)


def build_head(variant: str) -> nn.Module:
    if variant == "shared_attention":
        return SharedAttentionHead()
    if variant == "mean_pool":
        return PoolingHead("mean")
    if variant == "max_pool":
        return PoolingHead("max")
    raise ValueError(variant)


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
        raise RuntimeError("could not reproduce requested bootstrap")
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
    assert manifest["schema_version"] == 1 and manifest["experiment_id"] == "P1-B"
    config = manifest["config"]
    variants = config["variants"]
    assert tuple(variants) == EXPECTED_VARIANTS

    study_path = args.cache_dir / "dino_feature_studies.csv"
    target_path = args.cache_dir / "training_targets.csv"
    studies = pd.read_csv(study_path, dtype={ID: str})
    targets = pd.read_csv(target_path, dtype={ID: str})
    metadata = pd.read_csv(args.train_csv, dtype={ID: str})
    study_ids = studies[ID].to_numpy(str)
    targets = targets.set_index(ID).loc[study_ids].reset_index()
    metadata = metadata.set_index(ID).loc[study_ids].reset_index()
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    assert len(study_ids) == 4_407 and len(gold_indices) == 58 and len(pseudo_indices) == 4_349

    hashes = metadata["Report"].map(report_hash).to_numpy(str)
    splitter = KFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    fold_summary = pd.read_csv(args.output_dir / "fold_summary.csv")
    assert len(fold_summary) == len(variants) * config["folds"]
    expected_folds = np.full(len(gold_indices), -1, dtype=int)
    valid_positions = {}
    exclusions = 0
    for fold, (_, valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[valid_pos]
        valid_hashes = set(hashes[valid_index])
        excluded = [index for index in pseudo_indices if hashes[index] in valid_hashes]
        keep_pseudo = np.setdiff1d(pseudo_indices, excluded, assume_unique=True)
        train_gold = np.setdiff1d(gold_indices, valid_index, assume_unique=True)
        train_index = np.concatenate([keep_pseudo, train_gold])
        assert not (set(hashes[train_index]) & valid_hashes)
        for variant in variants:
            row = fold_summary[
                (fold_summary["variant"] == variant) & (fold_summary["fold"] == fold)
            ].iloc[0]
            assert int(row["train_rows"]) == len(train_index)
            assert int(row["excluded_pseudo_rows"]) == len(excluded)
            assert int(row["report_hash_overlap"]) == 0
        expected_folds[valid_pos] = fold
        valid_positions[fold] = valid_pos
        exclusions += len(excluded)
    assert exclusions == manifest["data"]["excluded_pseudo_studies_total_across_folds"] == 4

    features = np.load(args.cache_dir / "dino_token_features.npy", mmap_mode="r")
    mask = np.load(args.cache_dir / "dino_window_mask.npy", mmap_mode="r").astype(bool)
    assert list(features.shape) == manifest["data"]["token_shape"] == [4_407, 6, 3, 768]
    assert mask.shape == (4_407, 6, 3) and np.isfinite(features).all() and mask.all()
    labels = targets.loc[is_gold, TARGETS].to_numpy(np.float32)
    p0_frame = pd.read_csv(args.p0_dir / "p0_b_oof_predictions.csv", dtype={ID: str})
    assert p0_frame[ID].tolist() == study_ids[gold_indices].tolist()
    assert np.array_equal(p0_frame[TARGETS].to_numpy(np.float32), labels)
    p0_predictions = p0_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    p0_scores = metrics(labels, p0_predictions)
    assert np.isclose(p0_scores["MACRO"], manifest["baseline"]["macro_auc"])

    comparison = pd.read_csv(args.output_dir / "target_comparison.csv")
    variant_summary = pd.read_csv(args.output_dir / "variant_summary.csv").set_index("variant")
    head_audit = pd.read_csv(args.output_dir / "head_audit.csv").set_index("variant")
    gold_features = torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32))
    gold_mask = torch.from_numpy(mask[gold_indices])
    validated_results = {}

    for variant_index, variant in enumerate(variants):
        frame = pd.read_csv(
            args.output_dir / f"p1_b_{variant}_oof_predictions.csv", dtype={ID: str}
        )
        assert frame[ID].tolist() == study_ids[gold_indices].tolist()
        assert frame["fold"].to_numpy(int).tolist() == expected_folds.tolist()
        assert np.array_equal(frame[TARGETS].to_numpy(np.float32), labels)
        predictions = frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
        assert np.isfinite(predictions).all()
        scores = metrics(labels, predictions)
        recorded = pd.read_csv(args.output_dir / f"p1_b_{variant}_metrics.csv").set_index(
            "target"
        )["auc"]
        target_comparison = comparison[comparison["variant"] == variant].set_index("target")
        for target in [*TARGETS, "MACRO"]:
            assert np.isclose(scores[target], recorded.loc[target], atol=1e-12)
            assert np.isclose(
                scores[target], target_comparison.loc[target, "auc__candidate"], atol=1e-12
            )
            assert np.isclose(
                p0_scores[target], target_comparison.loc[target, "auc__p0_b"], atol=1e-12
            )

        replayed = np.full_like(predictions, np.nan)
        parameter_count = None
        for fold, valid_pos in valid_positions.items():
            row = fold_summary[
                (fold_summary["variant"] == variant) & (fold_summary["fold"] == fold)
            ].iloc[0]
            state = torch.load(
                args.output_dir / str(row["head_file"]), map_location="cpu", weights_only=True
            )
            assert all(torch.isfinite(value).all() for value in state.values())
            head = build_head(variant).eval()
            head.load_state_dict(state, strict=True)
            parameter_count = sum(parameter.numel() for parameter in head.parameters())
            with torch.inference_mode():
                replayed[valid_pos] = torch.sigmoid(
                    head(gold_features[valid_pos], gold_mask[valid_pos])
                ).numpy()
        assert np.allclose(replayed, predictions, atol=1e-7, rtol=1e-6)
        assert parameter_count == int(head_audit.loc[variant, "parameter_count"])

        bootstrap = paired_bootstrap(
            labels,
            p0_predictions,
            predictions,
            config["bootstrap_repeats"],
            config["seed"] + 300_000 + variant_index * 10_000,
        )
        for key, value in bootstrap.items():
            assert np.isclose(value, manifest["paired_bootstrap"][variant][key], atol=1e-12)
        assert np.isclose(scores["MACRO"], manifest["results"][variant]["macro_auc"])
        assert np.isclose(scores["MACRO"], variant_summary.loc[variant, "candidate_macro_auc"])
        validated_results[variant] = {
            "macro_auc": scores["MACRO"],
            "delta": scores["MACRO"] - p0_scores["MACRO"],
            "bootstrap_ci": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
            "saved_heads_reproduced": True,
        }

    observed_winner = max(validated_results, key=lambda name: validated_results[name]["macro_auc"])
    assert observed_winner == manifest["winner_by_observed_macro"]

    source_lookup = {
        "dino_token_features.npy": args.cache_dir / "dino_token_features.npy",
        "dino_window_mask.npy": args.cache_dir / "dino_window_mask.npy",
        "training_targets.csv": args.cache_dir / "training_targets.csv",
        "dino_pseudo_adapted_model.pt": args.cache_dir / "dino_pseudo_adapted_model.pt",
        "dino_feature_studies.csv": args.cache_dir / "dino_feature_studies.csv",
        "train.csv": args.train_csv,
        "P0_manifest.json": args.p0_dir / "manifest.json",
        "P0_B_oof_predictions.csv": args.p0_dir / "p0_b_oof_predictions.csv",
    }
    for name, expected in manifest["source_sha256"].items():
        assert sha256(source_lookup[name]) == expected
    for name, expected in manifest["output_sha256_excluding_manifest"].items():
        assert sha256(args.output_dir / name) == expected

    print(
        json.dumps(
            {
                "validated": True,
                "expert_oof_rows_per_variant": len(gold_indices),
                "excluded_pseudo_studies": exclusions,
                "p0_b_macro_auc": p0_scores["MACRO"],
                "results": validated_results,
                "winner_by_observed_macro": observed_winner,
                "saved_heads_reproduced": True,
                "bootstraps_recomputed": True,
                "source_and_output_hashes_verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
