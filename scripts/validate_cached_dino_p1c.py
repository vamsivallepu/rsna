#!/usr/bin/env python3
"""Independently validate P1-C ASL artifacts, heads, metrics, and bootstrap."""

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
import torch.nn.functional as F
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
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[:, None, :] + self.position_embedding[None, :, :]
        hidden = hidden.reshape(features.shape[0], N_SLOTS * N_POS, HIDDEN)
        flat_mask = mask.reshape(features.shape[0], N_SLOTS * N_POS).bool()
        scores = torch.einsum("bkh,th->btk", hidden, self.query) / (HIDDEN**0.5)
        attention = scores.masked_fill(~flat_mask[:, None, :], -1e4).softmax(dim=-1)
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
    return parser.parse_args()


def soft_asl(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
    eps: float,
) -> torch.Tensor:
    positive = torch.sigmoid(logits)
    negative = 1.0 - positive
    if clip > 0:
        negative = (negative + clip).clamp(max=1.0)
    likelihood = labels * torch.log(positive.clamp(min=eps))
    likelihood = likelihood + (1.0 - labels) * torch.log(negative.clamp(min=eps))
    probability_true = positive * labels + negative * (1.0 - labels)
    gamma = gamma_pos * labels + gamma_neg * (1.0 - labels)
    raw = -likelihood * torch.pow((1.0 - probability_true).clamp(min=0.0), gamma)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


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


def validate_loss_contract(config: dict[str, object]) -> dict[str, float | bool]:
    logits = torch.tensor(
        [[-3.0, -1.0, 0.0, 1.0], [3.0, 0.5, -0.5, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor(
        [[0.0, 0.25, 0.5, 1.0], [1.0, 0.75, 0.2, 0.0]], dtype=torch.float64
    )
    weights = torch.tensor(
        [[1.0, 0.5, 0.75, 2.0], [2.0, 1.0, 0.5, 1.0]], dtype=torch.float64
    )
    bce_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    bce = (bce_raw * weights).sum() / weights.sum()
    identity = soft_asl(logits, labels, weights, 0.0, 0.0, 0.0, float(config["eps"]))
    identity_error = float((bce - identity).abs().detach())
    actual = soft_asl(
        logits,
        labels,
        weights,
        float(config["gamma_neg"]),
        float(config["gamma_pos"]),
        float(config["clip"]),
        float(config["eps"]),
    )
    actual.backward()
    return {
        "gamma_zero_clip_zero_matches_weighted_bce": identity_error <= 1e-7,
        "identity_max_abs_error": identity_error,
        "official_default_loss_finite": bool(torch.isfinite(actual)),
        "official_default_gradient_finite": bool(torch.isfinite(logits.grad).all()),
        "official_default_gradient_max_abs": float(logits.grad.abs().max()),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["experiment_id"] == "P1-C"
    config = manifest["config"]
    assert config["gamma_neg"] == 4.0 and config["gamma_pos"] == 1.0
    assert config["clip"] == 0.05 and config["focal_weight_gradient_detached"] is False

    recomputed_contract = validate_loss_contract(config)
    recorded_contract = json.loads((args.output_dir / "loss_contract.json").read_text())
    for key, value in recomputed_contract.items():
        assert np.isclose(value, recorded_contract[key], atol=1e-12)
        assert np.isclose(value, manifest["loss_contract"][key], atol=1e-12)

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
    assert len(fold_summary) == config["folds"] and fold_summary["expert_valid_rows"].sum() == 58
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
        row = fold_summary[fold_summary["fold"] == fold].iloc[0]
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

    candidate_frame = pd.read_csv(
        args.output_dir / "p1_c_asl_oof_predictions.csv", dtype={ID: str}
    )
    p0_frame = pd.read_csv(args.p0_dir / "p0_b_oof_predictions.csv", dtype={ID: str})
    for frame in (candidate_frame, p0_frame):
        assert frame[ID].tolist() == study_ids[gold_indices].tolist()
        assert np.array_equal(frame[TARGETS].to_numpy(np.float32), labels)
    assert candidate_frame["fold"].to_numpy(int).tolist() == expected_folds.tolist()
    predictions = candidate_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    p0_predictions = p0_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(predictions).all() and np.isfinite(p0_predictions).all()
    scores = metrics(labels, predictions)
    p0_scores = metrics(labels, p0_predictions)
    recorded = pd.read_csv(args.output_dir / "p1_c_asl_metrics.csv").set_index("target")["auc"]
    comparison = pd.read_csv(args.output_dir / "target_comparison.csv").set_index("target")
    for target in [*TARGETS, "MACRO"]:
        assert np.isclose(scores[target], recorded.loc[target], atol=1e-12)
        assert np.isclose(scores[target], comparison.loc[target, "auc__p1_c"], atol=1e-12)
        assert np.isclose(p0_scores[target], comparison.loc[target, "auc__p0_b"], atol=1e-12)
    assert np.isclose(scores["MACRO"], manifest["result"]["macro_auc"])
    assert np.isclose(p0_scores["MACRO"], manifest["baseline"]["macro_auc"])

    gold_features = torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32))
    gold_mask = torch.from_numpy(mask[gold_indices])
    replayed = np.full_like(predictions, np.nan)
    for fold, valid_pos in valid_positions.items():
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
    assert np.allclose(replayed, predictions, atol=1e-7, rtol=1e-6)

    bootstrap = paired_bootstrap(
        labels,
        p0_predictions,
        predictions,
        config["bootstrap_repeats"],
        config["seed"] + 400_000,
    )
    for key, value in bootstrap.items():
        assert np.isclose(value, manifest["paired_bootstrap"][key], atol=1e-12)

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

    delta = scores["MACRO"] - p0_scores["MACRO"]
    assert np.isclose(delta, manifest["result"]["macro_delta_vs_p0_b"], atol=1e-12)
    print(
        json.dumps(
            {
                "validated": True,
                "expert_oof_rows": len(gold_indices),
                "excluded_pseudo_studies": exclusions,
                "loss_contract_recomputed": True,
                "p0_b_macro_auc": p0_scores["MACRO"],
                "p1_c_asl_macro_auc": scores["MACRO"],
                "macro_delta": delta,
                "bootstrap_ci": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
                "saved_heads_reproduced": True,
                "bootstrap_recomputed": True,
                "source_and_output_hashes_verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
