#!/usr/bin/env python3
"""Independently validate P1-D regularization and repeated-seed artifacts."""

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
DEFAULT_CONFIG = {"dropout": 0.20, "weight_decay": 1e-3}
SCREEN_CONFIGS = {
    "dropout_0_10": {"dropout": 0.10, "weight_decay": 1e-3},
    "dropout_0_30": {"dropout": 0.30, "weight_decay": 1e-3},
    "weight_decay_3e_4": {"dropout": 0.20, "weight_decay": 3e-4},
    "weight_decay_3e_3": {"dropout": 0.20, "weight_decay": 3e-3},
}


class RegularizedTokenHead(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM),
            nn.Linear(TOKEN_DIM, HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
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


def read_predictions(path: Path, study_ids: list[str], folds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, dtype={ID: str})
    assert frame[ID].tolist() == study_ids
    assert frame["fold"].to_numpy(int).tolist() == folds.tolist()
    assert np.array_equal(frame[TARGETS].to_numpy(np.float32), labels)
    predictions = frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    assert np.isfinite(predictions).all()
    return predictions


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["experiment_id"] == "P1-D"
    config = manifest["config"]
    assert config["default_configuration"] == DEFAULT_CONFIG
    assert config["screen_configurations"] == SCREEN_CONFIGS
    assert len(config["repeat_seeds"]) == 2 and len(set(config["repeat_seeds"])) == 2
    assert config["fixed_expert_folds_across_seeds"] is True
    assert config["objective"] == "weighted BCE"

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
    labels = targets.loc[is_gold, TARGETS].to_numpy(np.float32)

    hashes = metadata["Report"].map(report_hash).to_numpy(str)
    splitter = KFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    expected_folds = np.full(len(gold_indices), -1, dtype=int)
    valid_positions: dict[int, np.ndarray] = {}
    train_indices: dict[int, np.ndarray] = {}
    excluded_indices: dict[int, list[int]] = {}
    exclusions = 0
    for fold, (_, valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[valid_pos]
        valid_hashes = set(hashes[valid_index])
        excluded = [index for index in pseudo_indices if hashes[index] in valid_hashes]
        keep_pseudo = np.setdiff1d(pseudo_indices, excluded, assume_unique=True)
        train_gold = np.setdiff1d(gold_indices, valid_index, assume_unique=True)
        train_index = np.concatenate([keep_pseudo, train_gold])
        assert not (set(hashes[train_index]) & valid_hashes)
        expected_folds[valid_pos] = fold
        valid_positions[fold] = valid_pos
        train_indices[fold] = train_index
        excluded_indices[fold] = excluded
        exclusions += len(excluded)
    assert exclusions == manifest["data"]["excluded_pseudo_studies_total_across_folds"] == 4

    features = np.load(args.cache_dir / "dino_token_features.npy", mmap_mode="r")
    mask = np.load(args.cache_dir / "dino_window_mask.npy", mmap_mode="r").astype(bool)
    assert list(features.shape) == manifest["data"]["token_shape"] == [4_407, 6, 3, 768]
    assert mask.shape == (4_407, 6, 3) and np.isfinite(features).all() and mask.all()
    gold_features = torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32))
    gold_mask = torch.from_numpy(mask[gold_indices])

    p0_predictions = read_predictions(
        args.p0_dir / "p0_b_oof_predictions.csv",
        study_ids[gold_indices].tolist(),
        expected_folds,
        labels,
    )
    p0_scores = metrics(labels, p0_predictions)
    assert np.isclose(p0_scores["MACRO"], manifest["screen"]["baseline_macro_auc"])

    selected = manifest["screen"]["selected_configuration"]
    assert selected in SCREEN_CONFIGS
    primary_seed = int(config["seed"])
    repeat_seeds = tuple(int(value) for value in config["repeat_seeds"])
    expected_runs: list[tuple[str, int, str]] = [
        (name, primary_seed, "screen") for name in SCREEN_CONFIGS
    ]
    expected_runs.extend(
        (name, seed, "confirmation")
        for seed in repeat_seeds
        for name in ("default", selected)
    )
    fold_summary = pd.read_csv(args.output_dir / "fold_summary.csv")
    run_summary = pd.read_csv(args.output_dir / "run_summary.csv")
    assert len(fold_summary) == len(expected_runs) * config["folds"]
    assert len(run_summary) == len(expected_runs)

    predictions_by_run: dict[tuple[str, int], np.ndarray] = {
        ("default", primary_seed): p0_predictions
    }
    scores_by_run: dict[tuple[str, int], dict[str, float]] = {
        ("default", primary_seed): p0_scores
    }
    for name, run_seed, stage in expected_runs:
        run_row = run_summary[
            (run_summary["configuration"] == name) & (run_summary["run_seed"] == run_seed)
        ].iloc[0]
        assert run_row["stage"] == stage
        run_config = DEFAULT_CONFIG if name == "default" else SCREEN_CONFIGS[name]
        assert np.isclose(run_row["dropout"], run_config["dropout"])
        assert np.isclose(run_row["weight_decay"], run_config["weight_decay"])
        predictions = read_predictions(
            args.output_dir / str(run_row["prediction_file"]),
            study_ids[gold_indices].tolist(),
            expected_folds,
            labels,
        )
        scores = metrics(labels, predictions)
        recorded_metrics = pd.read_csv(
            args.output_dir / str(run_row["metrics_file"])
        ).set_index("target")["auc"]
        for target in [*TARGETS, "MACRO"]:
            assert np.isclose(scores[target], recorded_metrics.loc[target], atol=1e-12)
        assert np.isclose(scores["MACRO"], run_row["macro_auc"], atol=1e-12)

        replayed = np.full_like(predictions, np.nan)
        for fold, valid_pos in valid_positions.items():
            fold_row = fold_summary[
                (fold_summary["configuration"] == name)
                & (fold_summary["run_seed"] == run_seed)
                & (fold_summary["fold"] == fold)
            ].iloc[0]
            assert fold_row["stage"] == stage
            assert int(fold_row["train_rows"]) == len(train_indices[fold])
            assert int(fold_row["excluded_pseudo_rows"]) == len(excluded_indices[fold])
            assert int(fold_row["report_hash_overlap"]) == 0
            assert np.isclose(fold_row["dropout"], run_config["dropout"])
            assert np.isclose(fold_row["weight_decay"], run_config["weight_decay"])
            state = torch.load(
                args.output_dir / str(fold_row["head_file"]),
                map_location="cpu",
                weights_only=True,
            )
            assert all(torch.isfinite(value).all() for value in state.values())
            head = RegularizedTokenHead(run_config["dropout"]).eval()
            head.load_state_dict(state, strict=True)
            with torch.inference_mode():
                replayed[valid_pos] = torch.sigmoid(
                    head(gold_features[valid_pos], gold_mask[valid_pos])
                ).numpy()
        assert np.allclose(replayed, predictions, atol=1e-7, rtol=1e-6)
        predictions_by_run[(name, run_seed)] = predictions
        scores_by_run[(name, run_seed)] = scores

    observed_selected = max(
        SCREEN_CONFIGS,
        key=lambda name: scores_by_run[(name, primary_seed)]["MACRO"],
    )
    assert observed_selected == selected
    assert np.isclose(
        scores_by_run[(selected, primary_seed)]["MACRO"],
        manifest["screen"]["selected_primary_seed_macro_auc"],
    )
    screen_summary = pd.read_csv(args.output_dir / "screen_summary.csv").set_index(
        "configuration"
    )
    for name, run_config in SCREEN_CONFIGS.items():
        candidate_scores = scores_by_run[(name, primary_seed)]
        deltas = np.asarray(
            [candidate_scores[target] - p0_scores[target] for target in TARGETS]
        )
        row = screen_summary.loc[name]
        assert np.isclose(row["dropout"], run_config["dropout"])
        assert np.isclose(row["weight_decay"], run_config["weight_decay"])
        assert np.isclose(row["candidate_macro_auc"], candidate_scores["MACRO"])
        assert np.isclose(row["macro_delta"], candidate_scores["MACRO"] - p0_scores["MACRO"])
        assert int(row["nonnegative_targets"]) == int((deltas >= 0).sum())
        assert np.isclose(row["worst_target_delta"], deltas.min())

    all_seeds = (primary_seed, *repeat_seeds)
    seed_summary = pd.read_csv(args.output_dir / "seed_summary.csv").set_index("run_seed")
    seed_deltas = []
    for run_seed in all_seeds:
        default_scores = scores_by_run[("default", run_seed)]
        candidate_scores = scores_by_run[(selected, run_seed)]
        target_deltas = np.asarray(
            [candidate_scores[target] - default_scores[target] for target in TARGETS]
        )
        macro_delta = candidate_scores["MACRO"] - default_scores["MACRO"]
        row = seed_summary.loc[run_seed]
        assert np.isclose(row["default_macro_auc"], default_scores["MACRO"])
        assert np.isclose(row["candidate_macro_auc"], candidate_scores["MACRO"])
        assert np.isclose(row["macro_delta"], macro_delta)
        assert int(row["nonnegative_targets"]) == int((target_deltas >= 0).sum())
        assert np.isclose(row["worst_target_delta"], target_deltas.min())
        seed_deltas.append(macro_delta)

    default_ensemble = np.mean(
        np.stack([predictions_by_run[("default", seed)] for seed in all_seeds]), axis=0
    ).astype(np.float32)
    candidate_ensemble = np.mean(
        np.stack([predictions_by_run[(selected, seed)] for seed in all_seeds]), axis=0
    ).astype(np.float32)
    recorded_default_ensemble = read_predictions(
        args.output_dir / "default_seed_ensemble_oof_predictions.csv",
        study_ids[gold_indices].tolist(),
        expected_folds,
        labels,
    )
    recorded_candidate_ensemble = read_predictions(
        args.output_dir / "candidate_seed_ensemble_oof_predictions.csv",
        study_ids[gold_indices].tolist(),
        expected_folds,
        labels,
    )
    assert np.array_equal(default_ensemble, recorded_default_ensemble)
    assert np.array_equal(candidate_ensemble, recorded_candidate_ensemble)
    default_ensemble_scores = metrics(labels, default_ensemble)
    candidate_ensemble_scores = metrics(labels, candidate_ensemble)
    recorded_default_metrics = pd.read_csv(
        args.output_dir / "default_seed_ensemble_metrics.csv"
    ).set_index("target")["auc"]
    recorded_candidate_metrics = pd.read_csv(
        args.output_dir / "candidate_seed_ensemble_metrics.csv"
    ).set_index("target")["auc"]
    comparison = pd.read_csv(args.output_dir / "ensemble_target_comparison.csv").set_index(
        "target"
    )
    for target in [*TARGETS, "MACRO"]:
        assert np.isclose(default_ensemble_scores[target], recorded_default_metrics.loc[target])
        assert np.isclose(candidate_ensemble_scores[target], recorded_candidate_metrics.loc[target])
        assert np.isclose(default_ensemble_scores[target], comparison.loc[target, "auc__default"])
        assert np.isclose(
            candidate_ensemble_scores[target], comparison.loc[target, "auc__candidate"]
        )

    bootstrap = paired_bootstrap(
        labels,
        default_ensemble,
        candidate_ensemble,
        config["bootstrap_repeats"],
        primary_seed + 500_000,
    )
    for key, value in bootstrap.items():
        assert np.isclose(value, manifest["confirmation"]["paired_bootstrap"][key], atol=1e-12)

    ensemble_delta = candidate_ensemble_scores["MACRO"] - default_ensemble_scores["MACRO"]
    ensemble_target_deltas = np.asarray(
        [candidate_ensemble_scores[target] - default_ensemble_scores[target] for target in TARGETS]
    )
    seed_wins = int((np.asarray(seed_deltas) > 0).sum())
    mean_seed_delta = float(np.mean(seed_deltas))
    nonnegative_targets = int((ensemble_target_deltas >= 0).sum())
    worst_target_delta = float(ensemble_target_deltas.min())
    rule = manifest["promotion_rule"]
    criteria = {
        "mean_seed_delta_positive": mean_seed_delta > 0,
        "seed_wins_at_least_2_of_3": seed_wins >= rule["minimum_seed_wins"],
        "ensemble_delta_positive": ensemble_delta > 0,
        "nonnegative_targets_at_least_8_of_12": (
            nonnegative_targets >= rule["minimum_nonnegative_targets"]
        ),
        "worst_target_delta_above_minus_0_02": (
            worst_target_delta > rule["minimum_worst_target_delta"]
        ),
        "bootstrap_probability_at_least_0_80": (
            bootstrap["probability_delta_gt_zero"]
            >= rule["minimum_bootstrap_probability_delta_gt_zero"]
        ),
    }
    promoted = all(criteria.values())
    confirmation = manifest["confirmation"]
    assert seed_wins == confirmation["seed_wins"]
    assert np.isclose(mean_seed_delta, confirmation["mean_seed_macro_delta"])
    assert np.isclose(default_ensemble_scores["MACRO"], confirmation["default_ensemble_macro_auc"])
    assert np.isclose(
        candidate_ensemble_scores["MACRO"], confirmation["candidate_ensemble_macro_auc"]
    )
    assert np.isclose(ensemble_delta, confirmation["ensemble_macro_delta"])
    assert nonnegative_targets == confirmation["nonnegative_targets"]
    assert np.isclose(worst_target_delta, confirmation["worst_target_delta"])
    assert criteria == confirmation["criteria"]
    assert promoted is confirmation["promoted"]

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
                "expert_oof_rows_per_run": len(gold_indices),
                "trained_runs": len(expected_runs),
                "saved_heads_reproduced": len(expected_runs) * config["folds"],
                "excluded_pseudo_studies": exclusions,
                "selected_configuration": selected,
                "seed_macro_deltas": seed_deltas,
                "ensemble_macro_delta": ensemble_delta,
                "bootstrap_ci": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
                "promotion_criteria": criteria,
                "promoted": promoted,
                "bootstrap_recomputed": True,
                "source_and_output_hashes_verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
