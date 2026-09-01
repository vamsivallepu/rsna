#!/usr/bin/env python3
"""Run P1-D: global regularization screen with matched repeated-seed confirmation."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from run_cached_dino_p0 import (
    HIDDEN,
    ID,
    N_POS,
    N_SLOTS,
    TARGETS,
    TOKEN_DIM,
    calculate_metrics,
    git_state,
    paired_bootstrap,
    report_hash,
    save_predictions,
    sha256,
    weighted_bce,
)


DEFAULT_CACHE = Path("outputs/kaggle/rsna-dinov2-pseudo-adaptation")
DEFAULT_P0 = Path("outputs/experiments/P0-report-group-purged-dino-oof")
DEFAULT_OUTPUT = Path("outputs/experiments/P1-D-regularization-seed-confirmation")
DEFAULT_CONFIG = {"dropout": 0.20, "weight_decay": 1e-3}
SCREEN_CONFIGS = {
    "dropout_0_10": {"dropout": 0.10, "weight_decay": 1e-3},
    "dropout_0_30": {"dropout": 0.30, "weight_decay": 1e-3},
    "weight_decay_3e_4": {"dropout": 0.20, "weight_decay": 3e-4},
    "weight_decay_3e_3": {"dropout": 0.20, "weight_decay": 3e-3},
}
DEFAULT_REPEAT_SEEDS = (20260901, 20260902)
MIN_SEED_WINS = 2
MIN_NONNEGATIVE_TARGETS = 8
MIN_WORST_TARGET_DELTA = -0.02
MIN_BOOTSTRAP_WIN_PROBABILITY = 0.80


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


def parse_repeat_seeds(value: str) -> tuple[int, int]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise argparse.ArgumentTypeError("repeat seeds must contain exactly two distinct integers")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--repeat-seeds",
        type=parse_repeat_seeds,
        default=DEFAULT_REPEAT_SEEDS,
        help="Two comma-separated training seeds; expert folds remain fixed to --seed.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.folds < 2:
        raise ValueError("epochs and batch size must be positive; folds must be at least two")
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    if args.seed in args.repeat_seeds:
        raise ValueError("repeat seeds must differ from the primary seed")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; choose another directory or pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()

    p0_manifest_path = args.p0_dir / "manifest.json"
    p0_prediction_path = args.p0_dir / "p0_b_oof_predictions.csv"
    p0_manifest = json.loads(p0_manifest_path.read_text())
    if p0_manifest["experiment_id"] != "P0":
        raise ValueError("the baseline directory does not contain P0")
    paired_config = {
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    for key, value in paired_config.items():
        if p0_manifest["config"][key] != value:
            raise ValueError(f"P0/P1-D config mismatch for {key}")
    if p0_manifest["config"]["weight_decay"] != DEFAULT_CONFIG["weight_decay"]:
        raise ValueError("P0 does not use the declared default weight decay")

    feature_path = args.cache_dir / "dino_token_features.npy"
    mask_path = args.cache_dir / "dino_window_mask.npy"
    target_path = args.cache_dir / "training_targets.csv"
    model_path = args.cache_dir / "dino_pseudo_adapted_model.pt"
    study_path = args.cache_dir / "dino_feature_studies.csv"
    source_paths = {
        "dino_token_features.npy": feature_path,
        "dino_window_mask.npy": mask_path,
        "training_targets.csv": target_path,
        "dino_pseudo_adapted_model.pt": model_path,
        "dino_feature_studies.csv": study_path,
        "train.csv": args.train_csv,
        "P0_manifest.json": p0_manifest_path,
        "P0_B_oof_predictions.csv": p0_prediction_path,
    }
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    for name, expected in p0_manifest["source_sha256"].items():
        if source_hashes[name] != expected:
            raise ValueError(f"P0 source hash drift: {name}")

    studies = pd.read_csv(study_path, dtype={ID: str})
    targets = pd.read_csv(target_path, dtype={ID: str})
    metadata = pd.read_csv(args.train_csv, dtype={ID: str})
    for name, frame in (("studies", studies), ("targets", targets), ("metadata", metadata)):
        if frame[ID].duplicated().any():
            raise ValueError(f"duplicate {ID} in {name}")
    study_ids = studies[ID].to_numpy(str)
    targets = targets.set_index(ID).loc[study_ids].reset_index()
    metadata = metadata.set_index(ID).loc[study_ids].reset_index()
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    if len(study_ids) != 4_407 or len(gold_indices) != 58 or len(pseudo_indices) != 4_349:
        raise ValueError("expected 4,407 studies, 58 expert rows, and 4,349 pseudo rows")
    labels = targets[TARGETS].to_numpy(np.float32)
    weights = targets[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
    if not np.isfinite(labels).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid labels or agreement weights")

    report_hashes = metadata["Report"].map(report_hash).to_numpy(str)
    features_np = np.load(feature_path, mmap_mode="r")
    mask_np = np.load(mask_path, mmap_mode="r")
    if features_np.shape != (4_407, N_SLOTS, N_POS, TOKEN_DIM):
        raise ValueError(f"unexpected feature shape {features_np.shape}")
    if mask_np.shape != (4_407, N_SLOTS, N_POS):
        raise ValueError(f"unexpected mask shape {mask_np.shape}")
    if not np.isfinite(features_np).all() or not mask_np.astype(bool).all():
        raise ValueError("invalid feature cache")
    print("loading cached tokens into CPU float32 memory", flush=True)
    feature_tensor = torch.from_numpy(np.asarray(features_np).astype(np.float32))
    mask_tensor = torch.from_numpy(np.asarray(mask_np).astype(bool))
    label_tensor = torch.from_numpy(labels)
    weight_tensor = torch.from_numpy(weights)

    full_state = torch.load(model_path, map_location="cpu", weights_only=True)
    initial_head_state = {
        key.removeprefix("head."): value.detach().cpu().clone()
        for key, value in full_state.items()
        if key.startswith("head.")
    }
    probe = RegularizedTokenHead(DEFAULT_CONFIG["dropout"])
    probe.load_state_dict(initial_head_state, strict=True)
    del probe, full_state

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_definitions = []
    gold_fold = np.full(len(gold_indices), -1, dtype=int)
    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[gold_valid_pos]
        valid_hashes = set(report_hashes[valid_index])
        excluded_pseudo = np.asarray(
            [index for index in pseudo_indices if report_hashes[index] in valid_hashes], dtype=int
        )
        keep_pseudo = np.setdiff1d(pseudo_indices, excluded_pseudo, assume_unique=True)
        train_index = np.concatenate([keep_pseudo, gold_indices[gold_train_pos]])
        if set(report_hashes[train_index]) & valid_hashes:
            raise RuntimeError(f"fold {fold} retains report-hash overlap")
        gold_fold[gold_valid_pos] = fold
        fold_definitions.append(
            (fold, gold_train_pos, gold_valid_pos, train_index, valid_index, excluded_pseudo)
        )
    if (gold_fold < 0).any():
        raise AssertionError("incomplete expert fold assignment")

    gold_labels = labels[gold_indices]
    gold_ids = study_ids[gold_indices]
    p0_frame = pd.read_csv(p0_prediction_path, dtype={ID: str})
    if p0_frame[ID].tolist() != gold_ids.tolist():
        raise ValueError("P0-B prediction study order differs")
    if p0_frame["fold"].to_numpy(int).tolist() != gold_fold.tolist():
        raise ValueError("P0-B fold assignment differs")
    if not np.array_equal(p0_frame[TARGETS].to_numpy(np.float32), gold_labels):
        raise ValueError("P0-B labels differ")
    p0_predictions = p0_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    p0_metrics = calculate_metrics(gold_labels, p0_predictions)

    predictions_by_run: dict[tuple[str, int], np.ndarray] = {
        ("default", args.seed): p0_predictions
    }
    metrics_by_run: dict[tuple[str, int], pd.DataFrame] = {
        ("default", args.seed): p0_metrics
    }
    fold_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    def train_run(name: str, config: dict[str, float], run_seed: int, stage: str) -> None:
        oof = np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32)
        for fold, gold_train_pos, _, train_index, valid_index, excluded in fold_definitions:
            fold_seed = run_seed + fold
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            head = RegularizedTokenHead(config["dropout"])
            head.load_state_dict(initial_head_state, strict=True)
            optimizer = torch.optim.AdamW(
                head.parameters(),
                lr=args.learning_rate,
                weight_decay=config["weight_decay"],
            )
            generator = torch.Generator().manual_seed(fold_seed)
            train_cpu = torch.as_tensor(train_index, dtype=torch.long)
            fold_started = time.time()
            final_loss = float("nan")
            for _ in range(args.epochs):
                head.train()
                permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
                losses = []
                for start in range(0, len(permutation), args.batch_size):
                    index = permutation[start : start + args.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = head(feature_tensor[index], mask_tensor[index])
                    loss = weighted_bce(logits, label_tensor[index], weight_tensor[index])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                final_loss = float(np.mean(losses))
            head.eval()
            with torch.inference_mode():
                prediction = torch.sigmoid(
                    head(feature_tensor[valid_index], mask_tensor[valid_index])
                ).numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"non-finite predictions for {name} seed {run_seed} fold {fold}")
            oof[valid_index] = prediction
            head_path = args.output_dir / f"p1-d_{name}_seed_{run_seed}_fold_{fold}_head.pt"
            torch.save(head.state_dict(), head_path)
            elapsed = time.time() - fold_started
            fold_rows.append(
                {
                    "stage": stage,
                    "configuration": name,
                    "run_seed": run_seed,
                    "fold": fold,
                    "dropout": config["dropout"],
                    "weight_decay": config["weight_decay"],
                    "train_rows": len(train_index),
                    "pseudo_rows": len(train_index) - len(gold_train_pos),
                    "expert_train_rows": len(gold_train_pos),
                    "expert_valid_rows": len(valid_index),
                    "excluded_pseudo_rows": len(excluded),
                    "excluded_pseudo_ids": ";".join(study_ids[excluded]),
                    "report_hash_overlap": 0,
                    "final_loss": final_loss,
                    "elapsed_seconds": elapsed,
                    "head_file": head_path.name,
                }
            )
            print(
                f"P1-D {stage} config={name} seed={run_seed} fold={fold} "
                f"loss={final_loss:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        gold_predictions = oof[gold_indices]
        if not np.isfinite(gold_predictions).all():
            raise RuntimeError(f"incomplete OOF predictions for {name} seed {run_seed}")
        predictions_by_run[(name, run_seed)] = gold_predictions
        metrics = calculate_metrics(gold_labels, gold_predictions)
        metrics_by_run[(name, run_seed)] = metrics
        prediction_file = f"p1_d_{name}_seed_{run_seed}_oof_predictions.csv"
        metrics_file = f"p1_d_{name}_seed_{run_seed}_metrics.csv"
        save_predictions(
            args.output_dir / prediction_file,
            gold_ids,
            gold_labels,
            gold_predictions,
            gold_fold,
        )
        metrics.to_csv(args.output_dir / metrics_file, index=False)
        run_rows.append(
            {
                "stage": stage,
                "configuration": name,
                "run_seed": run_seed,
                "dropout": config["dropout"],
                "weight_decay": config["weight_decay"],
                "macro_auc": float(metrics.loc[metrics["target"] == "MACRO", "auc"].iloc[0]),
                "prediction_file": prediction_file,
                "metrics_file": metrics_file,
            }
        )

    for name, config in SCREEN_CONFIGS.items():
        train_run(name, config, args.seed, "screen")

    p0_scores = p0_metrics.set_index("target")["auc"]
    screen_rows = []
    for name, config in SCREEN_CONFIGS.items():
        scores = metrics_by_run[(name, args.seed)].set_index("target")["auc"]
        deltas = scores.loc[TARGETS] - p0_scores.loc[TARGETS]
        screen_rows.append(
            {
                "configuration": name,
                "dropout": config["dropout"],
                "weight_decay": config["weight_decay"],
                "p0_b_macro_auc": p0_scores.loc["MACRO"],
                "candidate_macro_auc": scores.loc["MACRO"],
                "macro_delta": scores.loc["MACRO"] - p0_scores.loc["MACRO"],
                "nonnegative_targets": int((deltas >= 0).sum()),
                "worst_target_delta": float(deltas.min()),
            }
        )
    screen_summary = pd.DataFrame(screen_rows)
    screen_summary.to_csv(args.output_dir / "screen_summary.csv", index=False)
    selected_name = max(
        SCREEN_CONFIGS,
        key=lambda name: float(
            metrics_by_run[(name, args.seed)]
            .set_index("target")
            .loc["MACRO", "auc"]
        ),
    )
    selected_config = SCREEN_CONFIGS[selected_name]
    print(f"P1-D selected screen candidate: {selected_name}", flush=True)

    for run_seed in args.repeat_seeds:
        train_run("default", DEFAULT_CONFIG, run_seed, "confirmation")
        train_run(selected_name, selected_config, run_seed, "confirmation")

    seed_rows = []
    all_seeds = (args.seed, *args.repeat_seeds)
    for run_seed in all_seeds:
        default_scores = metrics_by_run[("default", run_seed)].set_index("target")["auc"]
        candidate_scores = metrics_by_run[(selected_name, run_seed)].set_index("target")["auc"]
        target_deltas = candidate_scores.loc[TARGETS] - default_scores.loc[TARGETS]
        seed_rows.append(
            {
                "run_seed": run_seed,
                "default_macro_auc": default_scores.loc["MACRO"],
                "candidate_macro_auc": candidate_scores.loc["MACRO"],
                "macro_delta": candidate_scores.loc["MACRO"] - default_scores.loc["MACRO"],
                "nonnegative_targets": int((target_deltas >= 0).sum()),
                "worst_target_delta": float(target_deltas.min()),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(args.output_dir / "seed_summary.csv", index=False)

    default_ensemble = np.mean(
        np.stack([predictions_by_run[("default", seed)] for seed in all_seeds]), axis=0
    ).astype(np.float32)
    candidate_ensemble = np.mean(
        np.stack([predictions_by_run[(selected_name, seed)] for seed in all_seeds]), axis=0
    ).astype(np.float32)
    default_ensemble_metrics = calculate_metrics(gold_labels, default_ensemble)
    candidate_ensemble_metrics = calculate_metrics(gold_labels, candidate_ensemble)
    default_ensemble_metrics.to_csv(args.output_dir / "default_seed_ensemble_metrics.csv", index=False)
    candidate_ensemble_metrics.to_csv(
        args.output_dir / "candidate_seed_ensemble_metrics.csv", index=False
    )
    save_predictions(
        args.output_dir / "default_seed_ensemble_oof_predictions.csv",
        gold_ids,
        gold_labels,
        default_ensemble,
        gold_fold,
    )
    save_predictions(
        args.output_dir / "candidate_seed_ensemble_oof_predictions.csv",
        gold_ids,
        gold_labels,
        candidate_ensemble,
        gold_fold,
    )
    ensemble_comparison = default_ensemble_metrics.merge(
        candidate_ensemble_metrics,
        on="target",
        suffixes=("__default", "__candidate"),
    )
    ensemble_comparison["delta__candidate_minus_default"] = (
        ensemble_comparison["auc__candidate"] - ensemble_comparison["auc__default"]
    )
    ensemble_comparison.to_csv(args.output_dir / "ensemble_target_comparison.csv", index=False)
    bootstrap = paired_bootstrap(
        gold_labels,
        default_ensemble,
        candidate_ensemble,
        args.bootstrap_repeats,
        args.seed + 500_000,
    )
    (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_summary.csv", index=False)
    pd.DataFrame(run_rows).to_csv(args.output_dir / "run_summary.csv", index=False)

    seed_wins = int((seed_summary["macro_delta"] > 0).sum())
    mean_seed_delta = float(seed_summary["macro_delta"].mean())
    ensemble_macro_delta = float(
        ensemble_comparison.loc[
            ensemble_comparison["target"] == "MACRO", "delta__candidate_minus_default"
        ].iloc[0]
    )
    target_deltas = ensemble_comparison.set_index("target").loc[
        TARGETS, "delta__candidate_minus_default"
    ]
    nonnegative_targets = int((target_deltas >= 0).sum())
    worst_target_delta = float(target_deltas.min())
    criteria = {
        "mean_seed_delta_positive": mean_seed_delta > 0,
        "seed_wins_at_least_2_of_3": seed_wins >= MIN_SEED_WINS,
        "ensemble_delta_positive": ensemble_macro_delta > 0,
        "nonnegative_targets_at_least_8_of_12": (
            nonnegative_targets >= MIN_NONNEGATIVE_TARGETS
        ),
        "worst_target_delta_above_minus_0_02": worst_target_delta > MIN_WORST_TARGET_DELTA,
        "bootstrap_probability_at_least_0_80": (
            bootstrap["probability_delta_gt_zero"] >= MIN_BOOTSTRAP_WIN_PROBABILITY
        ),
    }
    promoted = all(criteria.values())

    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "P1-D",
        "description": "factor-at-a-time global regularization screen and seed confirmation",
        "created_unix": time.time(),
        "git": git_state(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "threads": args.threads,
        },
        "config": {
            **paired_config,
            "repeat_seeds": list(args.repeat_seeds),
            "bootstrap_repeats": args.bootstrap_repeats,
            "default_configuration": DEFAULT_CONFIG,
            "screen_configurations": SCREEN_CONFIGS,
            "selected_by": "highest primary-seed macro AUC; ties use declared order",
            "fixed_expert_folds_across_seeds": True,
            "objective": "weighted BCE",
            "label_weights": "current three-source agreement weights",
            "report_group_purge": True,
        },
        "promotion_rule": {
            "minimum_seed_wins": MIN_SEED_WINS,
            "minimum_nonnegative_targets": MIN_NONNEGATIVE_TARGETS,
            "minimum_worst_target_delta": MIN_WORST_TARGET_DELTA,
            "minimum_bootstrap_probability_delta_gt_zero": MIN_BOOTSTRAP_WIN_PROBABILITY,
            "all_criteria_required": True,
        },
        "data": {
            "studies": len(study_ids),
            "expert_studies": len(gold_indices),
            "pseudo_studies": len(pseudo_indices),
            "excluded_pseudo_studies_total_across_folds": int(
                sum(len(item[-1]) for item in fold_definitions)
            ),
            "token_shape": list(features_np.shape),
        },
        "screen": {
            "baseline_macro_auc": float(p0_scores.loc["MACRO"]),
            "selected_configuration": selected_name,
            "selected_primary_seed_macro_auc": float(
                metrics_by_run[(selected_name, args.seed)]
                .set_index("target")
                .loc["MACRO", "auc"]
            ),
        },
        "confirmation": {
            "seeds": list(all_seeds),
            "seed_wins": seed_wins,
            "mean_seed_macro_delta": mean_seed_delta,
            "default_ensemble_macro_auc": float(
                default_ensemble_metrics.set_index("target").loc["MACRO", "auc"]
            ),
            "candidate_ensemble_macro_auc": float(
                candidate_ensemble_metrics.set_index("target").loc["MACRO", "auc"]
            ),
            "ensemble_macro_delta": ensemble_macro_delta,
            "nonnegative_targets": nonnegative_targets,
            "worst_target_delta": worst_target_delta,
            "paired_bootstrap": bootstrap,
            "criteria": criteria,
            "promoted": promoted,
        },
        "elapsed_seconds": time.time() - started,
        "source_sha256": source_hashes,
        "output_sha256_excluding_manifest": output_hashes,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": "P1-D",
                "selected_configuration": selected_name,
                "seed_macro_deltas": seed_summary["macro_delta"].tolist(),
                "mean_seed_macro_delta": mean_seed_delta,
                "ensemble_macro_delta": ensemble_macro_delta,
                "bootstrap_ci": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
                "bootstrap_probability_delta_gt_zero": bootstrap[
                    "probability_delta_gt_zero"
                ],
                "promotion_criteria": criteria,
                "promoted": promoted,
                "elapsed_seconds": manifest["elapsed_seconds"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
