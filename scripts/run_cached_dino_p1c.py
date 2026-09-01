#!/usr/bin/env python3
"""Run P1-C: soft-label asymmetric focal loss on purged cached DINO tokens."""

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
from sklearn.model_selection import KFold

from run_cached_dino_p0 import (
    ID,
    N_POS,
    N_SLOTS,
    TARGETS,
    TOKEN_DIM,
    TokenHead,
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
DEFAULT_OUTPUT = Path("outputs/experiments/P1-C-asymmetric-focal-loss")
GAMMA_NEG = 4.0
GAMMA_POS = 1.0
CLIP = 0.05
EPS = 1e-8


def weighted_soft_asymmetric_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_weights: torch.Tensor,
    gamma_neg: float = GAMMA_NEG,
    gamma_pos: float = GAMMA_POS,
    clip: float = CLIP,
    eps: float = EPS,
) -> torch.Tensor:
    """Official ASL formula extended continuously to soft labels, with weighted mean."""
    positive_probability = torch.sigmoid(logits)
    negative_probability = 1.0 - positive_probability
    if clip > 0:
        negative_probability = (negative_probability + clip).clamp(max=1.0)
    log_likelihood = labels * torch.log(positive_probability.clamp(min=eps))
    log_likelihood = log_likelihood + (1.0 - labels) * torch.log(
        negative_probability.clamp(min=eps)
    )
    probability_true = positive_probability * labels + negative_probability * (1.0 - labels)
    gamma = gamma_pos * labels + gamma_neg * (1.0 - labels)
    focal_weight = torch.pow((1.0 - probability_true).clamp(min=0.0), gamma)
    raw = -log_likelihood * focal_weight
    return (raw * label_weights).sum() / label_weights.sum().clamp_min(1.0)


def validate_loss_contract() -> dict[str, float | bool]:
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
    bce = weighted_bce(logits, labels, weights)
    asl_identity = weighted_soft_asymmetric_loss(
        logits, labels, weights, gamma_neg=0.0, gamma_pos=0.0, clip=0.0
    )
    identity_error = float((bce - asl_identity).abs().detach())
    if identity_error > 1e-7:
        raise AssertionError(f"ASL identity contract failed: {identity_error}")
    actual = weighted_soft_asymmetric_loss(logits, labels, weights)
    actual.backward()
    gradient_max = float(logits.grad.abs().max())
    if not torch.isfinite(actual) or not torch.isfinite(logits.grad).all() or gradient_max <= 0:
        raise AssertionError("ASL value or gradient is invalid")
    return {
        "gamma_zero_clip_zero_matches_weighted_bce": True,
        "identity_max_abs_error": identity_error,
        "official_default_loss_finite": True,
        "official_default_gradient_finite": True,
        "official_default_gradient_max_abs": gradient_max,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.folds < 2:
        raise ValueError("epochs and batch size must be positive; folds must be at least two")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; choose another directory or pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    loss_contract = validate_loss_contract()
    (args.output_dir / "loss_contract.json").write_text(json.dumps(loss_contract, indent=2) + "\n")

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
        "weight_decay": args.weight_decay,
    }
    for key, value in paired_config.items():
        if p0_manifest["config"][key] != value:
            raise ValueError(f"P0/P1-C config mismatch for {key}")

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
    probe = TokenHead()
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

    p0_frame = pd.read_csv(p0_prediction_path, dtype={ID: str})
    gold_labels = labels[gold_indices]
    if p0_frame[ID].tolist() != study_ids[gold_indices].tolist():
        raise ValueError("P0-B prediction study order differs")
    if p0_frame["fold"].to_numpy(int).tolist() != gold_fold.tolist():
        raise ValueError("P0-B fold assignment differs")
    if not np.array_equal(p0_frame[TARGETS].to_numpy(np.float32), gold_labels):
        raise ValueError("P0-B labels differ")
    p0_predictions = p0_frame[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)

    oof = np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32)
    fold_rows = []
    for fold, gold_train_pos, _, train_index, valid_index, excluded_pseudo in fold_definitions:
        fold_seed = args.seed + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        head = TokenHead()
        head.load_state_dict(initial_head_state, strict=True)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
                loss = weighted_soft_asymmetric_loss(
                    logits, label_tensor[index], weight_tensor[index]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            final_loss = float(np.mean(losses))
        head.eval()
        with torch.inference_mode():
            predictions = torch.sigmoid(
                head(feature_tensor[valid_index], mask_tensor[valid_index])
            ).numpy()
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"non-finite fold {fold} predictions")
        oof[valid_index] = predictions
        head_path = args.output_dir / f"p1-c_asl_fold_{fold}_head.pt"
        torch.save(head.state_dict(), head_path)
        elapsed = time.time() - fold_started
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "pseudo_rows": len(train_index) - len(gold_train_pos),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": len(excluded_pseudo),
                "excluded_pseudo_ids": ";".join(study_ids[excluded_pseudo]),
                "report_hash_overlap": 0,
                "final_loss": final_loss,
                "elapsed_seconds": elapsed,
                "head_file": head_path.name,
            }
        )
        print(
            f"P1-C ASL fold={fold} train={len(train_index)} valid={len(valid_index)} "
            f"loss={final_loss:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    predictions = oof[gold_indices]
    if not np.isfinite(predictions).all():
        raise RuntimeError("incomplete P1-C OOF predictions")
    metrics = calculate_metrics(gold_labels, predictions)
    metrics.to_csv(args.output_dir / "p1_c_asl_metrics.csv", index=False)
    save_predictions(
        args.output_dir / "p1_c_asl_oof_predictions.csv",
        study_ids[gold_indices],
        gold_labels,
        predictions,
        gold_fold,
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_summary.csv", index=False)

    p0_metrics = calculate_metrics(gold_labels, p0_predictions)
    comparison = p0_metrics.merge(metrics, on="target", suffixes=("__p0_b", "__p1_c"))
    comparison["delta__p1_c_minus_p0_b"] = comparison["auc__p1_c"] - comparison["auc__p0_b"]
    comparison.to_csv(args.output_dir / "target_comparison.csv", index=False)
    bootstrap = paired_bootstrap(
        gold_labels,
        p0_predictions,
        predictions,
        args.bootstrap_repeats,
        args.seed + 400_000,
    )
    (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")

    p0_macro = float(p0_metrics.loc[p0_metrics["target"] == "MACRO", "auc"].iloc[0])
    candidate_macro = float(metrics.loc[metrics["target"] == "MACRO", "auc"].iloc[0])
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "P1-C",
        "description": "official-default asymmetric loss extended to soft pseudo labels",
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
            "bootstrap_repeats": args.bootstrap_repeats,
            "objective": "soft-label asymmetric loss",
            "gamma_neg": GAMMA_NEG,
            "gamma_pos": GAMMA_POS,
            "clip": CLIP,
            "eps": EPS,
            "focal_weight_gradient_detached": False,
            "label_weights": "current three-source agreement weights",
            "report_group_purge": True,
        },
        "loss_contract": loss_contract,
        "data": {
            "studies": len(study_ids),
            "expert_studies": len(gold_indices),
            "pseudo_studies": len(pseudo_indices),
            "excluded_pseudo_studies_total_across_folds": int(
                sum(len(item[-1]) for item in fold_definitions)
            ),
            "token_shape": list(features_np.shape),
        },
        "baseline": {"id": "P0-B weighted BCE", "macro_auc": p0_macro},
        "result": {
            "macro_auc": candidate_macro,
            "macro_delta_vs_p0_b": candidate_macro - p0_macro,
        },
        "paired_bootstrap": bootstrap,
        "elapsed_seconds": time.time() - started,
        "source_sha256": source_hashes,
        "output_sha256_excluding_manifest": output_hashes,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": "P1-C",
                "p0_b_macro_auc": p0_macro,
                "p1_c_asl_macro_auc": candidate_macro,
                "delta": candidate_macro - p0_macro,
                "bootstrap_ci": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
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
