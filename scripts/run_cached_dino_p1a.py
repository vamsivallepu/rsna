#!/usr/bin/env python3
"""Run P1-A: raw not-addressed confidence weights on purged cached DINO tokens."""

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
DEFAULT_RAW_LABELS = Path("data/sample/public_weak_labels/llm_labels_full.csv")
DEFAULT_OUTPUT = Path("outputs/experiments/P1-A-raw-not-addressed-weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--raw-labels", type=Path, default=DEFAULT_RAW_LABELS)
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

    p0_manifest_path = args.p0_dir / "manifest.json"
    p0_manifest = json.loads(p0_manifest_path.read_text())
    if p0_manifest["experiment_id"] != "P0":
        raise ValueError("the baseline directory does not contain a P0 manifest")
    expected_config = {
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    for key, value in expected_config.items():
        if p0_manifest["config"][key] != value:
            raise ValueError(f"P0/P1-A config mismatch for {key}")

    feature_path = args.cache_dir / "dino_token_features.npy"
    mask_path = args.cache_dir / "dino_window_mask.npy"
    target_path = args.cache_dir / "training_targets.csv"
    model_path = args.cache_dir / "dino_pseudo_adapted_model.pt"
    source_paths = {
        "dino_token_features.npy": feature_path,
        "dino_window_mask.npy": mask_path,
        "training_targets.csv": target_path,
        "dino_pseudo_adapted_model.pt": model_path,
        "train.csv": args.train_csv,
        "llm_labels_full.csv": args.raw_labels,
        "P0_manifest.json": p0_manifest_path,
    }
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    for name, expected in p0_manifest["source_sha256"].items():
        if source_hashes[name] != expected:
            raise ValueError(f"P0 source hash drift: {name}")

    studies = pd.read_csv(args.cache_dir / "dino_feature_studies.csv", dtype={ID: str})
    targets = pd.read_csv(target_path, dtype={ID: str})
    metadata = pd.read_csv(args.train_csv, dtype={ID: str})
    raw_labels = pd.read_csv(args.raw_labels, dtype={ID: str})
    for name, frame in (
        ("studies", studies),
        ("targets", targets),
        ("metadata", metadata),
        ("raw labels", raw_labels),
    ):
        if frame[ID].duplicated().any():
            raise ValueError(f"duplicate {ID} in {name}")
    study_ids = studies[ID].to_numpy(str)
    targets = targets.set_index(ID).loc[study_ids].reset_index()
    metadata = metadata.set_index(ID).loc[study_ids].reset_index()
    raw_labels = raw_labels.set_index(ID).loc[study_ids].reset_index()

    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    if len(study_ids) != 4_407 or len(gold_indices) != 58 or len(pseudo_indices) != 4_349:
        raise ValueError("expected 4,407 studies, 58 expert rows, and 4,349 pseudo rows")
    labels = targets[TARGETS].to_numpy(np.float32)
    current_weights = targets[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
    raw_values = raw_labels[TARGETS].to_numpy(np.float32)
    if not np.isfinite(raw_values).all() or raw_values.min() < 0 or raw_values.max() > 1:
        raise ValueError("raw label probabilities must be finite and inside [0, 1]")
    raw_confidence = 2.0 * np.abs(raw_values - 0.5)
    experiment_weights = raw_confidence.copy()
    experiment_weights[gold_indices] = 2.0
    if not np.isfinite(experiment_weights).all() or experiment_weights.min() < 0:
        raise ValueError("invalid P1-A weights")

    weight_rows = []
    for target_index, target in enumerate(TARGETS):
        pseudo_raw = raw_values[pseudo_indices, target_index]
        pseudo_confidence = raw_confidence[pseudo_indices, target_index]
        nonzero = pseudo_confidence[pseudo_confidence > 0]
        weight_rows.append(
            {
                "target": target,
                "not_addressed_fraction": float(np.isclose(pseudo_raw, 0.5).mean()),
                "raw_confidence_mean": float(pseudo_confidence.mean()),
                "raw_confidence_nonzero_mean": float(nonzero.mean()),
                "current_agreement_weight_mean": float(
                    current_weights[pseudo_indices, target_index].mean()
                ),
            }
        )
    weight_audit = pd.DataFrame(weight_rows)
    weight_audit.to_csv(args.output_dir / "weight_audit.csv", index=False)

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
    experiment_weight_tensor = torch.from_numpy(experiment_weights)

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
        overlap = set(report_hashes[train_index]) & valid_hashes
        if overlap:
            raise RuntimeError(f"fold {fold} retains {len(overlap)} report-hash overlaps")
        gold_fold[gold_valid_pos] = fold
        fold_definitions.append(
            (fold, gold_train_pos, gold_valid_pos, train_index, valid_index, excluded_pseudo)
        )
    if (gold_fold < 0).any():
        raise AssertionError("incomplete gold fold assignment")

    p0_frame = pd.read_csv(args.p0_dir / "p0_b_oof_predictions.csv", dtype={ID: str})
    if p0_frame[ID].tolist() != study_ids[gold_indices].tolist():
        raise ValueError("P0-B prediction study order differs")
    if p0_frame["fold"].to_numpy(int).tolist() != gold_fold.tolist():
        raise ValueError("P0-B fold assignment differs")
    gold_labels = labels[gold_indices]
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
                loss = weighted_bce(
                    logits, label_tensor[index], experiment_weight_tensor[index]
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
        head_path = args.output_dir / f"p1-a_fold_{fold}_head.pt"
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
                "zero_weight_cell_fraction": float(
                    (experiment_weights[train_index] == 0).mean()
                ),
                "final_loss": final_loss,
                "elapsed_seconds": elapsed,
                "head_file": head_path.name,
            }
        )
        print(
            f"P1-A fold={fold} train={len(train_index)} valid={len(valid_index)} "
            f"loss={final_loss:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    p1_predictions = oof[gold_indices]
    if not np.isfinite(p1_predictions).all():
        raise RuntimeError("incomplete P1-A OOF predictions")
    p1_metrics = calculate_metrics(gold_labels, p1_predictions)
    p1_metrics.to_csv(args.output_dir / "p1_a_metrics.csv", index=False)
    save_predictions(
        args.output_dir / "p1_a_oof_predictions.csv",
        study_ids[gold_indices],
        gold_labels,
        p1_predictions,
        gold_fold,
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_summary.csv", index=False)

    p0_metrics = calculate_metrics(gold_labels, p0_predictions)
    comparison = p0_metrics.merge(p1_metrics, on="target", suffixes=("__p0_b", "__p1_a"))
    comparison["delta__p1_a_minus_p0_b"] = comparison["auc__p1_a"] - comparison["auc__p0_b"]
    comparison.to_csv(args.output_dir / "target_comparison.csv", index=False)
    bootstrap = paired_bootstrap(
        gold_labels,
        p0_predictions,
        p1_predictions,
        args.bootstrap_repeats,
        args.seed + 200_000,
    )
    (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")

    p0_macro = float(p0_metrics.loc[p0_metrics["target"] == "MACRO", "auc"].iloc[0])
    p1_macro = float(p1_metrics.loc[p1_metrics["target"] == "MACRO", "auc"].iloc[0])
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "P1-A",
        "description": "fixed consensus targets with raw Steven not-addressed confidence weights",
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
            **expected_config,
            "bootstrap_repeats": args.bootstrap_repeats,
            "weight_formula": "gold=2.0; pseudo=2*abs(raw_steven_v1_probability-0.5)",
            "targets_changed": False,
            "report_group_purge": True,
        },
        "data": {
            "studies": len(study_ids),
            "expert_studies": len(gold_indices),
            "pseudo_studies": len(pseudo_indices),
            "zero_weight_pseudo_cell_fraction": float(
                (experiment_weights[pseudo_indices] == 0).mean()
            ),
            "excluded_pseudo_studies_total_across_folds": int(
                sum(len(item[-1]) for item in fold_definitions)
            ),
            "token_shape": list(features_np.shape),
        },
        "results": {
            "P0-B": {"macro_auc": p0_macro, "role": "paired purged baseline"},
            "P1-A": {"macro_auc": p1_macro, "role": "raw confidence weights"},
        },
        "macro_delta__p1_a_minus_p0_b": p1_macro - p0_macro,
        "paired_bootstrap": bootstrap,
        "elapsed_seconds": time.time() - started,
        "source_sha256": source_hashes,
        "output_sha256_excluding_manifest": output_hashes,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": "P1-A",
                "p0_b_macro_auc": p0_macro,
                "p1_a_macro_auc": p1_macro,
                "delta": p1_macro - p0_macro,
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
