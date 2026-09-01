#!/usr/bin/env python3
"""Run P1-B pooling/head controls on the purged cached DINO token representation."""

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
DEFAULT_OUTPUT = Path("outputs/experiments/P1-B-head-pooling-controls")
VARIANTS = ("shared_attention", "mean_pool", "max_pool")


class HeadCore(nn.Module):
    """Parameters shared with the adapted target-query head."""

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
        batch = features.shape[0]
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[:, None, :] + self.position_embedding[None, :, :]
        return hidden.reshape(batch, N_SLOTS * N_POS, HIDDEN)

    def output(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bh,th->bt", pooled, self.target_weight) + self.target_bias


class SharedAttentionHead(HeadCore):
    """One diagnosis-agnostic attention distribution with target-specific outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_query = nn.Parameter(torch.empty(HIDDEN))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        flat_mask = mask.reshape(features.shape[0], N_SLOTS * N_POS).bool()
        scores = torch.einsum("bkh,h->bk", hidden, self.shared_query) / (HIDDEN**0.5)
        attention = scores.masked_fill(~flat_mask, -1e4).softmax(dim=-1)
        pooled = torch.einsum("bk,bkh->bh", attention, hidden)
        return self.output(pooled)


class PoolingHead(HeadCore):
    """Masked global mean or max pooling with target-specific outputs."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"mean", "max"}:
            raise ValueError(mode)
        self.mode = mode

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        flat_mask = mask.reshape(features.shape[0], N_SLOTS * N_POS).bool()
        if self.mode == "mean":
            numerator = (hidden * flat_mask[:, :, None]).sum(dim=1)
            pooled = numerator / flat_mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            pooled = hidden.masked_fill(~flat_mask[:, :, None], -1e4).amax(dim=1)
        return self.output(pooled)


def build_head(variant: str, adapted_state: dict[str, torch.Tensor]) -> nn.Module:
    if variant == "shared_attention":
        head: nn.Module = SharedAttentionHead()
    elif variant == "mean_pool":
        head = PoolingHead("mean")
    elif variant == "max_pool":
        head = PoolingHead("max")
    else:
        raise ValueError(f"unknown variant: {variant}")
    compatible = {
        key: value.detach().cpu().clone()
        for key, value in adapted_state.items()
        if key in head.state_dict() and head.state_dict()[key].shape == value.shape
    }
    missing, unexpected = head.load_state_dict(compatible, strict=False)
    expected_missing = ["shared_query"] if variant == "shared_attention" else []
    if sorted(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"unexpected initialization mismatch for {variant}: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if variant == "shared_attention":
        with torch.no_grad():
            head.shared_query.copy_(adapted_state["query"].mean(dim=0))
    return head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
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
    variants = list(dict.fromkeys(args.variants))
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
            raise ValueError(f"P0/P1-B config mismatch for {key}")

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
    adapted_head_state = {
        key.removeprefix("head."): value.detach().cpu().clone()
        for key, value in full_state.items()
        if key.startswith("head.")
    }
    for variant in variants:
        probe = build_head(variant, adapted_head_state)
        with torch.no_grad():
            probe_output = probe(feature_tensor[:2], mask_tensor[:2])
        if probe_output.shape != (2, len(TARGETS)) or not torch.isfinite(probe_output).all():
            raise RuntimeError(f"invalid {variant} initialization")
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
    p0_metrics = calculate_metrics(gold_labels, p0_predictions)
    p0_macro = float(p0_metrics.loc[p0_metrics["target"] == "MACRO", "auc"].iloc[0])

    fold_rows = []
    head_rows = []
    comparison_rows = []
    summary_rows = []
    bootstraps = {}
    variant_results = {}

    for variant in variants:
        oof = np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32)
        sample_head = build_head(variant, adapted_head_state)
        parameter_count = sum(parameter.numel() for parameter in sample_head.parameters())
        head_rows.append(
            {
                "variant": variant,
                "parameter_count": parameter_count,
                "adapted_common_parameters_loaded": True,
                "shared_query_initialized_from_target_query_mean": variant == "shared_attention",
            }
        )
        del sample_head
        for fold, gold_train_pos, _, train_index, valid_index, excluded_pseudo in fold_definitions:
            fold_seed = args.seed + fold
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            head = build_head(variant, adapted_head_state)
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
                    loss = weighted_bce(logits, label_tensor[index], weight_tensor[index])
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
                raise RuntimeError(f"non-finite {variant} fold {fold} predictions")
            oof[valid_index] = predictions
            head_path = args.output_dir / f"p1-b_{variant}_fold_{fold}_head.pt"
            torch.save(head.state_dict(), head_path)
            elapsed = time.time() - fold_started
            fold_rows.append(
                {
                    "variant": variant,
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
                f"P1-B {variant} fold={fold} train={len(train_index)} "
                f"valid={len(valid_index)} loss={final_loss:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        predictions = oof[gold_indices]
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"incomplete {variant} OOF predictions")
        metrics = calculate_metrics(gold_labels, predictions)
        metrics_path = args.output_dir / f"p1_b_{variant}_metrics.csv"
        prediction_path = args.output_dir / f"p1_b_{variant}_oof_predictions.csv"
        metrics.to_csv(metrics_path, index=False)
        save_predictions(
            prediction_path,
            study_ids[gold_indices],
            gold_labels,
            predictions,
            gold_fold,
        )
        merged = p0_metrics.merge(metrics, on="target", suffixes=("__p0_b", "__candidate"))
        for row in merged.itertuples(index=False):
            comparison_rows.append(
                {
                    "variant": variant,
                    "target": row.target,
                    "auc__p0_b": row.auc__p0_b,
                    "auc__candidate": row.auc__candidate,
                    "delta__candidate_minus_p0_b": row.auc__candidate - row.auc__p0_b,
                }
            )
        candidate_macro = float(metrics.loc[metrics["target"] == "MACRO", "auc"].iloc[0])
        bootstrap = paired_bootstrap(
            gold_labels,
            p0_predictions,
            predictions,
            args.bootstrap_repeats,
            args.seed + 300_000 + variants.index(variant) * 10_000,
        )
        bootstraps[variant] = bootstrap
        summary_rows.append(
            {
                "variant": variant,
                "p0_b_macro_auc": p0_macro,
                "candidate_macro_auc": candidate_macro,
                "delta": candidate_macro - p0_macro,
                "ci_2_5": bootstrap["ci_2_5"],
                "ci_97_5": bootstrap["ci_97_5"],
                "probability_delta_gt_zero": bootstrap["probability_delta_gt_zero"],
                "parameter_count": parameter_count,
            }
        )
        variant_results[variant] = {
            "macro_auc": candidate_macro,
            "macro_delta_vs_p0_b": candidate_macro - p0_macro,
            "metrics_file": metrics_path.name,
            "predictions_file": prediction_path.name,
        }

    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_summary.csv", index=False)
    pd.DataFrame(head_rows).to_csv(args.output_dir / "head_audit.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "target_comparison.csv", index=False)
    variant_summary = pd.DataFrame(summary_rows).sort_values("candidate_macro_auc", ascending=False)
    variant_summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    (args.output_dir / "paired_bootstrap.json").write_text(
        json.dumps(bootstraps, indent=2) + "\n"
    )

    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "P1-B",
        "description": "shared-attention, masked-mean, and masked-max pooling controls",
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
            "variants": variants,
            "label_weights": "current three-source agreement weights",
            "report_group_purge": True,
            "initialization": (
                "adapted projection, slot/position embeddings, target weights and bias; "
                "shared query is mean of 12 adapted target queries"
            ),
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
        "baseline": {"id": "P0-B", "macro_auc": p0_macro},
        "results": variant_results,
        "winner_by_observed_macro": str(variant_summary.iloc[0]["variant"]),
        "paired_bootstrap": bootstraps,
        "elapsed_seconds": time.time() - started,
        "source_sha256": source_hashes,
        "output_sha256_excluding_manifest": output_hashes,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": "P1-B",
                "p0_b_macro_auc": p0_macro,
                "variants": variant_results,
                "winner_by_observed_macro": manifest["winner_by_observed_macro"],
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
