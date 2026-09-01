#!/usr/bin/env python3
"""Run the paired P0 report-group-purged DINO head experiment on cached tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
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
DEFAULT_CACHE = Path("outputs/kaggle/rsna-dinov2-pseudo-adaptation")
DEFAULT_OUTPUT = Path("outputs/experiments/P0-report-group-purged-dino-oof")


class TokenHead(nn.Module):
    """Architecture used by the completed Kaggle DINO adaptation notebook."""

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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output directory. Existing files may be replaced.",
    )
    return parser.parse_args()


def normalize_report(report: object) -> str:
    """Normalize formatting variants while preserving multilingual letters and digits."""
    text = unicodedata.normalize("NFKC", str(report)).casefold()
    return "".join(character for character in text if character.isalnum())


def report_hash(report: object) -> str:
    normalized = normalize_report(report)
    if not normalized:
        raise ValueError("empty report after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * label_weights).sum() / label_weights.sum().clamp_min(1.0)


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    rows = [
        {
            "target": target,
            "auc": float(roc_auc_score(labels[:, target_index], predictions[:, target_index])),
        }
        for target_index, target in enumerate(TARGETS)
    ]
    rows.append({"target": "MACRO", "auc": float(np.mean([row["auc"] for row in rows]))})
    return pd.DataFrame(rows)


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    purged: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    maximum_attempts = max(repeats * 20, repeats)
    while len(deltas) < repeats and attempts < maximum_attempts:
        attempts += 1
        sampled = rng.integers(0, len(labels), size=len(labels))
        target_deltas = []
        for target_index in range(len(TARGETS)):
            sampled_labels = labels[sampled, target_index]
            if np.unique(sampled_labels).size != 2:
                break
            target_deltas.append(
                roc_auc_score(sampled_labels, purged[sampled, target_index])
                - roc_auc_score(sampled_labels, baseline[sampled, target_index])
            )
        if len(target_deltas) == len(TARGETS):
            deltas.append(float(np.mean(target_deltas)))
    if not deltas:
        raise RuntimeError("no valid paired study-bootstrap samples")
    values = np.asarray(deltas)
    return {
        "requested_repeats": repeats,
        "valid_repeats": len(deltas),
        "attempts": attempts,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci_2_5": float(np.quantile(values, 0.025)),
        "ci_97_5": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float((values > 0).mean()),
    }


def save_predictions(
    path: Path,
    study_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    folds: np.ndarray,
) -> None:
    frame = pd.DataFrame({ID: study_ids, "fold": folds})
    for target_index, target in enumerate(TARGETS):
        frame[target] = labels[:, target_index]
        frame[f"pred__{target}"] = predictions[:, target_index]
    frame.to_csv(path, index=False)


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
    device = torch.device("cpu")
    started = time.time()

    studies = pd.read_csv(args.cache_dir / "dino_feature_studies.csv", dtype={ID: str})
    targets = pd.read_csv(args.cache_dir / "training_targets.csv", dtype={ID: str})
    metadata = pd.read_csv(args.train_csv, dtype={ID: str})
    for name, frame in (("studies", studies), ("targets", targets), ("metadata", metadata)):
        if frame[ID].duplicated().any():
            raise ValueError(f"duplicate {ID} in {name}")
    study_ids = studies[ID].to_numpy(str)
    targets = targets.set_index(ID).loc[study_ids].reset_index()
    metadata = metadata.set_index(ID).loc[study_ids].reset_index()
    if targets[ID].tolist() != metadata[ID].tolist():
        raise AssertionError("target, metadata, and feature study ordering differ")

    report_hashes = metadata["Report"].map(report_hash).to_numpy(str)
    report_groups = pd.DataFrame({ID: study_ids, "report_hash": report_hashes})
    report_groups["is_gold"] = targets["is_gold"].astype(bool).to_numpy()
    group_audit = (
        report_groups.groupby("report_hash", sort=True)
        .agg(
            studies=(ID, "size"),
            gold_studies=("is_gold", "sum"),
            study_ids=(ID, lambda values: ";".join(sorted(values))),
        )
        .reset_index()
    )
    group_audit["pseudo_studies"] = group_audit["studies"] - group_audit["gold_studies"]
    group_audit = group_audit[
        ["report_hash", "studies", "gold_studies", "pseudo_studies", "study_ids"]
    ]
    duplicate_groups = group_audit[group_audit["studies"] > 1].copy()
    cross_label_groups = duplicate_groups[
        (duplicate_groups["gold_studies"] > 0) & (duplicate_groups["pseudo_studies"] > 0)
    ]
    expert_duplicate_groups = duplicate_groups[duplicate_groups["gold_studies"] > 1]
    duplicate_groups.to_csv(args.output_dir / "report_group_audit.csv", index=False)
    if not expert_duplicate_groups.empty:
        raise RuntimeError(
            "expert-expert duplicate report groups require grouped fold assignment before P0"
        )

    feature_path = args.cache_dir / "dino_token_features.npy"
    mask_path = args.cache_dir / "dino_window_mask.npy"
    model_path = args.cache_dir / "dino_pseudo_adapted_model.pt"
    features_np = np.load(feature_path, mmap_mode="r")
    mask_np = np.load(mask_path, mmap_mode="r")
    if features_np.shape != (len(study_ids), N_SLOTS, N_POS, TOKEN_DIM):
        raise ValueError(f"unexpected token shape: {features_np.shape}")
    if mask_np.shape != (len(study_ids), N_SLOTS, N_POS):
        raise ValueError(f"unexpected mask shape: {mask_np.shape}")
    if not np.isfinite(features_np).all() or not mask_np.astype(bool).all():
        raise ValueError("token cache contains non-finite or missing windows")

    y = targets[TARGETS].to_numpy(np.float32)
    weights = targets[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid labels or label weights")
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    if len(gold_indices) != 58 or len(pseudo_indices) != 4_349:
        raise ValueError("expected 58 expert and 4,349 pseudo studies")

    print("loading 4407 x 18 cached tokens into CPU float32 memory", flush=True)
    feature_tensor = torch.from_numpy(np.asarray(features_np).astype(np.float32))
    mask_tensor = torch.from_numpy(np.asarray(mask_np).astype(bool))
    y_tensor = torch.from_numpy(y)
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
    fold_definitions: list[dict[str, object]] = []
    gold_fold = np.full(len(gold_indices), -1, dtype=int)
    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[gold_valid_pos]
        valid_hashes = set(report_hashes[valid_index])
        excluded_pseudo = np.asarray(
            [index for index in pseudo_indices if report_hashes[index] in valid_hashes], dtype=int
        )
        gold_fold[gold_valid_pos] = fold
        fold_definitions.append(
            {
                "fold": fold,
                "gold_train_pos": gold_train_pos,
                "gold_valid_pos": gold_valid_pos,
                "valid_index": valid_index,
                "excluded_pseudo": excluded_pseudo,
            }
        )
    if (gold_fold < 0).any():
        raise AssertionError("not every gold study received a fold")

    def train_head(
        arm: str,
        fold: int,
        train_index: np.ndarray,
        valid_index: np.ndarray,
    ) -> tuple[np.ndarray, float, float, Path]:
        fold_seed = args.seed + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        head = TokenHead().to(device)
        head.load_state_dict(initial_head_state, strict=True)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        generator = torch.Generator().manual_seed(fold_seed)
        train_cpu = torch.as_tensor(train_index, dtype=torch.long)
        fold_started = time.time()
        final_loss = float("nan")
        for epoch in range(args.epochs):
            head.train()
            permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
            losses = []
            for start in range(0, len(permutation), args.batch_size):
                index = permutation[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = head(feature_tensor[index], mask_tensor[index])
                loss = weighted_bce(logits, y_tensor[index], weight_tensor[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            final_loss = float(np.mean(losses))
        head.eval()
        with torch.inference_mode():
            prediction = torch.sigmoid(head(feature_tensor[valid_index], mask_tensor[valid_index]))
            prediction_np = prediction.numpy().astype(np.float32)
        if not np.isfinite(prediction_np).all():
            raise RuntimeError(f"non-finite predictions for {arm} fold {fold}")
        head_path = args.output_dir / f"{arm.lower()}_fold_{fold}_head.pt"
        torch.save(head.state_dict(), head_path)
        elapsed = time.time() - fold_started
        print(
            f"{arm} fold={fold} train={len(train_index)} valid={len(valid_index)} "
            f"loss={final_loss:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return prediction_np, final_loss, elapsed, head_path

    arm_predictions = {
        "P0-A": np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32),
        "P0-B": np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32),
    }
    fold_rows: list[dict[str, object]] = []
    head_paths: dict[tuple[str, int], Path] = {}

    for definition in fold_definitions:
        fold = int(definition["fold"])
        gold_train_pos = np.asarray(definition["gold_train_pos"])
        valid_index = np.asarray(definition["valid_index"])
        excluded_pseudo = np.asarray(definition["excluded_pseudo"])
        baseline_train = np.concatenate([pseudo_indices, gold_indices[gold_train_pos]])
        baseline_prediction, baseline_loss, baseline_seconds, baseline_head_path = train_head(
            "P0-A", fold, baseline_train, valid_index
        )
        arm_predictions["P0-A"][valid_index] = baseline_prediction
        head_paths[("P0-A", fold)] = baseline_head_path
        fold_rows.append(
            {
                "arm": "P0-A",
                "fold": fold,
                "train_rows": len(baseline_train),
                "pseudo_rows": len(pseudo_indices),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": 0,
                "excluded_pseudo_ids": "",
                "report_hash_overlap": len(set(report_hashes[baseline_train]) & set(report_hashes[valid_index])),
                "reused_from_p0_a": False,
                "final_loss": baseline_loss,
                "elapsed_seconds": baseline_seconds,
                "head_file": baseline_head_path.name,
            }
        )

        if len(excluded_pseudo):
            keep_pseudo = np.setdiff1d(pseudo_indices, excluded_pseudo, assume_unique=True)
            purged_train = np.concatenate([keep_pseudo, gold_indices[gold_train_pos]])
            purged_prediction, purged_loss, purged_seconds, purged_head_path = train_head(
                "P0-B", fold, purged_train, valid_index
            )
            reused = False
            head_paths[("P0-B", fold)] = purged_head_path
        else:
            purged_train = baseline_train
            purged_prediction = baseline_prediction.copy()
            purged_loss = baseline_loss
            purged_seconds = 0.0
            reused = True
            purged_head_path = baseline_head_path
            head_paths[("P0-B", fold)] = baseline_head_path
            print(f"P0-B fold={fold} has no cross-boundary twins; reusing paired P0-A head", flush=True)
        overlap = set(report_hashes[purged_train]) & set(report_hashes[valid_index])
        if overlap:
            raise RuntimeError(f"P0-B fold {fold} retains {len(overlap)} report-hash overlaps")
        arm_predictions["P0-B"][valid_index] = purged_prediction
        fold_rows.append(
            {
                "arm": "P0-B",
                "fold": fold,
                "train_rows": len(purged_train),
                "pseudo_rows": len(purged_train) - len(gold_train_pos),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": len(excluded_pseudo),
                "excluded_pseudo_ids": ";".join(study_ids[excluded_pseudo]),
                "report_hash_overlap": 0,
                "reused_from_p0_a": reused,
                "final_loss": purged_loss,
                "elapsed_seconds": purged_seconds,
                "head_file": purged_head_path.name,
            }
        )

    gold_labels = y[gold_indices]
    gold_ids = study_ids[gold_indices]
    results: dict[str, dict[str, object]] = {}
    metrics_by_arm: dict[str, pd.DataFrame] = {}
    for arm, predictions in arm_predictions.items():
        gold_predictions = predictions[gold_indices]
        if not np.isfinite(gold_predictions).all():
            raise RuntimeError(f"incomplete OOF predictions for {arm}")
        slug = arm.lower().replace("-", "_")
        metrics = calculate_metrics(gold_labels, gold_predictions)
        metrics.to_csv(args.output_dir / f"{slug}_metrics.csv", index=False)
        save_predictions(
            args.output_dir / f"{slug}_oof_predictions.csv",
            gold_ids,
            gold_labels,
            gold_predictions,
            gold_fold,
        )
        metrics_by_arm[arm] = metrics
        results[arm] = {
            "macro_auc": float(metrics.loc[metrics["target"] == "MACRO", "auc"].iloc[0]),
            "predictions_file": f"{slug}_oof_predictions.csv",
            "metrics_file": f"{slug}_metrics.csv",
        }

    target_comparison = metrics_by_arm["P0-A"].merge(
        metrics_by_arm["P0-B"], on="target", suffixes=("__p0_a", "__p0_b")
    )
    target_comparison["delta__p0_b_minus_p0_a"] = (
        target_comparison["auc__p0_b"] - target_comparison["auc__p0_a"]
    )
    target_comparison.to_csv(args.output_dir / "target_comparison.csv", index=False)
    bootstrap = paired_bootstrap(
        gold_labels,
        arm_predictions["P0-A"][gold_indices],
        arm_predictions["P0-B"][gold_indices],
        args.bootstrap_repeats,
        args.seed + 100_000,
    )
    (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_summary.csv", index=False)

    source_hashes = {
        "dino_token_features.npy": sha256(feature_path),
        "dino_window_mask.npy": sha256(mask_path),
        "training_targets.csv": sha256(args.cache_dir / "training_targets.csv"),
        "dino_pseudo_adapted_model.pt": sha256(model_path),
        "train.csv": sha256(args.train_csv),
    }
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    elapsed = time.time() - started
    manifest = {
        "schema_version": 1,
        "experiment_id": "P0",
        "description": "paired cached-DINO head replay versus fold-wise report-group purge",
        "created_unix": time.time(),
        "git": git_state(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "threads": args.threads,
        },
        "config": {
            "seed": args.seed,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "bootstrap_repeats": args.bootstrap_repeats,
            "report_normalization": "NFKC casefold then retain Unicode alphanumeric characters",
        },
        "data": {
            "studies": len(study_ids),
            "expert_studies": len(gold_indices),
            "pseudo_studies": len(pseudo_indices),
            "duplicate_report_groups": len(duplicate_groups),
            "studies_in_duplicate_groups": int(duplicate_groups["studies"].sum()),
            "cross_expert_pseudo_groups": len(cross_label_groups),
            "expert_expert_duplicate_groups": len(expert_duplicate_groups),
            "excluded_pseudo_studies_total_across_folds": int(
                sum(len(np.asarray(item["excluded_pseudo"])) for item in fold_definitions)
            ),
            "token_shape": list(features_np.shape),
        },
        "results": results,
        "macro_delta__p0_b_minus_p0_a": (
            float(results["P0-B"]["macro_auc"]) - float(results["P0-A"]["macro_auc"])
        ),
        "paired_bootstrap": bootstrap,
        "elapsed_seconds": elapsed,
        "source_sha256": source_hashes,
        "output_sha256_excluding_manifest": output_hashes,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": "P0",
                "p0_a_macro_auc": results["P0-A"]["macro_auc"],
                "p0_b_macro_auc": results["P0-B"]["macro_auc"],
                "delta": manifest["macro_delta__p0_b_minus_p0_a"],
                "bootstrap_ci": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
                "elapsed_seconds": elapsed,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
