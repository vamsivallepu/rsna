#!/usr/bin/env python3
"""Run cheap, study-level OOF ablations on the downloaded RadImageNet cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


SEED = 20260831
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
DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1_000.0)
DEFAULT_MODES = ("mean_raw", "mean_l2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    return parser.parse_args()


def build_features(raw: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    mask_float = mask.astype(np.float32)
    count = mask_float.sum(axis=1, keepdims=True).clip(min=1.0)
    slot_norm = raw / np.linalg.norm(raw, axis=2, keepdims=True).clip(min=1e-6)
    slot_norm *= mask_float[:, :, None]

    if mode == "mean_raw":
        result = (raw * mask_float[:, :, None]).sum(axis=1) / count
    elif mode == "mean_l2":
        result = slot_norm.sum(axis=1) / count
    elif mode == "mean_std_l2":
        mean = slot_norm.sum(axis=1) / count
        variance = (((slot_norm - mean[:, None, :]) ** 2) * mask_float[:, :, None]).sum(axis=1) / count
        result = np.concatenate([mean, np.sqrt(variance.clip(min=0.0))], axis=1)
    elif mode == "flatten_l2":
        result = np.concatenate([slot_norm.reshape(len(raw), -1), mask_float], axis=1)
    else:
        raise ValueError(mode)

    return np.asarray(result, dtype=np.float32)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.cache_dir

    raw = np.load(root / "study_features.npy", mmap_mode="r")
    mask = np.load(root / "study_slot_mask.npy", mmap_mode="r").astype(bool)
    studies = pd.read_csv(root / "feature_studies.csv")
    targets = pd.read_csv(root / "training_targets.csv").set_index("StudyInstanceUID").loc[
        studies["StudyInstanceUID"]
    ]
    y = targets[TARGETS].to_numpy(np.float32)
    weight_columns = [f"weight__{target}" for target in TARGETS]
    row_weight = targets[weight_columns].mean(axis=1).to_numpy(np.float32)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    assert len(raw) == len(mask) == len(targets) == 4_407
    assert len(gold_indices) == 58
    assert mask.any(axis=1).all()

    fold_indices = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        train_index = np.concatenate([pseudo_indices, gold_indices[gold_train_pos]])
        valid_index = gold_indices[gold_valid_pos]
        fold_indices.append((fold, train_index, valid_index))

    summary_rows: list[dict[str, float | str]] = []
    target_rows: list[dict[str, float | str]] = []
    oof_outputs: dict[str, np.ndarray] = {}

    for mode in args.modes:
        features = build_features(raw, mask, mode)
        print(f"mode={mode} shape={features.shape}", flush=True)
        mode_oof = {
            alpha: np.full((len(features), len(TARGETS)), np.nan, dtype=np.float32)
            for alpha in args.alphas
        }

        for fold, train_index, valid_index in fold_indices:
            scaler = StandardScaler(copy=False)
            train_x = scaler.fit_transform(features[train_index].copy())
            valid_x = scaler.transform(features[valid_index].copy())
            for alpha in args.alphas:
                model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=1_000)
                model.fit(train_x, y[train_index], sample_weight=row_weight[train_index])
                mode_oof[alpha][valid_index] = model.predict(valid_x)
            print(f"  fold={fold} complete", flush=True)

        for alpha, predictions in mode_oof.items():
            assert np.isfinite(predictions[gold_indices]).all()
            aucs = {
                target: float(roc_auc_score(y[gold_indices, index], predictions[gold_indices, index]))
                for index, target in enumerate(TARGETS)
            }
            macro = float(np.mean(list(aucs.values())))
            candidate = f"ridge__{mode}__alpha_{alpha:g}"
            summary_rows.append({"candidate": candidate, "macro_auc": macro})
            target_rows.extend(
                {"candidate": candidate, "target": target, "auc": auc}
                for target, auc in aucs.items()
            )
            oof_outputs[candidate] = predictions[gold_indices]
            print(f"  {candidate}: macro_auc={macro:.6f}", flush=True)

        del features

    summary = pd.DataFrame(summary_rows).sort_values("macro_auc", ascending=False)
    per_target = pd.DataFrame(target_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    per_target.to_csv(args.output_dir / "target_auc.csv", index=False)

    oof_frame = studies.loc[gold_indices, ["StudyInstanceUID"]].reset_index(drop=True)
    for candidate, predictions in oof_outputs.items():
        for target_index, target in enumerate(TARGETS):
            oof_frame[f"{candidate}__{target}"] = predictions[:, target_index]
    oof_frame.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    manifest = {
        "seed": SEED,
        "folds": 5,
        "gold_rows": len(gold_indices),
        "pseudo_rows_per_fold": len(pseudo_indices),
        "candidate_count": len(summary),
        "selection_warning": "Diagnostic only; 58 expert rows are too small for reliable model selection.",
        "best_observed": summary.iloc[0].to_dict(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
