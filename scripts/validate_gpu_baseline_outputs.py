#!/usr/bin/env python3
"""Validate persisted outputs from the full Kaggle 2.5D baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--teacher-targets", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest = json.loads((root / "image_baseline_manifest.json").read_text())
    metrics = pd.read_csv(root / "cv_metrics.csv").set_index("target")["auc"]
    oof = pd.read_csv(root / "oof_predictions.csv")
    studies = pd.read_csv(root / "feature_studies.csv")
    folds = pd.read_csv(root / "fold_summary.csv")
    predictions = pd.read_csv(root / "train_image_predictions.csv")
    failures = pd.read_csv(root / "feature_failures.csv")
    features = np.load(root / "study_features.npy", mmap_mode="r")
    slot_mask = np.load(root / "study_slot_mask.npy", mmap_mode="r")

    assert manifest["full_run"] is True
    assert manifest["studies"] == manifest["usable_studies"] == 4_407
    assert manifest["failed_studies"] == 0 and failures.empty
    assert features.shape == (4_407, 6, 2_048)
    assert slot_mask.shape == (4_407, 6)
    assert np.isfinite(features).all()
    assert slot_mask.any(axis=1).all()
    assert studies["StudyInstanceUID"].nunique() == len(studies) == 4_407
    assert predictions["StudyInstanceUID"].nunique() == len(predictions) == 4_407
    assert oof["StudyInstanceUID"].nunique() == len(oof) == 58
    assert oof[[*TARGETS, *[f"pred__{target}" for target in TARGETS]]].notna().all().all()
    assert folds["expert_valid_rows"].sum() == 58
    assert folds.shape[0] == 5

    recomputed = {
        target: roc_auc_score(oof[target], oof[f"pred__{target}"])
        for target in TARGETS
    }
    recomputed_macro = float(np.mean(list(recomputed.values())))
    assert np.isclose(recomputed_macro, metrics.loc["MACRO"], atol=1e-12)
    assert np.isclose(recomputed_macro, manifest["cv_result"]["macro_auc"], atol=1e-12)
    for target, auc in recomputed.items():
        assert np.isclose(auc, metrics.loc[target], atol=1e-12)

    checkpoint_keys = None
    for fold in range(5):
        state = torch.load(root / f"fold_{fold}_attention_head.pt", map_location="cpu", weights_only=True)
        keys = tuple(sorted(state))
        checkpoint_keys = checkpoint_keys or keys
        assert keys == checkpoint_keys
        assert all(torch.isfinite(value).all() for value in state.values())

    gpu_targets_sha256 = sha256(root / "training_targets.csv")
    if args.teacher_targets:
        assert gpu_targets_sha256 == sha256(args.teacher_targets)

    log = (root / "rsna-knee-2-5d-radimagenet-baseline.log").read_text(errors="replace")
    assert "Traceback (most recent call last)" not in log
    assert "encoded 4,407/4,407 studies" in log

    summary = {
        "validated": True,
        "studies": 4_407,
        "expert_oof_rows": 58,
        "failed_studies": 0,
        "mean_populated_slots": float(slot_mask.sum(axis=1).mean()),
        "macro_auc": recomputed_macro,
        "training_targets_sha256": gpu_targets_sha256,
        "checkpoint_parameter_tensors": len(checkpoint_keys or ()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
