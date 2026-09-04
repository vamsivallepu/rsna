#!/usr/bin/env python3
"""Independently validate full P5-A artifacts, metrics, folds, and saved heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import p5_orthofoundation_speed_smoke as common


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def macro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score(labels[:, index], predictions[:, index])
                for index in range(labels.shape[1])
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    manifest = json.loads((root / "p5_manifest.json").read_text())
    assert manifest["run_id"] == "P5A-ORTHOFOUNDATION-FROZEN-v1"
    assert manifest["execution_revision"] == "p5a_frozen_t4x2_full_v2"
    assert manifest["studies"] == 4_407
    assert manifest["pseudo_studies"] == 4_349
    assert manifest["expert_studies"] == 58
    assert manifest["world_size"] == 2
    assert manifest["feature_shape"] == [4_407, 32, 2, 1_024]
    assert manifest["checkpoint_sha256"] == common.MODEL_SHA256
    assert manifest["checkpoint_namespace"] == "backbone."
    assert manifest["representation_gold_rows"] == 0
    assert manifest["pseudo_report_group_overlap"] == 0
    assert manifest["fold_report_hash_overlap"] == 0
    assert max(manifest["peak_memory_mib_by_rank"]) < 14_500
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file(), path
        assert sha256_file(path) == expected, f"hash mismatch: {name}"

    features = np.load(root / "p5_orthofoundation_features.npy", mmap_mode="r")
    masks = np.load(root / "p5_feature_mask.npy").astype(bool)
    slots = np.load(root / "p5_feature_slots.npy").astype(np.int64)
    positions = np.load(root / "p5_feature_positions.npy").astype(np.float32)
    studies = pd.read_csv(root / "p5_feature_studies.csv", dtype={common.ID: str})
    oof = pd.read_csv(root / "p5_oof_predictions.csv", dtype={common.ID: str})
    assert features.shape == (4_407, 32, 2, 1_024)
    assert masks.shape == slots.shape == positions.shape == (4_407, 32)
    assert len(studies) == 4_407 and studies[common.ID].is_unique
    assert len(oof) == 58 and oof[common.ID].is_unique
    assert set(oof["fold"]) == set(range(5))
    assert np.isfinite(features).all() and masks.any(axis=1).all()

    study_lookup = pd.Series(np.arange(len(studies)), index=studies[common.ID])
    gold_indices = study_lookup.loc[oof[common.ID]].to_numpy(int)
    label_columns = [f"label__{target}" for target in common.TARGETS]
    prediction_columns = [f"pred__{target}" for target in common.TARGETS]
    labels = oof[label_columns].to_numpy(np.float32)
    predictions = oof[prediction_columns].to_numpy(np.float32)
    replay = np.full_like(predictions, np.nan)
    feature_tensor = torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32))
    mask_tensor = torch.from_numpy(masks[gold_indices])
    slot_tensor = torch.from_numpy(slots[gold_indices])
    position_tensor = torch.from_numpy(positions[gold_indices])
    for fold in range(5):
        selected = np.where(oof["fold"].to_numpy(int) == fold)[0]
        head = common.HierarchicalDiagnosisHead()
        head.load_state_dict(
            torch.load(root / f"p5_fold_{fold}_head.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        head.eval()
        with torch.inference_mode():
            replay[selected] = torch.sigmoid(
                head(
                    feature_tensor[selected],
                    mask_tensor[selected],
                    slot_tensor[selected],
                    position_tensor[selected],
                )
            ).numpy()
    assert np.isfinite(replay).all()
    assert np.allclose(replay, predictions, atol=2e-3, rtol=2e-3)
    replay_macro = macro_auc(labels, predictions)
    assert abs(replay_macro - manifest["fold_oof_macro_auc"]) < 1e-12

    comparison = json.loads((root / "p5_comparison.json").read_text())
    assert comparison == manifest["comparison"]
    primary = pd.read_csv(root / "p5_primary_blend_oof_predictions.csv")
    primary_predictions = primary[prediction_columns].to_numpy(np.float32)
    primary_macro = macro_auc(labels, primary_predictions)
    assert abs(primary_macro - comparison["primary_p5_25pct_macro_auc"]) < 1e-9
    folds = pd.read_csv(root / "p5_fold_summary.csv")
    assert len(folds) == 5
    assert (folds["report_hash_overlap"] == 0).all()
    assert folds["expert_valid_rows"].sum() == 58
    print(
        json.dumps(
            {
                "validated": True,
                "p5_oof_macro_auc": replay_macro,
                "primary_blend_macro_auc": primary_macro,
                "promotion_gate": comparison["promotion_gate"],
                "lora_gate": comparison["lora_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
