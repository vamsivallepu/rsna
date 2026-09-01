#!/usr/bin/env python3
"""Independently validate the completed Kaggle DINOv2 adaptation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
N_SLOTS = 6
N_POS = 3
TOKEN_DIM = 768
HIDDEN = 384


class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, HIDDEN), nn.GELU(), nn.Dropout(0.20)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aucs(labels: np.ndarray, predictions: np.ndarray) -> tuple[dict[str, float], float]:
    scores = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    return scores, float(np.mean(list(scores.values())))


def assert_metrics(expected: pd.Series, scores: dict[str, float], macro: float) -> None:
    for target, value in scores.items():
        assert np.isclose(value, expected.loc[target], atol=1e-12)
    assert np.isclose(macro, expected.loc["MACRO"], atol=1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--teacher-targets", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest = json.loads((root / "dino_adaptation_manifest.json").read_text())
    studies = pd.read_csv(root / "dino_feature_studies.csv")
    targets = pd.read_csv(root / "training_targets.csv").set_index("StudyInstanceUID").loc[
        studies["StudyInstanceUID"]
    ]
    failures = pd.read_csv(root / "dino_feature_failures.csv")
    folds = pd.read_csv(root / "dino_fold_summary.csv")
    oof = pd.read_csv(root / "dino_oof_predictions.csv")
    direct_metrics = pd.read_csv(root / "dino_direct_gold_metrics.csv").set_index("target")["auc"]
    oof_metrics = pd.read_csv(root / "dino_oof_metrics.csv").set_index("target")["auc"]
    features = np.load(root / "dino_token_features.npy", mmap_mode="r")
    mask = np.load(root / "dino_window_mask.npy", mmap_mode="r").astype(bool)

    assert manifest["full_run"] is True and manifest["adaptation_gold_rows"] == 0
    assert manifest["studies"] == manifest["usable_studies"] == 4_407
    assert manifest["expert_studies"] == 58 and manifest["pseudo_studies"] == 4_349
    assert failures.empty
    assert features.shape == (4_407, N_SLOTS, N_POS, TOKEN_DIM)
    assert mask.shape == (4_407, N_SLOTS, N_POS)
    assert np.isfinite(features).all() and mask.all()
    assert studies["StudyInstanceUID"].nunique() == len(studies) == 4_407
    assert folds.shape[0] == 5 and folds["expert_valid_rows"].sum() == 58
    assert oof["StudyInstanceUID"].nunique() == len(oof) == 58
    assert oof.notna().all().all()
    assert not (root / "dino_pixels.npy").exists()

    gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(gold)[0]
    labels = targets.loc[gold, TARGETS].to_numpy(np.float32)
    assert len(gold_indices) == 58

    full_state = torch.load(
        root / "dino_pseudo_adapted_model.pt", map_location="cpu", weights_only=True
    )
    head_state = {
        key.removeprefix("head."): value
        for key, value in full_state.items()
        if key.startswith("head.")
    }
    head = TokenHead().eval()
    head.load_state_dict(head_state, strict=True)
    with torch.inference_mode():
        direct_predictions = torch.sigmoid(
            head(
                torch.from_numpy(np.asarray(features[gold_indices]).astype(np.float32)),
                torch.from_numpy(mask[gold_indices]),
            )
        ).numpy()
    direct_scores, direct_macro = aucs(labels, direct_predictions)
    assert_metrics(direct_metrics, direct_scores, direct_macro)
    assert np.isclose(direct_macro, manifest["direct_gold_result"]["macro_auc"], atol=1e-12)

    oof_labels = oof[TARGETS].to_numpy(np.float32)
    oof_predictions = oof[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
    oof_scores, oof_macro = aucs(oof_labels, oof_predictions)
    assert_metrics(oof_metrics, oof_scores, oof_macro)
    assert np.isclose(oof_macro, manifest["fold_head_cv_result"]["macro_auc"], atol=1e-12)

    fold_keys = None
    for fold in range(5):
        state = torch.load(root / f"dino_fold_{fold}_head.pt", map_location="cpu", weights_only=True)
        keys = tuple(sorted(state))
        fold_keys = fold_keys or keys
        assert keys == fold_keys
        assert all(torch.isfinite(value).all() for value in state.values())

    target_hash = sha256(root / "training_targets.csv")
    if args.teacher_targets:
        assert target_hash == sha256(args.teacher_targets)

    log = (root / "rsna-knee-dinov2-pseudo-adaptation.log").read_text(errors="replace")
    assert "Traceback (most recent call last)" not in log
    assert "cached 4,407/4,407 studies" in log
    assert "fold_head_oof_macro_auc" in log

    summary = {
        "validated": True,
        "studies": 4_407,
        "failed_studies": 0,
        "valid_windows_per_study": int(mask.sum(axis=(1, 2)).min()),
        "direct_untouched_gold_macro_auc": direct_macro,
        "fold_head_oof_macro_auc": oof_macro,
        "training_targets_sha256": target_hash,
        "fold_head_parameter_tensors": len(fold_keys or ()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
