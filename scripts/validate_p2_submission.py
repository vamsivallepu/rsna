#!/usr/bin/env python3
"""Independently validate P2 submission-kernel outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
EXPECTED_P2_MANIFEST_SHA256 = (
    "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest = json.loads((root / "p2_submission_manifest.json").read_text())
    assert manifest["run_id"] == "P2-SUBMISSION-v1"
    assert manifest["source_p2_run_id"] == "P2-COATNET-v1"
    assert manifest["source_p2_manifest_sha256"] == EXPECTED_P2_MANIFEST_SHA256
    assert manifest["source_p2_best_epoch"] == 8
    assert manifest["targets"] == TARGETS
    assert manifest["ensemble"] == (
        "mean probabilities from five report-purged fold heads"
    )
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha256(root / name) == expected, name

    submission = pd.read_csv(root / "submission.csv", dtype={ID: str})
    direct = pd.read_csv(
        root / "p2_test_direct_predictions.csv", dtype={ID: str}
    )
    assert submission.columns.tolist() == [ID, *TARGETS]
    assert direct.columns.tolist() == [ID, *TARGETS]
    assert submission[ID].is_unique and len(submission) == manifest["studies"]
    assert direct[ID].tolist() == submission[ID].tolist()
    if args.sample_submission is not None:
        sample = pd.read_csv(args.sample_submission, dtype={ID: str})
        assert sample.columns.tolist() == submission.columns.tolist()
        assert sample[ID].tolist() == submission[ID].tolist()

    predictions = submission[TARGETS].to_numpy(np.float64)
    direct_predictions = direct[TARGETS].to_numpy(np.float64)
    fold_predictions = np.load(root / "p2_test_predictions_by_fold.npy")
    mask = np.load(root / "p2_test_feature_mask.npy").astype(bool)
    positions = np.load(root / "p2_test_feature_positions.npy")
    assert fold_predictions.shape == (5, len(submission), len(TARGETS))
    assert mask.shape == positions.shape == (len(submission), 32)
    assert np.isfinite(fold_predictions).all()
    assert np.isfinite(predictions).all() and np.isfinite(direct_predictions).all()
    assert ((predictions > 0) & (predictions < 1)).all()
    assert np.allclose(predictions, fold_predictions.mean(axis=0), atol=1e-7)
    assert mask.any(axis=1).all()
    assert (positions[mask] >= 1).all() and (positions[~mask] == -1).all()
    if len(submission) > 1:
        assert (np.ptp(predictions, axis=0) > 0).all()

    selections = pd.read_csv(root / "p2_test_selections.csv", dtype={ID: str})
    assert len(selections) == len(submission) * 5
    assert selections.groupby(ID).size().eq(5).all()
    assert set(selections[ID]) == set(submission[ID])
    assert selections["valid_slices"].ge(0).all()

    summary = {
        "validated": True,
        "studies": len(submission),
        "targets": len(TARGETS),
        "five_fold_mean_recomputed": True,
        "source_p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
        "submission_sha256": sha256(root / "submission.csv"),
        "prediction_ranges": {
            target: [
                float(submission[target].min()),
                float(submission[target].max()),
            ]
            for target in TARGETS
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
