#!/usr/bin/env python3
"""Independently validate P4 equal-rank blend submission outputs."""

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
EXPECTED_P4_MANIFEST_SHA256 = {
    "0": "6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932",
    "1": "9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5",
    "2": "93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac",
    "3": "9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c",
    "4": "e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228",
}


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


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index]).rank(
                method="average", pct=True
            ).to_numpy(np.float64)
            for index in range(predictions.shape[1])
        ]
    )


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest = json.loads(
        (root / "p4_blend_submission_manifest.json").read_text()
    )
    assert manifest["run_id"] == "P4-BLEND-SUBMISSION-v1"
    assert manifest["source_p2"]["manifest_sha256"] == (
        EXPECTED_P2_MANIFEST_SHA256
    )
    assert {
        fold: source["manifest_sha256"]
        for fold, source in manifest["source_p4"].items()
    } == EXPECTED_P4_MANIFEST_SHA256
    assert manifest["targets"] == TARGETS
    assert manifest["geometry"]["k_eval"] == 32
    assert manifest["ensemble"]["weights"] == {"p2": 0.5, "p4": 0.5}
    assert manifest["oof"]["equal_rank_p2_p4_macro_auc"] == (
        0.8958210633119559
    )
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha256(root / name) == expected, name

    submission = pd.read_csv(root / "submission.csv", dtype={ID: str})
    p2 = pd.read_csv(
        root / "p4_blend_p2_predictions.csv", dtype={ID: str}
    )
    p4 = pd.read_csv(
        root / "p4_blend_p4_predictions.csv", dtype={ID: str}
    )
    assert submission.columns.tolist() == [ID, *TARGETS]
    assert p2.columns.tolist() == p4.columns.tolist() == submission.columns.tolist()
    assert submission[ID].is_unique and len(submission) == manifest["studies"]
    assert p2[ID].tolist() == p4[ID].tolist() == submission[ID].tolist()
    if args.sample_submission is not None:
        sample = pd.read_csv(args.sample_submission, dtype={ID: str})
        assert sample.columns.tolist() == submission.columns.tolist()
        assert sample[ID].tolist() == submission[ID].tolist()

    p2_values = p2[TARGETS].to_numpy(np.float64)
    p4_values = p4[TARGETS].to_numpy(np.float64)
    submitted = submission[TARGETS].to_numpy(np.float64)
    by_fold = np.load(root / "p4_blend_p4_predictions_by_fold.npy")
    expected = np.clip(
        (rank_columns(p2_values) + rank_columns(p4_values)) / 2.0,
        1e-6,
        1.0 - 1e-6,
    )
    assert by_fold.shape == (5, len(submission), len(TARGETS))
    assert np.isfinite(by_fold).all()
    assert np.allclose(p4_values, by_fold.mean(axis=0), atol=1e-7)
    assert np.allclose(submitted, expected, atol=1e-12)
    assert np.isfinite(submitted).all()
    assert ((submitted > 0) & (submitted < 1)).all()
    if len(submission) > 1:
        assert (np.ptp(p2_values, axis=0) > 0).all()
        assert (np.ptp(p4_values, axis=0) > 0).all()

    mask = np.load(root / "p4_blend_feature_mask.npy").astype(bool)
    positions = np.load(root / "p4_blend_feature_positions.npy")
    assert mask.shape == positions.shape == (len(submission), 32)
    assert mask.any(axis=1).all()
    assert (positions[mask] >= 1).all() and (positions[~mask] == -1).all()
    selections = pd.read_csv(
        root / "p4_blend_test_selections.csv", dtype={ID: str}
    )
    assert len(selections) == len(submission) * 5
    assert selections.groupby(ID).size().eq(5).all()
    assert set(selections[ID]) == set(submission[ID])
    assert selections["valid_slices"].ge(0).all()

    summary = {
        "validated": True,
        "studies": len(submission),
        "targets": len(TARGETS),
        "five_p4_fold_mean_recomputed": True,
        "equal_rank_blend_recomputed": True,
        "source_p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
        "source_p4_manifest_sha256": EXPECTED_P4_MANIFEST_SHA256,
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
