#!/usr/bin/env python3
"""Independently validate P5 diagnostic submission outputs."""

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
EXPECTED_P5_MANIFEST_SHA256 = (
    "3b739d18f85347bbc457885bba72b1a20712d2ab56dda67ac68fc35840b3163d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    manifest = json.loads((root / "p5_submission_manifest.json").read_text())
    assert manifest["run_id"] == "P5-DIAGNOSTIC-SUBMISSION-v1"
    assert manifest["source_p5_manifest_sha256"] == EXPECTED_P5_MANIFEST_SHA256
    assert manifest["source_p5_oof_macro_auc"] == 0.7688217022757898
    assert manifest["geometry"]["slot_quotas"] == [9, 7, 6, 4, 6]
    assert manifest["geometry"]["slices_per_study"] == 32
    assert manifest["environment"]["world_size"] == 2
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha256_file(root / name) == expected, name

    submission = pd.read_csv(root / "submission.csv", dtype={ID: str})
    assert submission.columns.tolist() == [ID, *TARGETS]
    assert submission[ID].is_unique and len(submission) == manifest["studies"]
    if args.sample_submission is not None:
        sample = pd.read_csv(args.sample_submission, dtype={ID: str})
        assert submission.columns.tolist() == sample.columns.tolist()
        assert submission[ID].tolist() == sample[ID].tolist()
    predictions = submission[TARGETS].to_numpy(np.float64)
    by_fold = np.load(root / "p5_test_predictions_by_fold.npy")
    assert by_fold.shape == (5, len(submission), len(TARGETS))
    assert np.isfinite(by_fold).all()
    expected = np.clip(by_fold.mean(axis=0), 1e-6, 1.0 - 1e-6)
    assert np.allclose(predictions, expected, atol=1e-12)
    assert ((predictions > 0) & (predictions < 1)).all()
    if len(submission) > 1:
        assert (np.ptp(predictions, axis=0) > 0).all()

    mask = np.load(root / "p5_test_mask.npy").astype(bool)
    slots = np.load(root / "p5_test_slots.npy")
    positions = np.load(root / "p5_test_positions.npy")
    assert mask.shape == slots.shape == positions.shape == (len(submission), 32)
    assert mask.any(axis=1).all()
    assert ((slots >= 0) & (slots < 5)).all()
    assert ((positions[mask] >= -1) & (positions[mask] <= 1)).all()
    selections = pd.read_csv(root / "p5_test_selections.csv", dtype={ID: str})
    assert len(selections) == len(submission) * 5
    assert selections.groupby(ID).size().eq(5).all()
    assert set(selections[ID]) == set(submission[ID])
    print(
        json.dumps(
            {
                "validated": True,
                "studies": len(submission),
                "five_fold_mean_recomputed": True,
                "submission_sha256": sha256_file(root / "submission.csv"),
                "prediction_ranges": {
                    target: [
                        float(submission[target].min()),
                        float(submission[target].max()),
                    ]
                    for target in TARGETS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
