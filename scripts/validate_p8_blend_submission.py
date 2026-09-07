#!/usr/bin/env python3
"""Independently validate the P8 blended-submission canary outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
EXPECTED_P2_MANIFEST_SHA256 = (
    "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"
)
EXPECTED_P8_MANIFEST_SHA256 = {
    "0": "f242c61269bff702fe601e35f6bc1bd7267f7d5f94c265ce684e665a4b37baf6",
    "1": "1120c6173b9044ca84c035e47070cb5e79e40484eddea158cdd114f809eb10eb",
    "2": "78925624ab4cebbe7044da362e1015c4d39b0ab514f04625f3d031f2a952c04d",
    "3": "4d7f4a4090b7eb00517f29b8ea88822a0b8871519a593b64fb464fdaa71cd095",
    "4": "94bacb27bc1ffa637ba3adbad6be0c960f19f69a0b59563a2b1fd6efe9f7ac2b",
}
EXPECTED_P8_CHECKPOINT_SHA256 = {
    "0": "5883762c8f4161e68eea33766aa737b34b772b180e18e75572937eeabf366c5f",
    "1": "68df88b583cca94d6f777476f0b2696c9fcd850591d863cc0255a85f248003d0",
    "2": "b2b42a0c44f83986839f90ec4547038ac0ca9705a30923be3da1e209c5196498",
    "3": "a2c1600b173a4abc3b94e339e92de0db92e93a0d3a5a456db1f415e0c0db3639",
    "4": "d0a154d4ab0844793c195ac92beba5fb036c9ea51dfba20bdef99194fdcb868f",
}
EXPECTED_P8_TEACHER_TARGET_SHA256 = {
    "0": "ec306bb842e8aa43a81d200eaba6b9d3d372fc48792f3ca902a576d663e9790b",
    "1": "fdc1165235bcc974be415f6bf068d31a84367f15206d10cd07e5523a84b55596",
    "2": "522a8478a7bd8dc49226cc923c22a859bc80e1236a74c41360deaf62b6c2d7b4",
    "3": "27df74518c118a61aa556f65231aa8fb532794cba090855d6beaa76ecfcdea04",
    "4": "10245d23ae189656823d983000fab8631ed6432e3a859c333b8d01cb25213b30",
}
EXPECTED_P8_TEACHER_MANIFEST_SHA256 = (
    "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
)
EXPECTED_P6_MANIFEST_SHA256 = (
    "885b58c5a9af1bffaac1f52c9703db3538d25f8d9660b036b563b351b1067bce"
)
HIDDEN_STUDIES = 1300
MAX_GPU_HOURS = 9.0
MINIMUM_HEADROOM_HOURS = 0.9


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
            pd.Series(predictions[:, index])
            .rank(method="average", pct=True)
            .to_numpy(np.float64)
            for index in range(predictions.shape[1])
        ]
    )


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest_path = root / "p8_blend_submission_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P8-BLEND-SUBMISSION-v1"
    assert manifest["source_p2"]["manifest_sha256"] == (
        EXPECTED_P2_MANIFEST_SHA256
    )
    source_p8 = manifest["source_p8"]
    assert {
        fold: source["manifest_sha256"] for fold, source in source_p8.items()
    } == EXPECTED_P8_MANIFEST_SHA256
    assert {
        fold: source["checkpoint_sha256"] for fold, source in source_p8.items()
    } == EXPECTED_P8_CHECKPOINT_SHA256
    assert {
        fold: source["teacher_target_sha256"] for fold, source in source_p8.items()
    } == EXPECTED_P8_TEACHER_TARGET_SHA256
    assert all(
        source["teacher_manifest_sha256"]
        == EXPECTED_P8_TEACHER_MANIFEST_SHA256
        for source in source_p8.values()
    )
    assert manifest["source_p6"]["manifest_sha256"] == (
        EXPECTED_P6_MANIFEST_SHA256
    )
    assert manifest["targets"] == TARGETS
    assert manifest["geometry"]["k_eval"] == 32
    assert manifest["ensemble"]["formula"] == (
        "0.375 rank(P2) + 0.375 rank(P8) + 0.25 rank(P6)"
    )
    assert manifest["ensemble"]["weights"] == {
        "p2_p8_equal_rank": 0.75,
        "p6_rank": 0.25,
    }
    assert manifest["oof"]["p8_macro_auc"] == 0.9029118074637345
    assert manifest["oof"]["equal_rank_p2_p8_macro_auc"] == 0.9022377777850243
    assert manifest["oof"][
        "fixed_75pct_p2_p8_25pct_p6_macro_auc"
    ] == 0.9097982866758175
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha256(root / name) == expected, name

    frames = {
        "submission": pd.read_csv(root / "submission.csv", dtype={ID: str}),
        "p2": pd.read_csv(root / "p8_blend_p2_predictions.csv", dtype={ID: str}),
        "p8": pd.read_csv(root / "p8_blend_p8_predictions.csv", dtype={ID: str}),
        "p6": pd.read_csv(root / "p8_blend_p6_predictions.csv", dtype={ID: str}),
    }
    submission = frames["submission"]
    expected_columns = [ID, *TARGETS]
    assert all(frame.columns.tolist() == expected_columns for frame in frames.values())
    assert submission[ID].is_unique and len(submission) == manifest["studies"]
    assert all(
        frame[ID].tolist() == submission[ID].tolist() for frame in frames.values()
    )
    if args.sample_submission is not None:
        sample = pd.read_csv(args.sample_submission, dtype={ID: str})
        assert sample.columns.tolist() == expected_columns
        assert sample[ID].tolist() == submission[ID].tolist()

    p2 = frames["p2"][TARGETS].to_numpy(np.float64)
    p8 = frames["p8"][TARGETS].to_numpy(np.float64)
    p6 = frames["p6"][TARGETS].to_numpy(np.float64)
    submitted = submission[TARGETS].to_numpy(np.float64)
    by_fold = np.load(root / "p8_blend_p8_predictions_by_fold.npy")
    p2_p8 = np.clip(
        (rank_columns(p2) + rank_columns(p8)) / 2.0,
        1e-6,
        1.0 - 1e-6,
    )
    expected = np.clip(
        0.75 * p2_p8 + 0.25 * rank_columns(p6),
        1e-6,
        1.0 - 1e-6,
    )
    assert by_fold.shape == (5, len(submission), len(TARGETS))
    assert np.isfinite(by_fold).all()
    assert np.allclose(p8, by_fold.mean(axis=0), atol=1e-7)
    assert np.allclose(submitted, expected, atol=1e-12)
    assert np.isfinite(submitted).all()
    assert ((submitted > 0) & (submitted < 1)).all()
    if len(submission) > 1:
        assert all(np.ptp(values, axis=0).min() > 0 for values in (p2, p8, p6))

    mask = np.load(root / "p8_blend_feature_mask.npy").astype(bool)
    positions = np.load(root / "p8_blend_feature_positions.npy")
    assert mask.shape == positions.shape == (len(submission), 32)
    assert mask.any(axis=1).all()
    assert (positions[mask] >= 1).all() and (positions[~mask] == -1).all()
    selections = pd.read_csv(
        root / "p8_blend_test_selections.csv", dtype={ID: str}
    )
    assert len(selections) == len(submission) * 5
    assert selections.groupby(ID).size().eq(5).all()
    assert set(selections[ID]) == set(submission[ID])

    environment = manifest["environment"]
    assert environment["visible_gpus"] == 2
    assert environment["batch_size"] == 2
    assignments = environment["device_assignments"]
    assert set(assignments) == {"cuda:0", "cuda:1"}
    assert all("P6-alternating-batches" in values for values in assignments.values())
    assert sorted(
        value
        for values in assignments.values()
        for value in values
        if value.startswith("P8-fold-")
    ) == [f"P8-fold-{fold}" for fold in range(5)]
    inference_minutes = manifest["elapsed"]["dicom_and_model_inference_minutes"]
    projected_hours = inference_minutes / len(submission) * HIDDEN_STUDIES / 60.0
    headroom_hours = MAX_GPU_HOURS - projected_hours
    runtime_gate_passed = headroom_hours >= MINIMUM_HEADROOM_HOURS

    summary = {
        "validated": True,
        "studies": len(submission),
        "targets": len(TARGETS),
        "five_p8_fold_mean_recomputed": True,
        "fixed_p8_blend_recomputed": True,
        "dual_gpu_alternating_p6_verified": True,
        "submission_sha256": sha256(root / "submission.csv"),
        "manifest_sha256": sha256(manifest_path),
        "runtime": {
            "canary_inference_minutes": inference_minutes,
            "projected_hidden_studies": HIDDEN_STUDIES,
            "projected_hidden_hours": projected_hours,
            "nine_hour_headroom_hours": headroom_hours,
            "minimum_required_headroom_hours": MINIMUM_HEADROOM_HOURS,
            "gate_passed": runtime_gate_passed,
        },
    }
    print(json.dumps(summary, indent=2))
    if not runtime_gate_passed:
        raise RuntimeError(
            "P8 canary is valid but lacks the predeclared 0.9-hour runtime margin"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
