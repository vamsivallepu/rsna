#!/usr/bin/env python3
"""Independently validate full P6 localization-training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import timm
from safetensors.torch import load_file


ARCH = "coatnet_rmlp_2_rw_384"
TRAIN_ARCHIVE_SHA256 = (
    "1e8ab7f602bb0fea649736020aa9317ebf3ba9c417c211b286b85c81a28bce5f"
)
VAL_ARCHIVE_SHA256 = "cd65a27f25d11369ad73fd329ff0c68ea8d2510f98e9f46aaeff7fcaf0dba43b"
ANNOTATION_SHA256 = "c1f4a083646cec815b67f005f8d7764313d1b82a028b90c272a4d000330d065b"
REVIEWED_SHA256 = "4b09e5523709815d5314a4f652882c40f8c14af26b67a00ea769054ca7e1d6b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root
    manifest_path = root / "p6_localization_training.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["status"] != "validated":
        raise ValueError("P6 manifest status/schema is not validated")
    if not manifest.get("code_commit"):
        raise ValueError("P6 manifest is missing the deployed code commit")

    source = manifest["source"]
    expected_source = {
        "train_archive_sha256": TRAIN_ARCHIVE_SHA256,
        "validation_archive_sha256": VAL_ARCHIVE_SHA256,
        "annotations_sha256": ANNOTATION_SHA256,
        "reviewed_files_sha256": REVIEWED_SHA256,
        "image_field": "reconstruction_rss",
        "labeling_frame_transform": "pixels[:, ::-1, :]",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"Unexpected source field {key}: {source.get(key)}")

    artifact_paths = {
        "train_index_sha256": root / "p6_train_hdf5_index.csv",
        "validation_index_sha256": root / "p6_validation_hdf5_index.csv",
        "sanitation_sha256": root / "p6_full_box_sanitation.csv",
        "samples_sha256": root / "p6_source_samples.csv",
        "backbone_sha256": root / "p6_localization_backbone_best.safetensors",
        "head_sha256": root / "p6_localization_head_best.safetensors",
        "history_sha256": root / "p6_training_history.json",
    }
    for key, path in artifact_paths.items():
        observed = sha256_file(path)
        if manifest["artifacts"].get(key) != observed:
            raise ValueError(f"Artifact hash mismatch for {path.name}")

    train_index = pd.read_csv(artifact_paths["train_index_sha256"], dtype={"file": str})
    validation_index = pd.read_csv(
        artifact_paths["validation_index_sha256"], dtype={"file": str}
    )
    if (
        train_index["file"].duplicated().any()
        or validation_index["file"].duplicated().any()
    ):
        raise ValueError("Duplicated HDF5 identity in a split index")
    train_volumes = set(train_index["file"])
    validation_volumes = set(validation_index["file"])
    if len(train_volumes) != 973 or len(validation_volumes) != 199:
        raise ValueError(
            "Official archive volume counts are not 973 train / 199 validation"
        )
    if train_volumes & validation_volumes:
        raise ValueError("Train/validation HDF5 volume overlap")
    if len(train_volumes | validation_volumes) != 1172:
        raise ValueError("Full reviewed-volume coverage is not 1,172")

    samples = pd.read_csv(artifact_paths["samples_sha256"], dtype={"file": str})
    sample_train = set(samples.loc[samples["split"].eq("train"), "file"])
    sample_validation = set(samples.loc[samples["split"].eq("validation"), "file"])
    if sample_train & sample_validation:
        raise ValueError("Train/validation sample leakage")
    if not sample_train <= train_volumes or not sample_validation <= validation_volumes:
        raise ValueError("Sample split does not agree with archive split")
    data = manifest["data"]
    expected_counts = {
        "train_volumes": len(train_index),
        "validation_volumes": len(validation_index),
        "train_samples": int(samples["split"].eq("train").sum()),
        "validation_samples": int(samples["split"].eq("validation").sum()),
        "volume_overlap": 0,
    }
    for key, expected in expected_counts.items():
        if data.get(key) != expected:
            raise ValueError(f"Manifest data count mismatch for {key}")

    sanitation = pd.read_csv(artifact_paths["sanitation_sha256"])
    kept = sanitation["kept"].astype(str).str.lower().eq("true")
    if int(kept.sum()) != 16153 or int((~kept).sum()) != 1:
        raise ValueError("Unexpected full-release box sanitation counts")
    if data["kept_boxes"] != 16153 or data["discarded_boxes"] != 1:
        raise ValueError("Manifest box sanitation counts are inconsistent")
    if data.get("excluded_study_level_artifact_volumes") != 5:
        raise ValueError("Five study-level artifact-only volumes must be excluded")
    if data.get("study_level_artifact_volumes") != 13:
        raise ValueError("The release must contain 13 study-level artifact volumes")

    history = json.loads(artifact_paths["history_sha256"].read_text(encoding="utf-8"))
    trained = [row for row in history if row["epoch"] > 0]
    if len(trained) != manifest["training"]["epochs"]:
        raise ValueError("Training history length does not match configured epochs")
    selected = min(
        trained,
        key=lambda row: (
            row["validation_loss"],
            -row["validation_lesion_center_recall_topk_radius_2"],
        ),
    )
    selection = manifest["selection"]
    if selection["best_epoch"] != selected["epoch"]:
        raise ValueError("Selected epoch does not minimize validation loss")
    if not math.isclose(
        selection["best_validation_loss"], selected["validation_loss"], abs_tol=1e-10
    ):
        raise ValueError("Selected validation loss does not match history")

    backbone_state = load_file(str(artifact_paths["backbone_sha256"]), device="cpu")
    expected_backbone = timm.create_model(ARCH, pretrained=False).state_dict()
    if set(backbone_state) != set(expected_backbone):
        raise ValueError(
            "Selected backbone state keys do not strictly match P2 CoAtNet"
        )
    for key, value in backbone_state.items():
        if value.shape != expected_backbone[key].shape:
            raise ValueError(f"Backbone tensor shape mismatch: {key}")
    head_state = load_file(str(artifact_paths["head_sha256"]), device="cpu")
    if not head_state or not all(
        value.isfinite().all() for value in head_state.values()
    ):
        raise ValueError("Selected localization head is empty or non-finite")
    if not all(value.isfinite().all() for value in backbone_state.values()):
        raise ValueError("Selected backbone contains non-finite tensors")

    report = {
        "status": "validated",
        "code_commit": manifest["code_commit"],
        "train_volumes": len(train_volumes),
        "validation_volumes": len(validation_volumes),
        "train_samples": expected_counts["train_samples"],
        "validation_samples": expected_counts["validation_samples"],
        "kept_boxes": int(kept.sum()),
        "discarded_boxes": int((~kept).sum()),
        "best_epoch": selected["epoch"],
        "best_validation_loss": selected["validation_loss"],
        "backbone_tensors": len(backbone_state),
        "head_tensors": len(head_state),
        "all_artifact_hashes_match": True,
        "zero_volume_overlap": True,
        "strict_p2_backbone_compatibility": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
