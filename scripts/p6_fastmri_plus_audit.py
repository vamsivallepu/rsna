#!/usr/bin/env python3
"""Validate the official fastMRI+ knee annotation release for P6.

This audit intentionally operates on annotations only. It does not download,
open, or redistribute fastMRI image data, whose access is governed separately
from the MIT-licensed fastMRI+ annotation repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ANNOTATION_SHA256 = "c1f4a083646cec815b67f005f8d7764313d1b82a028b90c272a4d000330d065b"
REVIEWED_SHA256 = "4b09e5523709815d5314a4f652882c40f8c14af26b67a00ea769054ca7e1d6b7"
SOURCE_COMMIT = "67ed9a68ee33d3197f93893e8125688470c4a577"
EXPECTED_COLUMNS = ["file", "slice", "study_level", "x", "y", "width", "height", "label"]
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

# This crosswalk is an audit artifact, not permission to convert boxes into RSNA
# labels. Shared/proxy relationships are deliberately kept explicit.
CROSSWALK = [
    ("ACL", "Ligament - ACL Low Grade sprain", "direct_grouped", "ACL injury, low grade"),
    ("ACL", "Ligament - ACL High Grade Sprain", "direct_grouped", "ACL injury, high grade"),
    ("MCL", "Ligament - MCL Low-Mod Grade Sprain", "direct_grouped", "MCL injury, low/moderate grade"),
    ("MCL", "Ligament - MCL High Grade sprain", "direct_grouped", "MCL injury, high grade"),
    (
        "Medial Meniscus",
        "Meniscus Tear",
        "ambiguous_shared",
        "annotation does not identify medial versus lateral meniscus",
    ),
    (
        "Medial Meniscus",
        "Displaced Meniscal Tissue",
        "ambiguous_shared",
        "annotation does not identify medial versus lateral meniscus",
    ),
    (
        "Lateral Meniscus",
        "Meniscus Tear",
        "ambiguous_shared",
        "annotation does not identify medial versus lateral meniscus",
    ),
    (
        "Lateral Meniscus",
        "Displaced Meniscal Tissue",
        "ambiguous_shared",
        "annotation does not identify medial versus lateral meniscus",
    ),
    (
        "Medial OA",
        "Cartilage - Partial Thickness loss/defect",
        "ambiguous_shared",
        "cartilage compartment is not identified",
    ),
    (
        "Medial OA",
        "Cartilage - Full Thickness loss/defect",
        "ambiguous_shared",
        "cartilage compartment is not identified",
    ),
    (
        "Lateral OA",
        "Cartilage - Partial Thickness loss/defect",
        "ambiguous_shared",
        "cartilage compartment is not identified",
    ),
    (
        "Lateral OA",
        "Cartilage - Full Thickness loss/defect",
        "ambiguous_shared",
        "cartilage compartment is not identified",
    ),
    (
        "PF OA",
        "Cartilage - Partial Thickness loss/defect",
        "ambiguous_shared",
        "cartilage compartment is not identified",
    ),
    (
        "PF OA",
        "Cartilage - Full Thickness loss/defect",
        "ambiguous_shared",
        "cartilage compartment is not identified",
    ),
    ("Effusion", "Joint Effusion", "direct_grouped", "same finding family"),
    ("Synovitis", "", "none", "no released fastMRI+ knee label is a defensible direct match"),
    (
        "Baker's",
        "Periarticular cysts",
        "proxy",
        "Baker cyst is only one member of a broader periarticular-cyst label",
    ),
    (
        "Contusion",
        "Bone-Fracture/Contusion/dislocation",
        "ambiguous_shared",
        "combined label cannot separate contusion from fracture or dislocation",
    ),
    (
        "Contusion",
        "Bone- Subchondral edema",
        "proxy",
        "edema can accompany contusion but is not equivalent",
    ),
    (
        "Fracture",
        "Bone-Fracture/Contusion/dislocation",
        "ambiguous_shared",
        "combined label cannot separate fracture from contusion or dislocation",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help="Official knee.csv")
    parser.add_argument(
        "--reviewed-files", type=Path, required=True, help="Official knee_file_list.csv"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonable(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    annotation_hash = sha256_file(args.annotations)
    reviewed_hash = sha256_file(args.reviewed_files)
    if annotation_hash != ANNOTATION_SHA256:
        raise ValueError(f"Unexpected knee.csv SHA-256: {annotation_hash}")
    if reviewed_hash != REVIEWED_SHA256:
        raise ValueError(f"Unexpected knee_file_list.csv SHA-256: {reviewed_hash}")

    annotations = pd.read_csv(args.annotations)
    reviewed_raw = pd.read_csv(args.reviewed_files, header=None, names=["file"], dtype=str)
    if annotations.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected annotation columns: {annotations.columns.tolist()}")
    if reviewed_raw["file"].isna().any() or reviewed_raw["file"].duplicated().any():
        raise ValueError("Reviewed-file release contains a null or duplicate row")
    reviewed_id_mask = reviewed_raw["file"].str.fullmatch(r"file\d+")
    ignored_reviewed_rows = reviewed_raw.loc[~reviewed_id_mask, "file"].tolist()
    if ignored_reviewed_rows != ["Knee data"]:
        raise ValueError(f"Unexpected non-ID rows in reviewed-file release: {ignored_reviewed_rows}")
    reviewed = reviewed_raw.loc[reviewed_id_mask].reset_index(drop=True)
    if annotations["file"].isna().any() or annotations["label"].isna().any():
        raise ValueError("Annotation table contains a null file ID or label")
    if not set(annotations["file"]).issubset(set(reviewed["file"])):
        raise ValueError("An annotated volume is absent from the reviewed-volume list")
    if set(annotations["study_level"].unique()) != {"No", "Yes"}:
        raise ValueError("Unexpected study_level values")

    annotations["label_clean"] = annotations["label"].str.strip()
    if annotations.groupby("label_clean")["label"].nunique().max() != 1:
        raise ValueError("Whitespace normalization would merge distinct released labels")
    boxes = annotations.loc[annotations["study_level"].eq("No")].copy()
    study_rows = annotations.loc[annotations["study_level"].eq("Yes")].copy()
    if boxes[["x", "y", "width", "height"]].isna().any().any():
        raise ValueError("A slice-level annotation is missing box coordinates")
    if not study_rows[["x", "y", "width", "height"]].isna().all().all():
        raise ValueError("A study-level annotation unexpectedly has box coordinates")
    if set(study_rows["label_clean"]) != {"artifact"}:
        raise ValueError("Unexpected study-level label")

    label_stats = (
        boxes.groupby("label_clean", sort=True)
        .agg(
            box_count=("label_clean", "size"),
            volume_count=("file", "nunique"),
            slice_min=("slice", "min"),
            slice_max=("slice", "max"),
            x_min=("x", "min"),
            y_min=("y", "min"),
            width_max=("width", "max"),
            height_max=("height", "max"),
        )
        .reset_index()
        .rename(columns={"label_clean": "label"})
    )
    crosswalk = pd.DataFrame(CROSSWALK, columns=["target", "source_label", "relation", "note"])
    if set(crosswalk["target"]) != set(TARGETS):
        raise AssertionError("Crosswalk does not cover all competition targets")
    unknown_crosswalk_labels = set(crosswalk.loc[crosswalk["source_label"].ne(""), "source_label"]) - set(
        boxes["label_clean"]
    )
    if unknown_crosswalk_labels:
        raise AssertionError(f"Crosswalk contains unknown labels: {sorted(unknown_crosswalk_labels)}")

    observed = {
        "annotation_rows": int(len(annotations)),
        "bounding_boxes": int(len(boxes)),
        "study_level_artifact_rows": int(len(study_rows)),
        "released_labels_including_artifact": int(annotations["label"].nunique()),
        "slice_level_labels": int(boxes["label"].nunique()),
        "labels_with_surrounding_whitespace": int(
            annotations.loc[annotations["label"].ne(annotations["label_clean"]), "label"].nunique()
        ),
        "annotated_volumes": int(annotations["file"].nunique()),
        "reviewed_file_release_rows": int(len(reviewed_raw)),
        "reviewed_non_id_rows": int(len(ignored_reviewed_rows)),
        "reviewed_volumes": int(reviewed["file"].nunique()),
        "reviewed_without_any_annotation": int(
            reviewed.loc[~reviewed["file"].isin(set(annotations["file"])), "file"].nunique()
        ),
        "nonpositive_width_boxes": int((boxes["width"] <= 0).sum()),
        "nonpositive_height_boxes": int((boxes["height"] <= 0).sum()),
        "negative_coordinate_boxes": int(((boxes["x"] < 0) | (boxes["y"] < 0)).sum()),
    }
    expected_observed = {
        "annotation_rows": 16_167,
        "bounding_boxes": 16_154,
        "study_level_artifact_rows": 13,
        "released_labels_including_artifact": 22,
        "slice_level_labels": 21,
        "annotated_volumes": 974,
        "reviewed_file_release_rows": 1_173,
        "reviewed_non_id_rows": 1,
        "reviewed_volumes": 1_172,
        "reviewed_without_any_annotation": 198,
    }
    mismatches = {
        key: {"expected": expected, "observed": observed[key]}
        for key, expected in expected_observed.items()
        if observed[key] != expected
    }
    if mismatches:
        raise ValueError(f"Official-release invariant mismatch: {mismatches}")

    manifest = {
        "schema_version": 1,
        "experiment_id": "P6-AUDIT",
        "status": "validated_annotation_only",
        "source": {
            "repository": "https://github.com/microsoft/fastmri-plus",
            "commit": SOURCE_COMMIT,
            "annotations_sha256": annotation_hash,
            "reviewed_files_sha256": reviewed_hash,
        },
        "observed": observed,
        "reviewed_file_interpretation": {
            "paper_reported_reviewed_knee_volumes": 1_172,
            "released_csv_rows": observed["reviewed_file_release_rows"],
            "valid_file_ids": observed["reviewed_volumes"],
            "ignored_non_id_rows": ignored_reviewed_rows,
            "paper_count_matches_valid_file_ids": True,
        },
        "transfer_assessment": {
            "direct_target_count": int(
                crosswalk.loc[crosswalk["relation"].eq("direct_grouped"), "target"].nunique()
            ),
            "targets_without_direct_mapping": sorted(
                set(TARGETS)
                - set(crosswalk.loc[crosswalk["relation"].eq("direct_grouped"), "target"])
            ),
            "annotation_only_spatial_prior_approved": False,
            "reason": (
                "The CSV omits source image dimensions, spacing, and volume depth; labels were "
                "drawn after an up/down pixel flip; findings are sagittal and several competition "
                "targets have only ambiguous, proxy, or no mappings."
            ),
        },
        "access_gate": {
            "annotations_mit_licensed": True,
            "image_data_included": False,
            "official_fastmri_agreement_required_for_images": True,
            "unofficial_kaggle_image_mirror_approved": False,
        },
        "crosswalk_signature": sha256_jsonable(crosswalk.to_dict(orient="records")),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_stats.to_csv(args.output_dir / "p6_label_stats.csv", index=False)
    crosswalk.to_csv(args.output_dir / "p6_target_crosswalk.csv", index=False)
    with (args.output_dir / "p6_fastmri_plus_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote audit artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
