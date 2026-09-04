#!/usr/bin/env python3
"""Validate a deterministic fastMRI+ DICOM smoke cohort for P6-B.

The official DICOM package already contains the up/down flip used during labeling.
This script therefore evaluates boxes in the released pixel frame without another
flip. It never copies source images into the repository; output overlays belong in
an ignored, private experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pydicom
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


ANNOTATION_SHA256 = "c1f4a083646cec815b67f005f8d7764313d1b82a028b90c272a4d000330d065b"
REVIEWED_SHA256 = "4b09e5523709815d5314a4f652882c40f8c14af26b67a00ea769054ca7e1d6b7"
SOURCE_COMMIT = "67ed9a68ee33d3197f93893e8125688470c4a577"
FILE_RE = re.compile(r"^(file\d+)_(\d+)\.dcm$", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reviewed-files", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-volumes", type=int, default=128)
    parser.add_argument("--positive-fraction", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--overlay-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
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


def stable_sort(values: list[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{namespace}:{seed}:{value}".encode()).hexdigest(),
    )


def load_release(
    annotation_path: Path, reviewed_path: Path
) -> tuple[pd.DataFrame, set[str]]:
    annotation_hash = sha256_file(annotation_path)
    reviewed_hash = sha256_file(reviewed_path)
    if annotation_hash != ANNOTATION_SHA256:
        raise ValueError(f"Unexpected knee.csv SHA-256: {annotation_hash}")
    if reviewed_hash != REVIEWED_SHA256:
        raise ValueError(f"Unexpected knee_file_list.csv SHA-256: {reviewed_hash}")

    annotations = pd.read_csv(annotation_path)
    expected_columns = ["file", "slice", "study_level", "x", "y", "width", "height", "label"]
    if annotations.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected annotation columns: {annotations.columns.tolist()}")
    reviewed_raw = pd.read_csv(reviewed_path, header=None, names=["file"], dtype=str)
    valid = reviewed_raw["file"].str.fullmatch(r"file\d+")
    ignored = reviewed_raw.loc[~valid, "file"].tolist()
    if ignored != ["Knee data"]:
        raise ValueError(f"Unexpected reviewed-list footer: {ignored}")
    reviewed = set(reviewed_raw.loc[valid, "file"])
    if len(reviewed) != 1_172:
        raise ValueError(f"Expected 1,172 reviewed volumes, found {len(reviewed)}")
    return annotations, reviewed


def index_dicom_paths(dicom_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    malformed: list[str] = []
    for path in sorted(dicom_root.rglob("*.dcm")):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            malformed.append(path.relative_to(dicom_root).as_posix())
            continue
        rows.append(
            {
                "file": match.group(1),
                "slice": int(match.group(2)),
                "path": path.relative_to(dicom_root).as_posix(),
            }
        )
    if not rows:
        raise ValueError(f"No DICOM files found below {dicom_root}")
    index = pd.DataFrame(rows).sort_values(["file", "slice", "path"]).reset_index(drop=True)
    duplicated = index.duplicated(["file", "slice"], keep=False)
    if duplicated.any():
        examples = index.loc[duplicated, ["file", "slice", "path"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate volume/slice identities: {examples}")
    return index, malformed


def select_smoke_cohort(
    available: set[str],
    positive: set[str],
    requested: int,
    positive_fraction: float,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if requested <= 0 or requested > len(available):
        raise ValueError(f"Invalid smoke volume count {requested}; available={len(available)}")
    if not 0.0 < positive_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("Fractions must be strictly between zero and one")

    positives = stable_sort(sorted(available & positive), seed, "positive")
    negatives = stable_sort(sorted(available - positive), seed, "negative")
    want_positive = round(requested * positive_fraction)
    take_positive = min(want_positive, len(positives))
    take_negative = min(requested - take_positive, len(negatives))
    if take_positive + take_negative < requested:
        take_positive = min(len(positives), requested - take_negative)
    if take_positive + take_negative < requested:
        take_negative = min(len(negatives), requested - take_positive)
    if take_positive + take_negative != requested:
        raise ValueError("Could not construct the requested positive/negative smoke cohort")

    selected = [(value, True) for value in positives[:take_positive]]
    selected.extend((value, False) for value in negatives[:take_negative])
    records: list[dict[str, object]] = []
    for has_annotation in (True, False):
        stratum = [value for value, flag in selected if flag is has_annotation]
        ordered = stable_sort(stratum, seed, f"split-{has_annotation}")
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation = set(ordered[:validation_count])
        records.extend(
            {
                "file": value,
                "has_annotation": has_annotation,
                "split": "validation" if value in validation else "train",
            }
            for value in ordered
        )
    return pd.DataFrame(records).sort_values("file").reset_index(drop=True)


def validate_and_decode(
    dicom_root: Path, selected_index: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]]]:
    audited_rows: list[dict[str, object]] = []
    geometry: dict[str, tuple[int, int]] = {}

    for volume_id, group in selected_index.groupby("file", sort=True):
        slices = group["slice"].tolist()
        if slices != list(range(len(slices))):
            raise ValueError(
                f"Noncontiguous or non-zero-based slices for {volume_id}: "
                f"first={slices[:3]}, last={slices[-3:]}"
            )
        expected_depth = len(group)
        for row in group.itertuples(index=False):
            path = dicom_root / row.path
            dataset = pydicom.dcmread(path, force=False)
            transfer_syntax = str(dataset.file_meta.TransferSyntaxUID)
            rows = int(dataset.Rows)
            columns = int(dataset.Columns)
            instance = int(dataset.InstanceNumber)
            images = int(dataset.ImagesInAcquisition)
            patient_id = str(dataset.PatientID)
            patient_name = str(dataset.PatientName)
            if transfer_syntax != "1.2.840.10008.1.2.1":
                raise ValueError(f"Unexpected transfer syntax for {row.path}: {transfer_syntax}")
            if dataset.PhotometricInterpretation != "MONOCHROME2" or int(dataset.SamplesPerPixel) != 1:
                raise ValueError(f"Unexpected pixel interpretation for {row.path}")
            if instance != int(row.slice) + 1 or images != expected_depth:
                raise ValueError(
                    f"Slice metadata mismatch for {row.path}: "
                    f"instance={instance}, images={images}, expected_depth={expected_depth}"
                )
            if patient_id != volume_id or patient_name != volume_id:
                raise ValueError(f"Volume identity mismatch for {row.path}")
            if volume_id in geometry and geometry[volume_id] != (rows, columns):
                raise ValueError(f"Inconsistent geometry within {volume_id}")
            geometry[volume_id] = (rows, columns)

            pixels = np.asarray(dataset.pixel_array)
            if pixels.shape != (rows, columns) or not np.isfinite(pixels).all():
                raise ValueError(f"Invalid decoded pixels for {row.path}: shape={pixels.shape}")
            audited_rows.append(
                {
                    "file": volume_id,
                    "slice": int(row.slice),
                    "path": row.path,
                    "rows": rows,
                    "columns": columns,
                    "instance_number": instance,
                    "images_in_acquisition": images,
                    "transfer_syntax_uid": transfer_syntax,
                    "pixel_min": float(pixels.min()),
                    "pixel_max": float(pixels.max()),
                }
            )
    return pd.DataFrame(audited_rows), geometry


def sanitize_boxes(
    annotations: pd.DataFrame, selected: set[str], geometry: dict[str, tuple[int, int]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    boxes = annotations.loc[
        annotations["study_level"].eq("No") & annotations["file"].isin(selected)
    ].copy()
    for annotation_index, row in boxes.iterrows():
        volume_id = str(row["file"])
        slice_index = int(row["slice"])
        image_rows, image_columns = geometry[volume_id]
        x = float(row["x"])
        y = float(row["y"])
        width = float(row["width"])
        height = float(row["height"])
        x1 = min(max(x, 0.0), float(image_columns))
        y1 = min(max(y, 0.0), float(image_rows))
        x2 = min(max(x + width, 0.0), float(image_columns))
        y2 = min(max(y + height, 0.0), float(image_rows))
        kept = x2 > x1 and y2 > y1
        label_clean = str(row["label"]).strip()
        rows.append(
            {
                "annotation_row": int(annotation_index),
                "file": volume_id,
                "slice": slice_index,
                "label_raw": row["label"],
                "label_clean": label_clean,
                "x_raw": x,
                "y_raw": y,
                "width_raw": width,
                "height_raw": height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "kept": kept,
                "coordinates_adjusted": not (
                    math.isclose(x1, x)
                    and math.isclose(y1, y)
                    and math.isclose(x2, x + width)
                    and math.isclose(y2, y + height)
                ),
                "label_whitespace_adjusted": label_clean != str(row["label"]),
            }
        )
    sanitation = pd.DataFrame(rows)
    if sanitation.empty:
        raise ValueError("The selected positive cohort has no slice-level boxes")
    return sanitation


def render_overlays(
    dicom_root: Path,
    dicom_index: pd.DataFrame,
    sanitation: pd.DataFrame,
    output_dir: Path,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    kept = sanitation.loc[sanitation["kept"]].copy()
    identities = [f"{row.file}:{row.slice}" for row in kept.itertuples(index=False)]
    selected_identities = stable_sort(sorted(set(identities)), seed, "overlay")[:count]
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    path_lookup = {
        (row.file, int(row.slice)): row.path for row in dicom_index.itertuples(index=False)
    }
    for identity in selected_identities:
        volume_id, raw_slice = identity.split(":")
        slice_index = int(raw_slice)
        source_path = dicom_root / path_lookup[(volume_id, slice_index)]
        pixels = np.asarray(pydicom.dcmread(source_path).pixel_array, dtype=np.float32)
        low, high = np.percentile(pixels, [0.5, 99.5])
        if high <= low:
            normalized = np.zeros_like(pixels, dtype=np.uint8)
        else:
            normalized = np.clip((pixels - low) / (high - low), 0.0, 1.0)
            normalized = np.round(normalized * 255.0).astype(np.uint8)
        slice_boxes = kept.loc[
            kept["file"].eq(volume_id) & kept["slice"].eq(slice_index)
        ]

        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(normalized, cmap="gray", vmin=0, vmax=255)
        for box in slice_boxes.itertuples(index=False):
            axis.add_patch(
                Rectangle(
                    (box.x1, box.y1),
                    box.x2 - box.x1,
                    box.y2 - box.y1,
                    linewidth=1.5,
                    edgecolor="lime",
                    facecolor="none",
                )
            )
            axis.text(
                box.x1,
                max(0.0, box.y1 - 3.0),
                box.label_clean,
                color="yellow",
                fontsize=6,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
            )
        axis.set_title(f"{volume_id} slice {slice_index} — released DICOM frame")
        axis.axis("off")
        output_path = overlay_dir / f"{volume_id}_{slice_index:03d}.png"
        figure.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        records.append(
            {
                "file": volume_id,
                "slice": slice_index,
                "box_count": int(len(slice_boxes)),
                "path": output_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(output_path),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations, reviewed = load_release(args.annotations, args.reviewed_files)
    path_index, malformed = index_dicom_paths(args.dicom_root)
    available = set(path_index["file"]) & reviewed
    annotation_volumes = set(
        annotations.loc[annotations["study_level"].eq("No"), "file"].astype(str)
    )
    cohort = select_smoke_cohort(
        available,
        annotation_volumes,
        args.smoke_volumes,
        args.positive_fraction,
        args.validation_fraction,
        args.seed,
    )
    selected = set(cohort["file"])
    selected_index = path_index.loc[path_index["file"].isin(selected)].copy()
    decoded_index, geometry = validate_and_decode(args.dicom_root, selected_index)

    sanitation = sanitize_boxes(annotations, selected, geometry)
    known_slices = set(zip(decoded_index["file"], decoded_index["slice"], strict=True))
    missing_box_slices = [
        (row.file, int(row.slice), int(row.annotation_row))
        for row in sanitation.itertuples(index=False)
        if (row.file, int(row.slice)) not in known_slices
    ]
    if missing_box_slices:
        raise ValueError(f"Annotations reference absent DICOM slices: {missing_box_slices[:10]}")

    cohort_path = args.output_dir / "p6_smoke_volumes.csv"
    index_path = args.output_dir / "p6_dicom_index.csv"
    sanitation_path = args.output_dir / "p6_box_sanitation.csv"
    cohort.to_csv(cohort_path, index=False)
    decoded_index.to_csv(index_path, index=False)
    sanitation.to_csv(sanitation_path, index=False)
    overlays = render_overlays(
        args.dicom_root,
        decoded_index,
        sanitation,
        args.output_dir,
        args.overlay_count,
        args.seed,
    )

    manifest = {
        "schema_version": 1,
        "experiment_id": "P6B-FASTMRI-PLUS-DICOM-SMOKE-v1",
        "status": "validated",
        "source": {
            "fastmri_plus_commit": SOURCE_COMMIT,
            "annotations_sha256": ANNOTATION_SHA256,
            "reviewed_files_sha256": REVIEWED_SHA256,
            "dicom_frame": "official released DICOM; labeling-time up/down flip already applied",
            "additional_flip_applied": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pydicom": pydicom.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "archive_inventory": {
            "dicom_files": int(len(path_index)),
            "archive_volumes": int(path_index["file"].nunique()),
            "malformed_dicom_filenames": malformed,
            "reviewed_volumes_available": int(len(available)),
            "unreviewed_archive_volumes_excluded": int(
                len(set(path_index["file"]) - reviewed)
            ),
        },
        "smoke": {
            "seed": args.seed,
            "volumes": int(len(cohort)),
            "annotated_volumes": int(cohort["has_annotation"].sum()),
            "reviewed_negative_volumes": int((~cohort["has_annotation"]).sum()),
            "train_volumes": int(cohort["split"].eq("train").sum()),
            "validation_volumes": int(cohort["split"].eq("validation").sum()),
            "train_validation_overlap": int(
                len(
                    set(cohort.loc[cohort["split"].eq("train"), "file"])
                    & set(cohort.loc[cohort["split"].eq("validation"), "file"])
                )
            ),
            "decoded_slices": int(len(decoded_index)),
            "decoded_pixel_failures": 0,
            "selection_signature": sha256_jsonable(cohort.to_dict("records")),
        },
        "sanitation": {
            "selected_raw_boxes": int(len(sanitation)),
            "kept_boxes": int(sanitation["kept"].sum()),
            "discarded_zero_area_after_clamp": int((~sanitation["kept"]).sum()),
            "coordinate_adjustments": int(sanitation["coordinates_adjusted"].sum()),
            "label_whitespace_adjustments": int(
                sanitation["label_whitespace_adjusted"].sum()
            ),
            "missing_box_slices": 0,
        },
        "overlays": overlays,
        "artifacts": {
            "cohort_sha256": sha256_file(cohort_path),
            "dicom_index_sha256": sha256_file(index_path),
            "box_sanitation_sha256": sha256_file(sanitation_path),
        },
    }
    manifest_path = args.output_dir / "p6_dicom_smoke.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
