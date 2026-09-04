#!/usr/bin/env python3
"""Validate a deterministic fastMRI+ HDF5 smoke cohort for P6-B.

fastMRI+ labels refer to ``file*.h5`` volumes and were drawn on
``reconstruction_rss`` after an up/down flip. This script reproduces that exact
frame, validates every selected volume, sanitizes the released boxes, and writes
private overlay artifacts to an ignored output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


ANNOTATION_SHA256 = "c1f4a083646cec815b67f005f8d7764313d1b82a028b90c272a4d000330d065b"
REVIEWED_SHA256 = "4b09e5523709815d5314a4f652882c40f8c14af26b67a00ea769054ca7e1d6b7"
SOURCE_COMMIT = "67ed9a68ee33d3197f93893e8125688470c4a577"
FILE_RE = re.compile(r"^(file\d+)\.h5$", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-root", type=Path, required=True)
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


def index_hdf5_paths(hdf5_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    malformed: list[str] = []
    for path in sorted(hdf5_root.rglob("*.h5")):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            malformed.append(path.relative_to(hdf5_root).as_posix())
            continue
        rows.append(
            {
                "file": match.group(1),
                "path": path.relative_to(hdf5_root).as_posix(),
                "file_bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise ValueError(f"No file-ID HDF5 volumes found below {hdf5_root}")
    index = pd.DataFrame(rows).sort_values(["file", "path"]).reset_index(drop=True)
    duplicated = index["file"].duplicated(keep=False)
    if duplicated.any():
        examples = index.loc[duplicated, ["file", "path"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate fastMRI file identities: {examples}")
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
    hdf5_root: Path, selected_index: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[int, int, int]]]:
    audited_rows: list[dict[str, object]] = []
    geometry: dict[str, tuple[int, int, int]] = {}
    for row in selected_index.itertuples(index=False):
        path = hdf5_root / row.path
        with h5py.File(path, "r") as handle:
            required = {"kspace", "reconstruction_rss", "reconstruction_esc", "ismrmrd_header"}
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise ValueError(f"Missing HDF5 fields in {row.path}: {missing}")
            reconstruction = handle["reconstruction_rss"]
            if reconstruction.ndim != 3:
                raise ValueError(
                    f"Unexpected reconstruction_rss shape in {row.path}: {reconstruction.shape}"
                )
            depth, image_rows, image_columns = map(int, reconstruction.shape)
            pixels = np.asarray(reconstruction[:])
            if pixels.shape != (depth, image_rows, image_columns) or not np.isfinite(pixels).all():
                raise ValueError(f"Invalid reconstruction_rss pixels in {row.path}")
            # This is the exact transform in the pinned fastMRI+ example notebook.
            label_frame = pixels[:, ::-1, :]
            if not np.isfinite(label_frame).all():
                raise ValueError(f"Invalid pixels after labeling-frame flip in {row.path}")
            geometry[row.file] = (depth, image_rows, image_columns)
            audited_rows.append(
                {
                    "file": row.file,
                    "path": row.path,
                    "file_bytes": int(row.file_bytes),
                    "slices": depth,
                    "rows": image_rows,
                    "columns": image_columns,
                    "reconstruction_dtype": str(reconstruction.dtype),
                    "pixel_min": float(label_frame.min()),
                    "pixel_max": float(label_frame.max()),
                    "kspace_shape": json.dumps(list(map(int, handle["kspace"].shape))),
                }
            )
    return pd.DataFrame(audited_rows), geometry


def sanitize_boxes(
    annotations: pd.DataFrame,
    selected: set[str],
    geometry: dict[str, tuple[int, int, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    boxes = annotations.loc[
        annotations["study_level"].eq("No") & annotations["file"].isin(selected)
    ].copy()
    for annotation_index, row in boxes.iterrows():
        volume_id = str(row["file"])
        slice_index = int(row["slice"])
        depth, image_rows, image_columns = geometry[volume_id]
        if not 0 <= slice_index < depth:
            raise ValueError(
                f"Annotation row {annotation_index} references slice {slice_index} "
                f"outside depth {depth} for {volume_id}"
            )
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
    hdf5_root: Path,
    volume_index: pd.DataFrame,
    sanitation: pd.DataFrame,
    output_dir: Path,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    kept = sanitation.loc[sanitation["kept"]].copy()
    identities = [f"{row.file}:{row.slice}" for row in kept.itertuples(index=False)]
    selected_identities = stable_sort(sorted(set(identities)), seed, "overlay")[:count]
    path_lookup = {row.file: row.path for row in volume_index.itertuples(index=False)}
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for identity in selected_identities:
        volume_id, raw_slice = identity.split(":")
        slice_index = int(raw_slice)
        with h5py.File(hdf5_root / path_lookup[volume_id], "r") as handle:
            pixels = np.asarray(
                handle["reconstruction_rss"][slice_index, ::-1, :], dtype=np.float32
            )
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
        axis.set_title(f"{volume_id} slice {slice_index} — fastMRI+ labeling frame")
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
    path_index, malformed = index_hdf5_paths(args.hdf5_root)
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
    audited_index, geometry = validate_and_decode(args.hdf5_root, selected_index)
    sanitation = sanitize_boxes(annotations, selected, geometry)

    cohort_path = args.output_dir / "p6_smoke_volumes.csv"
    index_path = args.output_dir / "p6_hdf5_index.csv"
    sanitation_path = args.output_dir / "p6_box_sanitation.csv"
    cohort.to_csv(cohort_path, index=False)
    audited_index.to_csv(index_path, index=False)
    sanitation.to_csv(sanitation_path, index=False)
    overlays = render_overlays(
        args.hdf5_root,
        selected_index,
        sanitation,
        args.output_dir,
        args.overlay_count,
        args.seed,
    )

    manifest = {
        "schema_version": 1,
        "experiment_id": "P6B-FASTMRI-PLUS-HDF5-SMOKE-v1",
        "status": "validated",
        "source": {
            "fastmri_plus_commit": SOURCE_COMMIT,
            "annotations_sha256": ANNOTATION_SHA256,
            "reviewed_files_sha256": REVIEWED_SHA256,
            "image_field": "reconstruction_rss",
            "labeling_frame_transform": "pixels[:, ::-1, :]",
            "flip_applied_exactly_once": True,
        },
        "environment": {
            "python": platform.python_version(),
            "h5py": h5py.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "archive_inventory": {
            "hdf5_files": int(len(path_index)),
            "malformed_hdf5_filenames": malformed,
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
            "decoded_volumes": int(len(audited_index)),
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
            "hdf5_index_sha256": sha256_file(index_path),
            "box_sanitation_sha256": sha256_file(sanitation_path),
        },
    }
    manifest_path = args.output_dir / "p6_hdf5_smoke.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
