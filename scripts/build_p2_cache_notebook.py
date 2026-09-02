#!/usr/bin/env python3
"""Build one partition of the frozen P2-CACHE-v1 Kaggle notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("a", "b"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"output/jupyter-notebook/rsna-p2-wide-cache-{args.part}.ipynb"
    )
    start, stop = (0, 2204) if args.part == "a" else (2204, 4407)

    cells = [
        markdown(
            f"""
# RSNA P2-CACHE-v1 — wide physical cache, part {args.part.upper()}

**Objective.** Materialize rows `{start}:{stop}` of the frozen P2 pixel contract
without training a model. The cache uses five acquisition slots, 64 total slices,
2–98% volume coverage, a 140 mm physical crop, and 336 px uint8 storage.

The concatenated part A/B study order must exactly match `train.csv`. Each output
contains its IDs, validity mask, slot bounds, failures, and a cryptographic manifest.
No expert label is read by this notebook. Downstream 2.5D windows are allowed only
within a slot, never across a slot boundary.
"""
        ),
        code(
            f"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for candidate in (Path.cwd() / ".deps", Path.cwd().parent / ".deps", Path.cwd().parent.parent / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

PART = "{args.part}"
ROW_START = {start}
ROW_STOP = {stop}
ID = "StudyInstanceUID"
CACHE_ID = "P2-CACHE-v1"
SIZE = 336
CROP_MM = 140.0
LOW_PERCENTILE = 2.0
HIGH_PERCENTILE = 98.0
ON_KAGGLE = Path("/kaggle/working").is_dir()
FULL_RUN = ON_KAGGLE
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(
    f"outputs/p2_cache_smoke_{{PART}}"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SLOT_SPECS = [
    {{"name": "sagittal_fluid", "plane": "Sagittal", "kind": "fluid", "count": 18}},
    {{"name": "sagittal_structural", "plane": "Sagittal", "kind": "structural", "count": 14}},
    {{"name": "coronal_fluid", "plane": "Coronal", "kind": "fluid", "count": 12}},
    {{"name": "coronal_structural", "plane": "Coronal", "kind": "structural", "count": 8}},
    {{"name": "axial_any", "plane": "Axial", "kind": "any", "count": 12}},
]
assert sum(slot["count"] for slot in SLOT_SPECS) == 64

slot_bounds = []
cursor = 0
for slot in SLOT_SPECS:
    slot_bounds.append({{"name": slot["name"], "start": cursor, "stop": cursor + slot["count"]}})
    cursor += slot["count"]

print({{
    "part": PART, "rows": [ROW_START, ROW_STOP], "full_run": FULL_RUN,
    "python": platform.python_version(), "numpy": np.__version__,
    "pandas": pd.__version__, "pydicom": pydicom.__version__,
    "output_root": str(OUTPUT_ROOT),
}})
"""
        ),
        markdown("## Resolve the competition metadata and DICOM mount"),
        code(
            """
COMPETITION_SLUG = "rsna-knee-abnormality-detection"


def candidate_roots() -> list[Path]:
    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    if Path("/kaggle/input").is_dir():
        roots.append(Path("/kaggle/input"))
    return roots


def find_metadata_root() -> Path:
    candidates = [
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
    ]
    for root in candidate_roots():
        candidates.extend([root / "data/raw", root])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for pattern in ("*/train_series.csv", "*/*/train_series.csv"):
            candidates.extend(path.parent for path in kaggle_input.glob(pattern))
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series.csv").is_file():
            return path.resolve()
    raise FileNotFoundError("Could not locate competition metadata")


DATA_ROOT = find_metadata_root()
local_dicom = next(
    (root / "data/sample/files/train_series" for root in candidate_roots()
     if (root / "data/sample/files/train_series").is_dir()),
    None,
)
DICOM_ROOT = (DATA_ROOT / "train_series") if (DATA_ROOT / "train_series").is_dir() else local_dicom
assert DICOM_ROOT is not None and DICOM_ROOT.is_dir()

train = pd.read_csv(DATA_ROOT / "train.csv")
series = pd.read_csv(DATA_ROOT / "train_series.csv")
assert len(train) == 4_407 and len(series) == 24_371 and train[ID].is_unique
all_study_ids = train[ID].astype(str).tolist()
if FULL_RUN:
    study_ids = all_study_ids[ROW_START:ROW_STOP]
else:
    available = {path.name for path in DICOM_ROOT.iterdir() if path.is_dir()}
    study_ids = [uid for uid in all_study_ids if uid in available]
    assert study_ids
    # The richest local sample exercises all five selection branches.
    study_ids = [max(
        study_ids,
        key=lambda uid: sum(1 for _ in (DICOM_ROOT / uid).glob("*/*.dcm")),
    )]

series = series[series[ID].astype(str).isin(study_ids)].copy()
series_by_study = {uid: group.to_dict("records") for uid, group in series.groupby(ID, sort=False)}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


ORDER_SHA256 = sha256_bytes(("\\n".join(all_study_ids) + "\\n").encode())
PART_ORDER_SHA256 = sha256_bytes(("\\n".join(study_ids) + "\\n").encode())
TRAIN_CSV_SHA256 = sha256_file(DATA_ROOT / "train.csv")
SERIES_CSV_SHA256 = sha256_file(DATA_ROOT / "train_series.csv")
print({
    "data_root": str(DATA_ROOT), "dicom_root": str(DICOM_ROOT),
    "studies": len(study_ids), "order_sha256": ORDER_SHA256,
})
"""
        ),
        markdown("## Deterministic series selection and physical preprocessing"),
        code(
            """
HEADER_TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber"]


def spatial_key(path: Path) -> tuple[float, int, str]:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)
        orientation = np.asarray(ds.get("ImageOrientationPatient", []), dtype=float)
        position = np.asarray(ds.get("ImagePositionPatient", []), dtype=float)
        instance = int(ds.get("InstanceNumber", 0) or 0)
        if orientation.size == 6 and position.size == 3:
            location = float(np.dot(position, np.cross(orientation[:3], orientation[3:])))
            return (location, instance, path.name)
        return (float(instance), instance, path.name)
    except Exception:
        return (0.0, 0, path.name)


def ordered_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.glob("*.dcm"), key=spatial_key)


def sequence_score(record: dict, kind: str) -> int:
    if kind == "any":
        return 0
    fluid = int(record["Fluid_Sensitive"])
    fat = int(record["Fat_Suppression"])
    if kind == "fluid":
        return 3 if fluid == 1 and fat == 1 else 2 if fluid == 1 else 1 if fat == 1 else 0
    return 3 if fluid == 0 and fat == 0 else 2 if fluid == 0 else 1 if fat == 0 else 0


def choose_series(
    study_uid: str, records: list[dict], slot: dict, used: set[str]
) -> tuple[Path | None, str | None]:
    candidates = []
    for record in records:
        series_uid = str(record["SeriesInstanceUID"])
        if series_uid in used or str(record["Anatomical_Plane"]) != slot["plane"]:
            continue
        path = DICOM_ROOT / study_uid / series_uid
        count = sum(1 for _ in path.glob("*.dcm")) if path.is_dir() else 0
        if count:
            candidates.append((sequence_score(record, slot["kind"]), count, series_uid, path))
    if not candidates:
        return None, None
    _, _, series_uid, path = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    used.add(series_uid)
    return path, series_uid


def decode_raw(path: Path) -> tuple[np.ndarray, tuple[float, float]] | None:
    try:
        ds = pydicom.dcmread(path)
        array = ds.pixel_array.astype(np.float32)
        array = array * float(ds.get("RescaleSlope", 1.0)) + float(ds.get("RescaleIntercept", 0.0))
        if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            array = array.max() + array.min() - array
        spacing = ds.get("PixelSpacing", [CROP_MM / max(array.shape)] * 2)
        sy, sx = float(spacing[0]), float(spacing[1])
        if not np.isfinite(array).any() or sy <= 0 or sx <= 0:
            return None
        return array, (sy, sx)
    except Exception:
        return None


def crop_resize_uint8(
    array: np.ndarray, spacing: tuple[float, float], lo: float, hi: float
) -> np.ndarray | None:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    sy, sx = spacing
    crop_h = min(array.shape[0], max(16, int(round(CROP_MM / sy))))
    crop_w = min(array.shape[1], max(16, int(round(CROP_MM / sx))))
    cy, cx = array.shape[0] // 2, array.shape[1] // 2
    y0 = min(max(0, cy - crop_h // 2), array.shape[0] - crop_h)
    x0 = min(max(0, cx - crop_w // 2), array.shape[1] - crop_w)
    crop = array[y0:y0 + crop_h, x0:x0 + crop_w]
    scaled = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    image = Image.fromarray(np.round(scaled * 255.0).astype(np.uint8))
    return np.asarray(image.resize((SIZE, SIZE), resample=Image.Resampling.BILINEAR))


def build_slot(series_dir: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    files = ordered_files(series_dir)
    images = np.zeros((count, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros(count, dtype=np.uint8)
    if not files:
        return images, valid
    centers = np.round(np.linspace(0.02, 0.98, count) * (len(files) - 1)).astype(int)
    decoded = [decode_raw(files[int(index)]) for index in centers]
    finite_chunks = [item[0][np.isfinite(item[0])] for item in decoded if item is not None]
    if not finite_chunks:
        return images, valid
    finite = np.concatenate(finite_chunks)
    lo, hi = np.percentile(finite, [LOW_PERCENTILE, HIGH_PERCENTILE])
    for out_index, item in enumerate(decoded):
        if item is None:
            continue
        processed = crop_resize_uint8(item[0], item[1], float(lo), float(hi))
        if processed is not None:
            images[out_index] = processed
            valid[out_index] = 1
    return images, valid


def build_study(study_uid: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    images = np.zeros((64, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros(64, dtype=np.uint8)
    selections = []
    records = series_by_study.get(study_uid, [])
    used: set[str] = set()
    cursor = 0
    for slot in SLOT_SPECS:
        path, series_uid = choose_series(study_uid, records, slot, used)
        stop = cursor + slot["count"]
        if path is not None:
            images[cursor:stop], valid[cursor:stop] = build_slot(path, slot["count"])
        selections.append({
            "slot": slot["name"], "series_uid": series_uid,
            "valid_slices": int(valid[cursor:stop].sum()),
        })
        cursor = stop
    return images, valid, selections
"""
        ),
        markdown("## Smoke check, partition cache, and milestones"),
        code(
            """
sample_images, sample_valid, sample_selections = build_study(study_ids[0])
assert sample_images.shape == (64, SIZE, SIZE)
assert sample_valid.shape == (64,) and sample_valid.sum() > 0
assert all(
    not (bound["start"] == 0 and bound["stop"] == 0) for bound in slot_bounds
)
print({
    "sample_uid": study_ids[0], "valid_slices": int(sample_valid.sum()),
    "selections": sample_selections,
})

pixel_path = OUTPUT_ROOT / f"p2_pixels_part_{PART}.npy"
pixel_cache = np.lib.format.open_memmap(
    pixel_path, mode="w+", dtype=np.uint8, shape=(len(study_ids), 64, SIZE, SIZE)
)
valid_mask = np.zeros((len(study_ids), 64), dtype=np.uint8)
failures = []
selection_rows = []
started = time.time()

for index, uid in enumerate(study_ids):
    try:
        images, valid, selections = build_study(uid)
        pixel_cache[index] = images
        valid_mask[index] = valid
        for row in selections:
            selection_rows.append({ID: uid, **row})
        if valid.sum() == 0:
            failures.append({ID: uid, "reason": "no_decodable_slices"})
    except Exception as exc:
        pixel_cache[index] = 0
        failures.append({ID: uid, "reason": f"{type(exc).__name__}: {exc}"})
    interval = 100 if FULL_RUN else 1
    if (index + 1) % interval == 0 or index + 1 == len(study_ids):
        pixel_cache.flush()
        print(
            f"P2-CACHE-{PART.upper()} cached {index + 1:,}/{len(study_ids):,} "
            f"studies in {(time.time() - started) / 60:.1f} min; failures={len(failures)}",
            flush=True,
        )

usable = valid_mask.any(axis=1)
assert usable.all(), f"{int((~usable).sum())} studies have no usable slices"
if FULL_RUN:
    assert len(study_ids) == ROW_STOP - ROW_START
    assert study_ids == all_study_ids[ROW_START:ROW_STOP]

np.save(OUTPUT_ROOT / f"p2_valid_mask_part_{PART}.npy", valid_mask)
np.save(OUTPUT_ROOT / "p2_slot_bounds.npy", np.array(
    [[bound["start"], bound["stop"]] for bound in slot_bounds], dtype=np.int16
))
pd.DataFrame({ID: study_ids}).to_csv(
    OUTPUT_ROOT / f"p2_studies_part_{PART}.csv", index=False
)
pd.DataFrame(failures, columns=[ID, "reason"]).to_csv(
    OUTPUT_ROOT / f"p2_failures_part_{PART}.csv", index=False
)
pd.DataFrame(selection_rows).to_csv(
    OUTPUT_ROOT / f"p2_selections_part_{PART}.csv", index=False
)
print({
    "usable": int(usable.sum()), "pixel_gib": round(pixel_path.stat().st_size / 2**30, 3),
    "mean_valid_slices": float(valid_mask.sum(axis=1).mean()),
})
"""
        ),
        markdown("## Cryptographic manifest and cache signature"),
        code(
            """
contract = {
    "cache_id": CACHE_ID,
    "competition": COMPETITION_SLUG,
    "study_order_sha256": ORDER_SHA256,
    "train_csv_sha256": TRAIN_CSV_SHA256,
    "train_series_csv_sha256": SERIES_CSV_SHA256,
    "slot_specs": SLOT_SPECS,
    "slot_bounds": slot_bounds,
    "relative_slice_range": [0.02, 0.98],
    "physical_crop_mm": CROP_MM,
    "cache_image_size": SIZE,
    "intensity": {
        "rescale_slope_intercept": True,
        "monochrome1_inversion": True,
        "normalization_scope": "selected_series_sampled_slices",
        "percentiles": [LOW_PERCENTILE, HIGH_PERCENTILE],
    },
    "resize": "PIL bilinear",
    "window_policy": "three adjacent valid slices within slot only",
}
contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
cache_signature = sha256_bytes(contract_json.encode())

pixel_cache.flush()
files = {
    f"p2_pixels_part_{PART}.npy": sha256_file(pixel_path),
    f"p2_valid_mask_part_{PART}.npy": sha256_file(OUTPUT_ROOT / f"p2_valid_mask_part_{PART}.npy"),
    f"p2_studies_part_{PART}.csv": sha256_file(OUTPUT_ROOT / f"p2_studies_part_{PART}.csv"),
    "p2_slot_bounds.npy": sha256_file(OUTPUT_ROOT / "p2_slot_bounds.npy"),
}
manifest = {
    "schema_version": 1,
    "cache_id": CACHE_ID,
    "cache_signature_sha256": cache_signature,
    "part": PART,
    "full_run": FULL_RUN,
    "row_range": [ROW_START, ROW_STOP] if FULL_RUN else None,
    "studies": len(study_ids),
    "part_order_sha256": PART_ORDER_SHA256,
    "array_shape": list(pixel_cache.shape),
    "array_dtype": str(pixel_cache.dtype),
    "usable_studies": int(usable.sum()),
    "failures": len(failures),
    "mean_valid_slices": float(valid_mask.sum(axis=1).mean()),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "contract": contract,
    "files_sha256": files,
    "run_context": {"data_root": str(DATA_ROOT), "dicom_root": str(DICOM_ROOT)},
}
manifest_path = OUTPUT_ROOT / f"p2_cache_manifest_part_{PART}.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n")

assert manifest["cache_signature_sha256"] == sha256_bytes(contract_json.encode())
assert all(len(value) == 64 for value in manifest["files_sha256"].values())
print({
    "P2_CACHE_COMPLETE": PART.upper(),
    "cache_signature_sha256": cache_signature,
    "pixel_sha256": files[f"p2_pixels_part_{PART}.npy"],
    "elapsed_minutes": round(manifest["elapsed_minutes"], 2),
})
manifest
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
