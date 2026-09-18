"""P14 native-triplet cache with physical in-plane orientation and provenance."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .contracts import ID, VERSION, sha256, write_json

SLOTS = [("Sagittal", "fluid", 9, 18), ("Sagittal", "structural", 7, 14),
         ("Coronal", "fluid", 6, 12), ("Coronal", "structural", 4, 8), ("Axial", "any", 6, 12)]
AXES = {"Sagittal": (np.array([0., 1, 0]), np.array([0., 0, 1]), 0),
        "Coronal": (np.array([1., 0, 0]), np.array([0., 0, 1]), 1),
        "Axial": (np.array([1., 0, 0]), np.array([0., 1, 0]), 2)}


def canonical_basis(orientation, plane):
    o = np.asarray(orientation, float)
    if o.shape != (6,) or not np.isfinite(o).all():
        raise ValueError("Invalid ImageOrientationPatient")
    col, row = o[:3], o[3:]
    if abs(np.linalg.norm(col) - 1) > .01 or abs(np.linalg.norm(row) - 1) > .01 or abs(col @ row) > .01:
        raise ValueError("Non-orthonormal DICOM orientation")
    normal = np.cross(col, row)
    u0, v0, axis = AXES[plane]
    if abs(normal[axis]) < .85:
        raise ValueError("DICOM orientation disagrees with plane annotation")
    u = u0 - normal * (u0 @ normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    if v @ v0 < 0:
        v = -v
    return col, row, normal, u, v


def orient_crop(image, spacing, orientation, plane, size, lo, hi, crop_mm=140.):
    col, row, _, u, v = canonical_basis(orientation, plane)
    sy, sx = np.asarray(spacing, float)
    if not np.isfinite([sy, sx, lo, hi]).all() or min(sy, sx) <= 0 or hi <= lo:
        raise ValueError("Invalid spacing/intensity window")
    grid = (np.arange(size) + .5 - size / 2) * crop_mm / size
    xx, yy = np.meshgrid(grid, grid)
    cc = (image.shape[1] - 1) / 2 + (xx * (u @ col) + yy * (v @ col)) / sx
    rr = (image.shape[0] - 1) / 2 + (xx * (u @ row) + yy * (v @ row)) / sy
    sampled = map_coordinates(image.astype(np.float32), [rr, cc], order=1, mode="constant", cval=float(lo))
    return np.round(np.clip((sampled - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def native_indices(length, count):
    if length < 3:
        return np.empty((0, 3), int)
    center = np.clip(np.round(np.linspace(.02, .98, count) * (length - 1)).astype(int), 1, length - 2)
    return center[:, None] + np.array([-1, 0, 1])


def series_score(record, kind):
    fluid, fat = int(record["Fluid_Sensitive"]), int(record["Fat_Suppression"])
    if kind == "any":
        return 0
    if kind == "fluid":
        return 3 if fluid and fat else 2 if fluid else 1 if fat else 0
    return 3 if not fluid and not fat else 2 if not fluid else 1 if not fat else 0


def build_study(uid, records, dicom_root, size=336, axial_policy="longest"):
    import pydicom
    images = np.zeros((32, 3, size, size), np.uint8)
    mask = np.zeros(32, bool)
    slots = np.full(32, -1, np.int16)
    relative = np.zeros(32, np.float32)
    provenance, failures, used = [], [], set()
    cursor = 0
    for slot, (plane, kind, count, old_count) in enumerate(SLOTS):
        if plane == "Axial" and axial_policy == "fluid":
            kind = "fluid"
        candidates = []
        for record in records:
            sid = str(record["SeriesInstanceUID"])
            if record["Anatomical_Plane"] != plane or sid in used:
                continue
            files = list((Path(dicom_root) / uid / sid).glob("*.dcm"))
            if files:
                candidates.append((series_score(record, kind), len(files), sid, files))
        for _, _, sid, files in sorted(candidates, key=lambda v: v[:3], reverse=True):
            try:
                headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in files]
                _, _, normal, _, _ = canonical_basis(headers[0].ImageOrientationPatient, plane)
                # Channel order and normalized position always increase along the
                # same patient axis, even if an acquisition reverses its normal.
                if normal[AXES[plane][2]] < 0:
                    normal = -normal
                locations = []
                for h in headers:
                    canonical_basis(h.ImageOrientationPatient, plane)
                    if not np.allclose(h.ImageOrientationPatient, headers[0].ImageOrientationPatient, atol=.001):
                        raise ValueError("Orientation changes within series")
                    locations.append(float(np.asarray(h.ImagePositionPatient) @ normal))
                order = np.argsort(locations, kind="stable")
                files = [files[i] for i in order]
                locations = np.array(locations)[order]
                gaps = np.diff(locations)
                if len(gaps) < 2 or (gaps <= 0).any():
                    raise ValueError("Duplicate or missing slice positions")
                triplets = native_indices(len(files), count)
                norm_ix = np.round(np.linspace(.02, .98, old_count) * (len(files) - 1)).astype(int)
                decoded = {}
                for i in np.unique(np.r_[triplets.ravel(), norm_ix]):
                    ds = pydicom.dcmread(files[i])
                    a = ds.pixel_array.astype(np.float32)
                    a = a * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
                    if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
                        a = a.max() + a.min() - a
                    if a.ndim != 2 or not np.isfinite(a).all():
                        raise ValueError("Invalid decoded slice")
                    decoded[int(i)] = (a, ds)
                lo, hi = np.percentile(np.concatenate([decoded[int(i)][0].ravel() for i in norm_ix]), [2, 98])
                for j, triple in enumerate(triplets):
                    for channel, i in enumerate(triple):
                        a, ds = decoded[int(i)]
                        images[cursor + j, channel] = orient_crop(a, ds.PixelSpacing, ds.ImageOrientationPatient, plane, size, lo, hi)
                    slots[cursor + j], mask[cursor + j] = slot, True
                    relative[cursor + j] = 2 * triple[1] / (len(files) - 1) - 1
                    provenance.append({ID: uid, "window": cursor + j, "slot": slot, "plane": plane,
                        "series_uid": sid, "native_indices": triple.tolist(),
                        "sop_uids": [str(decoded[int(i)][1].SOPInstanceUID) for i in triple],
                        "slice_positions_mm": locations[triple].tolist(),
                        "adjacent_gaps_mm": np.diff(locations[triple]).tolist(),
                        "orientation": list(map(float, headers[0].ImageOrientationPatient)),
                        "pixel_spacing": list(map(float, decoded[int(triple[1])][1].PixelSpacing)),
                        "intensity_window": [float(lo), float(hi)]})
                used.add(sid)
                break
            except Exception as error:
                images[cursor:cursor + count] = 0
                mask[cursor:cursor + count] = False
                slots[cursor:cursor + count] = -1
                provenance = [r for r in provenance if r["window"] < cursor]
                failures.append({ID: uid, "series_uid": sid, "reason": str(error)})
        cursor += count
    return images, mask, slots, relative, provenance, failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--dicom-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--size", type=int, choices=[336, 384], default=336)
    p.add_argument("--axial-policy", choices=["longest", "fluid"], default="longest")
    args = p.parse_args()
    if not 0 <= args.part < args.parts:
        raise ValueError("Invalid cache partition")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Cache output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    meta, series = pd.read_csv(args.train), pd.read_csv(args.series)
    indices = np.array_split(np.arange(len(meta)), args.parts)[args.part]
    ids = meta.iloc[indices][ID].tolist()
    records = {uid: d.to_dict("records") for uid, d in series.groupby(ID)}
    volume = np.lib.format.open_memmap(args.output / "pixels.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(ids), 32, 3, args.size, args.size))
    masks = np.zeros((len(ids), 32), bool)
    slots, positions = np.full((len(ids), 32), -1, np.int16), np.zeros((len(ids), 32), np.float32)
    failures = []
    started = time.time()
    with (args.output / "geometry.jsonl").open("w") as f:
        for i, uid in enumerate(ids):
            x, masks[i], slots[i], positions[i], rows, errors = build_study(uid, records[uid], args.dicom_root, args.size, args.axial_policy)
            volume[i] = x
            for row in rows:
                f.write(json.dumps(row) + "\n")
            failures.extend(errors)
            if (i + 1) % 25 == 0:
                volume.flush()
                print(json.dumps({"studies": i + 1, "minutes": (time.time() - started) / 60}), flush=True)
    volume.flush()
    np.save(args.output / "mask.npy", masks)
    np.save(args.output / "slots.npy", slots)
    np.save(args.output / "positions.npy", positions)
    pd.DataFrame({ID: ids}).to_csv(args.output / "studies.csv", index=False)
    write_json(args.output / "failures.json", failures)
    contract = {"schema_version": VERSION, "cache_id": "P14-native-triplets-v1", "size": args.size,
                "crop_mm": 140, "axial_policy": args.axial_policy, "train_sha256": sha256(args.train),
                "series_sha256": sha256(args.series), "code_sha256": sha256(__file__),
                "windows": 32, "orientation": "DICOM physical in-plane canonical basis", "slot_windows": [9, 7, 6, 4, 6]}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    files = ["pixels.npy", "mask.npy", "slots.npy", "positions.npy", "studies.csv", "geometry.jsonl", "failures.json"]
    write_json(args.output / "manifest.json", {**contract, "cache_signature_sha256": signature,
        "part": args.part, "parts": args.parts, "studies_count": len(ids), "pixels": "pixels.npy", "mask": "mask.npy",
        "slots": "slots.npy", "positions": "positions.npy", "studies": "studies.csv",
        "unusable_studies": int((~masks.any(axis=1)).sum()), "requires_geometry_qa": True,
        "files_sha256": {name: sha256(args.output / name) for name in files}})
    if not masks.any(axis=1).all():
        raise ValueError("Unusable studies: cache cannot advance until resolved")


if __name__ == "__main__":
    main()
