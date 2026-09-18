"""Label-blind scanner/volume audit using one DICOM header per series."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ID, sha256, write_json


def main():
    import pydicom
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--dicom-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ids = pd.read_csv(args.train, usecols=[ID])[ID]
    series = pd.read_csv(args.series)
    rows, errors = [], []
    tags = ["Rows", "Columns", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
            "ImageOrientationPatient", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices"]
    for i, r in series.iterrows():
        sid = str(r.SeriesInstanceUID)
        try:
            files = sorted((args.dicom_root / r[ID] / sid).glob("*.dcm"))
            if not files:
                raise ValueError("No DICOM slices")
            h = pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=tags)
            orientation = np.asarray(h.get("ImageOrientationPatient", []), float)
            agreement = None
            if len(orientation) == 6:
                axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[r.Anatomical_Plane]
                agreement = float(abs(np.cross(orientation[:3], orientation[3:])[axis]))
            rows.append({ID: r[ID], "SeriesInstanceUID": sid, "plane": r.Anatomical_Plane,
                "fluid_sensitive": int(r.Fluid_Sensitive), "native_slices": len(files),
                "rows": int(h.Rows), "columns": int(h.Columns),
                "estimated_pixels": len(files) * int(h.Rows) * int(h.Columns),
                "manufacturer": str(h.get("Manufacturer", "unknown")),
                "scanner_model": str(h.get("ManufacturerModelName", "unknown")),
                "field_strength": str(h.get("MagneticFieldStrength", "unknown")),
                "pixel_spacing": json.dumps(list(map(float, h.get("PixelSpacing", [])))),
                "slice_thickness": str(h.get("SliceThickness", "unknown")),
                "between_slice_spacing": str(h.get("SpacingBetweenSlices", "unknown")),
                "plane_normal_agreement": agreement,
                "transfer_syntax": str(h.file_meta.TransferSyntaxUID)})
        except Exception as error:
            errors.append({ID: r[ID], "SeriesInstanceUID": sid, "error": str(error)})
        if (i + 1) % 1000 == 0:
            print(json.dumps({"series": i + 1, "errors": len(errors)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(args.output / "series_headers.csv", index=False)
    write_json(args.output / "errors.json", errors)
    studies = data.groupby(ID).agg(native_slices=("native_slices", "sum"),
        estimated_pixels=("estimated_pixels", "sum"), series_count=("SeriesInstanceUID", "size"),
        plane_count=("plane", "nunique"), fluid_series=("fluid_sensitive", "sum"),
        scanner_domain=("scanner_model", lambda x: "|".join(sorted(set(x)))))
    studies = studies.reindex(ids)
    studies.to_csv(args.output / "study_domains_sizes.csv")
    if studies.isna().any().any() or errors:
        write_json(args.output / "manifest.json", {"status": "incomplete_header_audit", "series_errors": len(errors)})
        raise ValueError("Resolve header errors before size-stratified benchmarking")
    ordered = studies.sort_values(["scanner_domain", "plane_count", "fluid_series", "estimated_pixels"])
    chosen = {studies.estimated_pixels.idxmin(), studies.estimated_pixels.idxmax()}
    for _, group in ordered.groupby("scanner_domain"):
        chosen.add(group.index[len(group) // 2])
    target_count = min(240, len(studies))
    if len(chosen) > target_count:
        raise ValueError("More scanner domains than benchmark capacity")
    remaining = ordered.drop(index=list(chosen))
    extra = np.round(np.linspace(0, len(remaining) - 1, target_count - len(chosen))).astype(int)
    chosen.update(remaining.index[extra])
    selected = pd.DataFrame({ID: ids[ids.isin(chosen)]})
    selected.to_csv(args.output / "benchmark_studies.csv", index=False)
    data.groupby(["manufacturer", "scanner_model", "field_strength"]).size().rename("series_count").to_csv(args.output / "scanner_counts.csv")
    write_json(args.output / "manifest.json", {"status": "completed_header_audit", "studies": len(studies),
        "series": len(data), "scanner_domains": int(studies.scanner_domain.nunique()),
        "native_slice_range": [int(studies.native_slices.min()), int(studies.native_slices.max())],
        "estimated_pixel_range": [int(studies.estimated_pixels.min()), int(studies.estimated_pixels.max())],
        "plane_disagreements": int((data.plane_normal_agreement < .85).sum()),
        "missing_orientations": int(data.plane_normal_agreement.isna().sum()),
        "benchmark_studies": len(selected), "train_sha256": sha256(args.train), "series_sha256": sha256(args.series),
        "label_blind": True, "limitation": "One header per series; volume size estimated from native slice count and first-slice dimensions",
        "files_sha256": {p.name: sha256(p) for p in args.output.glob("*.csv")}})


if __name__ == "__main__":
    main()
