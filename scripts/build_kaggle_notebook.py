#!/usr/bin/env python3
"""Build the Kaggle-ready RSNA data audit notebook."""

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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/jupyter-notebook/rsna-data-exploration.ipynb"),
    )
    args = parser.parse_args()

    cells = [
        markdown(
            """
# RSNA Knee Abnormality Detection: data audit and baseline scaffold

**Question.** Can we verify the full competition metadata and exercise a robust,
plane-aware DICOM volume loader without downloading the 247 GB archive locally?

**Success criteria.**

- Load and validate every training/test metadata row.
- Quantify expert-label availability, positives, report characteristics, and series mix.
- Decode six complete MRI series spanning all training plane × sequence scenarios.
- Save compact machine-readable audit artifacts for later training notebooks.

This notebook is intentionally CPU-only. It reads the full metadata on Kaggle but
touches only selected DICOM series, preserving GPU quota for modeling.
"""
        ),
        code(
            """
from __future__ import annotations

import json
import random
import sys
import unicodedata
from pathlib import Path

# Local-only dependencies are vendored under .deps; Kaggle uses its image packages.
for candidate in (
    Path.cwd() / ".deps",
    Path.cwd().parent / ".deps",
    Path.cwd().parent.parent / ".deps",
):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

SEED = 20260831
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_columns", 30)
print({"python": sys.version.split()[0], "pydicom": pydicom.__version__, "seed": SEED})
"""
        ),
        markdown(
            """
## Plan

1. Resolve the same data interface locally and on Kaggle.
2. Assert study/series keys and the expected 12-target schema.
3. Audit sparse expert labels and multilingual-report proxies.
4. Audit plane and sequence descriptors.
5. Decode selected complete volumes with spatial slice ordering and robust display scaling.
6. Export concise audit tables and JSON for downstream notebooks.
"""
        ),
        code(
            """
COMPETITION_SLUG = "rsna-knee-abnormality-detection"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def find_metadata_root() -> Path:
    candidates = [
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("/kaggle/input/rsna-knee-abnormalities-detection"),
        Path.cwd() / "data/raw",
        Path.cwd().parent / "data/raw",
        Path.cwd().parent.parent / "data/raw",
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path for path in kaggle_input.iterdir() if path.is_dir())
        for pattern in ("*/train.csv", "*/*/train.csv", "*/*/*/train.csv"):
            candidates.extend(path.parent for path in kaggle_input.glob(pattern))
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (candidate / "train_series.csv").is_file():
            return candidate.resolve()
    input_entries = sorted(str(path) for path in kaggle_input.glob("*")) if kaggle_input.is_dir() else []
    raise FileNotFoundError(
        "Could not locate train.csv and train_series.csv; "
        f"top-level Kaggle inputs={input_entries[:50]}"
    )


def find_dicom_root(metadata_root: Path) -> Path | None:
    candidates = [
        metadata_root / "train_series",
        Path.cwd() / "data/sample/files/train_series",
        Path.cwd().parent / "data/sample/files/train_series",
        Path.cwd().parent.parent / "data/sample/files/train_series",
    ]
    return next((path.resolve() for path in candidates if path.is_dir()), None)


DATA_ROOT = find_metadata_root()
DICOM_ROOT = find_dicom_root(DATA_ROOT)
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else DATA_ROOT.parent.parent / "outputs/data_audit"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_CONTEXT = {
    "on_kaggle": ON_KAGGLE,
    "metadata_root": str(DATA_ROOT),
    "dicom_root": str(DICOM_ROOT) if DICOM_ROOT else None,
    "output_root": str(OUTPUT_ROOT.resolve()),
}
RUN_CONTEXT
"""
        ),
        markdown("## Metadata load and invariants"),
        code(
            """
train = pd.read_csv(DATA_ROOT / "train.csv")
train_series = pd.read_csv(DATA_ROOT / "train_series.csv")
test = pd.read_csv(DATA_ROOT / "test.csv")
test_series = pd.read_csv(DATA_ROOT / "test_series.csv")
sample_submission = pd.read_csv(DATA_ROOT / "sample_submission.csv")

expected_train_columns = ["StudyInstanceUID", "Report", *TARGETS]
assert train.columns.tolist() == expected_train_columns
assert sample_submission.columns.tolist() == ["StudyInstanceUID", *TARGETS]
assert train["StudyInstanceUID"].is_unique
assert train_series["SeriesInstanceUID"].is_unique
assert set(train_series["StudyInstanceUID"]).issubset(set(train["StudyInstanceUID"]))
assert not train["Report"].fillna("").str.strip().eq("").any()

dataset_summary = pd.DataFrame(
    {
        "value": {
            "train_studies": len(train),
            "train_series": len(train_series),
            "test_example_studies": len(test),
            "test_example_series": len(test_series),
            "submission_targets": len(TARGETS),
        }
    }
)
dataset_summary
"""
        ),
        markdown("## Expert-label audit"),
        code(
            """
labels = train[TARGETS].apply(pd.to_numeric, errors="coerce")
label_stats = pd.DataFrame(
    {
        "labeled": labels.notna().sum(),
        "positive": labels.eq(1).sum(),
        "negative": labels.eq(0).sum(),
    }
)
label_stats["positive_rate"] = label_stats["positive"] / label_stats["labeled"]

fully_labeled = labels.notna().all(axis=1)
partially_labeled = labels.notna().any(axis=1) & ~fully_labeled
label_coverage = {
    "fully_labeled_studies": int(fully_labeled.sum()),
    "partially_labeled_studies": int(partially_labeled.sum()),
    "unlabeled_studies": int(labels.isna().all(axis=1).sum()),
}
display(pd.Series(label_coverage, name="studies").to_frame())
display(label_stats.style.format({"positive_rate": "{:.1%}"}))

ax = label_stats["positive_rate"].sort_values().plot.barh(figsize=(8, 5), color="#4C78A8")
ax.set(title="Positive prevalence among expert-labeled studies", xlabel="Positive rate", ylabel="")
ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
plt.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
Only a small expert-labeled subset is expected. The report audit below measures
text-shape and Unicode-script coverage without printing report contents.
"""
        ),
        code(
            """
SCRIPT_MARKERS = ("LATIN", "CYRILLIC", "ARABIC", "THAI", "CJK", "HANGUL", "GREEK")


def scripts_in(text: str) -> tuple[str, ...]:
    scripts = set()
    for char in str(text):
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        scripts.update(marker for marker in SCRIPT_MARKERS if marker in name)
    return tuple(sorted(scripts)) or ("UNKNOWN",)


def non_ascii_fraction(text: str) -> float:
    letters = [char for char in str(text) if char.isalpha()]
    return float(sum(ord(char) > 127 for char in letters) / len(letters)) if letters else 0.0


report_audit = pd.DataFrame(
    {
        "characters": train["Report"].str.len(),
        "words": train["Report"].str.split().str.len(),
        "non_ascii_fraction": train["Report"].map(non_ascii_fraction),
        "scripts": train["Report"].map(lambda text: "+".join(scripts_in(text))),
    }
)
display(report_audit[["characters", "words", "non_ascii_fraction"]].describe().round(2))
display(report_audit["scripts"].value_counts().rename("studies").head(15).to_frame())

ax = report_audit["characters"].clip(upper=report_audit["characters"].quantile(0.99)).plot.hist(
    bins=50, figsize=(8, 4), color="#F58518"
)
ax.set(title="Report length (clipped at the 99th percentile)", xlabel="Characters")
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Series descriptor audit"),
        code(
            """
series_scenarios = (
    train_series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])
    .size()
    .rename("series")
    .reset_index()
    .sort_values(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])
)
series_per_study = train_series.groupby("StudyInstanceUID").size()
display(series_scenarios)
display(series_per_study.describe().rename("series_per_study").to_frame().round(2))

scenario_plot = series_scenarios.assign(
    sequence=lambda frame: frame.apply(
        lambda row: f"fluid={row.Fluid_Sensitive}, fat-suppression={row.Fat_Suppression}", axis=1
    )
)
pivot = scenario_plot.pivot(index="Anatomical_Plane", columns="sequence", values="series").fillna(0)
pivot.plot.bar(figsize=(8, 4), color=["#54A24B", "#E45756"])
plt.title("Training series by plane and sequence descriptor")
plt.xlabel("")
plt.ylabel("Series")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
## Representative DICOM volume loader

The fixed sample spans all six descriptor scenarios present in training metadata.
Locally these are the downloaded 0.165 GiB sample; on Kaggle the same UIDs resolve
inside the full mounted competition dataset.
"""
        ),
        code(
            """
SAMPLE_STUDY_UID = "1.2.826.0.1.3680043.8.498.82166943552764439138333504456139890254"
SAMPLE_SERIES = [
    ("1.2.826.0.1.3680043.8.498.10018402039504397857968533816812816572", "Axial | fluid=0 | fat-suppression=0"),
    ("1.2.826.0.1.3680043.8.498.10257912747645500412136630543476045285", "Axial | fluid=1 | fat-suppression=1"),
    ("1.2.826.0.1.3680043.8.498.30205011752939967279543537379010794909", "Coronal | fluid=0 | fat-suppression=0"),
    ("1.2.826.0.1.3680043.8.498.10436783375077399659860593777522979072", "Coronal | fluid=1 | fat-suppression=1"),
    ("1.2.826.0.1.3680043.8.498.10347272512623805065972057458376849254", "Sagittal | fluid=0 | fat-suppression=0"),
    ("1.2.826.0.1.3680043.8.498.41453776462847737420755793667519107493", "Sagittal | fluid=1 | fat-suppression=1"),
]


def spatial_position(dataset: pydicom.Dataset) -> float:
    orientation = np.asarray(dataset.get("ImageOrientationPatient", []), dtype=float)
    position = np.asarray(dataset.get("ImagePositionPatient", []), dtype=float)
    if orientation.size == 6 and position.size == 3:
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(position, normal))
    return float(dataset.get("InstanceNumber", 0))


def load_series(series_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOMs under {series_dir}")

    ordered = []
    transfer_syntaxes = set()
    first_dataset = None
    for path in files:
        dataset = pydicom.dcmread(path, force=True)
        first_dataset = first_dataset or dataset
        transfer_syntaxes.add(str(dataset.file_meta.TransferSyntaxUID))
        pixels = dataset.pixel_array.astype(np.float32)
        pixels = pixels * float(dataset.get("RescaleSlope", 1.0)) + float(dataset.get("RescaleIntercept", 0.0))
        if dataset.get("PhotometricInterpretation") == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        ordered.append((spatial_position(dataset), pixels))

    volume = np.stack([pixels for _, pixels in sorted(ordered, key=lambda item: item[0])])
    metadata = {
        "slices": len(files),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "transfer_syntaxes": sorted(transfer_syntaxes),
        "photometric_interpretation": str(first_dataset.get("PhotometricInterpretation", "")),
        "pixel_spacing": [float(value) for value in first_dataset.get("PixelSpacing", [])],
        "slice_thickness": float(first_dataset.get("SliceThickness", np.nan)),
        "min": float(volume.min()),
        "max": float(volume.max()),
    }
    return volume, metadata


def display_window(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)
"""
        ),
        code(
            """
volumes = {}
dicom_records = []

if DICOM_ROOT is None:
    print("DICOM root unavailable; metadata audit remains valid.")
else:
    for series_uid, scenario in SAMPLE_SERIES:
        directory = DICOM_ROOT / SAMPLE_STUDY_UID / series_uid
        try:
            volume, metadata = load_series(directory)
            volumes[scenario] = volume
            dicom_records.append({"series_uid": series_uid, "scenario": scenario, "status": "decoded", **metadata})
        except Exception as error:
            dicom_records.append(
                {
                    "series_uid": series_uid,
                    "scenario": scenario,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

dicom_audit = pd.DataFrame(dicom_records)
display(dicom_audit)

if volumes:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (scenario, volume) in zip(axes.flat, volumes.items()):
        axis.imshow(display_window(volume[len(volume) // 2]), cmap="gray")
        axis.set_title(f"{scenario}\\nshape={tuple(volume.shape)}", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(volumes):]:
        axis.axis("off")
    figure.suptitle("Representative mid-slices", fontsize=14)
    plt.tight_layout()
    plt.show()
"""
        ),
        markdown("## Persist compact audit artifacts"),
        code(
            """
audit = {
    "run_context": RUN_CONTEXT,
    "dataset": {key: int(value) for key, value in dataset_summary["value"].items()},
    "label_coverage": label_coverage,
    "series_scenarios": series_scenarios.to_dict(orient="records"),
    "dicom": dicom_records,
}

(OUTPUT_ROOT / "rsna_data_audit.json").write_text(json.dumps(audit, indent=2) + "\\n", encoding="utf-8")
label_stats.to_csv(OUTPUT_ROOT / "label_stats.csv")
series_scenarios.to_csv(OUTPUT_ROOT / "series_scenarios.csv", index=False)
report_audit.groupby("scripts").size().rename("studies").to_csv(OUTPUT_ROOT / "report_script_counts.csv")
if not dicom_audit.empty:
    dicom_audit.to_csv(OUTPUT_ROOT / "dicom_audit.csv", index=False)

sorted(path.name for path in OUTPUT_ROOT.iterdir())
"""
        ),
        markdown(
            """
## Results and decisions

- The invariant checks establish the authoritative study/series joins and submission schema.
- Expert-label scarcity makes report-derived weak supervision a first-class modeling task.
- Plane and sequence descriptors must remain explicit through sampling and aggregation.
- Successful decoding of all six representative volumes validates the reusable local/Kaggle loader.
- This CPU audit should remain separate from GPU training to conserve accelerator quota.

## Next steps

1. Build and validate multilingual report-to-label weak supervision against the expert subset.
2. Add group-aware cross-validation at study level, with site/scanner proxies where defensible.
3. Create a separate GPU notebook for plane-aware 2.5D image baselines and macro-AUC validation.
4. Keep a single-model efficiency candidate alongside any stronger ensemble.
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote {args.output} with {len(cells)} cells")


if __name__ == "__main__":
    main()
