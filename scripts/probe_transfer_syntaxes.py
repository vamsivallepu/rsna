#!/usr/bin/env python3
"""Find and retain one decodable DICOM for each declared transfer syntax."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps"))

import pydicom
from pydicom.uid import UID

from download_representative_sample import download_one, list_series_files
from kaggle.api.kaggle_api_extended import KaggleApi


TARGET_SYNTAXES = {
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless, Non-Hierarchical, First-Order Prediction",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Image Compression (Lossless Only)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/train_series.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/sample/files"))
    parser.add_argument("--manifest", type=Path, default=Path("data/sample/transfer_syntax_samples.json"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-probes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    import csv

    with args.metadata.open(encoding="utf-8", newline="") as handle:
        series = list(csv.DictReader(handle))
    random.Random(args.seed).shuffle(series)

    found: dict[str, dict[str, object]] = {}
    for path in args.output.rglob("*.dcm"):
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        syntax = str(dataset.file_meta.TransferSyntaxUID)
        if syntax in TARGET_SYNTAXES and syntax not in found:
            found[syntax] = {
                "uid": syntax,
                "name": TARGET_SYNTAXES[syntax],
                "path": str(path),
                "StudyInstanceUID": str(dataset.get("StudyInstanceUID", "")),
                "SeriesInstanceUID": str(dataset.get("SeriesInstanceUID", "")),
                "SOPInstanceUID": str(dataset.get("SOPInstanceUID", "")),
                "decoded_shape": None,
            }

    api = KaggleApi()
    api.authenticate()
    competition = "rsna-knee-abnormality-detection"
    probed = 0

    with tempfile.TemporaryDirectory(prefix="rsna-transfer-syntax-") as temporary:
        temporary_root = Path(temporary)
        for start in range(0, min(len(series), args.max_probes), args.batch_size):
            if set(found) == set(TARGET_SYNTAXES):
                break
            batch = series[start : start + args.batch_size]
            planned = []
            for record in batch:
                series_path = (
                    f"train_series/{record['StudyInstanceUID']}/{record['SeriesInstanceUID']}"
                )
                available = list_series_files(api, competition, series_path)
                if available:
                    planned.append(available[len(available) // 2])

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(download_one, competition, temporary_root, item): item
                    for item in planned
                }
                for future in as_completed(futures):
                    source = future.result()
                    probed += 1
                    dataset = pydicom.dcmread(source, force=True)
                    syntax = str(dataset.file_meta.TransferSyntaxUID)
                    if syntax not in TARGET_SYNTAXES or syntax in found:
                        continue
                    pixels = dataset.pixel_array
                    relative = Path(str(futures[future]["relative_url"]))
                    destination = args.output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    found[syntax] = {
                        "uid": syntax,
                        "name": TARGET_SYNTAXES[syntax],
                        "pydicom_name": UID(syntax).name,
                        "path": str(destination),
                        "relative_url": str(relative),
                        "StudyInstanceUID": str(dataset.get("StudyInstanceUID", "")),
                        "SeriesInstanceUID": str(dataset.get("SeriesInstanceUID", "")),
                        "SOPInstanceUID": str(dataset.get("SOPInstanceUID", "")),
                        "decoded_shape": list(pixels.shape),
                        "decoded_dtype": str(pixels.dtype),
                    }
                    print(f"found {syntax}: {TARGET_SYNTAXES[syntax]}", flush=True)

            print(
                f"probed={probed} found={len(found)}/{len(TARGET_SYNTAXES)}",
                flush=True,
            )

    payload = {
        "competition": competition,
        "probed_files": probed,
        "complete": set(found) == set(TARGET_SYNTAXES),
        "samples": [found[syntax] for syntax in TARGET_SYNTAXES if syntax in found],
        "missing": [
            {"uid": syntax, "name": name}
            for syntax, name in TARGET_SYNTAXES.items()
            if syntax not in found
        ],
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if args.strict and not payload["complete"]:
        raise RuntimeError(f"Missing transfer syntaxes: {payload['missing']}")


if __name__ == "__main__":
    main()
