#!/usr/bin/env python3
"""Independently validate P7 BGE-M3 report-embedding artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ID = "StudyInstanceUID"
EXPECTED_MODEL_SOURCE = "yethukmutt/bge-m3/Transformers/m3/1"
EXPECTED_STUDIES = 4_407
EXPECTED_DIMENSION = 1_024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--train-csv", type=Path, default=Path("data/raw/train.csv")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_report(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest_path = root / "p7_report_embeddings_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P7-REPORT-EMBEDDINGS-v1"
    assert manifest["studies"] == manifest["unique_studies"] == EXPECTED_STUDIES
    assert manifest["nonempty_reports"] == EXPECTED_STUDIES
    assert manifest["model"]["source"] == EXPECTED_MODEL_SOURCE
    assert manifest["model"]["upstream"] == "BAAI/bge-m3"
    assert manifest["model"]["license"] == "MIT"
    assert manifest["model"]["embedding_dimension"] == EXPECTED_DIMENSION
    assert manifest["model"]["extraction_max_tokens"] == 4096
    assert manifest["tokens"]["truncated_reports"] == 0
    assert manifest["tokens"]["maximum"] <= 4096
    assert manifest["execution"]["world_size"] == 2
    assert manifest["execution"]["visible_gpus"] == 2
    assert len(manifest["execution"]["peak_allocated_mib_by_rank"]) == 2
    assert all(
        value < 14_500
        for value in manifest["execution"]["peak_allocated_mib_by_rank"]
    )
    assert manifest["model"]["source_files_sha256"]
    assert any(
        name.endswith((".bin", ".safetensors"))
        for name in manifest["model"]["source_files_sha256"]
    )
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha256_file(root / name) == expected, name

    train = pd.read_csv(args.train_csv, dtype={ID: str})
    studies = pd.read_csv(root / "p7_bge_m3_report_studies.csv", dtype={ID: str})
    assert len(train) == EXPECTED_STUDIES and train[ID].is_unique
    assert studies.columns.tolist() == [ID]
    assert studies[ID].tolist() == train[ID].tolist()
    reports = [normalize_report(value) for value in train["Report"]]
    assert manifest["train_csv_sha256"] == sha256_file(args.train_csv)
    assert manifest["study_order_sha256"] == sha256_lines(
        train[ID].astype(str).tolist()
    )
    assert manifest["normalized_report_order_sha256"] == sha256_lines(reports)

    embeddings = np.load(root / "p7_bge_m3_report_embeddings.npy", mmap_mode="r")
    lengths = np.load(root / "p7_bge_m3_token_lengths.npy")
    assert embeddings.shape == (EXPECTED_STUDIES, EXPECTED_DIMENSION)
    assert embeddings.dtype == np.float16 and np.isfinite(embeddings).all()
    assert lengths.shape == (EXPECTED_STUDIES,) and (lengths > 0).all()
    assert int(lengths.max()) == manifest["tokens"]["maximum"]
    norms = np.linalg.norm(np.asarray(embeddings).astype(np.float32), axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    assert max_norm_error < 0.002
    assert np.std(np.asarray(embeddings).astype(np.float32), axis=0).mean() > 1e-4

    summary = {
        "validated": True,
        "studies": len(studies),
        "dimension": EXPECTED_DIMENSION,
        "maximum_tokens": int(lengths.max()),
        "max_embedding_norm_error": max_norm_error,
        "model_source_files": len(manifest["model"]["source_files_sha256"]),
        "elapsed_minutes_by_rank": manifest["execution"][
            "elapsed_minutes_by_rank"
        ],
        "peak_allocated_mib_by_rank": manifest["execution"][
            "peak_allocated_mib_by_rank"
        ],
        "manifest_sha256": sha256_file(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
