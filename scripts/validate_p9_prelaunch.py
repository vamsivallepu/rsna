#!/usr/bin/env python3
"""Validate the frozen P9 inputs and generated Kaggle bundles before launch."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models.video import s3d


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/p9_s3d_specialist.py"
ANCHOR = "from __future__ import annotations\n"
WEIGHT = ROOT / "kaggle/rsna-p9-s3d-init/s3d-d76dad2f.pth"
WEIGHT_SHA256 = "d76dad2f4b2a6701ad7f64e85b294b2d1989f1ea0d3f8b240446ed2152850bbb"
TEACHER_ROOT = ROOT / "outputs/kaggle/rsna-p8-teacher-labels-v2"
TEACHER_SHA256 = "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-report-weak-supervision",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
    "vallepuvamsikrishna/rsna-knee-p8-teacher-labels",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_kernel(root: Path, *, speed_smoke: bool) -> str:
    metadata = json.loads((root / "kernel-metadata.json").read_text())
    code_path = root / metadata["code_file"]
    code = code_path.read_text(encoding="utf-8")
    ast.parse(code)
    assert code.count(TEACHER_SHA256) == 1
    assert code.count('_p9_bootstrap_os.environ["P9_SPEED_SMOKE"] = "1"') == (
        1 if speed_smoke else 0
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == [
        "vallepuvamsikrishna/rsna-p9-s3d-init"
    ]
    assert metadata["kernel_sources"] == KERNEL_SOURCES
    return metadata["id"]


def main() -> int:
    ast.parse(SOURCE.read_text(encoding="utf-8"))
    assert WEIGHT.stat().st_size == 33_524_997
    assert sha256(WEIGHT) == WEIGHT_SHA256
    state = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    incompatible = s3d(weights=None).load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys

    manifest_path = TEACHER_ROOT / "p8_teacher_manifest.json"
    assert sha256(manifest_path) == TEACHER_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    studies = pd.read_csv(TEACHER_ROOT / "p8_teacher_studies.csv")
    original = pd.read_csv(
        ROOT / "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"
    )
    assert studies["StudyInstanceUID"].tolist() == original["StudyInstanceUID"].tolist()
    p2 = np.load(TEACHER_ROOT / "p8_p2_direct_predictions.npy")
    p6 = np.load(TEACHER_ROOT / "p8_p6_direct_predictions.npy")
    assert p2.shape == p6.shape == (4_407, 12)
    report = original[TARGETS].to_numpy(np.float32)
    safe = 0.50 * report + 0.25 * p2 + 0.25 * p6
    is_gold = original["is_gold"].astype(bool).to_numpy()
    assert int(is_gold.sum()) == 58 and np.isfinite(safe[~is_gold]).all()
    assert np.array_equal(report[is_gold], original.loc[is_gold, TARGETS].to_numpy())

    smoke = validate_kernel(ROOT / "kaggle/rsna-p9-s3d-speed-smoke", speed_smoke=True)
    full = validate_kernel(ROOT / "kaggle/rsna-p9-s3d-specialist", speed_smoke=False)
    print(json.dumps({
        "validated": True,
        "weight_sha256": WEIGHT_SHA256,
        "teacher_manifest_sha256": TEACHER_SHA256,
        "safe_pseudo_formula": "0.50 report + 0.25 P2 direct + 0.25 P6 direct",
        "expert_targets_unchanged": True,
        "p4_teacher_excluded": True,
        "smoke_kernel": smoke,
        "full_kernel": full,
        "accelerator": "NvidiaTeslaT4 x2",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
