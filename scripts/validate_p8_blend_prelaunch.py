#!/usr/bin/env python3
"""Validate the immutable P8 blend canary bundle before Kaggle launch."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "kaggle/rsna-p8-blend-canary"
SCRIPT = BUNDLE / "rsna-p8-blend-canary.py"
METADATA = BUNDLE / "kernel-metadata.json"
EXPECTED_KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-p2-coatnet-rmlp2",
    *[
        f"vallepuvamsikrishna/rsna-knee-p8-noisy-student-fold-{fold}"
        for fold in range(5)
    ],
    "vallepuvamsikrishna/rsna-knee-p6-p2-transfer",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    metadata = json.loads(METADATA.read_text())
    assert metadata == {
        "id": "vallepuvamsikrishna/rsna-knee-p8-blend-canary",
        "title": "RSNA Knee P8 Blend Canary",
        "code_file": SCRIPT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "kernel_sources": EXPECTED_KERNEL_SOURCES,
        "model_sources": [],
    }
    text = SCRIPT.read_text()
    ast.parse(text)
    for required in (
        'RUN_ID = "P8-BLEND-SUBMISSION-v1"',
        "ENABLE_P6 = True",
        "EXPECTED_P8_MANIFEST_SHA256",
        "EXPECTED_P8_CHECKPOINT_SHA256",
        "EXPECTED_P8_TEACHER_MANIFEST_SHA256",
        'output_prefix = "p8_blend"',
        'effective_run_id = RUN_ID',
        '"P8_BLEND_SUBMISSION_COMPLETE"',
        '"p8_blend_submission_manifest.json"',
        '"p8_blend_p6_predictions.csv"',
        "0.375 rank(P2) + 0.375 rank(P8) + 0.25 rank(P6)",
    ):
        assert required in text, required
    assert "rsna-knee-p8-noisy-student-fold-{fold}" in text
    assert 'p8_fold_{fold}_manifest.json' in text
    assert "rsna-knee-p4-coatnet-e2e-fold-" not in text
    assert "EXPECTED_P4_MANIFEST_SHA256" not in text
    assert "p4_fold_" not in text
    print(
        json.dumps(
            {
                "validated": True,
                "run_id": "P8-BLEND-SUBMISSION-v1",
                "script_sha256": sha256(SCRIPT),
                "metadata_sha256": sha256(METADATA),
                "kernel_sources": len(EXPECTED_KERNEL_SOURCES),
                "seven_backbones": True,
                "dual_t4": True,
                "internet_disabled": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
