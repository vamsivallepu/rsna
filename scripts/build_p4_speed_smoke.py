#!/usr/bin/env python3
"""Materialize the exact fold-0 P4 path as a short dual-T4 Kaggle smoke."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("scripts/p2_coatnet_ddp.py")
OUTPUT = Path("kaggle/rsna-p4-speed-smoke")
CODE_NAME = "rsna-p4-coatnet-e2e-speed-smoke.py"
FOLD_MARKER = (
    'P4_FOLD = int(os.environ["P4_FOLD"]) if "P4_FOLD" in os.environ else None'
)
SMOKE_MARKER = 'P4_SPEED_SMOKE = os.environ.get("P4_SPEED_SMOKE") == "1"'


def main() -> int:
    template = SOURCE.read_text()
    if template.count(FOLD_MARKER) != 1 or template.count(SMOKE_MARKER) != 1:
        raise RuntimeError("canonical P4 smoke markers are missing or ambiguous")
    code = template.replace(FOLD_MARKER, "P4_FOLD = 0")
    code = code.replace(SMOKE_MARKER, "P4_SPEED_SMOKE = True")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / CODE_NAME).write_text(code)
    metadata = {
        "id": "vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-speed-smoke",
        "title": "RSNA Knee P4 CoAtNet E2E Speed Smoke",
        "code_file": CODE_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [
            "vallepuvamsikrishna/rsna-coatnet-rmlp2-imagenet-init"
        ],
        "kernel_sources": [
            "vallepuvamsikrishna/rsna-knee-report-weak-supervision",
            "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
            "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
        ],
        "model_sources": [],
    }
    (OUTPUT / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
