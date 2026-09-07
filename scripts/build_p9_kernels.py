#!/usr/bin/env python3
"""Build the P9 S3D speed-smoke and full Kaggle script kernels."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("scripts/p9_s3d_specialist.py")
ANCHOR = "from __future__ import annotations\n"
TEACHER_MANIFEST_SHA256 = (
    "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
)
KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-report-weak-supervision",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
    "vallepuvamsikrishna/rsna-knee-p8-teacher-labels",
]
DATASET_SOURCES = ["vallepuvamsikrishna/rsna-p9-s3d-init"]


def render(source: str, *, speed_smoke: bool) -> str:
    bootstrap = (
        f"{ANCHOR}\n"
        "import os as _p9_bootstrap_os\n\n"
        "_p9_bootstrap_os.environ[\n"
        "    \"P9_TEACHER_MANIFEST_SHA256\"\n"
        f"] = \"{TEACHER_MANIFEST_SHA256}\"\n"
    )
    if speed_smoke:
        bootstrap += '_p9_bootstrap_os.environ["P9_SPEED_SMOKE"] = "1"\n'
    return source.replace(ANCHOR, bootstrap, 1)


def write_bundle(
    root: Path, *, code_name: str, kernel_id: str, title: str, code: str
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    compile(code, code_name, "exec")
    (root / code_name).write_text(code, encoding="utf-8")
    metadata = {
        "id": f"vallepuvamsikrishna/{kernel_id}",
        "title": title,
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": KERNEL_SOURCES,
        "model_sources": [],
    }
    (root / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(root)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(ANCHOR) != 1:
        raise ValueError("canonical source future-import anchor is missing or ambiguous")
    write_bundle(
        Path("kaggle/rsna-p9-s3d-speed-smoke"),
        code_name="rsna-p9-s3d-speed-smoke.py",
        kernel_id="rsna-knee-p9-s3d-speed-smoke",
        title="RSNA Knee P9 S3D Speed Smoke",
        code=render(source, speed_smoke=True),
    )
    write_bundle(
        Path("kaggle/rsna-p9-s3d-specialist"),
        code_name="rsna-p9-s3d-specialist.py",
        kernel_id="rsna-knee-p9-s3d-specialist",
        title="RSNA Knee P9 S3D Specialist",
        code=render(source, speed_smoke=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
