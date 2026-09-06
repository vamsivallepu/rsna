#!/usr/bin/env python3
"""Build the P8 student speed-smoke and five fold-specific Kaggle kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE = Path("scripts/p2_coatnet_ddp.py")
ANCHOR = "from __future__ import annotations\n"
KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
    "vallepuvamsikrishna/rsna-knee-p8-teacher-labels",
]
DATASET_SOURCES = ["vallepuvamsikrishna/rsna-p6-fastmri-localization-init"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest-sha256", required=True)
    return parser.parse_args()


def render(source: str, *, fold: int, speed_smoke: bool, manifest_hash: str) -> str:
    bootstrap = (
        f"{ANCHOR}\n"
        "import os as _p8_bootstrap_os\n\n"
        f'_p8_bootstrap_os.environ["P8_FOLD"] = "{fold}"\n'
        "_p8_bootstrap_os.environ[\n"
        '    "P8_TEACHER_MANIFEST_SHA256"\n'
        f'] = "{manifest_hash}"\n'
    )
    if speed_smoke:
        bootstrap += '_p8_bootstrap_os.environ["P8_SPEED_SMOKE"] = "1"\n'
    return source.replace(ANCHOR, bootstrap, 1)


def write_bundle(
    root: Path,
    *,
    code_name: str,
    kernel_id: str,
    title: str,
    code: str,
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
        json.dumps(metadata, indent=2) + "\n"
    )
    print(root)


def main() -> int:
    args = parse_args()
    manifest_hash = args.teacher_manifest_sha256.lower()
    if len(manifest_hash) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_hash
    ):
        raise ValueError("teacher manifest SHA-256 must be 64 lowercase hex characters")
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(ANCHOR) != 1:
        raise ValueError("canonical source future-import anchor is missing or ambiguous")

    smoke_code = render(
        source, fold=0, speed_smoke=True, manifest_hash=manifest_hash
    )
    write_bundle(
        Path("kaggle/rsna-p8-student-speed-smoke"),
        code_name="rsna-p8-student-speed-smoke.py",
        kernel_id="rsna-knee-p8-student-speed-smoke",
        title="RSNA Knee P8 Student Speed Smoke",
        code=smoke_code,
    )
    for fold in range(5):
        code = render(
            source, fold=fold, speed_smoke=False, manifest_hash=manifest_hash
        )
        write_bundle(
            Path("kaggle/rsna-p8-student-folds") / f"fold-{fold}",
            code_name=f"rsna-p8-noisy-student-fold-{fold}.py",
            kernel_id=f"rsna-knee-p8-noisy-student-fold-{fold}",
            title=f"RSNA Knee P8 Noisy Student Fold {fold}",
            code=code,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
