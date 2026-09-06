#!/usr/bin/env python3
"""Validate generated P8 student kernels and teacher inputs before launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/p2_coatnet_ddp.py"
ANCHOR = "from __future__ import annotations\n"
ID = "StudyInstanceUID"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
    "vallepuvamsikrishna/rsna-knee-p8-teacher-labels",
]
DATASET_SOURCES = ["vallepuvamsikrishna/rsna-p6-fastmri-localization-init"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p8-teacher-labels-v1",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(*, fold: int, speed_smoke: bool, manifest_hash: str) -> str:
    rendered = (
        f"{ANCHOR}\n"
        "import os as _p8_bootstrap_os\n\n"
        f'_p8_bootstrap_os.environ["P8_FOLD"] = "{fold}"\n'
        "_p8_bootstrap_os.environ[\n"
        '    "P8_TEACHER_MANIFEST_SHA256"\n'
        f'] = "{manifest_hash}"\n'
    )
    if speed_smoke:
        rendered += '_p8_bootstrap_os.environ["P8_SPEED_SMOKE"] = "1"\n'
    return rendered


def validate_kernel(
    root: Path,
    *,
    fold: int,
    speed_smoke: bool,
    manifest_hash: str,
) -> str:
    metadata = json.loads((root / "kernel-metadata.json").read_text())
    code_path = root / metadata["code_file"]
    source = SOURCE.read_text(encoding="utf-8")
    expected = source.replace(
        ANCHOR,
        bootstrap(fold=fold, speed_smoke=speed_smoke, manifest_hash=manifest_hash),
        1,
    )
    assert code_path.read_text(encoding="utf-8") == expected
    compile(expected, str(code_path), "exec")
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == DATASET_SOURCES
    assert metadata["kernel_sources"] == KERNEL_SOURCES
    assert metadata["model_sources"] == []
    return metadata["id"]


def main() -> int:
    args = parse_args()
    teacher_manifest_path = args.teacher_root / "p8_teacher_manifest.json"
    manifest_hash = sha256(teacher_manifest_path)
    manifest = json.loads(teacher_manifest_path.read_text())
    assert manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert manifest["expert_audit_macro_auc"]["relabel"] == 0.9209204490507412
    assert manifest["fold_safety"]["expert_validation_teacher_gradient_rows"] == 0

    source = SOURCE.read_text(encoding="utf-8")
    assert source.count(ANCHOR) == 1
    compile(source, str(SOURCE), "exec")
    assert 'P8_SEED_BASE = 20260908' in source
    assert 'USES_P6_INITIALIZATION = P6_TRANSFER or P7_REPORT_AUX or P8_MODE' in source
    assert 'P8-NOISY-STUDENT' in source

    original = pd.read_csv(
        ROOT / "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv",
        dtype={ID: str},
    )
    is_gold = original["is_gold"].astype(bool).to_numpy()
    non_target = [column for column in original if column not in TARGETS]
    target_hashes: dict[str, str] = {}
    for fold in range(5):
        name = f"p8_fold_{fold}_training_targets.csv"
        path = args.teacher_root / name
        target_hashes[str(fold)] = sha256(path)
        assert target_hashes[str(fold)] == manifest["artifacts_sha256"][name]
        targets = pd.read_csv(path, dtype={ID: str})
        pd.testing.assert_frame_equal(
            targets[non_target], original[non_target], check_dtype=False, check_exact=True
        )
        assert np.array_equal(
            targets.loc[is_gold, TARGETS].to_numpy(np.float64),
            original.loc[is_gold, TARGETS].to_numpy(np.float64),
        )
        assert not np.array_equal(
            targets.loc[~is_gold, TARGETS].to_numpy(np.float64),
            original.loc[~is_gold, TARGETS].to_numpy(np.float64),
        )

    kernels = [
        validate_kernel(
            ROOT / "kaggle/rsna-p8-student-speed-smoke",
            fold=0,
            speed_smoke=True,
            manifest_hash=manifest_hash,
        )
    ]
    kernels.extend(
        validate_kernel(
            ROOT / f"kaggle/rsna-p8-student-folds/fold-{fold}",
            fold=fold,
            speed_smoke=False,
            manifest_hash=manifest_hash,
        )
        for fold in range(5)
    )
    print(
        json.dumps(
            {
                "validated": True,
                "teacher_manifest_sha256": manifest_hash,
                "teacher_target_sha256": target_hashes,
                "expert_targets_unchanged": True,
                "non_target_columns_unchanged": True,
                "p6_localization_initialization": True,
                "generated_kernels_match_canonical_source": True,
                "kernels": kernels,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
