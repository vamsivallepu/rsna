#!/usr/bin/env python3
"""Validate the P6 checkpoint payload and matched downstream Kaggle kernels."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/p2_coatnet_ddp.py"
BUILDER = ROOT / "scripts/build_p6_p2_kernels.py"
DATASET = ROOT / "kaggle/rsna-p6-fastmri-localization-init"
CONTROL = ROOT / "kaggle/rsna-p6-p2-matched-control"
TRANSFER = ROOT / "kaggle/rsna-p6-p2-transfer"
ANCHOR = "from __future__ import annotations\n"
CHECKPOINT_SHA256 = "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195"
MANIFEST_SHA256 = "f6fb96ee2434ee8ead9e9e586be7d85bb8cae3fd6c71e059f552a67fb02452ef"
HISTORY_SHA256 = "7b94a65787d45e84589b5f77134bdc405fcf5e72019ccd8504e00be5a82a6d5c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rendered_source(environment_name: str) -> str:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count(ANCHOR) == 1
    bootstrap = (
        f"{ANCHOR}\n"
        "import os as _p6_bootstrap_os\n\n"
        f'_p6_bootstrap_os.environ["{environment_name}"] = "1"\n'
    )
    return source.replace(ANCHOR, bootstrap, 1)


def load_kernel(root: Path, environment_name: str) -> tuple[dict[str, object], str]:
    metadata = json.loads((root / "kernel-metadata.json").read_text())
    code = (root / str(metadata["code_file"])).read_text(encoding="utf-8")
    assert code == rendered_source(environment_name)
    assert code.count(f'_p6_bootstrap_os.environ["{environment_name}"] = "1"') == 1
    other = (
        "P6_TRANSFER"
        if environment_name == "P6_MATCHED_CONTROL"
        else "P6_MATCHED_CONTROL"
    )
    assert code.count(f'_p6_bootstrap_os.environ["{other}"] = "1"') == 0
    ast.parse(code)
    return metadata, code


def main() -> int:
    ast.parse(SOURCE.read_text(encoding="utf-8"))
    ast.parse(BUILDER.read_text(encoding="utf-8"))

    expected_files = {
        "README.md",
        "PROVENANCE.json",
        "dataset-metadata.json",
        "p6_localization_backbone_best.safetensors",
        "p6_localization_training.json",
        "p6_training_history.json",
    }
    assert {path.name for path in DATASET.iterdir() if path.is_file()} == expected_files
    checkpoint = DATASET / "p6_localization_backbone_best.safetensors"
    manifest_path = DATASET / "p6_localization_training.json"
    history_path = DATASET / "p6_training_history.json"
    assert checkpoint.stat().st_size == 295_695_280
    assert sha256(checkpoint) == CHECKPOINT_SHA256
    assert sha256(manifest_path) == MANIFEST_SHA256
    assert sha256(history_path) == HISTORY_SHA256

    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "validated"
    assert manifest["experiment_id"] == "P6B-FASTMRI-PLUS-LOCALIZATION-v1"
    assert manifest["code_commit"] == "e2be22d"
    assert manifest["selection"]["best_epoch"] == 2
    assert manifest["artifacts"]["backbone_sha256"] == CHECKPOINT_SHA256
    assert manifest["data"]["train_volumes"] == 973
    assert manifest["data"]["validation_volumes"] == 199
    assert manifest["data"]["volume_overlap"] == 0

    dataset_metadata = json.loads((DATASET / "dataset-metadata.json").read_text())
    assert (
        dataset_metadata["id"]
        == "vallepuvamsikrishna/rsna-p6-fastmri-localization-init"
    )
    provenance = json.loads((DATASET / "PROVENANCE.json").read_text())
    assert provenance["visibility"] == "private"
    assert provenance["contains_medical_images"] is False
    assert provenance["checkpoint"]["sha256"] == CHECKPOINT_SHA256

    control_meta, control_code = load_kernel(CONTROL, "P6_MATCHED_CONTROL")
    transfer_meta, transfer_code = load_kernel(TRANSFER, "P6_TRANSFER")
    for metadata in (control_meta, transfer_meta):
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is True
        assert metadata["enable_internet"] is False
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
        assert metadata["kernel_sources"] == [
            "vallepuvamsikrishna/rsna-knee-report-weak-supervision",
            "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
            "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
        ]
    assert control_meta["dataset_sources"] == [
        "vallepuvamsikrishna/rsna-coatnet-rmlp2-imagenet-init"
    ]
    assert transfer_meta["dataset_sources"] == [
        "vallepuvamsikrishna/rsna-p6-fastmri-localization-init"
    ]

    normalized_control = control_code.replace(
        '_p6_bootstrap_os.environ["P6_MATCHED_CONTROL"] = "1"',
        '_p6_bootstrap_os.environ["P6_ARM"] = "1"',
        1,
    )
    normalized_transfer = transfer_code.replace(
        '_p6_bootstrap_os.environ["P6_TRANSFER"] = "1"',
        '_p6_bootstrap_os.environ["P6_ARM"] = "1"',
        1,
    )
    assert normalized_control == normalized_transfer

    print(
        json.dumps(
            {
                "validated": True,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "source_manifest_sha256": MANIFEST_SHA256,
                "dataset_contains_medical_images": False,
                "generated_kernels_match_canonical_source": True,
                "normalized_kernel_sources_identical": True,
                "only_intended_arm_difference": "initialization checkpoint",
                "accelerator": "NvidiaTeslaT4 x2",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
