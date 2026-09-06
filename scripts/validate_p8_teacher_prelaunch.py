#!/usr/bin/env python3
"""Validate the frozen P8 teacher-label kernel before a Kaggle launch."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/p8_teacher_labels.py"
BUILDER = ROOT / "scripts/build_p8_teacher_labels.py"
KERNEL_ROOT = ROOT / "kaggle/rsna-p8-teacher-labels"
METADATA = KERNEL_ROOT / "kernel-metadata.json"
EXPECTED_KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-report-weak-supervision",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
    "vallepuvamsikrishna/rsna-knee-p2-coatnet-rmlp2",
    "vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-0",
    "vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-1",
    "vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-2",
    "vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-3",
    "vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-4",
    "vallepuvamsikrishna/rsna-knee-p6-p2-transfer",
]
P2_ROOT = ROOT / "outputs/kaggle/rsna-p2-coatnet-rmlp2-v2"
P6_ROOT = ROOT / "outputs/kaggle/rsna-p6-p2-transfer-v1"
P4_ROOT = ROOT / "outputs/kaggle/rsna-p4-coatnet-e2e-v1"
WEAK_ROOT = ROOT / "outputs/kaggle/rsna-report-weak-supervision"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def validate_manifest(root: Path, name: str, expected_hash: str) -> dict[str, object]:
    path = root / name
    assert sha256(path) == expected_hash, name
    manifest = json.loads(path.read_text())
    for artifact, expected in manifest["artifacts_sha256"].items():
        artifact_path = root / artifact
        assert artifact_path.is_file(), artifact_path
        assert sha256(artifact_path) == expected, artifact
    return manifest


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    generated = KERNEL_ROOT / "rsna-p8-teacher-labels.py"
    assert generated.read_text(encoding="utf-8") == source
    ast.parse(source)
    ast.parse(BUILDER.read_text(encoding="utf-8"))
    values = constants(SOURCE)

    assert values["RUN_ID"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert values["N_FOLDS"] == 5
    assert values["FOLD_SEED"] == 20260831
    assert values["REPORT_WEIGHT"] == 0.5
    assert values["IMAGE_WEIGHT"] == 0.5
    assert "IMAGE_COMPONENT_WEIGHT = 1.0 / 3.0" in source
    assert values["EXPECTED_RELABEL_MACRO"] == 0.9209204490507412
    assert source.count("batch_size=2") == 2
    assert '"expert_replay_preflight"' in source
    assert '"canonical_expert_values_used_for_audit": True' in source
    assert values["DIRECT_REPLAY_ATOL"] == 1e-5
    assert values["P4_REPLAY_MAX_ATOL"] == 5e-3
    assert values["P4_REPLAY_MEAN_ATOL"] == 5e-4
    assert "target_specific_adjustments\": False" in source
    assert "expert_validation_teacher_gradient_rows\": 0" in source
    assert "p4_teacher_matches_student_fold\": True" in source
    assert "torch.distributed.run" in source and "--nproc_per_node=2" in source

    metadata = json.loads(METADATA.read_text())
    assert metadata == {
        "id": "vallepuvamsikrishna/rsna-knee-p8-teacher-labels",
        "title": "RSNA Knee P8 Teacher Labels",
        "code_file": generated.name,
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

    weak_manifest = WEAK_ROOT / "weak_label_manifest.json"
    assert sha256(weak_manifest) == values["WEAK_MANIFEST_SHA256"]
    assert sha256(WEAK_ROOT / "training_targets.csv") == values[
        "TRAINING_TARGETS_SHA256"
    ]
    assert sha256(WEAK_ROOT / "consensus_weak_labels.csv") == values[
        "CONSENSUS_SHA256"
    ]

    p2 = validate_manifest(
        P2_ROOT, "p2_manifest.json", str(values["P2_MANIFEST_SHA256"])
    )
    p6 = validate_manifest(
        P6_ROOT, "p6_transfer_manifest.json", str(values["P6_MANIFEST_SHA256"])
    )
    assert p2["run_id"] == "P2-COATNET-v1"
    assert p6["run_id"] == "P6-P2-TRANSFER-v1"

    expected_p4 = values["P4_MANIFEST_SHA256"]
    assert isinstance(expected_p4, dict)
    for fold in range(5):
        manifest = validate_manifest(
            P4_ROOT / f"fold-{fold}",
            f"p4_fold_{fold}_manifest.json",
            str(expected_p4[fold]),
        )
        assert manifest["run_id"] == f"P4-COATNET-E2E-FOLD-{fold}-v1"
        assert manifest["fold"] == fold
        assert manifest["expert_valid_gradient_rows"] == 0
        assert manifest["selection_report_overlap"] == 0

    print(
        json.dumps(
            {
                "validated": True,
                "kernel": metadata["id"],
                "generated_kernel_matches_canonical_source": True,
                "teacher_formula": "0.50 report + 0.50 mean(P2 direct, P4 fold, P6 direct)",
                "folds": 5,
                "expert_validation_teacher_gradient_rows": 0,
                "visible_gpus_required": 2,
                "study_batch_per_gpu": 2,
                "expert_replay_preflight_before_full_inference": True,
                "canonical_expert_values_used_for_audit": True,
                "source_manifests_and_artifacts_validated": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
