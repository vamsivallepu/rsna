#!/usr/bin/env python3
"""Validate P7 embedding evidence and generated Kaggle kernels before launch."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/p2_coatnet_ddp.py"
BUILDER = ROOT / "scripts/build_p7_kernels.py"
EMBEDDING_OUTPUT = ROOT / "outputs/kaggle/rsna-p7-report-embeddings-v1"
FULL = ROOT / "kaggle/rsna-p7-report-image-aux"
SMOKE = ROOT / "kaggle/rsna-p7-report-image-speed-smoke"
ANCHOR = "from __future__ import annotations\n"
EMBEDDING_MANIFEST_SHA256 = (
    "f82ddf5f5b95224a21cb83b2f1e8b3ed1bf3bad2e8495396d5c398956a9556a8"
)
COMMON_KERNEL_SOURCES = [
    "vallepuvamsikrishna/rsna-knee-report-weak-supervision",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
    "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
    "vallepuvamsikrishna/rsna-knee-p7-report-embeddings",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(*, speed_smoke: bool) -> str:
    rendered = (
        f"{ANCHOR}\n"
        "import os as _p7_bootstrap_os\n\n"
        '_p7_bootstrap_os.environ["P7_REPORT_AUX"] = "1"\n'
        "_p7_bootstrap_os.environ[\n"
        '    "P7_REPORT_EMBEDDINGS_MANIFEST_SHA256"\n'
        f'] = "{EMBEDDING_MANIFEST_SHA256}"\n'
    )
    if speed_smoke:
        rendered += '_p7_bootstrap_os.environ["P2_SPEED_SMOKE"] = "1"\n'
    return rendered


def validate_kernel(root: Path, *, speed_smoke: bool) -> dict[str, object]:
    metadata = json.loads((root / "kernel-metadata.json").read_text())
    code_path = root / str(metadata["code_file"])
    code = code_path.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    expected = source.replace(ANCHOR, bootstrap(speed_smoke=speed_smoke), 1)
    assert code == expected
    ast.parse(code)
    assert code.count('_p7_bootstrap_os.environ["P7_REPORT_AUX"] = "1"') == 1
    assert code.count(EMBEDDING_MANIFEST_SHA256) == 1
    expected_speed_count = 1 if speed_smoke else 0
    assert code.count('_p7_bootstrap_os.environ["P2_SPEED_SMOKE"] = "1"') == (
        expected_speed_count
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == [
        "vallepuvamsikrishna/rsna-p6-fastmri-localization-init"
    ]
    assert metadata["kernel_sources"] == COMMON_KERNEL_SOURCES
    assert metadata["model_sources"] == []
    return metadata


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count(ANCHOR) == 1
    ast.parse(source)
    ast.parse(BUILDER.read_text(encoding="utf-8"))
    assert sha256(
        EMBEDDING_OUTPUT / "p7_report_embeddings_manifest.json"
    ) == EMBEDDING_MANIFEST_SHA256
    embedding_manifest = json.loads(
        (EMBEDDING_OUTPUT / "p7_report_embeddings_manifest.json").read_text()
    )
    assert embedding_manifest["run_id"] == "P7-REPORT-EMBEDDINGS-v1"
    assert embedding_manifest["studies"] == 4_407
    assert embedding_manifest["tokens"]["truncated_reports"] == 0
    assert embedding_manifest["model"]["license"] == "MIT"
    assert embedding_manifest["execution"]["world_size"] == 2
    for name, expected in embedding_manifest["artifacts_sha256"].items():
        assert sha256(EMBEDDING_OUTPUT / name) == expected

    full = validate_kernel(FULL, speed_smoke=False)
    smoke = validate_kernel(SMOKE, speed_smoke=True)
    assert full["id"] == (
        "vallepuvamsikrishna/rsna-knee-p7-report-image-auxiliary"
    )
    print(
        json.dumps(
            {
                "validated": True,
                "embedding_manifest_sha256": EMBEDDING_MANIFEST_SHA256,
                "embedding_rows": embedding_manifest["studies"],
                "maximum_report_tokens": embedding_manifest["tokens"]["maximum"],
                "generated_kernels_match_canonical_source": True,
                "training_text_model_input_attached": False,
                "training_embedding_output_attached": True,
                "full_kernel": full["id"],
                "speed_smoke_kernel": smoke["id"],
                "accelerator": "NvidiaTeslaT4 x2",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
