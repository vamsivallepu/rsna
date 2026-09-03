#!/usr/bin/env python3
"""Materialize the five fold-specific P4 Kaggle DDP script bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FOLD_MARKER = (
    'P4_FOLD = int(os.environ["P4_FOLD"]) if "P4_FOLD" in os.environ else None'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("scripts/p2_coatnet_ddp.py")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("kaggle/rsna-p4-coatnet-folds")
    )
    parser.add_argument(
        "--fold", type=int, action="append", choices=range(5), dest="folds"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    template = source.read_text()
    if template.count(FOLD_MARKER) != 1:
        raise RuntimeError("canonical P4 fold marker is missing or ambiguous")

    folds = sorted(set(args.folds if args.folds is not None else range(5)))
    for fold in folds:
        bundle = args.output_root / f"fold-{fold}"
        bundle.mkdir(parents=True, exist_ok=True)
        code_name = f"rsna-p4-coatnet-e2e-fold-{fold}.py"
        code = template.replace(FOLD_MARKER, f"P4_FOLD = {fold}")
        (bundle / code_name).write_text(code)
        metadata = {
            "id": f"vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-{fold}",
            "title": f"RSNA Knee P4 CoAtNet E2E Fold {fold}",
            "code_file": code_name,
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
        (bundle / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
