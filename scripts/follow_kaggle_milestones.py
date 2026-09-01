#!/usr/bin/env python3
"""Follow one Kaggle kernel while printing only useful execution milestones."""

from __future__ import annotations

import argparse
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi


DEFAULT_TERMS = (
    "teacher_origin",
    "selected_studies",
    "sample_uid",
    "encoded ",
    "cached ",
    "temporary_pixel_gib",
    "trainable_backbone_millions",
    "epoch ",
    "pseudo_only_direct_gold_macro_auc",
    "token batches ",
    "token_feature_mib",
    "fold_head_oof_macro_auc",
    "Fold ",
    "fold ",
    "macro_auc",
    "image_baseline_manifest",
    "completed",
    "failed",
    "Traceback",
    "ERROR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Kaggle kernel as owner/slug")
    parser.add_argument("--retries", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = KaggleApi()
    api.authenticate()

    seen = 0
    failures = 0
    while failures <= args.retries:
        before = seen
        try:
            for index, event in enumerate(api.kernels_logs_stream(args.kernel)):
                if index < seen:
                    continue
                seen = index + 1
                data = str(event.get("data", ""))
                if data and any(term in data for term in DEFAULT_TERMS):
                    print(data, flush=True)
            return 0
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # Network streams can terminate mid-chunk.
            failures = 0 if seen > before else failures + 1
            if failures > args.retries:
                print(f"log stream stopped after {failures} retries: {exc}", file=sys.stderr)
                return 1
            time.sleep(min(5, failures + 1))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
