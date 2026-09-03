#!/usr/bin/env python3
"""Materialize the canonical P3 DDP script in its Kaggle kernel folder.

P3 uses a Kaggle script kernel because torchrun/DDP is materially faster and more
reliable than notebook-driven DataParallel. The experiment remains structured as
a reproducible notebook-style run with a frozen protocol and compact milestones.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("scripts/p3_dinov2_hmil_ddp.py")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kaggle/rsna-p3-dinov2-hmil/rsna-p3-dinov2-hmil.py"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
