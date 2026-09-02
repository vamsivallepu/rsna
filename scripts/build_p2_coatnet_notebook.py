#!/usr/bin/env python3
"""Materialize the canonical P2 DDP script in its Kaggle kernel folder.

The historical filename is retained so existing workflow references keep working.
P2 now runs as a Kaggle script because torchrun/DDP is materially faster and more
reliable than driving two GPUs from an in-notebook ``nn.DataParallel`` process.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("scripts/p2_coatnet_ddp.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "kaggle/rsna-p2-coatnet-rmlp2/rsna-p2-coatnet-rmlp2.py"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
