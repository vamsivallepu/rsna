#!/usr/bin/env python3
"""Materialize the canonical P5-A speed smoke in its Kaggle script folder."""

from __future__ import annotations

import shutil
from pathlib import Path


SOURCE = Path("scripts/p5_orthofoundation_speed_smoke.py")
OUTPUT = Path("kaggle/rsna-p5-orthofoundation-speed-smoke/rsna-p5-speed-smoke.py")


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
