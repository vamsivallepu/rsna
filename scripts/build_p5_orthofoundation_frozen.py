#!/usr/bin/env python3
"""Build the full P5-A Kaggle script bundle from canonical sources."""

from __future__ import annotations

import shutil
from pathlib import Path


SOURCE = Path("scripts/p5_orthofoundation_frozen.py")
COMMON = Path("scripts/p5_orthofoundation_speed_smoke.py")
OUTPUT_ROOT = Path("kaggle/rsna-p5-orthofoundation-frozen")


def main() -> int:
    for path in (SOURCE, COMMON):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, OUTPUT_ROOT / "rsna-p5-orthofoundation-frozen.py")
    shutil.copyfile(COMMON, OUTPUT_ROOT / "p5_orthofoundation_speed_smoke.py")
    print(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
