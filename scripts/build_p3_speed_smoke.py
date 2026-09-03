#!/usr/bin/env python3
"""Build the short dual-T4 P3 throughput and memory smoke script."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("scripts/p3_dinov2_hmil_ddp.py")
OUTPUT = Path("kaggle/rsna-p3-speed-smoke/rsna-p3-speed-smoke.py")


def main() -> int:
    source = SOURCE.read_text()
    needle = "import os\n"
    replacement = 'import os\nos.environ["P3_SPEED_SMOKE"] = "1"\n'
    if source.count(needle) != 1:
        raise ValueError("unexpected canonical source structure")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source.replace(needle, replacement, 1))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
