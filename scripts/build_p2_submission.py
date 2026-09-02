#!/usr/bin/env python3
"""Materialize the canonical P2 submission script in its Kaggle kernel folder."""

from pathlib import Path


SOURCE = Path("scripts/p2_submission.py")
OUTPUT = Path("kaggle/rsna-p2-submission/rsna-p2-submission.py")


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(SOURCE.read_text())
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
