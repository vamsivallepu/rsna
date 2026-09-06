#!/usr/bin/env python3
"""Materialize the P8 fold-safe teacher-label Kaggle script."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("scripts/p8_teacher_labels.py")
OUTPUT = Path(
    "kaggle/rsna-p8-teacher-labels/rsna-p8-teacher-labels.py"
)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    compile(source, str(SOURCE), "exec")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
