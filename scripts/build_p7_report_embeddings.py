#!/usr/bin/env python3
"""Materialize the P7 report-embedding Kaggle script."""

from pathlib import Path


SOURCE = Path("scripts/p7_report_embeddings.py")
OUTPUT = Path(
    "kaggle/rsna-p7-report-embeddings/rsna-p7-report-embeddings.py"
)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    assert 'RUN_ID = "P7-REPORT-EMBEDDINGS-v1"' in source
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
