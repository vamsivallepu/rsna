#!/usr/bin/env python3
"""Materialize the self-contained P6 blend canary Kaggle script."""

from pathlib import Path


COMMON = Path("scripts/p2_submission.py")
SOURCE = Path("scripts/p4_blend_submission.py")
OUTPUT = Path(
    "kaggle/rsna-p6-blend-canary/rsna-p6-blend-canary.py"
)
IMPORT_START = "# BEGIN COMMON IMPORTS"
IMPORT_END = "# END COMMON IMPORTS"
P6_SWITCH = "ENABLE_P6 = False  # The P6 deployment builder changes this immutable switch."


def main() -> int:
    common = COMMON.read_text()
    marker = "\ndef main() -> int:"
    assert marker in common
    common_prefix = common.split(marker, maxsplit=1)[0].rstrip()

    source = SOURCE.read_text()
    assert source.count(IMPORT_START) == source.count(IMPORT_END) == 1
    assert source.count(P6_SWITCH) == 1
    source = source.replace(P6_SWITCH, "ENABLE_P6 = True", 1)
    before, remainder = source.split(IMPORT_START, maxsplit=1)
    _, after = remainder.split(IMPORT_END, maxsplit=1)
    deployment = before.rstrip() + "\n" + after.lstrip("\n")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(common_prefix + "\n\n" + deployment)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
