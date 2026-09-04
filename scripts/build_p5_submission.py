#!/usr/bin/env python3
"""Materialize the self-contained P5 diagnostic submission script."""

from pathlib import Path


SOURCE = Path("scripts/p5_submission.py")
P2_COMMON = Path("scripts/p2_submission.py")
P5_COMMON = Path("scripts/p5_orthofoundation_speed_smoke.py")
OUTPUT = Path("kaggle/rsna-p5-submission/rsna-p5-submission.py")


def main() -> int:
    source = SOURCE.read_text()
    imports = (
        "import p2_submission as prep\n"
        "import p5_orthofoundation_speed_smoke as p5\n"
    )
    if source.count(imports) != 1:
        raise ValueError("unexpected P5 submission import structure")
    p2_source = P2_COMMON.read_text()
    p5_source = P5_COMMON.read_text()
    bootstrap = (
        "import types as _bundle_types\n"
        "prep = _bundle_types.ModuleType('p5_embedded_p2_preprocessing')\n"
        "prep.__file__ = '<embedded-p2-preprocessing>'\n"
        f"exec(compile({p2_source!r}, prep.__file__, 'exec'), prep.__dict__)\n"
        "p5 = _bundle_types.ModuleType('p5_embedded_common')\n"
        "p5.__file__ = '<embedded-p5-common>'\n"
        f"exec(compile({p5_source!r}, p5.__file__, 'exec'), p5.__dict__)\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source.replace(imports, bootstrap, 1))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
