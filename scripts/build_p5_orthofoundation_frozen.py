#!/usr/bin/env python3
"""Build the full P5-A Kaggle script bundle from canonical sources."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("scripts/p5_orthofoundation_frozen.py")
COMMON = Path("scripts/p5_orthofoundation_speed_smoke.py")
OUTPUT_ROOT = Path("kaggle/rsna-p5-orthofoundation-frozen")


def main() -> int:
    for path in (SOURCE, COMMON):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    needle = "import p5_orthofoundation_speed_smoke as common\n"
    if source.count(needle) != 1:
        raise ValueError("unexpected P5 full source import structure")
    common_source = COMMON.read_text()
    bootstrap = (
        "import types as _p5_types\n"
        "common = _p5_types.ModuleType('p5_orthofoundation_embedded_common')\n"
        "common.__file__ = '<embedded-p5-common>'\n"
        f"exec(compile({common_source!r}, common.__file__, 'exec'), common.__dict__)\n"
    )
    output = OUTPUT_ROOT / "rsna-p5-orthofoundation-frozen.py"
    output.write_text(source.replace(needle, bootstrap, 1))
    print(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
