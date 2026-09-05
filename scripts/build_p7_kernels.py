#!/usr/bin/env python3
"""Materialize the frozen P7 report-image auxiliary Kaggle kernels."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("scripts/p2_coatnet_ddp.py")
ANCHOR = "from __future__ import annotations\n"
EMBEDDING_MANIFEST_SHA256 = (
    "f82ddf5f5b95224a21cb83b2f1e8b3ed1bf3bad2e8495396d5c398956a9556a8"
)
OUTPUTS = {
    False: Path(
        "kaggle/rsna-p7-report-image-aux/rsna-p7-report-image-aux.py"
    ),
    True: Path(
        "kaggle/rsna-p7-report-image-speed-smoke/"
        "rsna-p7-report-image-speed-smoke.py"
    ),
}


def render(source: str, *, speed_smoke: bool) -> str:
    bootstrap = (
        f"{ANCHOR}\n"
        "import os as _p7_bootstrap_os\n\n"
        '_p7_bootstrap_os.environ["P7_REPORT_AUX"] = "1"\n'
        "_p7_bootstrap_os.environ[\n"
        '    "P7_REPORT_EMBEDDINGS_MANIFEST_SHA256"\n'
        f'] = "{EMBEDDING_MANIFEST_SHA256}"\n'
    )
    if speed_smoke:
        bootstrap += '_p7_bootstrap_os.environ["P2_SPEED_SMOKE"] = "1"\n'
    return source.replace(ANCHOR, bootstrap, 1)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(ANCHOR) != 1:
        raise ValueError("Could not locate the unique future-import injection anchor")
    for speed_smoke, output in OUTPUTS.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(source, speed_smoke=speed_smoke), encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
