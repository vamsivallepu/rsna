#!/usr/bin/env python3
"""Materialize the two matched P6 downstream Kaggle kernels."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("scripts/p2_coatnet_ddp.py")
ANCHOR = "from __future__ import annotations\n"
OUTPUTS = {
    "P6_MATCHED_CONTROL": Path(
        "kaggle/rsna-p6-p2-matched-control/rsna-p6-p2-matched-control.py"
    ),
    "P6_TRANSFER": Path("kaggle/rsna-p6-p2-transfer/rsna-p6-p2-transfer.py"),
}


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(ANCHOR) != 1:
        raise ValueError("Could not locate the unique future-import injection anchor")
    for environment_name, output in OUTPUTS.items():
        bootstrap = (
            f"{ANCHOR}\n"
            "import os as _p6_bootstrap_os\n\n"
            f'_p6_bootstrap_os.environ["{environment_name}"] = "1"\n'
        )
        rendered = source.replace(ANCHOR, bootstrap, 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
