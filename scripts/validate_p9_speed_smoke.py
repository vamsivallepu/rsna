#!/usr/bin/env python3
"""Validate P9 dual-T4 speed/memory smoke output and promotion gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    root = parser.parse_args().output_dir
    result = json.loads((root / "p9_speed_smoke.json").read_text())
    assert result["run_id"] == "P9-S3D-SPEED-SMOKE-v1"
    assert result["world_size"] == 2
    assert result["global_study_batch"] == 2
    assert result["clips_per_study"] == 5
    assert result["frames_per_clip"] == 16
    assert result["image_size"] == 160
    assert result["train_studies"] == 64
    assert result["valid_studies"] == 16
    assert result["global_train_steps"] == 32
    assert math.isfinite(result["train_weighted_bce"])
    assert math.isfinite(result["pseudo_valid_weighted_bce"])
    assert len(result["peak_memory_mib_by_rank"]) == 2
    assert max(result["peak_memory_mib_by_rank"]) < 14_500
    projection = float(result["projected_six_epoch_training_hours"])
    assert 0 < projection < 4.0
    logs = list(root.glob("*.log"))
    assert len(logs) == 1
    log = logs[0].read_text(errors="replace")
    assert "P9_S3D_DDP_READY" in log
    assert "P9_SPEED_SMOKE_COMPLETE" in log
    assert "'rank': 0" in log and "'rank': 1" in log
    assert "Traceback" not in log and "CUDA out of memory" not in log
    print(json.dumps({
        "validated": True,
        "projected_six_epoch_training_hours": projection,
        "seconds_per_global_step": result["seconds_per_global_step"],
        "peak_memory_mib_by_rank": result["peak_memory_mib_by_rank"],
        "advance_to_full_run": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
