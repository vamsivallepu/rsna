#!/usr/bin/env python3
"""Validate P7 two-GPU report-auxiliary speed-smoke output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    root = parse_args().output_dir
    result = json.loads((root / "p2_speed_smoke.json").read_text())
    assert result["execution_revision"] == "p7_report_aux_ddp_t4_v1"
    assert result["world_size"] == 2
    assert result["global_study_batch"] == 2
    assert result["gradient_accumulation"] == 4
    assert result["global_steps"] == 32
    assert result["p7_report_aux"] is True
    assert result["p7_report_temperature"] == 0.07
    assert result["p7_report_aux_weight"] == 0.10
    assert result["p7_report_bank_rows"] > 3_800
    assert math.isfinite(result["train_report_contrastive"])
    assert result["train_report_contrastive"] > 0
    assert math.isfinite(result["train_combined_loss"])
    assert result["elapsed_minutes"] < 20
    assert len(result["peak_memory_mib_by_rank"]) == 2
    assert max(result["peak_memory_mib_by_rank"]) < 14_500

    logs = list(root.glob("*.log"))
    assert len(logs) == 1
    log = logs[0].read_text(errors="replace")
    assert "P7_REPORT_EMBEDDINGS_VALIDATED" in log
    assert "P7_REPORT_BANK_READY" in log
    assert "P2_SPEED_SMOKE_COMPLETE" in log
    assert "Traceback" not in log and "CUDA out of memory" not in log

    full_global_steps = math.ceil(result["p7_report_bank_rows"] / 2) * 8
    projected_representation_hours = (
        result["seconds_per_global_step"] * full_global_steps / 3_600
    )
    summary = {
        "validated": True,
        "world_size": result["world_size"],
        "report_bank_rows": result["p7_report_bank_rows"],
        "report_contrastive_loss": result["train_report_contrastive"],
        "seconds_per_global_step": result["seconds_per_global_step"],
        "projected_representation_hours": projected_representation_hours,
        "peak_memory_mib_by_rank": result["peak_memory_mib_by_rank"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
