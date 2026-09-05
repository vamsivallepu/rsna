#!/usr/bin/env python3
"""Validate P7 two-GPU report-auxiliary speed-smoke output."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np


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
    bank_indices = np.load(root / "p7_report_bank_indices.npy")
    group_ids = np.load(root / "p7_report_group_ids.npy")
    assert len(bank_indices) == result["p7_report_bank_rows"] == 3_926
    assert group_ids.shape == (4_407,)
    assert len(np.unique(group_ids[bank_indices])) == 3_761

    logs = list(root.glob("*.log"))
    assert len(logs) == 1
    log = logs[0].read_text(errors="replace")
    assert "P7_REPORT_EMBEDDINGS_VALIDATED" in log
    assert "P7_REPORT_BANK_READY" in log
    assert "P2_SPEED_SMOKE_COMPLETE" in log
    assert "'rank': 0" in log and "'rank': 1" in log
    assert "Traceback" not in log and "CUDA out of memory" not in log

    progress = []
    for event in json.loads(log):
        data = str(event.get("data", "")).strip()
        if "P2_SPEED_SMOKE_PROGRESS" in data:
            progress.append(ast.literal_eval(data)["P2_SPEED_SMOKE_PROGRESS"])
    assert [row["step"] for row in progress] == list(range(4, 33, 4))
    steady_seconds_per_global_step = (
        progress[-1]["elapsed_seconds"] - progress[0]["elapsed_seconds"]
    ) / (progress[-1]["step"] - progress[0]["step"])
    assert 0 < steady_seconds_per_global_step < result["seconds_per_global_step"]

    full_global_steps = math.ceil(result["p7_report_bank_rows"] / 2) * 8
    conservative_projection_hours = (
        result["seconds_per_global_step"] * full_global_steps / 3_600
    )
    steady_projection_hours = (
        progress[0]["elapsed_seconds"]
        + steady_seconds_per_global_step * (full_global_steps - progress[0]["step"])
    ) / 3_600
    summary = {
        "validated": True,
        "world_size": result["world_size"],
        "report_bank_rows": result["p7_report_bank_rows"],
        "report_contrastive_loss": result["train_report_contrastive"],
        "seconds_per_global_step": result["seconds_per_global_step"],
        "steady_seconds_per_global_step_after_warmup": (
            steady_seconds_per_global_step
        ),
        "projected_representation_hours": {
            "steady_after_warmup": steady_projection_hours,
            "conservative_all_in": conservative_projection_hours,
        },
        "peak_memory_mib_by_rank": result["peak_memory_mib_by_rank"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
