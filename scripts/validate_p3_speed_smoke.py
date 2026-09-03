#!/usr/bin/env python3
"""Validate the P3 dual-T4 throughput, memory, and numerical gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    result = json.loads((root / "p3_speed_smoke.json").read_text())
    assert result["run_id"] == "P3-DINOV2-HMIL-v1"
    assert result["execution_revision"] == "p3_ddp_t4_v1"
    assert result["world_size"] == 2
    assert result["global_study_batch"] == 2
    assert result["gradient_accumulation"] == 4
    assert result["smoke_train_studies"] == 64
    assert result["smoke_valid_studies"] == 8
    assert result["k_train"] == 8 and result["k_eval"] == 32
    assert result["region_tokens"] == 5
    assert 3_000_000 < result["trainable_parameters"] < 6_000_000
    numeric = [
        result["training_seconds"],
        result["validation_seconds"],
        result["seconds_per_global_study"],
        result["pseudo_valid_weighted_bce"],
        result["projected_full_minutes"],
        *result["peak_memory_mib_by_rank"],
    ]
    assert np.isfinite(numeric).all()
    assert 0.0 < result["pseudo_valid_weighted_bce"] < 2.0
    assert result["projected_full_minutes"] < 480.0
    assert len(result["peak_memory_mib_by_rank"]) == 2
    assert max(result["peak_memory_mib_by_rank"]) < 13_500.0

    telemetry = pd.read_csv(root / "p3_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    assert telemetry.notna().all().all()
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] >= 80.0).all()
    summary = {
        "validated": True,
        "projected_full_minutes": result["projected_full_minutes"],
        "peak_memory_mib_by_rank": result["peak_memory_mib_by_rank"],
        "gpu_utilization": utilization.to_dict("index"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
