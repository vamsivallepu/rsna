#!/usr/bin/env python3
"""Validate P5-A's exact two-T4 feature-extraction speed gate."""

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
    result = json.loads((args.output_dir / "p5_speed_smoke.json").read_text())
    assert result["run_id"] == "P5A-ORTHOFOUNDATION-FROZEN-v1"
    assert result["execution_revision"] == "p5a_frozen_t4x2_smoke_v1"
    assert result["world_size"] == 2
    assert result["smoke_studies"] == 128
    assert result["slices_per_study"] == 32
    assert result["slot_quotas"] == [9, 7, 6, 4, 6]
    assert result["image_batch_per_rank"] == 64
    assert result["feature_shape_per_study"] == [32, 2, 1024]
    assert result["checkpoint_sha256"] == (
        "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"
    )
    assert result["checkpoint_bytes"] == 1_213_056_638
    assert 300_000_000 < result["model_parameters"] < 310_000_000
    numeric = [
        result["images_per_second_two_gpu"],
        result["projected_full_feature_minutes"],
        result["projected_hidden_feature_minutes"],
        *result["model_load_seconds_by_rank"],
        *result["extraction_seconds_by_rank"],
        *result["head_seconds_by_rank"],
        *result["head_final_loss_by_rank"],
        *result["peak_memory_mib_by_rank"],
    ]
    assert np.isfinite(numeric).all()
    assert result["finite"] is True
    assert result["images_per_second_two_gpu"] > 0
    assert result["projected_full_feature_minutes"] < 360.0
    assert result["projected_hidden_feature_minutes"] < 240.0
    assert max(result["peak_memory_mib_by_rank"]) < 14_500.0
    assert all(0 < value < 2 for value in result["head_final_loss_by_rank"])

    telemetry = pd.read_csv(args.output_dir / "p5_speed_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    assert telemetry.notna().all().all()
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] >= 80.0).all()
    print(
        json.dumps(
            {
                "validated": True,
                "projected_full_feature_minutes": result["projected_full_feature_minutes"],
                "projected_hidden_feature_minutes": result["projected_hidden_feature_minutes"],
                "peak_memory_mib_by_rank": result["peak_memory_mib_by_rank"],
                "gpu_utilization": utilization.to_dict("index"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
