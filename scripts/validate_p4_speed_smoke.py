#!/usr/bin/env python3
"""Independently validate the fold-0 P4 dual-T4 execution smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "kaggle/rsna-p4-speed-smoke/rsna-p4-coatnet-e2e-speed-smoke.py"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_hash(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = "".join(character for character in normalized if character.isalnum())
    if not normalized:
        raise ValueError("empty normalized report")
    return hashlib.sha256(normalized.encode()).hexdigest()


def main() -> int:
    args = parse_args()
    root = args.output_dir
    manifest = json.loads((root / "p4_fold_0_manifest.json").read_text())
    assert manifest["run_id"] == "P4-COATNET-E2E-FOLD-0-SPEED-SMOKE-v1"
    assert manifest["full_run"] is False and manifest["speed_smoke"] is True
    assert manifest["fold"] == 0
    assert manifest["representation_rows"] == 64
    assert manifest["pseudo_train_rows"] == 62
    assert manifest["expert_train_rows"] == 2
    assert manifest["pseudo_valid_rows"] == 8
    assert manifest["expert_valid_rows"] == 12
    assert manifest["planned_pseudo_train_rows"] == 3914
    assert manifest["planned_pseudo_valid_rows"] == 435
    assert manifest["planned_expert_train_rows"] == 46
    assert manifest["checkpoint_selection_gold_rows"] == 0
    assert manifest["expert_valid_gradient_rows"] == 0
    assert manifest["expert_valid_report_overlap"] == 0
    assert manifest["selection_report_overlap"] == 0
    assert manifest["visible_gpus"] == 2 and manifest["world_size"] == 2
    assert manifest["data_parallel_backend"] == "DistributedDataParallel"
    assert manifest["distributed_backend"] == "nccl"
    assert manifest["execution_revision"] == "p4_e2e_fold_ddp_t4_speed_smoke_v1"
    assert manifest["representation_epochs"] == 1
    assert manifest["best_epoch"] == 1
    assert np.isfinite(manifest["best_pseudo_valid_weighted_bce"])
    assert sha256(args.source) == manifest["script_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file() and sha256(path) == expected, name

    train = pd.read_csv(args.train, dtype={ID: str})
    targets = pd.read_csv(args.targets, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist()
    split = pd.read_csv(root / "p4_fold_0_split.csv", dtype={ID: str})
    assert split[ID].tolist() == train[ID].tolist()
    assert split["report_hash"].tolist() == train["Report"].map(report_hash).tolist()
    assert split[ID].nunique() == len(split) == 4_407
    assert int((split["role"] == "expert_valid").sum()) == 12
    assert int((split["role"] == "expert_train").sum()) == 46
    assert int((split["role"] == "pseudo_valid").sum()) == 435
    train_hashes = set(split.loc[split["role"].isin(["pseudo_train", "expert_train"]), "report_hash"])
    selection_hashes = set(split.loc[split["role"] == "pseudo_valid", "report_hash"])
    expert_valid_hashes = set(split.loc[split["role"] == "expert_valid", "report_hash"])
    assert not train_hashes & selection_hashes
    assert not train_hashes & expert_valid_hashes
    assert not selection_hashes & expert_valid_hashes

    predictions = pd.read_csv(
        root / "p4_fold_0_oof_predictions.csv", dtype={ID: str}
    )
    assert predictions[ID].nunique() == len(predictions) == 12
    assert (predictions["fold"] == 0).all()
    expected_ids = split.loc[split["role"] == "expert_valid", ID].tolist()
    assert predictions[ID].tolist() == expected_ids
    target_by_id = targets.set_index(ID)
    for target in TARGETS:
        assert np.array_equal(
            predictions[f"label__{target}"].to_numpy(np.float32),
            target_by_id.loc[expected_ids, target].to_numpy(np.float32),
        )
        values = predictions[f"pred__{target}"].to_numpy(float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all()

    history = pd.read_csv(root / "p4_fold_0_history.csv")
    assert history["epoch"].tolist() == [1]
    assert history[["train_loss", "pseudo_valid_weighted_bce"]].notna().all().all()
    telemetry = pd.read_csv(root / "p4_fold_0_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    gpu = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (gpu["max"] > 0).all()
    print(json.dumps({
        "validated": True,
        "representation_rows": manifest["representation_rows"],
        "pseudo_valid_weighted_bce": manifest["best_pseudo_valid_weighted_bce"],
        "expert_valid_rows": len(predictions),
        "gpu_utilization": gpu.to_dict("index"),
        "elapsed": manifest["elapsed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
