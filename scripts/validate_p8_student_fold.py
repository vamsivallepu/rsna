#!/usr/bin/env python3
"""Independently validate one completed P8 noisy-student fold or speed smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from aggregate_p4_outputs import (
    CACHE_SIGNATURE,
    FOLD_SEED,
    ID,
    PSEUDO_SPLIT_SEED,
    TARGETS,
    expected_contract,
    report_hash,
    sha256,
    stable_group_split,
)


ROOT = Path(__file__).resolve().parents[1]
P6_MODEL_SHA256 = "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--speed-smoke", action="store_true")
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/train.csv")
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "outputs/kaggle/rsna-p8-teacher-labels-v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fold = args.fold
    root = args.output_dir
    train = pd.read_csv(args.train, dtype={ID: str})
    target_path = args.teacher_root / f"p8_fold_{fold}_training_targets.csv"
    targets = pd.read_csv(target_path, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist() and len(train) == 4_407
    teacher_manifest_path = args.teacher_root / "p8_teacher_manifest.json"
    teacher_manifest_sha256 = sha256(teacher_manifest_path)
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    assert teacher_manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert sha256(target_path) == teacher_manifest["artifacts_sha256"][target_path.name]

    hashes = train["Report"].map(report_hash).to_numpy(str)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold = np.where(is_gold)[0]
    pseudo = np.where(~is_gold)[0]
    assert len(gold) == 58 and len(pseudo) == 4_349
    base_pseudo_train, base_pseudo_valid = stable_group_split(
        pseudo,
        hashes[pseudo],
        valid_fraction=0.10,
        seed=PSEUDO_SPLIT_SEED,
    )
    folds = list(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(gold))
    expert_train_pos, expert_valid_pos = folds[fold]
    expert_train = gold[expert_train_pos]
    expert_valid = gold[expert_valid_pos]
    roles, representation, pseudo_valid = expected_contract(
        fold,
        train=train,
        hashes=hashes,
        is_gold=is_gold,
        base_pseudo_train=base_pseudo_train,
        base_pseudo_valid=base_pseudo_valid,
        expert_train=expert_train,
        expert_valid=expert_valid,
    )

    manifest_path = root / f"p8_fold_{fold}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_run = (
        f"P8-NOISY-STUDENT-FOLD-{fold}-SPEED-SMOKE-v1"
        if args.speed_smoke
        else f"P8-NOISY-STUDENT-FOLD-{fold}-v1"
    )
    assert manifest["run_id"] == expected_run
    assert manifest["full_run"] is (not args.speed_smoke)
    assert manifest["speed_smoke"] is args.speed_smoke
    assert manifest["fold"] == fold
    assert manifest["seeds"] == {
        "pseudo_split": PSEUDO_SPLIT_SEED,
        "expert_folds": FOLD_SEED,
        "representation": 20260908 + fold,
    }
    assert manifest["model_sha256"] == P6_MODEL_SHA256
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    assert manifest["teacher_manifest_sha256"] == teacher_manifest_sha256
    assert manifest["teacher_target_sha256"] == sha256(target_path)
    assert manifest["teacher_target_file"] == target_path.name
    assert manifest["pseudo_target_formula"] == (
        "0.50 report + 0.50 mean(P2 direct, P4 fold, P6 direct)"
    )
    assert manifest["planned_expert_train_rows"] == len(expert_train)
    assert manifest["expert_valid_rows"] == len(expert_valid)
    assert manifest["checkpoint_selection_gold_rows"] == 0
    assert manifest["expert_valid_gradient_rows"] == 0
    assert manifest["expert_valid_report_overlap"] == 0
    assert manifest["selection_report_overlap"] == 0
    expected_revision = (
        "p8_e2e_fold_ddp_t4_speed_smoke_v1"
        if args.speed_smoke
        else "p8_e2e_fold_ddp_t4_v1"
    )
    assert manifest["execution_revision"] == expected_revision
    assert manifest["visible_gpus"] == 2 and manifest["world_size"] == 2
    assert manifest["data_parallel_backend"] == "DistributedDataParallel"
    assert manifest["distributed_backend"] == "nccl"
    assert manifest["representation_epochs"] == (1 if args.speed_smoke else 8)
    if args.speed_smoke:
        assert manifest["representation_rows"] == 64
        assert manifest["pseudo_train_rows"] == 62
        assert manifest["expert_train_rows"] == 2
        assert manifest["pseudo_valid_rows"] == 8
        source = ROOT / "kaggle/rsna-p8-student-speed-smoke/rsna-p8-student-speed-smoke.py"
    else:
        assert manifest["representation_rows"] == len(representation)
        assert manifest["representation_gold_rows"] == len(expert_train)
        assert manifest["expert_train_rows"] == len(expert_train)
        assert manifest["pseudo_valid_rows"] == len(pseudo_valid)
        source = (
            ROOT
            / f"kaggle/rsna-p8-student-folds/fold-{fold}/"
            f"rsna-p8-noisy-student-fold-{fold}.py"
        )
    assert sha256(source) == manifest["script_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        artifact = root / name
        assert artifact.is_file() and sha256(artifact) == expected, name

    split = pd.read_csv(root / f"p8_fold_{fold}_split.csv", dtype={ID: str})
    assert split[ID].tolist() == train[ID].tolist()
    assert split["report_hash"].tolist() == hashes.tolist()
    assert np.array_equal(split["is_gold"].astype(bool).to_numpy(), is_gold)
    assert split["role"].tolist() == roles.tolist()
    train_hashes = set(
        split.loc[
            split["role"].isin(["pseudo_train", "expert_train"]), "report_hash"
        ]
    )
    selection_hashes = set(
        split.loc[split["role"] == "pseudo_valid", "report_hash"]
    )
    valid_hashes = set(split.loc[split["role"] == "expert_valid", "report_hash"])
    assert not train_hashes & selection_hashes
    assert not train_hashes & valid_hashes
    assert not selection_hashes & valid_hashes

    history = pd.read_csv(root / f"p8_fold_{fold}_history.csv")
    assert history["epoch"].tolist() == list(
        range(1, (1 if args.speed_smoke else 8) + 1)
    )
    assert history[["train_loss", "pseudo_valid_weighted_bce"]].notna().all().all()
    best = history.loc[history["pseudo_valid_weighted_bce"].idxmin()]
    assert manifest["best_epoch"] == int(best["epoch"])
    assert np.isclose(
        manifest["best_pseudo_valid_weighted_bce"],
        best["pseudo_valid_weighted_bce"],
        atol=1e-12,
    )

    predictions = pd.read_csv(
        root / f"p8_fold_{fold}_oof_predictions.csv", dtype={ID: str}
    )
    expected_ids = train.iloc[expert_valid][ID].tolist()
    assert predictions[ID].tolist() == expected_ids
    assert predictions[ID].nunique() == len(predictions) == len(expert_valid)
    assert (predictions["fold"] == fold).all()
    for target in TARGETS:
        assert np.array_equal(
            predictions[f"label__{target}"].to_numpy(np.float32),
            targets.iloc[expert_valid][target].to_numpy(np.float32),
        )
        values = predictions[f"pred__{target}"].to_numpy(float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all()

    telemetry = pd.read_csv(root / f"p8_fold_{fold}_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] > 0).all()
    if not args.speed_smoke:
        assert (utilization["max"] >= 90).all()
    print(
        json.dumps(
            {
                "validated": True,
                "fold": fold,
                "speed_smoke": args.speed_smoke,
                "representation_rows": manifest["representation_rows"],
                "pseudo_valid_rows": manifest["pseudo_valid_rows"],
                "expert_train_rows": manifest["expert_train_rows"],
                "expert_valid_rows": manifest["expert_valid_rows"],
                "best_epoch": manifest["best_epoch"],
                "best_pseudo_valid_weighted_bce": manifest[
                    "best_pseudo_valid_weighted_bce"
                ],
                "elapsed": manifest["elapsed"],
                "gpu_utilization": utilization.to_dict("index"),
                "manifest_sha256": sha256(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
