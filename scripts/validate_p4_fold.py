#!/usr/bin/env python3
"""Independently validate one completed full P4 end-to-end fold."""

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
    MODEL_SHA256,
    PSEUDO_SPLIT_SEED,
    TARGETS,
    expected_contract,
    report_hash,
    sha256,
    stable_group_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"
        ),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("kaggle/rsna-p4-coatnet-folds"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fold = args.fold
    root = args.output_dir
    train = pd.read_csv(args.train, dtype={ID: str})
    targets = pd.read_csv(args.targets, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist() and len(train) == 4_407
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

    manifest_path = root / f"p4_fold_{fold}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == f"P4-COATNET-E2E-FOLD-{fold}-v1"
    assert manifest["full_run"] is True and manifest["speed_smoke"] is False
    assert manifest["fold"] == fold
    assert manifest["seeds"] == {
        "pseudo_split": PSEUDO_SPLIT_SEED,
        "expert_folds": FOLD_SEED,
        "representation": 20260904 + fold,
    }
    assert manifest["model_sha256"] == MODEL_SHA256
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    assert manifest["representation_rows"] == len(representation)
    assert manifest["representation_gold_rows"] == len(expert_train)
    assert manifest["expert_train_rows"] == len(expert_train)
    assert manifest["expert_valid_rows"] == len(expert_valid)
    assert manifest["pseudo_valid_rows"] == len(pseudo_valid)
    assert manifest["checkpoint_selection_gold_rows"] == 0
    assert manifest["expert_valid_gradient_rows"] == 0
    assert manifest["expert_valid_report_overlap"] == 0
    assert manifest["selection_report_overlap"] == 0
    assert manifest["execution_revision"] == "p4_e2e_fold_ddp_t4_v1"
    assert manifest["visible_gpus"] == 2 and manifest["world_size"] == 2
    assert manifest["data_parallel_backend"] == "DistributedDataParallel"
    assert manifest["distributed_backend"] == "nccl"
    assert manifest["representation_epochs"] == 8
    source = (
        args.bundle_root
        / f"fold-{fold}"
        / f"rsna-p4-coatnet-e2e-fold-{fold}.py"
    )
    assert sha256(source) == manifest["script_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        path = root / name
        assert path.is_file() and sha256(path) == expected, name

    split = pd.read_csv(root / f"p4_fold_{fold}_split.csv", dtype={ID: str})
    assert split[ID].tolist() == train[ID].tolist()
    assert split["report_hash"].tolist() == hashes.tolist()
    assert np.array_equal(split["is_gold"].astype(bool).to_numpy(), is_gold)
    assert split["role"].tolist() == roles.tolist()
    assert split.loc[split["role"] == "expert_valid", ID].tolist() == (
        train.iloc[expert_valid][ID].tolist()
    )
    train_hashes = set(
        split.loc[split["role"].isin(["pseudo_train", "expert_train"]), "report_hash"]
    )
    selection_hashes = set(
        split.loc[split["role"] == "pseudo_valid", "report_hash"]
    )
    valid_hashes = set(split.loc[split["role"] == "expert_valid", "report_hash"])
    assert not train_hashes & selection_hashes
    assert not train_hashes & valid_hashes
    assert not selection_hashes & valid_hashes

    history = pd.read_csv(root / f"p4_fold_{fold}_history.csv")
    assert history["epoch"].tolist() == list(range(1, 9))
    assert history[["train_loss", "pseudo_valid_weighted_bce"]].notna().all().all()
    best = history.loc[history["pseudo_valid_weighted_bce"].idxmin()]
    assert manifest["best_epoch"] == int(best["epoch"])
    assert np.isclose(
        manifest["best_pseudo_valid_weighted_bce"],
        best["pseudo_valid_weighted_bce"],
        atol=1e-12,
    )

    predictions = pd.read_csv(
        root / f"p4_fold_{fold}_oof_predictions.csv", dtype={ID: str}
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

    telemetry = pd.read_csv(root / f"p4_fold_{fold}_gpu_telemetry.csv")
    assert set(telemetry["index"].unique()) == {0, 1}
    utilization = telemetry.groupby("index")["utilization_gpu_percent"].agg(
        ["count", "mean", "median", "max"]
    )
    assert (utilization["max"] >= 90).all()
    print(json.dumps({
        "validated": True,
        "fold": fold,
        "representation_rows": len(representation),
        "pseudo_valid_rows": len(pseudo_valid),
        "expert_train_rows": len(expert_train),
        "expert_valid_rows": len(expert_valid),
        "best_epoch": manifest["best_epoch"],
        "best_pseudo_valid_weighted_bce": manifest[
            "best_pseudo_valid_weighted_bce"
        ],
        "elapsed": manifest["elapsed"],
        "gpu_utilization": utilization.to_dict("index"),
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
