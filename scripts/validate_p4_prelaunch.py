#!/usr/bin/env python3
"""Independently validate P4 fold bundles and report-group split contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


ID = "StudyInstanceUID"
PSEUDO_SPLIT_SEED = 20260902
FOLD_SEED = 20260831
N_FOLDS = 5
EXPECTED_CACHE_SIGNATURE = (
    "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def report_hash(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = "".join(character for character in normalized if character.isalnum())
    if not normalized:
        raise ValueError("empty normalized report")
    return hashlib.sha256(normalized.encode()).hexdigest()


def stable_group_split(
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups, dtype=str)
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_by_group = dict(zip(unique_groups.tolist(), counts.tolist(), strict=True))
    ordered_groups = sorted(
        unique_groups.tolist(),
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    target_rows = round(len(indices) * valid_fraction)
    valid_groups: set[str] = set()
    valid_rows = 0
    for group in ordered_groups:
        candidate_rows = valid_rows + count_by_group[group]
        if abs(candidate_rows - target_rows) <= abs(valid_rows - target_rows):
            valid_groups.add(group)
            valid_rows = candidate_rows
        if valid_rows == target_rows:
            break
    valid_mask = np.asarray([group in valid_groups for group in groups], dtype=bool)
    assert valid_mask.any() and (~valid_mask).any()
    assert not set(groups[valid_mask]) & set(groups[~valid_mask])
    return indices[~valid_mask], indices[valid_mask]


def main() -> int:
    args = parse_args()
    train = pd.read_csv(args.train, dtype={ID: str})
    targets = pd.read_csv(args.targets, dtype={ID: str})
    assert train[ID].tolist() == targets[ID].tolist()
    assert len(train) == 4_407
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold = np.where(is_gold)[0]
    pseudo = np.where(~is_gold)[0]
    assert len(gold) == 58 and len(pseudo) == 4_349
    hashes = train["Report"].map(report_hash).to_numpy(str)

    base_pseudo_train, base_pseudo_valid = stable_group_split(
        pseudo,
        hashes[pseudo],
        valid_fraction=0.10,
        seed=PSEUDO_SPLIT_SEED,
    )
    assert not set(hashes[base_pseudo_train]) & set(hashes[base_pseudo_valid])

    seen_valid: list[int] = []
    summaries: list[dict[str, object]] = []
    expert_splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (expert_train_pos, expert_valid_pos) in enumerate(
        expert_splitter.split(gold)
    ):
        expert_train = gold[expert_train_pos]
        expert_valid = gold[expert_valid_pos]
        seen_valid.extend(expert_valid.tolist())
        valid_hashes = set(hashes[expert_valid])
        train_hashes = set(hashes[expert_train])
        pseudo_train = np.asarray(
            [index for index in base_pseudo_train if hashes[index] not in valid_hashes],
            dtype=int,
        )
        pseudo_valid = np.asarray(
            [
                index
                for index in base_pseudo_valid
                if hashes[index] not in valid_hashes
                and hashes[index] not in train_hashes
            ],
            dtype=int,
        )
        representation = np.concatenate([pseudo_train, expert_train]).astype(int)
        dropped: list[str] = []
        if len(representation) % 2:
            pseudo_positions = np.where(~is_gold[representation])[0]
            assert len(pseudo_positions) > 0
            drop_position = int(pseudo_positions[-1])
            dropped.append(train.iloc[representation[drop_position]][ID])
            representation = np.delete(representation, drop_position)

        assert len(expert_valid) in {11, 12}
        assert len(expert_train) + len(expert_valid) == 58
        assert len(pseudo_valid) > 400
        assert len(representation) % 2 == 0
        assert not set(hashes[representation]) & valid_hashes
        assert not set(hashes[pseudo_valid]) & set(hashes[representation])
        assert not set(hashes[pseudo_valid]) & valid_hashes

        bundle = args.bundle_root / f"fold-{fold}"
        metadata_path = bundle / "kernel-metadata.json"
        code_path = bundle / f"rsna-p4-coatnet-e2e-fold-{fold}.py"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["id"] == (
            f"vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-{fold}"
        )
        assert metadata["code_file"] == code_path.name
        assert metadata["enable_gpu"] is True
        assert metadata["enable_internet"] is False
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        code = code_path.read_text()
        assert f"P4_FOLD = {fold}\n" in code
        assert f"P4_FOLD = {(fold + 1) % N_FOLDS}\n" not in code
        assert EXPECTED_CACHE_SIGNATURE in code
        py_compile.compile(str(code_path), doraise=True)

        summaries.append({
            "fold": fold,
            "representation_rows": len(representation),
            "pseudo_train_rows": len(pseudo_train) - len(dropped),
            "pseudo_valid_rows": len(pseudo_valid),
            "expert_train_rows": len(expert_train),
            "expert_valid_rows": len(expert_valid),
            "expert_valid_report_twins_removed": int(
                sum(hashes[index] in valid_hashes for index in pseudo)
            ),
            "pseudo_valid_expert_train_twins_removed": int(
                sum(
                    hashes[index] in train_hashes
                    and hashes[index] not in valid_hashes
                    for index in base_pseudo_valid
                )
            ),
            "ddp_balance_dropped_studies": dropped,
        })

    assert len(seen_valid) == len(set(seen_valid)) == 58
    assert set(seen_valid) == set(gold.tolist())
    print(json.dumps({
        "validated": True,
        "studies": len(train),
        "pseudo_base_train": len(base_pseudo_train),
        "pseudo_base_valid": len(base_pseudo_valid),
        "expert_oof_coverage": len(seen_valid),
        "folds": summaries,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
