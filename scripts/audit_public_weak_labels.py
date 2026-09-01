#!/usr/bin/env python3
"""Audit public report-derived weak labels without fitting to the 58 gold rows.

This local helper mirrors the core of the Kaggle weak-supervision notebook. It
uses fixed, predeclared blends so the tiny expert-labelled subset is an audit
set, not a source of blend weights or thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ID = "StudyInstanceUID"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--pilkwang", type=Path, required=True)
    parser.add_argument("--steven", type=Path, required=True)
    parser.add_argument("--lixin", type=Path, required=True)
    parser.add_argument("--nli", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_source(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[ID, *TARGETS])
    if frame[ID].duplicated().any():
        raise ValueError(f"{name}: duplicate {ID} values")
    if frame[TARGETS].isna().any().any():
        raise ValueError(f"{name}: missing target values")
    return frame.rename(columns={target: f"{name}__{target}" for target in TARGETS})


def rank01(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    if train[ID].duplicated().any():
        raise ValueError(f"train: duplicate {ID} values")

    source_paths = {
        "pilkwang": args.pilkwang,
        "steven": args.steven,
        "lixin": args.lixin,
    }
    if args.nli:
        source_paths["nli"] = args.nli

    merged = train[[ID, *TARGETS]].copy()
    for name, path in source_paths.items():
        merged = merged.merge(load_source(path, name), on=ID, how="left", validate="one_to_one")

    source_names = list(source_paths)
    if len(merged) != len(train):
        raise AssertionError("Merge changed the training row count")
    print("Source row coverage:")
    for name in source_names:
        coverage = merged[[f"{name}__{target}" for target in TARGETS]].notna().all(axis=1)
        print(f"  {name:10s} {int(coverage.sum()):,}/{len(merged):,}")

    llm_names = ["pilkwang", "steven", "lixin"]
    output = train[[ID]].copy()
    for target in TARGETS:
        llm_cols = [f"{name}__{target}" for name in llm_names]
        llm_values = merged[llm_cols]
        output[f"median__{target}"] = llm_values.median(axis=1)
        output[f"mean__{target}"] = llm_values.mean(axis=1)
        output[f"rank__{target}"] = pd.concat(
            [rank01(merged[column]) for column in llm_cols], axis=1
        ).mean(axis=1)
        votes = (llm_values > 0.5).astype(float).where(llm_values.notna())
        output[f"vote_fraction__{target}"] = votes.mean(axis=1)
        output[f"agreement__{target}"] = (votes.mean(axis=1) - 0.5).abs() * 2.0
        output[f"spread__{target}"] = llm_values.max(axis=1) - llm_values.min(axis=1)
        if "nli" in source_names:
            four_cols = [*llm_cols, f"nli__{target}"]
            output[f"rank4__{target}"] = pd.concat(
                [rank01(merged[column]) for column in four_cols], axis=1
            ).mean(axis=1)

    gold_mask = merged[TARGETS].notna().all(axis=1)
    gold = merged.loc[gold_mask].reset_index(drop=True)
    gold_output = output.loc[gold_mask].reset_index(drop=True)
    print(f"Studies: {len(train):,}; complete expert rows: {len(gold):,}")

    methods = [*source_names, "median", "mean", "rank"]
    if "nli" in source_names:
        methods.append("rank4")
    rows: list[dict[str, object]] = []
    for method in methods:
        aucs = []
        for target in TARGETS:
            prediction = (
                gold[f"{method}__{target}"]
                if method in source_names
                else gold_output[f"{method}__{target}"]
            )
            y_true = gold[target].astype(int)
            valid = prediction.notna()
            auc = (
                roc_auc_score(y_true[valid], prediction[valid])
                if valid.any() and y_true[valid].nunique() == 2
                else np.nan
            )
            rows.append({"method": method, "target": target, "auc": auc})
            aucs.append(auc)
        rows.append({"method": method, "target": "MACRO", "auc": np.nanmean(aucs)})

    scores = pd.DataFrame(rows)
    table = scores.pivot(index="method", columns="target", values="auc")
    ordered = [*TARGETS, "MACRO"]
    print(table[ordered].round(4).to_string())

    print("\nThree-LLM unanimous vote fraction by target:")
    unanimous = {
        target: float((output[f"agreement__{target}"] == 1.0).mean()) for target in TARGETS
    }
    print(pd.Series(unanimous).sort_values().round(4).to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        scores.to_csv(args.output.with_name(f"{args.output.stem}_audit.csv"), index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
