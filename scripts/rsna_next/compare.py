"""Apply predeclared paired two-seed promotion gates to saved OOF predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ID, TARGETS, exclusion_groups, read_splits, sha256, write_json
from .infer import blend_predictions
from .validate import promotion


def read_prediction(path, ids):
    df = pd.read_csv(path)
    if df[ID].duplicated().any() or set(df[ID]) != set(ids):
        raise ValueError("OOF coverage must match every expert exactly once")
    columns = ["pred__" + t for t in TARGETS] if "pred__ACL" in df else TARGETS
    values = df.set_index(ID).loc[ids, columns].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite OOF predictions")
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--baseline", type=Path, nargs=2, required=True)
    p.add_argument("--candidate", type=Path, nargs=2, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--release-fingerprint", required=True)
    p.add_argument("--diversity", action="store_true", help="Frozen 80/20 champion/candidate rank blend")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    metadata = pd.read_csv(args.train)
    splits = read_splits(args.splits, metadata)
    gold = splits.is_gold
    ids = splits.loc[gold, ID]
    baseline = [read_prediction(path, ids) for path in args.baseline]
    candidate = [read_prediction(path, ids) for path in args.candidate]
    if args.diversity:
        candidate = [blend_predictions([b, c], [.8, .2]) for b, c in zip(baseline, candidate)]
        baseline = [blend_predictions([b], [1.]) for b in baseline]
    runtime = json.loads(args.runtime.read_text())
    if (runtime.get("release_fingerprint") != args.release_fingerprint or not runtime.get("benchmark") or
            runtime.get("studies", 0) < 200):
        raise ValueError("Runtime evidence must refer to this exact release and at least 200 studies")
    result = promotion(metadata.loc[gold, TARGETS].to_numpy(), baseline, candidate,
                       exclusion_groups(splits)[gold], runtime["projected_hidden_hours"])
    result.update(release_fingerprint=args.release_fingerprint, split_sha256=sha256(args.splits),
                  baseline_sha256=[sha256(p) for p in args.baseline], candidate_sha256=[sha256(p) for p in args.candidate],
                  runtime_sha256=sha256(args.runtime), diversity_80_20=args.diversity,
                  status="statistical_gates_only", submission_eligible=False,
                  note="Requires P10 membership validation and independent replay for every source model; 58-row selection uncertainty remains")
    write_json(args.output, result)


if __name__ == "__main__":
    main()
