"""Independent split/artifact validation, OOF aggregation and promotion gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ID, TARGETS, exclusion_groups, metrics, paired_bootstrap, read_splits, sha256, write_json


def validate_run(root, splits, split_path, allow_smoke=False):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if m["split_sha256"] != sha256(split_path):
        raise ValueError("Split hash mismatch")
    smoke = m["status"] == "smoke_complete"
    if smoke and not allow_smoke:
        raise ValueError("Smoke predictions cannot count as OOF")
    for name, key in (("best.pt", "checkpoint_sha256"), ("predictions.csv", "predictions_sha256")):
        if sha256(root / name) != m[key]:
            raise ValueError(f"Artifact mismatch: {name}")
    fold, mode = m["fold"], m["mode"]
    if mode == "refit":
        raise ValueError("Full-data refit is not validation evidence")
    train_allowed = set(splits.loc[splits[f"{mode}_train_{fold}"], ID])
    train = set(m["train_ids"])
    if (not smoke and train != train_allowed) or not train <= train_allowed:
        raise ValueError("Training membership mismatch")
    pseudo_valid = set(splits.loc[splits.role == "pseudo_valid", ID])
    if (not smoke and set(m["validation_ids"]) != pseudo_valid) or not set(m["validation_ids"]) <= pseudo_valid:
        raise ValueError("Validation membership mismatch")
    indexed = splits.set_index(ID)
    groups = exclusion_groups(indexed)
    forbidden = set(groups.loc[m["validation_ids"]])
    if mode == "student":
        expected = set(splits.loc[splits.is_gold & (splits.expert_fold == fold), ID])
    else:
        expected = set(splits.loc[((splits.role == "pseudo_train") & (splits.teacher_fold == fold)) |
                                 (splits.role == "pseudo_valid") | splits.is_gold, ID])
    pred = pd.read_csv(root / "predictions.csv")
    if pred[ID].duplicated().any() or (not smoke and set(pred[ID]) != expected) or not set(pred[ID]) <= expected:
        raise ValueError("Prediction coverage mismatch")
    forbidden |= set(groups.loc[pred[ID]])
    if set(groups.loc[list(train)]) & forbidden:
        raise ValueError("Training/prediction group overlap")
    values = pred[TARGETS].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid probabilities")
    history = pd.read_csv(root / "history.csv")
    if not np.isfinite(history[["train_bce", "original_report_valid_bce"]]).all().all():
        raise ValueError("Nonfinite losses")
    best = int(history.loc[history.original_report_valid_bce.idxmin(), "epoch"])
    if m["best_epoch"] != best:
        raise ValueError("Checkpoint not selected by original report BCE")
    return m, pred


def aggregate(roots, splits, split_path, metadata):
    rows, manifests, seen = [], [], set()
    for root in roots:
        m, pred = validate_run(root, splits, split_path)
        if m["mode"] != "student" or m["fold"] in seen:
            raise ValueError("Need exactly five student folds")
        seen.add(m["fold"])
        pred["fold"] = m["fold"]
        rows.append(pred)
        manifests.append(m)
    if seen != set(range(5)) or any(m["config"] != manifests[0]["config"] for m in manifests[1:]):
        raise ValueError("Five matched folds required")
    combined = pd.concat(rows).set_index(ID)
    gold = splits.is_gold
    combined = combined.loc[splits.loc[gold, ID]]
    y = metadata.loc[gold, TARGETS].to_numpy()
    return combined, metrics(y, combined[TARGETS].to_numpy())


def promotion(y, baselines, candidates, groups, runtime_hours, repeats=5000):
    if len(baselines) != 2 or len(candidates) != 2:
        raise ValueError("Promotion requires two matched seeds")
    deltas = [metrics(y, c)["MACRO"] - metrics(y, b)["MACRO"] for b, c in zip(baselines, candidates)]
    base, candidate = np.mean(baselines, axis=0), np.mean(candidates, axis=0)
    bm, cm = metrics(y, base), metrics(y, candidate)
    target_deltas = {t: cm[t] - bm[t] for t in TARGETS}
    boot = paired_bootstrap(y, base, candidate, groups, repeats)
    gates = {"both_seed_wins": all(d > 0 for d in deltas),
             "mean_gain_positive": float(np.mean(deltas)) > 0,
             "seed_ensemble_gain_positive": cm["MACRO"] > bm["MACRO"],
             "bootstrap": boot["probability_positive"] >= .90,
             "target_floor": min(target_deltas.values()) >= -.02,
             "runtime": np.isfinite(runtime_hours) and 0 < runtime_hours <= 7.5}
    return {"seed_deltas": deltas, "target_deltas": target_deltas, "paired_bootstrap": boot,
            "gates": gates, "passed": all(gates.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = pd.read_csv(args.train)
    splits = read_splits(args.splits, meta)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        if len(args.roots) != 1:
            raise ValueError("One smoke run required")
        manifest, predictions = validate_run(args.roots[0], splits, args.splits, True)
        if manifest["status"] != "smoke_complete":
            raise ValueError("Not a smoke run")
        gates = {"two_gpus": manifest["world_size"] == 2,
                 "memory": manifest["peak_cuda_memory_mib"] < 14500,
                 "predictions": len(predictions) > 0}
        result = {"status": "validated_smoke" if all(gates.values()) else "failed_smoke", "gates": gates,
                  "config": manifest["config"], "manifest_sha256": sha256(args.roots[0] / "manifest.json"),
                  "split_sha256": sha256(args.splits), "independent_image_replay": False}
        write_json(args.output / "validation.json", result)
        if not all(gates.values()):
            raise ValueError("Smoke resource gates failed")
    else:
        predictions, scores = aggregate(args.roots, splits, args.splits, meta)
        predictions.to_csv(args.output / "oof.csv")
        write_json(args.output / "validation.json", {"status": "artifact_and_membership_validated",
                   "metrics": scores, "oof_sha256": sha256(args.output / "oof.csv"),
                   "independent_image_replay": False, "promotion_eligible": False})


if __name__ == "__main__":
    main()
