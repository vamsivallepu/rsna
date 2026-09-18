"""Run P11 two-seed cached CoAtNet head comparisons with saved-head replay."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .contracts import (BOUNDS, ID, TARGETS, VERSION, metrics, paired_bootstrap,
                        read_splits, save_predictions, sha256, write_json)
from .models import AnatomicalHead


def positions_to_metadata(centers, mask):
    slots = np.full(centers.shape, -1, dtype=np.int64)
    relative = np.zeros(centers.shape, dtype=np.float32)
    for slot, (start, stop) in enumerate(BOUNDS):
        valid = mask & (centers > start) & (centers < stop - 1)
        slots[valid] = slot
        relative[valid] = 2 * (centers[valid] - start) / (stop - start - 1) - 1
    if (mask & (slots < 0)).any():
        raise ValueError("Invalid window position or slot-crossing triplet")
    return slots, relative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=Path("outputs/experiments/P10-contract/splits.csv"))
    p.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--targets", type=Path, default=Path("outputs/kaggle/rsna-report-weak-supervision/training_targets.csv"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260831, 20260901])
    p.add_argument("--variants", nargs="+", choices=["baseline", "position", "hierarchical"],
                   default=["baseline", "position", "hierarchical"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.train)
    splits = read_splits(args.splits, metadata)
    targets = pd.read_csv(args.targets).set_index(ID).loc[metadata[ID]]
    studies = pd.read_csv(args.cache / "p2_coatnet_feature_studies.csv")
    if studies[ID].tolist() != metadata[ID].tolist():
        raise ValueError("Cache row order mismatch")
    x = np.load(args.cache / "p2_coatnet_features.npy", mmap_mode="r")
    mask = np.load(args.cache / "p2_coatnet_feature_mask.npy").astype(bool)
    centers = np.load(args.cache / "p2_coatnet_feature_positions.npy")
    if x.shape != (len(splits), 32, 1024) or not np.isfinite(x).all():
        raise ValueError("Invalid feature cache")
    slots, relative = positions_to_metadata(centers, mask)
    y = targets[TARGETS].to_numpy(np.float32)
    w = targets[["weight__" + t for t in TARGETS]].to_numpy(np.float32)
    gold = splits.is_gold.to_numpy()
    if not np.array_equal(y[gold], metadata.loc[gold, TARGETS].to_numpy()):
        raise ValueError("Gold targets changed")
    state = torch.load(args.cache / "p2_coatnet_pseudo_best.pt", map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if k.startswith("head.")}
    source_hashes = {name: sha256(args.cache / name) for name in (
        "p2_coatnet_features.npy", "p2_coatnet_feature_mask.npy", "p2_coatnet_feature_positions.npy",
        "p2_coatnet_feature_studies.csv", "p2_coatnet_pseudo_best.pt")}
    config = {"version": VERSION, "epochs": args.epochs, "batch_size": args.batch_size,
              "seeds": args.seeds, "variants": args.variants, "source_hashes": source_hashes,
              "targets_sha256": sha256(args.targets), "splits_sha256": sha256(args.splits),
              "code_sha256": {n: sha256(Path(__file__).with_name(n)) for n in ("screen.py", "models.py", "contracts.py")},
              "representation_is_historical": True, "promotion_eligible": False,
              "head_training": "historical all-pseudo with held-out expert report purge"}
    manifest_path = args.output / "config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:
        raise ValueError("Refusing to mix different runs in the same output directory")
    write_json(manifest_path, config)
    gold_ix = np.flatnonzero(gold)
    results = {}

    def batch(indices):
        return (torch.from_numpy(np.array(x[indices], dtype=np.float32)),
                torch.from_numpy(mask[indices]), torch.from_numpy(slots[indices]),
                torch.from_numpy(relative[indices]))

    for seed in args.seeds:
        for variant in args.variants:
            run = args.output / f"{variant}-seed-{seed}"
            run.mkdir(exist_ok=True)
            prediction = np.zeros((len(splits), len(TARGETS)), np.float32)
            for fold in range(5):
                started = time.time()
                fold_seed = seed + fold
                random.seed(fold_seed)
                np.random.seed(fold_seed)
                torch.manual_seed(fold_seed)
                model = AnatomicalHead(variant=variant)
                model.load_adapted(state)
                valid = np.flatnonzero(gold & (splits.expert_fold.to_numpy() == fold))
                excluded_groups = set(splits.loc[valid, "report_group"])
                train = np.flatnonzero(~splits.report_group.isin(excluded_groups).to_numpy())
                checkpoint = run / f"fold-{fold}.pt"
                if not checkpoint.exists():
                    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
                    # Reset RNG after variant construction for matched batch order/dropout.
                    torch.manual_seed(fold_seed)
                    rng = np.random.default_rng(fold_seed)
                    for epoch in range(args.epochs):
                        model.train()
                        order = rng.permutation(train)
                        for start in range(0, len(order), args.batch_size):
                            ix = order[start:start + args.batch_size]
                            logits = model(*batch(ix))
                            weights = torch.from_numpy(w[ix])
                            loss = (F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[ix]), reduction="none") * weights).sum() / weights.sum()
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            opt.step()
                        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                            print(json.dumps({"variant": variant, "seed": seed, "fold": fold,
                                              "epoch": epoch + 1, "elapsed_seconds": time.time() - started}), flush=True)
                    torch.save(model.state_dict(), checkpoint.with_suffix(".tmp"))
                    checkpoint.with_suffix(".tmp").replace(checkpoint)
                model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                model.eval()
                with torch.inference_mode():
                    prediction[valid] = model(*batch(valid)).sigmoid().numpy()
                write_json(run / f"fold-{fold}.json", {"checkpoint_sha256": sha256(checkpoint),
                           "train_ids": splits.iloc[train][ID].tolist(), "valid_ids": splits.iloc[valid][ID].tolist()})
            pred = prediction[gold_ix]
            save_predictions(run / "oof.csv", splits.loc[gold, ID], y[gold], pred, splits.loc[gold, "expert_fold"])
            result = {"metrics": metrics(y[gold], pred), "oof_sha256": sha256(run / "oof.csv")}
            results[f"{variant}-{seed}"] = result
            if variant != "baseline" and (args.output / f"baseline-seed-{seed}" / "oof.csv").exists():
                base = pd.read_csv(args.output / f"baseline-seed-{seed}" / "oof.csv")
                bp = base[["pred__" + t for t in TARGETS]].to_numpy()
                result["paired_bootstrap"] = paired_bootstrap(y[gold], bp, pred,
                    splits.loc[gold, "report_group"], repeats=args.bootstrap_repeats)
            write_json(run / "result.json", result)
            write_json(args.output / "results.json", {"status": "screening_only", "promotion_eligible": False, "runs": results})
            print(json.dumps({"completed": str(run), **result}), flush=True)


if __name__ == "__main__":
    main()
