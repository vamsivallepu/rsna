"""Reconstruct image predictions from saved checkpoints and immutable pixels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .contracts import ID, TARGETS, read_splits, sha256, write_json
from .data import PixelCache, StudyDataset
from .train import INIT_HASHES, ImageModel
from .validate import validate_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--cache-manifests", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    meta = pd.read_csv(args.train)
    splits = read_splits(args.splits, meta)
    manifest, pred = validate_run(args.run, splits, args.splits, allow_smoke=True)
    cfg = manifest["config"]
    if sha256(args.weights) != INIT_HASHES[cfg["initialization"]]:
        raise ValueError("Initialization hash mismatch")
    cache = PixelCache(args.cache_manifests, meta[ID])
    if cache.signature != manifest["cache_signature"]:
        raise ValueError("Replay cache signature mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageModel(cfg["initialization"], args.weights, cfg.get("head", "baseline")).to(device)
    model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True, map_location=device), strict=True)
    model.eval()
    pred = pred.iloc[:args.limit] if args.limit else pred
    lookup = dict(zip(meta[ID], range(len(meta))))
    indices = np.array([lookup[uid] for uid in pred[ID]])
    zeros = np.zeros((len(meta), 12), np.float32)
    loader = DataLoader(StudyDataset(cache, indices, zeros, zeros), batch_size=1)
    values = []
    with torch.inference_mode():
        for _, x, mask, slots, pos, _, _ in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                values.append(model(x.to(device), mask.to(device), slots.to(device), pos.to(device)).sigmoid().float().cpu().numpy())
    replay = np.concatenate(values)
    delta = float(np.max(np.abs(replay - pred[TARGETS].to_numpy())))
    if not np.isfinite(replay).all() or delta > 5e-4:
        raise ValueError(f"Saved checkpoint replay failed: max error {delta}")
    write_json(args.output, {"status": "replay_validated" if not args.limit else "partial_replay_validated",
        "rows": len(pred), "maximum_absolute_error": delta, "tolerance": 5e-4,
        "run_manifest_sha256": sha256(args.run / "manifest.json"),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "predictions_sha256": manifest["predictions_sha256"]})


if __name__ == "__main__":
    main()
