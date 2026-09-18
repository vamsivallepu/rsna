"""Create a versioned report-and-image-group split without changing expert folds."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .contracts import ID, make_splits, read_splits, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--pixel-matches", type=Path, required=True)
    p.add_argument("--verification", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta, matches = pd.read_csv(args.train), pd.read_csv(args.pixel_matches)
    pairs = []
    for _, rows in matches.groupby("selected_pixels_sha256"):
        ids = rows[ID].tolist()
        pairs.extend([[ids[0], uid] for uid in ids[1:]])
    if not pairs:
        raise ValueError("No duplicate image groups")
    verified = False
    if args.verification:
        v = json.loads(args.verification.read_text())
        covered = {r[ID] for r in v["studies"]}
        images = {r[ID]: r for r in v["studies"]}
        verified = v["status"] == "verified_image_overlap" and v["source_train_sha256"] == sha256(args.train)
        verified &= all(set(pair) <= covered for pair in pairs)
        if verified:
            verified &= all(images[a]["reconstructed_cache_match"] and images[b]["reconstructed_cache_match"] and
                            images[a]["reconstructed_windows_sha256"] == images[b]["reconstructed_windows_sha256"]
                            for a, b in pairs)
        if not verified:
            raise ValueError("Raw verification does not cover the image-group split")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "splits.csv"
    split = make_splits(meta, pairs)
    if path.exists() and path.read_text() != split.to_csv(index=False):
        raise ValueError("Refusing to overwrite a different split")
    split.to_csv(path, index=False)
    manifest = {"split_revision": "P10-report-image-v2", "image_pairs": pairs,
                "split_sha256": sha256(path), "train_sha256": sha256(args.train),
                "pixel_matches_sha256": sha256(args.pixel_matches),
                "verification_sha256": sha256(args.verification) if args.verification else None,
                "full_training_allowed": bool(verified),
                "status": "verified_image_group_split" if verified else "quarantined_pending_raw_verification"}
    write_json(path.with_suffix(".groups.json"), manifest)
    read_splits(path, meta)
    old = make_splits(meta)
    assert split.expert_fold.equals(old.expert_fold)
    assert set(split.loc[split.role == "pseudo_valid", ID]) == set(old.loc[old.role == "pseudo_valid", ID])
    write_json(args.output / "manifest.json", {**manifest, "roles": split.role.value_counts().to_dict(),
               "teacher_fold_counts": split.loc[split.role == "pseudo_train"].teacher_fold.value_counts().to_dict(),
               "expert_folds_unchanged": True, "original_validation_membership_unchanged": True})
    print(json.dumps({"status": manifest["status"], "roles": split.role.value_counts().to_dict()}))


if __name__ == "__main__":
    main()
