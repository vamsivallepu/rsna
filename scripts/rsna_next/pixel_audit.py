"""Label-blind cache duplicate candidates and acquisition coverage audit."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .contracts import ID, sha256, write_json
from .data import PixelCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--cache-manifests", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = pd.read_csv(args.train, usecols=[ID])
    cache = PixelCache(args.cache_manifests, meta[ID])
    fingerprints, records = [], []
    for i, uid in enumerate(meta[ID]):
        x, mask, slots, _ = cache.windows(i, 32)
        if not mask.any():
            raise ValueError("Empty study cannot be treated as a duplicate")
        digest = hashlib.sha256(x[mask].tobytes() + slots[mask].tobytes()).hexdigest()
        vectors = []
        for slot in range(5):
            selected = mask & (slots == slot)
            mean = x[selected, 1].mean(axis=0) if selected.any() else np.zeros(x.shape[-2:])
            ix = np.linspace(0, mean.shape[0] - 1, 16).astype(int)
            iy = np.linspace(0, mean.shape[1] - 1, 16).astype(int)
            vectors.append(mean[np.ix_(ix, iy)].ravel())
        vec = np.concatenate(vectors).astype(np.float32)
        vec -= vec.mean()
        vec /= max(float(np.linalg.norm(vec)), 1e-6)
        fingerprints.append(vec)
        records.append({ID: uid, "selected_pixels_sha256": digest,
                        **{f"slot_{s}_windows": int((mask & (slots == s)).sum()) for s in range(5)}})
        if (i + 1) % 100 == 0:
            print({"audited": i + 1}, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(records)
    records.to_csv(args.output / "study_fingerprints.csv", index=False)
    records[records.selected_pixels_sha256.duplicated(keep=False)].to_csv(args.output / "exact_selected_pixel_matches.csv", index=False)
    x = np.array(fingerprints)
    nn = NearestNeighbors(n_neighbors=min(4, len(x)), metric="cosine").fit(x)
    distances, neighbors = nn.kneighbors(x)
    candidates = []
    for i, (dd, jj) in enumerate(zip(distances, neighbors)):
        for d, j in zip(dd, jj):
            if j > i and d <= .005:
                candidates.append({"study_a": meta.iloc[i][ID], "study_b": meta.iloc[j][ID], "cosine_similarity": 1 - float(d)})
    pd.DataFrame(candidates, columns=["study_a", "study_b", "cosine_similarity"]).to_csv(args.output / "near_duplicate_candidates.csv", index=False)
    series = pd.read_csv(args.series)
    series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).size().rename("series_count").to_csv(args.output / "acquisition_counts.csv")
    write_json(args.output / "manifest.json", {"status": "candidate_audit_only", "studies": len(meta),
        "cache_signature": cache.signature, "train_sha256": sha256(args.train), "near_candidates": len(candidates),
        "not_patient_identity_proof": True, "requires_candidate_review_before_group_changes": True})


if __name__ == "__main__":
    main()
