"""Immutable P2/P14 pixel readers and reproducible slot-balanced sampling."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .contracts import BOUNDS, ID, sha256

P2_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"


def sample_windows(slots, count, rng=None, balanced=False):
    slots = np.asarray(slots)
    available = np.flatnonzero(slots >= 0)
    if not len(available):
        return np.zeros(count, int), np.zeros(count, bool)
    if rng is None:
        selected = available[np.round(np.linspace(0, len(available) - 1, min(count, len(available)))).astype(int)]
    elif balanced:
        populated = np.unique(slots[available])
        if count < len(populated):
            raise ValueError("Not enough windows for every populated slot")
        selected = np.array([rng.choice(np.flatnonzero(slots == slot)) for slot in populated])
        remainder = np.setdiff1d(available, selected)
        n = count - len(selected)
        if n:
            pool = remainder if len(remainder) else available
            selected = np.r_[selected, rng.choice(pool, n, replace=len(pool) < n)]
        rng.shuffle(selected)
    else:
        selected = rng.choice(available, count, replace=len(available) < count)
    result = np.zeros(count, int)
    valid = np.zeros(count, bool)
    result[:len(selected)], valid[:len(selected)] = selected, True
    return result, valid


class PixelCache:
    def __init__(self, manifests, study_ids, verify=True):
        self.parts, ids, self.manifest_hashes = [], [], {}
        self.kind = None
        signature = None
        for path in map(Path, manifests):
            m = json.loads(path.read_text())
            kind = "p14" if m["cache_id"].startswith("P14") else "p2"
            if self.kind is not None and self.kind != kind:
                raise ValueError("Cannot mix cache formats")
            self.kind = kind
            current = m["cache_signature_sha256"]
            if (kind == "p2" and current != P2_SIGNATURE) or (signature is not None and signature != current):
                raise ValueError("Cache signature mismatch")
            signature = current
            if verify:
                for name, expected in m["files_sha256"].items():
                    if sha256(path.parent / name) != expected:
                        raise ValueError(f"Corrupt cache artifact {name}")
            self.manifest_hashes[str(path)] = sha256(path)
            if kind == "p2":
                suffix = path.stem.rsplit("_", 1)[-1]
                names = [f"p2_pixels_part_{suffix}.npy", f"p2_valid_mask_part_{suffix}.npy", f"p2_studies_part_{suffix}.csv"]
            else:
                names = [m["pixels"], m["mask"], m["studies"]]
            volume = np.load(path.parent / names[0], mmap_mode="r")
            mask = np.load(path.parent / names[1]).astype(bool)
            part_ids = pd.read_csv(path.parent / names[2])[ID].tolist()
            extra = None
            if kind == "p14":
                extra = (np.load(path.parent / m["slots"]), np.load(path.parent / m["positions"]))
            self.parts.append((len(ids), volume, mask, extra))
            ids.extend(part_ids)
        if ids != list(study_ids):
            raise ValueError("Cache study order does not match metadata")
        self.signature = signature

    def windows(self, index, count, rng=None, balanced=False):
        for offset, volume, masks, extra in reversed(self.parts):
            if index >= offset:
                i = index - offset
                if self.kind == "p14":
                    all_slots, positions = extra[0][i].copy(), extra[1][i]
                    all_slots[~masks[i]] = -1
                    pick, valid = sample_windows(all_slots, count, rng, balanced)
                    return np.array(volume[i, pick]), valid, all_slots[pick], positions[pick].astype(np.float32)
                centers, slots, positions = [], [], []
                for slot, (start, stop) in enumerate(BOUNDS):
                    for center in range(start + 1, stop - 1):
                        if masks[i, center - 1:center + 2].all():
                            centers.append(center)
                            slots.append(slot)
                            positions.append(2 * (center - start) / (stop - start - 1) - 1)
                if not centers:
                    size = volume.shape[-1]
                    return np.zeros((count, 3, size, size), np.uint8), np.zeros(count, bool), np.full(count, -1), np.zeros(count, np.float32)
                pick, valid = sample_windows(slots, count, rng, balanced)
                windows = np.stack([volume[i, centers[k] - 1:centers[k] + 2] for k in pick])
                return windows, valid, np.array(slots)[pick], np.array(positions, np.float32)[pick]
        raise IndexError(index)


class StudyDataset(Dataset):
    def __init__(self, cache, indices, labels, weights, count=32, seed=None, balanced=False):
        self.cache, self.indices, self.labels, self.weights = cache, indices, labels, weights
        self.count, self.seed, self.balanced = count, seed, balanced

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        rng = None if self.seed is None else np.random.default_rng(self.seed + 10007 * index + position)
        x, mask, slots, relative = self.cache.windows(index, self.count, rng, self.balanced)
        return index, x, mask, slots, relative, self.labels[index], self.weights[index]
