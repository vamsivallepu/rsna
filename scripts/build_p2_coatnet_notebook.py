#!/usr/bin/env python3
"""Build the frozen P2 wide-coverage CoAtNet-RMLP2 experiment notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/jupyter-notebook/rsna-p2-coatnet-rmlp2.ipynb"),
    )
    args = parser.parse_args()

    cells = [
        markdown(
            """
# P2 experiment: wide-coverage CoAtNet-RMLP2

**Objective.** Test the one frozen configuration in `docs/P2_PROTOCOL.md` against
the honest P0-B DINO OOF baseline (`0.7967883`). This notebook consumes the
immutable two-part `P2-CACHE-v1` cache; it never decodes the full DICOM corpus.

**Leakage boundary.** The CoAtNet representation is trained on 4,349 pseudo-labelled
studies only and selected solely by report-grouped pseudo-validation weighted BCE.
All 58 expert studies remain untouched until the direct gate. Five subsequent
feature-only heads use fold-wise report-group purging to create one honest OOF
prediction per expert study.

**Initialization.** Only the official Apache-2.0 timm ImageNet-12k → ImageNet-1k
checkpoint is accepted. Its SHA-256 is hard-coded below. Public competition-trained
weights and gold-selected public procedures are prohibited.
"""
        ),
        markdown("## Setup and frozen constants"),
        code(
            r"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import unicodedata
from pathlib import Path

for candidate in (Path.cwd() / ".deps", Path.cwd().parent / ".deps", Path.cwd().parent.parent / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch.utils.data import DataLoader, Dataset

ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ARCH = "coatnet_rmlp_2_rw_384"
MODEL_ID = "timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
MODEL_SHA256 = "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
CACHE_ID = "P2-CACHE-v1"
EXPECTED_CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
P2_SEED = 20260902
FOLD_SEED = 20260831
N_FOLDS = 5
N_TARGETS = len(TARGETS)
FEATURE_DIM = 1024
CACHE_SIZE = 336
MODEL_SIZE = 384
K_TRAIN = 8
K_EVAL = 32
REP_EPOCHS = 8
HEAD_EPOCHS = 30
STUDY_BATCH = 2
GRAD_ACCUM = 4
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3
REP_WEIGHT_DECAY = 0.02
HEAD_FOLD_LR = 2e-4
HEAD_FOLD_WEIGHT_DECAY = 1e-3
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
ON_KAGGLE = Path("/kaggle/working").is_dir()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_RUN = ON_KAGGLE and DEVICE.type == "cuda"
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/p2_coatnet_smoke")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if not FULL_RUN:
    REP_EPOCHS = 1
    HEAD_EPOCHS = 1
    STUDY_BATCH = 1
    GRAD_ACCUM = 1
    K_TRAIN = 1
    K_EVAL = 1

random.seed(P2_SEED)
np.random.seed(P2_SEED)
torch.manual_seed(P2_SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(P2_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print({
    "python": platform.python_version(), "torch": torch.__version__,
    "timm": timm.__version__, "device": str(DEVICE), "full_run": FULL_RUN,
    "rep_epochs": REP_EPOCHS, "k_train": K_TRAIN, "k_eval": K_EVAL,
})
if ON_KAGGLE:
    assert DEVICE.type == "cuda", "P2 training requires a Kaggle GPU"
    print(torch.cuda.get_device_name(0))
"""
        ),
        markdown("## Resolve and cryptographically validate every input"),
        code(
            r"""
def candidate_roots() -> list[Path]:
    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    if Path("/kaggle/input").is_dir():
        roots.append(Path("/kaggle/input"))
    roots.extend([Path("/tmp/rsna-coatnet-rmlp2-init")])
    return roots


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_named_file(filename: str, minimum_bytes: int = 1) -> Path:
    direct = []
    for root in candidate_roots():
        direct.extend([
            root / filename,
            root / "data/raw" / filename,
            root / "outputs/kaggle/rsna-report-weak-supervision" / filename,
            root / "kaggle/rsna-p2-wide-cache-a/outputs/p2_cache_smoke_a" / filename,
        ])
    for path in direct:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for path in kaggle_input.glob(f"**/{filename}"):
            if path.is_file() and path.stat().st_size >= minimum_bytes:
                return path.resolve()
    raise FileNotFoundError(filename)


TRAIN_PATH = find_named_file("train.csv", 100_000)
TARGET_PATH = find_named_file("training_targets.csv", 100_000)
MODEL_PATH = find_named_file("model.safetensors", 250_000_000)
assert sha256_file(MODEL_PATH) == MODEL_SHA256, "official model hash mismatch"


def load_cache_part(part: str) -> dict:
    manifest_path = find_named_file(f"p2_cache_manifest_part_{part}.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_id"] == CACHE_ID
    assert manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
    root = manifest_path.parent
    names = {
        "pixels": f"p2_pixels_part_{part}.npy",
        "mask": f"p2_valid_mask_part_{part}.npy",
        "studies": f"p2_studies_part_{part}.csv",
    }
    paths = {key: root / value for key, value in names.items()}
    for key, path in paths.items():
        assert path.is_file(), path
        expected = manifest["files_sha256"][names[key]]
        observed = sha256_file(path)
        assert observed == expected, f"{path.name} hash mismatch"
    return {"manifest": manifest, "root": root, **paths}


part_a = load_cache_part("a")
if FULL_RUN:
    part_b = load_cache_part("b")
    assert part_a["manifest"]["row_range"] == [0, 2204]
    assert part_b["manifest"]["row_range"] == [2204, 4407]
    assert part_a["manifest"]["contract"] == part_b["manifest"]["contract"]
else:
    part_b = None

train_all = pd.read_csv(TRAIN_PATH, dtype={ID: str})
targets_all = pd.read_csv(TARGET_PATH, dtype={ID: str})
ids_a = pd.read_csv(part_a["studies"], dtype={ID: str})[ID].tolist()
ids_b = pd.read_csv(part_b["studies"], dtype={ID: str})[ID].tolist() if part_b else []
study_ids = ids_a + ids_b
if FULL_RUN:
    assert len(study_ids) == 4_407 and study_ids == train_all[ID].tolist()
else:
    assert len(study_ids) == 1

metadata = train_all.set_index(ID).loc[study_ids].reset_index()
targets = targets_all.set_index(ID).loc[study_ids].reset_index()
assert metadata[ID].tolist() == targets[ID].tolist() == study_ids
assert targets[TARGETS].notna().all().all()

pixels_a = np.load(part_a["pixels"], mmap_mode="r")
mask_a = np.load(part_a["mask"], mmap_mode="r").astype(bool)
if part_b:
    pixels_b = np.load(part_b["pixels"], mmap_mode="r")
    mask_b = np.load(part_b["mask"], mmap_mode="r").astype(bool)
else:
    pixels_b = mask_b = None
slot_bounds = np.load(part_a["root"] / "p2_slot_bounds.npy").astype(int)
assert slot_bounds.tolist() == [[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]]

print({
    "train": str(TRAIN_PATH), "targets": str(TARGET_PATH),
    "model": str(MODEL_PATH), "model_sha256": MODEL_SHA256,
    "studies": len(study_ids), "cache_signature": EXPECTED_CACHE_SIGNATURE,
    "cache_parts": [str(part_a["root"]), str(part_b["root"]) if part_b else None],
})
"""
        ),
        markdown("## Boundary-safe window dataset and grouped pseudo split"),
        code(
            r"""
def normalize_report(report: object) -> str:
    text = unicodedata.normalize("NFKC", str(report)).casefold()
    return "".join(character for character in text if character.isalnum())


def report_hash(report: object) -> str:
    normalized = normalize_report(report)
    if not normalized:
        raise ValueError("empty report after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


report_hashes = metadata["Report"].map(report_hash).to_numpy(str)
y = targets[TARGETS].to_numpy(np.float32)
weights = targets[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
is_gold = targets["is_gold"].astype(bool).to_numpy()
gold_indices = np.where(is_gold)[0]
pseudo_indices = np.where(~is_gold)[0]

if FULL_RUN:
    assert len(gold_indices) == 58 and len(pseudo_indices) == 4_349
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=P2_SEED)
    train_pos, valid_pos = next(splitter.split(pseudo_indices, groups=report_hashes[pseudo_indices]))
    adapt_indices = pseudo_indices[train_pos]
    pseudo_valid_indices = pseudo_indices[valid_pos]
    overlap = set(report_hashes[adapt_indices]) & set(report_hashes[pseudo_valid_indices])
    assert not overlap
else:
    adapt_indices = np.arange(len(study_ids))
    pseudo_valid_indices = np.arange(len(study_ids))


def read_study(index: int) -> tuple[np.ndarray, np.ndarray]:
    if index < len(pixels_a):
        return np.asarray(pixels_a[index]), np.asarray(mask_a[index])
    assert pixels_b is not None and mask_b is not None
    local = index - len(pixels_a)
    return np.asarray(pixels_b[local]), np.asarray(mask_b[local])


def valid_centers(mask: np.ndarray) -> np.ndarray:
    centers = []
    for start, stop in slot_bounds:
        for center in range(start + 1, stop - 1):
            if mask[center - 1:center + 2].all():
                centers.append(center)
    return np.asarray(centers, dtype=np.int16)


def choose_centers(
    mask: np.ndarray, count: int, *, stochastic: bool, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    available = valid_centers(mask)
    if len(available) == 0:
        return np.zeros(count, dtype=np.int16), np.zeros(count, dtype=bool)
    if stochastic:
        rng = np.random.default_rng(seed)
        selected = rng.choice(available, size=count, replace=len(available) < count)
        return selected.astype(np.int16), np.ones(count, dtype=bool)
    if len(available) <= count:
        selected = np.zeros(count, dtype=np.int16)
        selected[:len(available)] = available
        valid = np.zeros(count, dtype=bool)
        valid[:len(available)] = True
        return selected, valid
    positions = np.round(np.linspace(0, len(available) - 1, count)).astype(int)
    return available[positions], np.ones(count, dtype=bool)


def build_windows(index: int, count: int, stochastic: bool, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    volume, mask = read_study(index)
    centers, window_mask = choose_centers(
        mask, count, stochastic=stochastic, seed=P2_SEED + 10_007 * epoch + index
    )
    windows = np.zeros((count, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    for out_index, center in enumerate(centers):
        if window_mask[out_index]:
            windows[out_index] = volume[int(center) - 1:int(center) + 2]
    return windows, window_mask


class StudyDataset(Dataset):
    def __init__(self, indices: np.ndarray, count: int, stochastic: bool):
        self.indices = np.asarray(indices, dtype=int)
        self.count = count
        self.stochastic = stochastic
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        windows, window_mask = build_windows(index, self.count, self.stochastic, self.epoch)
        return index, windows, window_mask, y[index], weights[index]


probe_windows, probe_mask = build_windows(0, K_TRAIN, stochastic=False, epoch=0)
assert probe_windows.shape == (K_TRAIN, 3, CACHE_SIZE, CACHE_SIZE)
assert probe_mask.any()
print({
    "pseudo_train": len(adapt_indices), "pseudo_valid": len(pseudo_valid_indices),
    "gold": len(gold_indices), "probe_valid_windows": int(probe_mask.sum()),
    "pseudo_group_overlap": 0 if FULL_RUN else None,
})
"""
        ),
        markdown("## Official CoAtNet and target-specific attention MIL"),
        code(
            r"""
class TargetAttentionHead(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, dropout: float = 0.20):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(), nn.Dropout(dropout), nn.Linear(256, N_TARGETS)
        )
        self.target_weight = nn.Parameter(torch.randn(N_TARGETS, feature_dim) * 0.02)
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(features)
        scores = self.attention(hidden).transpose(1, 2)
        scores = scores.masked_fill(~mask[:, None, :].bool(), -1e4)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("btk,bkf->btf", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


class CoAtNetMIL(nn.Module):
    def __init__(self, weight_path: Path):
        super().__init__()
        backbone = timm.create_model(ARCH, pretrained=False)
        state = load_file(str(weight_path), device="cpu")
        incompatible = backbone.load_state_dict(state, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        backbone.reset_classifier(0, global_pool="avg")
        assert backbone.num_features == FEATURE_DIM
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(enable=True)
        self.backbone = backbone
        self.head = TargetAttentionHead()
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1))

    def encode(self, windows: torch.Tensor, augment: bool = False) -> torch.Tensor:
        batch, count = windows.shape[:2]
        pixels = windows.reshape(batch * count, 3, CACHE_SIZE, CACHE_SIZE).float().div(255.0)
        pixels = F.interpolate(pixels, size=(MODEL_SIZE, MODEL_SIZE), mode="bicubic", align_corners=False)
        if augment:
            gain = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(0.90, 1.10)
            offset = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(-0.05, 0.05)
            pixels = (pixels * gain + offset).clamp(0.0, 1.0)
        pixels = (pixels - self.mean) / self.std
        return self.backbone(pixels).reshape(batch, count, FEATURE_DIM)

    def forward(self, windows: torch.Tensor, mask: torch.Tensor, augment: bool = False) -> torch.Tensor:
        return self.head(self.encode(windows, augment=augment), mask)


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * label_weights).sum() / label_weights.sum().clamp_min(1.0)


model = CoAtNetMIL(MODEL_PATH).to(DEVICE)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert 73_000_000 < parameter_count < 76_000_000
print({"parameters": parameter_count, "feature_dim": model.backbone.num_features})
"""
        ),
        markdown("## Fixed pseudo-only representation training"),
        code(
            r"""
def move_batch(batch):
    index, windows, window_mask, labels, label_weights = batch
    return (
        index.numpy(),
        windows.to(DEVICE, non_blocking=True),
        window_mask.to(DEVICE, non_blocking=True).bool(),
        labels.to(DEVICE, non_blocking=True),
        label_weights.to(DEVICE, non_blocking=True),
    )


@torch.inference_mode()
def validation_loss(indices: np.ndarray) -> float:
    dataset = StudyDataset(indices, K_EVAL, stochastic=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    numerator = 0.0
    denominator = 0.0
    for batch in loader:
        _, windows, window_mask, labels, label_weights = move_batch(batch)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(windows, window_mask, augment=False)
            raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        numerator += float((raw.float() * label_weights).sum())
        denominator += float(label_weights.sum())
    return numerator / max(denominator, 1.0)


backbone_parameters = list(model.backbone.parameters())
optimizer = torch.optim.AdamW([
    {"params": backbone_parameters, "lr": BACKBONE_LR},
    {"params": model.head.parameters(), "lr": HEAD_LR},
], weight_decay=REP_WEIGHT_DECAY)
train_dataset = StudyDataset(adapt_indices, K_TRAIN, stochastic=True)
updates_per_epoch = math.ceil(math.ceil(len(train_dataset) / STUDY_BATCH) / GRAD_ACCUM)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[BACKBONE_LR, HEAD_LR],
    total_steps=REP_EPOCHS * updates_per_epoch,
    pct_start=0.15,
)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
history = []
best_loss = float("inf")
best_epoch = -1
best_state = None
training_started = time.time()

for epoch in range(REP_EPOCHS):
    train_dataset.epoch = epoch
    generator = torch.Generator().manual_seed(P2_SEED + epoch)
    loader = DataLoader(
        train_dataset, batch_size=STUDY_BATCH, shuffle=True, generator=generator,
        num_workers=0, pin_memory=DEVICE.type == "cuda",
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(loader):
        _, windows, window_mask, labels, label_weights = move_batch(batch)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(windows, window_mask, augment=True)
            loss = weighted_bce(logits, labels, label_weights)
        scaler.scale(loss / GRAD_ACCUM).backward()
        should_update = (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        losses.append(float(loss.detach()))
        if FULL_RUN and (step + 1) % 250 == 0:
            print(
                f"P2 train epoch {epoch + 1}/{REP_EPOCHS} step {step + 1}/{len(loader)} "
                f"loss={np.mean(losses[-50:]):.5f}", flush=True,
            )
    valid_loss = validation_loss(pseudo_valid_indices)
    row = {
        "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
        "pseudo_valid_weighted_bce": valid_loss,
        "elapsed_minutes": (time.time() - training_started) / 60.0,
    }
    history.append(row)
    if valid_loss < best_loss:
        best_loss = valid_loss
        best_epoch = epoch + 1
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    print({"P2_EPOCH_COMPLETE": row, "best_epoch": best_epoch}, flush=True)

assert best_state is not None and best_epoch >= 1
model.load_state_dict(best_state, strict=True)
torch.save(best_state, OUTPUT_ROOT / "p2_coatnet_pseudo_best.pt")
pd.DataFrame(history).to_csv(OUTPUT_ROOT / "p2_representation_history.csv", index=False)
del best_state
gc.collect()
print({"P2_REPRESENTATION_COMPLETE": True, "best_epoch": best_epoch, "best_loss": best_loss})
"""
        ),
        markdown("## Untouched expert gate and reusable CoAtNet feature cache"),
        code(
            r"""
feature_path = OUTPUT_ROOT / "p2_coatnet_features.npy"
feature_cache = np.lib.format.open_memmap(
    feature_path, mode="w+", dtype=np.float16,
    shape=(len(study_ids), K_EVAL, FEATURE_DIM),
)
feature_mask = np.zeros((len(study_ids), K_EVAL), dtype=np.uint8)
feature_positions = np.full((len(study_ids), K_EVAL), -1, dtype=np.int16)
extract_started = time.time()
model.eval()

with torch.inference_mode():
    for index in range(len(study_ids)):
        volume, slice_mask = read_study(index)
        centers, selected_mask = choose_centers(slice_mask, K_EVAL, stochastic=False, seed=0)
        windows = np.zeros((K_EVAL, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        for out_index, center in enumerate(centers):
            if selected_mask[out_index]:
                windows[out_index] = volume[int(center) - 1:int(center) + 2]
        windows_tensor = torch.from_numpy(windows[None]).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            encoded = model.encode(windows_tensor, augment=False)
        feature_cache[index] = encoded[0].float().cpu().numpy().astype(np.float16)
        feature_mask[index] = selected_mask.astype(np.uint8)
        feature_positions[index, selected_mask] = centers[selected_mask]
        if (index + 1) % (100 if FULL_RUN else 1) == 0 or index + 1 == len(study_ids):
            feature_cache.flush()
            print(
                f"P2 features {index + 1:,}/{len(study_ids):,} in "
                f"{(time.time() - extract_started) / 60:.1f} min", flush=True,
            )

assert np.isfinite(feature_cache).all() and feature_mask.any(axis=1).all()
np.save(OUTPUT_ROOT / "p2_coatnet_feature_mask.npy", feature_mask)
np.save(OUTPUT_ROOT / "p2_coatnet_feature_positions.npy", feature_positions)
pd.DataFrame({ID: study_ids}).to_csv(OUTPUT_ROOT / "p2_coatnet_feature_studies.csv", index=False)


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows = [
        {"target": target, "auc": float(roc_auc_score(labels[:, i], predictions[:, i]))}
        for i, target in enumerate(TARGETS)
    ]
    macro = float(np.mean([row["auc"] for row in rows]))
    rows.append({"target": "MACRO", "auc": macro})
    return pd.DataFrame(rows), macro


direct_result = None
if FULL_RUN:
    feature_tensor_gold = torch.from_numpy(
        np.asarray(feature_cache[gold_indices]).astype(np.float32)
    ).to(DEVICE)
    mask_tensor_gold = torch.from_numpy(feature_mask[gold_indices].astype(bool)).to(DEVICE)
    model.head.eval()
    with torch.inference_mode():
        direct_predictions = torch.sigmoid(
            model.head(feature_tensor_gold, mask_tensor_gold)
        ).cpu().numpy()
    direct_metrics, direct_macro = calculate_metrics(y[gold_indices], direct_predictions)
    direct_metrics.to_csv(OUTPUT_ROOT / "p2_direct_expert_metrics.csv", index=False)
    direct_frame = pd.DataFrame({ID: np.asarray(study_ids)[gold_indices]})
    for i, target in enumerate(TARGETS):
        direct_frame[f"label__{target}"] = y[gold_indices, i]
        direct_frame[f"pred__{target}"] = direct_predictions[:, i]
    direct_frame.to_csv(OUTPUT_ROOT / "p2_direct_expert_predictions.csv", index=False)
    direct_result = {"macro_auc": direct_macro}
    print({"P2_DIRECT_EXPERT_MACRO_AUC": direct_macro})
"""
        ),
        markdown("## Five report-group-purged cached-feature heads"),
        code(
            r"""
def paired_bootstrap(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    repeats: int = 5_000, seed: int = FOLD_SEED + 100_000,
) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), len(labels))
        target_deltas = []
        for target_index in range(labels.shape[1]):
            sampled_labels = labels[sampled, target_index]
            if np.unique(sampled_labels).size < 2:
                break
            base_auc = roc_auc_score(sampled_labels, baseline[sampled, target_index])
            cand_auc = roc_auc_score(sampled_labels, candidate[sampled, target_index])
            target_deltas.append(cand_auc - base_auc)
        if len(target_deltas) == labels.shape[1]:
            deltas.append(float(np.mean(target_deltas)))
    values = np.asarray(deltas)
    assert len(values) == repeats
    return {
        "repeats": repeats,
        "mean_delta": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float((values > 0).mean()),
    }


fold_result = None
if FULL_RUN:
    features_tensor = torch.from_numpy(np.asarray(feature_cache).astype(np.float32)).to(DEVICE)
    masks_tensor = torch.from_numpy(feature_mask.astype(bool)).to(DEVICE)
    labels_tensor = torch.from_numpy(y).to(DEVICE)
    weights_tensor = torch.from_numpy(weights).to(DEVICE)
    initial_head_state = {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}
    oof = np.full((len(study_ids), N_TARGETS), np.nan, dtype=np.float32)
    fold_rows = []
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    gold_fold = np.full(len(gold_indices), -1, dtype=int)

    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[gold_valid_pos]
        valid_hashes = set(report_hashes[valid_index])
        keep_pseudo = np.asarray(
            [index for index in pseudo_indices if report_hashes[index] not in valid_hashes], dtype=int
        )
        train_index = np.concatenate([keep_pseudo, gold_indices[gold_train_pos]])
        assert not (set(report_hashes[train_index]) & valid_hashes)
        gold_fold[gold_valid_pos] = fold

        fold_seed = FOLD_SEED + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        head = TargetAttentionHead().to(DEVICE)
        head.load_state_dict(initial_head_state, strict=True)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=HEAD_FOLD_LR, weight_decay=HEAD_FOLD_WEIGHT_DECAY
        )
        generator = torch.Generator().manual_seed(fold_seed)
        train_cpu = torch.as_tensor(train_index, dtype=torch.long)
        final_loss = float("nan")
        for epoch in range(HEAD_EPOCHS):
            head.train()
            permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
            losses = []
            for start in range(0, len(permutation), 256):
                index = permutation[start:start + 256].to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = head(features_tensor[index], masks_tensor[index])
                loss = weighted_bce(logits, labels_tensor[index], weights_tensor[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            final_loss = float(np.mean(losses))
        head.eval()
        with torch.inference_mode():
            oof[valid_index] = torch.sigmoid(
                head(features_tensor[valid_index], masks_tensor[valid_index])
            ).cpu().numpy()
        torch.save(head.state_dict(), OUTPUT_ROOT / f"p2_fold_{fold}_head.pt")
        excluded = len(pseudo_indices) - len(keep_pseudo)
        fold_rows.append({
            "fold": fold, "train_rows": len(train_index), "pseudo_rows": len(keep_pseudo),
            "expert_train_rows": len(gold_train_pos), "expert_valid_rows": len(valid_index),
            "excluded_pseudo_rows": excluded, "report_hash_overlap": 0,
            "final_loss": final_loss,
        })
        print({"P2_HEAD_FOLD_COMPLETE": fold_rows[-1]}, flush=True)

    assert (gold_fold >= 0).all() and np.isfinite(oof[gold_indices]).all()
    oof_metrics, oof_macro = calculate_metrics(y[gold_indices], oof[gold_indices])
    oof_metrics.to_csv(OUTPUT_ROOT / "p2_oof_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "p2_fold_summary.csv", index=False)
    oof_frame = pd.DataFrame({ID: np.asarray(study_ids)[gold_indices], "fold": gold_fold})
    for i, target in enumerate(TARGETS):
        oof_frame[f"label__{target}"] = y[gold_indices, i]
        oof_frame[f"pred__{target}"] = oof[gold_indices, i]
    oof_frame.to_csv(OUTPUT_ROOT / "p2_oof_predictions.csv", index=False)
    fold_result = {"macro_auc": oof_macro}
    print({"P2_OOF_MACRO_AUC": oof_macro})
    display(oof_metrics)
else:
    windows = torch.from_numpy(probe_windows[None]).to(DEVICE)
    window_mask = torch.from_numpy(probe_mask[None]).to(DEVICE)
    labels = torch.from_numpy(y[[0]]).to(DEVICE)
    label_weights = torch.from_numpy(weights[[0]]).to(DEVICE)
    model.train()
    logits = model(windows, window_mask, augment=True)
    loss = weighted_bce(logits, labels, label_weights)
    loss.backward()
    assert logits.shape == (1, N_TARGETS) and torch.isfinite(logits).all()
    (OUTPUT_ROOT / "p2_local_smoke.json").write_text(json.dumps({
        "study_uid": study_ids[0], "forward_shape": list(logits.shape),
        "loss": float(loss.detach()), "backward": True,
    }, indent=2) + "\n")
"""
        ),
        markdown("## Result manifest and next decision"),
        code(
            r"""
artifacts = [
    "p2_coatnet_pseudo_best.pt", "p2_representation_history.csv",
    "p2_coatnet_features.npy", "p2_coatnet_feature_mask.npy",
    "p2_coatnet_feature_positions.npy", "p2_coatnet_feature_studies.csv",
]
if FULL_RUN:
    artifacts.extend([
        "p2_direct_expert_metrics.csv", "p2_direct_expert_predictions.csv",
        "p2_oof_metrics.csv", "p2_oof_predictions.csv", "p2_fold_summary.csv",
        *[f"p2_fold_{fold}_head.pt" for fold in range(N_FOLDS)],
    ])
else:
    artifacts.append("p2_local_smoke.json")
for name in artifacts:
    assert (OUTPUT_ROOT / name).is_file(), name

manifest = {
    "schema_version": 1,
    "run_id": "P2-COATNET-v1",
    "full_run": FULL_RUN,
    "seeds": {"representation": P2_SEED, "folds": FOLD_SEED},
    "studies": len(study_ids),
    "pseudo_studies": int((~is_gold).sum()),
    "expert_studies": int(is_gold.sum()),
    "cache_id": CACHE_ID,
    "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
    "model_id": MODEL_ID,
    "model_sha256": MODEL_SHA256,
    "model_license": "Apache-2.0",
    "feature_dim": FEATURE_DIM,
    "cache_size": CACHE_SIZE,
    "model_size": MODEL_SIZE,
    "normalization": {"mean": MODEL_MEAN, "std": MODEL_STD},
    "k_train": K_TRAIN,
    "k_eval": K_EVAL,
    "representation_epochs": REP_EPOCHS,
    "best_epoch": best_epoch,
    "best_pseudo_valid_weighted_bce": best_loss,
    "direct_expert": direct_result,
    "fold_oof": fold_result,
    "representation_gold_rows": 0 if FULL_RUN else None,
    "pseudo_report_group_overlap": 0 if FULL_RUN else None,
    "artifacts_sha256": {name: sha256_file(OUTPUT_ROOT / name) for name in artifacts},
    "elapsed": {
        "representation_minutes": (extract_started - training_started) / 60.0,
        "feature_extraction_minutes": (time.time() - extract_started) / 60.0,
    },
    "run_context": {
        "device": str(DEVICE), "model_path": str(MODEL_PATH),
        "cache_a": str(part_a["root"]), "cache_b": str(part_b["root"]) if part_b else None,
    },
}
manifest_path = OUTPUT_ROOT / "p2_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print({
    "P2_COMPLETE": True, "best_epoch": best_epoch,
    "direct_macro": None if direct_result is None else direct_result["macro_auc"],
    "oof_macro": None if fold_result is None else fold_result["macro_auc"],
    "manifest_sha256": sha256_file(manifest_path),
})
manifest
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
