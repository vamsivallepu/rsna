#!/usr/bin/env python3
"""Build the leakage-safe DINOv2 pseudo-adaptation Kaggle notebook."""

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
        default=Path("output/jupyter-notebook/rsna-dinov2-pseudo-adaptation.ipynb"),
    )
    args = parser.parse_args()

    cells = [
        markdown(
            """
# RSNA Knee: DINOv2 pseudo-only adaptation

**Purpose.** Replace the weak frozen global-pool baseline with a spatially richer
image representation while preserving a clean expert gate.

The backbone is the official Apache-2.0 Meta DINOv2-small model hosted by Kaggle.
It sees three physically ordered 2.5D windows in each of six plane × sequence slots.
Only the 4,349 report-pseudo-labelled studies adapt the image model; the 58 expert
studies are never used for backbone adaptation or epoch selection. After that fixed
pseudo-only stage, five fold-specific heads may use the non-held-out expert rows, so
every expert study still receives exactly one leakage-safe OOF prediction.

Local execution uses the same DICOM/model/backward path on one representative study.
The full Kaggle run caches uint8 pixels temporarily, deletes that large cache before
exit, and persists only the adapted model, token features, fold heads, metrics, and
manifests.
"""
        ),
        code(
            """
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

for candidate in (Path.cwd() / ".deps", Path.cwd().parent / ".deps", Path.cwd().parent.parent / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

warnings.filterwarnings("ignore", message=".*mode.*deprecated.*", category=DeprecationWarning)

SEED = 20260831
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
SLOTS = [
    (plane, kind)
    for plane in ("Axial", "Coronal", "Sagittal")
    for kind in ("structural", "fluid")
]
SIZE = 224
CROP_MM = 130.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_KAGGLE = Path("/kaggle/working").is_dir()
FULL_RUN = ON_KAGGLE and DEVICE.type == "cuda"
POSITIONS = np.array([0.15, 0.50, 0.85] if FULL_RUN else [0.50], dtype=np.float32)
N_POS = len(POSITIONS)
ADAPT_EPOCHS = 3 if FULL_RUN else 1
HEAD_EPOCHS = 30 if FULL_RUN else 1
STUDY_BATCH = 4 if FULL_RUN else 1
TOKEN_DIM = 768
HIDDEN = 384
N_FOLDS = 5

print({
    "python": sys.version.split()[0], "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
    "pydicom": pydicom.__version__, "device": str(DEVICE),
    "on_kaggle": ON_KAGGLE, "full_run": FULL_RUN,
    "positions": POSITIONS.tolist(),
})
if ON_KAGGLE:
    assert DEVICE.type == "cuda", "Kaggle execution requires a visible GPU"
    print(torch.cuda.get_device_name(0))
"""
        ),
        markdown("## Resolve official inputs and the fixed report teacher"),
        code(
            """
COMPETITION_SLUG = "rsna-knee-abnormality-detection"


def candidate_roots() -> list[Path]:
    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    if Path("/kaggle/input").is_dir():
        roots.append(Path("/kaggle/input"))
    return roots


def find_metadata_root() -> Path:
    candidates = [
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
    ]
    for root in candidate_roots():
        candidates.extend([root / "data/raw", root])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for pattern in ("*/train_series.csv", "*/*/train_series.csv", "*/*/*/train_series.csv"):
            candidates.extend(path.parent for path in kaggle_input.glob(pattern))
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series.csv").is_file():
            return path.resolve()
    raise FileNotFoundError("Could not locate competition metadata")


def find_named_file(filename: str, minimum_bytes: int = 1) -> Path:
    for root in candidate_roots():
        direct = [
            root / filename,
            root / "data/sample/public_weak_labels" / filename,
            root / "outputs/kaggle/rsna-report-weak-supervision" / filename,
        ]
        for path in direct:
            if path.is_file() and path.stat().st_size >= minimum_bytes:
                return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for path in kaggle_input.glob(f"**/{filename}"):
            if path.is_file() and path.stat().st_size >= minimum_bytes:
                return path.resolve()
    raise FileNotFoundError(filename)


def find_dino_root() -> Path:
    direct = [
        Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1"),
        Path("/kaggle/input/dinov2/pytorch/small/1"),
    ]
    for root in candidate_roots():
        direct.append(root / "data/sample/models/dinov2-small")
    for path in direct:
        if (path / "config.json").is_file() and (path / "pytorch_model.bin").is_file():
            return path.resolve()
    raise FileNotFoundError("Official DINOv2-small model was not found")


DATA_ROOT = find_metadata_root()
local_dicom = next(
    (root / "data/sample/files/train_series" for root in candidate_roots()
     if (root / "data/sample/files/train_series").is_dir()),
    None,
)
DICOM_ROOT = (DATA_ROOT / "train_series") if (DATA_ROOT / "train_series").is_dir() else local_dicom
DINO_ROOT = find_dino_root()
try:
    TARGET_PATH = find_named_file("training_targets.csv", 100_000)
except FileNotFoundError:
    TARGET_PATH = None
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/rsna_dino_adaptation_smoke")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
assert DICOM_ROOT is not None and DICOM_ROOT.is_dir()

train = pd.read_csv(DATA_ROOT / "train.csv")
series = pd.read_csv(DATA_ROOT / "train_series.csv")


def build_teacher_targets() -> tuple[pd.DataFrame, dict[str, str]]:
    source_files = {
        "pilkwang": "report_labels_v2.csv",
        "steven": "llm_labels_v4_blend.csv",
        "lixin": "report_labels_gpt56sol.csv",
    }
    paths = {name: find_named_file(filename, 100_000) for name, filename in source_files.items()}
    merged = train[[ID, *TARGETS]].copy()
    for name, path in paths.items():
        source = pd.read_csv(path, usecols=[ID, *TARGETS]).rename(
            columns={target: f"{name}__{target}" for target in TARGETS}
        )
        merged = merged.merge(source, on=ID, how="left", validate="one_to_one")
    gold = train[TARGETS].notna().all(axis=1)
    result = train[[ID]].copy()
    result["is_gold"] = gold.astype(np.uint8)
    result["label_source"] = np.where(gold, "expert", "three_llm_mean")
    for target in TARGETS:
        columns = [f"{name}__{target}" for name in source_files]
        values = merged[columns]
        pseudo = values.mean(axis=1)
        votes = (values > 0.5).astype(float).where(values.notna()).mean(axis=1)
        agreement = (votes - 0.5).abs() * 2.0
        result[target] = train[target].where(gold, pseudo).astype(float)
        result[f"weight__{target}"] = np.where(gold, 2.0, 0.50 + 0.50 * agreement)
    return result, {name: str(path) for name, path in paths.items()}


if TARGET_PATH is not None:
    targets = pd.read_csv(TARGET_PATH)
    TEACHER_ORIGIN = {"type": "verified_kernel_output", "path": str(TARGET_PATH)}
else:
    targets, source_paths = build_teacher_targets()
    TEACHER_ORIGIN = {"type": "in_notebook_fixed_consensus", "paths": source_paths}
targets.to_csv(OUTPUT_ROOT / "training_targets.csv", index=False)

assert len(train) == len(targets) == 4_407
assert len(series) == 24_371
assert train[ID].is_unique and targets[ID].is_unique and series["SeriesInstanceUID"].is_unique
assert int(targets["is_gold"].sum()) == 58
assert targets[TARGETS].notna().all().all()
print({"teacher_origin": TEACHER_ORIGIN, "dino_root": str(DINO_ROOT)})
"""
        ),
        markdown("## Study selection, sequence slots, and physical 2.5D windows"),
        code(
            """
if FULL_RUN:
    study_ids = targets[ID].tolist()
else:
    available = [path.name for path in DICOM_ROOT.iterdir() if path.is_dir()]
    assert available
    study_ids = [max(
        available,
        key=lambda uid: sum(1 for _ in (DICOM_ROOT / uid).glob("*/*.dcm")),
    )]

series = series[series[ID].isin(study_ids)].copy()
series_by_study = {uid: group.to_dict("records") for uid, group in series.groupby(ID, sort=False)}
HEADER_TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber"]


def spatial_key(path: Path) -> float:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)
        orientation = np.asarray(ds.get("ImageOrientationPatient", []), dtype=float)
        position = np.asarray(ds.get("ImagePositionPatient", []), dtype=float)
        if orientation.size == 6 and position.size == 3:
            return float(np.dot(position, np.cross(orientation[:3], orientation[3:])))
        return float(ds.get("InstanceNumber", 0))
    except Exception:
        return 0.0


def ordered_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.glob("*.dcm"), key=spatial_key)


def resize_square(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    return np.asarray(image.resize((SIZE, SIZE), resample=Image.Resampling.BILINEAR))


def read_slice(path: Path) -> np.ndarray | None:
    try:
        ds = pydicom.dcmread(path)
        array = ds.pixel_array.astype(np.float32)
        array = array * float(ds.get("RescaleSlope", 1.0)) + float(ds.get("RescaleIntercept", 0.0))
        if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            array = array.max() + array.min() - array
        spacing_values = ds.get("PixelSpacing", [CROP_MM / max(array.shape)] * 2)
        sy, sx = float(spacing_values[0]), float(spacing_values[1])
        crop_h = min(array.shape[0], max(16, int(round(CROP_MM / max(sy, 1e-6)))))
        crop_w = min(array.shape[1], max(16, int(round(CROP_MM / max(sx, 1e-6)))))
        cy, cx = array.shape[0] // 2, array.shape[1] // 2
        y0 = min(max(0, cy - crop_h // 2), array.shape[0] - crop_h)
        x0 = min(max(0, cx - crop_w // 2), array.shape[1] - crop_w)
        crop = array[y0:y0 + crop_h, x0:x0 + crop_w]
        finite = crop[np.isfinite(crop)]
        if finite.size < 16:
            return None
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if hi <= lo:
            return None
        scaled = np.clip((crop - lo) / (hi - lo), 0, 1)
        return resize_square(np.round(scaled * 255).astype(np.uint8))
    except Exception:
        return None


def sequence_score(record: dict, kind: str) -> int:
    fluid = int(record["Fluid_Sensitive"])
    fat = int(record["Fat_Suppression"])
    if kind == "fluid":
        return 3 if fluid == 1 and fat == 1 else 2 if fluid == 1 else 1 if fat == 1 else 0
    return 3 if fluid == 0 and fat == 0 else 2 if fluid == 0 else 1 if fat == 0 else 0


def choose_series(study_uid: str, records: list[dict], slot: tuple[str, str]) -> Path | None:
    plane, kind = slot
    candidates = []
    for record in records:
        if record["Anatomical_Plane"] != plane:
            continue
        path = DICOM_ROOT / study_uid / str(record["SeriesInstanceUID"])
        count = sum(1 for _ in path.glob("*.dcm")) if path.is_dir() else 0
        if count:
            candidates.append((sequence_score(record, kind), count, path))
    return max(candidates, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]


def build_slot(series_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = ordered_files(series_dir)
    images = np.zeros((N_POS, 3, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros(N_POS, dtype=np.uint8)
    if not files:
        return images, valid
    centers = np.round(POSITIONS * (len(files) - 1)).astype(int)
    needed = sorted({int(np.clip(center + d, 0, len(files) - 1)) for center in centers for d in (-1, 0, 1)})
    decoded = {index: read_slice(files[index]) for index in needed}
    for out_index, center in enumerate(centers):
        channels = [decoded[int(np.clip(center + d, 0, len(files) - 1))] for d in (-1, 0, 1)]
        if channels[1] is None:
            continue
        channels = [channel if channel is not None else channels[1] for channel in channels]
        images[out_index] = np.stack(channels)
        valid[out_index] = 1
    return images, valid


def build_study(study_uid: str) -> tuple[np.ndarray, np.ndarray]:
    images = np.zeros((len(SLOTS), N_POS, 3, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros((len(SLOTS), N_POS), dtype=np.uint8)
    records = series_by_study.get(study_uid, [])
    for slot_index, slot in enumerate(SLOTS):
        path = choose_series(study_uid, records, slot)
        if path is not None:
            images[slot_index], valid[slot_index] = build_slot(path)
    return images, valid


sample_images, sample_mask = build_study(study_ids[0])
assert sample_images.shape == (6, N_POS, 3, SIZE, SIZE)
assert sample_mask.sum() > 0
print({"sample_uid": study_ids[0], "valid_windows": int(sample_mask.sum())})
"""
        ),
        markdown("## One-pass temporary uint8 pixel cache"),
        code(
            """
pixel_path = OUTPUT_ROOT / "dino_pixels.npy"
pixel_cache = np.lib.format.open_memmap(
    pixel_path,
    mode="w+",
    dtype=np.uint8,
    shape=(len(study_ids), len(SLOTS), N_POS, 3, SIZE, SIZE),
)
window_mask = np.zeros((len(study_ids), len(SLOTS), N_POS), dtype=np.uint8)
failures = []
started = time.time()

for index, uid in enumerate(study_ids):
    try:
        pixel_cache[index], window_mask[index] = build_study(uid)
        if window_mask[index].sum() == 0:
            failures.append({"study_uid": uid, "reason": "no_decodable_windows"})
    except Exception as exc:
        pixel_cache[index] = 0
        failures.append({"study_uid": uid, "reason": f"{type(exc).__name__}: {exc}"})
    if (index + 1) % (100 if FULL_RUN else 1) == 0 or index + 1 == len(study_ids):
        pixel_cache.flush()
        print(
            f"cached {index + 1:,}/{len(study_ids):,} studies in "
            f"{(time.time() - started) / 60:.1f} min; failures={len(failures)}",
            flush=True,
        )

usable = window_mask.reshape(len(study_ids), -1).any(axis=1)
target_rows = targets.set_index(ID).loc[study_ids].reset_index()
is_gold = target_rows["is_gold"].astype(bool).to_numpy()
assert usable[is_gold].sum() == is_gold.sum()
if FULL_RUN:
    assert len(study_ids) == 4_407 and int(is_gold.sum()) == 58
    assert int(usable.sum()) == 4_407

np.save(OUTPUT_ROOT / "dino_window_mask.npy", window_mask)
pd.DataFrame({ID: study_ids}).to_csv(OUTPUT_ROOT / "dino_feature_studies.csv", index=False)
pd.DataFrame(failures, columns=["study_uid", "reason"]).to_csv(
    OUTPUT_ROOT / "dino_feature_failures.csv", index=False
)
print({
    "usable_studies": int(usable.sum()), "failed_studies": int((~usable).sum()),
    "mean_valid_windows": float(window_mask[usable].sum(axis=(1, 2)).mean()),
    "temporary_pixel_gib": round(pixel_path.stat().st_size / 2**30, 2),
})
"""
        ),
        markdown("## DINOv2 token MIL with target-specific attention"),
        code(
            """
class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, HIDDEN), nn.GELU(), nn.Dropout(0.20)
        )
        self.slot_embedding = nn.Parameter(torch.randn(len(SLOTS), HIDDEN) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(N_POS, HIDDEN) * 0.02)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HIDDEN) * 0.02)
        self.target_weight = nn.Parameter(torch.randn(len(TARGETS), HIDDEN) * 0.02)
        self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding[:, None, :] + self.position_embedding[None, :, :]
        hidden = hidden.reshape(batch, len(SLOTS) * N_POS, HIDDEN)
        flat_mask = mask.reshape(batch, len(SLOTS) * N_POS).bool()
        scores = torch.einsum("bkh,th->btk", hidden, self.query) / (HIDDEN ** 0.5)
        scores = scores.masked_fill(~flat_mask[:, None, :], -1e4)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("btk,bkh->bth", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


class DinoMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(str(DINO_ROOT), local_files_only=True)
        self.head = TokenHead()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        flat = images.reshape(-1, 3, SIZE, SIZE)
        flat_mask = mask.reshape(-1).bool()
        features = torch.zeros((len(flat), TOKEN_DIM), device=images.device)
        selected = torch.where(flat_mask)[0]
        chunk = 48 if DEVICE.type == "cuda" else 2
        for start in range(0, len(selected), chunk):
            index = selected[start:start + chunk]
            pixels = flat[index].float().div(255.0)
            pixels = (pixels - self.mean) / self.std
            output = self.backbone(pixel_values=pixels).last_hidden_state
            encoded = torch.cat([output[:, 0], output[:, 1:].mean(dim=1)], dim=1)
            features[index] = encoded
        return features.reshape(batch, len(SLOTS), N_POS, TOKEN_DIM)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(images, mask), mask)


model = DinoMIL().to(DEVICE)
for parameter in model.backbone.parameters():
    parameter.requires_grad = False
n_unfrozen = 2 if FULL_RUN else 1
for block in model.backbone.encoder.layer[-n_unfrozen:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
for parameter in model.backbone.layernorm.parameters():
    parameter.requires_grad = True

trainable_backbone = sum(
    parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
)
print({
    "backbone": model.backbone.config.model_type,
    "hidden_size": model.backbone.config.hidden_size,
    "unfrozen_blocks": n_unfrozen,
    "trainable_backbone_millions": round(trainable_backbone / 1e6, 3),
})
assert model.backbone.config.hidden_size == 384
"""
        ),
        markdown("## Fixed pseudo-only backbone adaptation"),
        code(
            """
y = target_rows[TARGETS].to_numpy(np.float32)
weights = target_rows[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
pseudo_indices = np.where(~is_gold & usable)[0]
gold_indices = np.where(is_gold & usable)[0]

if FULL_RUN:
    assert len(pseudo_indices) == 4_349 and len(gold_indices) == 58
    adapt_indices = np.array([
        index for index in pseudo_indices
        if int(hashlib.sha1(study_ids[index].encode()).hexdigest()[:8], 16) % 10 != 0
    ])
    pseudo_valid_indices = np.setdiff1d(pseudo_indices, adapt_indices)
else:
    adapt_indices = np.arange(len(study_ids))
    pseudo_valid_indices = np.array([], dtype=int)


class IndexDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return int(self.indices[index])


def tensor_batch(indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    index = indices.numpy()
    images = torch.from_numpy(np.asarray(pixel_cache[index])).to(DEVICE, non_blocking=True)
    mask = torch.from_numpy(window_mask[index].astype(bool)).to(DEVICE, non_blocking=True)
    labels = torch.from_numpy(y[index]).to(DEVICE)
    label_weights = torch.from_numpy(weights[index]).to(DEVICE)
    return images, mask, labels, label_weights


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * label_weights).sum() / label_weights.sum().clamp_min(1.0)


backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([
    {"params": backbone_parameters, "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 3e-4},
], weight_decay=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
adapt_loader = DataLoader(
    IndexDataset(adapt_indices), batch_size=STUDY_BATCH, shuffle=True,
    generator=torch.Generator().manual_seed(SEED), num_workers=0,
)
adapt_history = []

for epoch in range(ADAPT_EPOCHS):
    model.train()
    losses = []
    for step, index in enumerate(adapt_loader):
        images, mask, labels, label_weights = tensor_batch(index)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(images, mask)
            loss = weighted_bce(logits, labels, label_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if FULL_RUN and (step + 1) % 250 == 0:
            print(f"epoch {epoch + 1}/{ADAPT_EPOCHS} step {step + 1}/{len(adapt_loader)} loss={np.mean(losses[-50:]):.5f}")
    adapt_history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
    print(adapt_history[-1], flush=True)

assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
torch.save(model.state_dict(), OUTPUT_ROOT / "dino_pseudo_adapted_model.pt")
pd.DataFrame(adapt_history).to_csv(OUTPUT_ROOT / "dino_adaptation_history.csv", index=False)
"""
        ),
        markdown("## Untouched expert gate and reusable token cache"),
        code(
            """
def predict_indices(indices: np.ndarray, head: nn.Module | None = None) -> np.ndarray:
    active_head = head if head is not None else model.head
    active_head.eval()
    model.backbone.eval()
    output = []
    loader = DataLoader(IndexDataset(indices), batch_size=8 if FULL_RUN else 1, shuffle=False)
    with torch.inference_mode():
        for index in loader:
            images, mask, _, _ = tensor_batch(index)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                features = model.encode(images, mask)
                output.append(torch.sigmoid(active_head(features, mask)).float().cpu().numpy())
    return np.concatenate(output)


def target_auc(labels: np.ndarray, predictions: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows = [
        {"target": target, "auc": float(roc_auc_score(labels[:, index], predictions[:, index]))}
        for index, target in enumerate(TARGETS)
    ]
    macro = float(np.mean([row["auc"] for row in rows]))
    rows.append({"target": "MACRO", "auc": macro})
    return pd.DataFrame(rows), macro


DIRECT_RESULT = None
if FULL_RUN:
    direct_predictions = predict_indices(gold_indices)
    direct_metrics, direct_macro = target_auc(y[gold_indices], direct_predictions)
    direct_metrics.to_csv(OUTPUT_ROOT / "dino_direct_gold_metrics.csv", index=False)
    DIRECT_RESULT = {"macro_auc": direct_macro}
    print({"pseudo_only_direct_gold_macro_auc": direct_macro})

feature_path = OUTPUT_ROOT / "dino_token_features.npy"
token_features = np.lib.format.open_memmap(
    feature_path, mode="w+", dtype=np.float16,
    shape=(len(study_ids), len(SLOTS), N_POS, TOKEN_DIM),
)
model.eval()
feature_loader = DataLoader(IndexDataset(np.arange(len(study_ids))), batch_size=8 if FULL_RUN else 1)
with torch.inference_mode():
    for batch_number, index in enumerate(feature_loader):
        images, mask, _, _ = tensor_batch(index)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            feature = model.encode(images, mask)
        token_features[index.numpy()] = feature.float().cpu().numpy().astype(np.float16)
        if (batch_number + 1) % (100 if FULL_RUN else 1) == 0 or batch_number + 1 == len(feature_loader):
            token_features.flush()
            print(f"token batches {batch_number + 1}/{len(feature_loader)}", flush=True)

assert np.isfinite(token_features).all()
print({"token_feature_mib": round(feature_path.stat().st_size / 2**20, 1)})
"""
        ),
        markdown("## Five fold-specific heads on the pseudo-adapted representation"),
        code(
            """
CV_RESULT = None
if FULL_RUN:
    feature_tensor = torch.from_numpy(np.asarray(token_features).astype(np.float32)).to(DEVICE)
    mask_tensor = torch.from_numpy(window_mask.astype(bool)).to(DEVICE)
    y_tensor = torch.from_numpy(y).to(DEVICE)
    weight_tensor = torch.from_numpy(weights).to(DEVICE)
    oof = np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32)
    fold_rows = []
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        train_index = np.concatenate([pseudo_indices, gold_indices[gold_train_pos]])
        valid_index = gold_indices[gold_valid_pos]
        head = copy.deepcopy(model.head).to(DEVICE)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-3)
        generator = torch.Generator().manual_seed(SEED + fold)
        train_cpu = torch.as_tensor(train_index, dtype=torch.long)
        for epoch in range(HEAD_EPOCHS):
            head.train()
            permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
            losses = []
            for start in range(0, len(permutation), 256):
                index = permutation[start:start + 256].to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = head(feature_tensor[index], mask_tensor[index])
                loss = weighted_bce(logits, y_tensor[index], weight_tensor[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            oof[valid_index] = torch.sigmoid(
                head(feature_tensor[valid_index], mask_tensor[valid_index])
            ).cpu().numpy()
        torch.save(head.state_dict(), OUTPUT_ROOT / f"dino_fold_{fold}_head.pt")
        fold_rows.append({
            "fold": fold, "train_rows": len(train_index),
            "expert_valid_rows": len(valid_index), "final_loss": float(np.mean(losses)),
        })
        print(f"fold {fold}: train={len(train_index)} valid={len(valid_index)} loss={np.mean(losses):.5f}")
        del head

    assert np.isfinite(oof[gold_indices]).all()
    cv_metrics, cv_macro = target_auc(y[gold_indices], oof[gold_indices])
    cv_metrics.to_csv(OUTPUT_ROOT / "dino_oof_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "dino_fold_summary.csv", index=False)
    oof_frame = target_rows.loc[is_gold, [ID, *TARGETS]].reset_index(drop=True)
    for index, target in enumerate(TARGETS):
        oof_frame[f"pred__{target}"] = oof[gold_indices, index]
    oof_frame.to_csv(OUTPUT_ROOT / "dino_oof_predictions.csv", index=False)
    CV_RESULT = {"macro_auc": cv_macro}
    print({"fold_head_oof_macro_auc": cv_macro})
    display(cv_metrics)
else:
    with torch.no_grad():
        index = torch.tensor([0])
        images, mask, labels, label_weights = tensor_batch(index)
        logits = model(images, mask)
        assert logits.shape == (1, len(TARGETS)) and torch.isfinite(logits).all()
    (OUTPUT_ROOT / "dino_local_smoke.json").write_text(json.dumps({
        "study_uid": study_ids[0], "valid_windows": int(window_mask[0].sum()),
        "forward_shape": list(logits.shape), "backward_and_update": True,
    }, indent=2))
"""
        ),
        markdown("## Manifest and cleanup"),
        code(
            """
manifest = {
    "schema_version": 1,
    "seed": SEED,
    "full_run": FULL_RUN,
    "studies": len(study_ids),
    "usable_studies": int(usable.sum()),
    "expert_studies": int(is_gold.sum()),
    "pseudo_studies": int((~is_gold).sum()),
    "backbone": "Meta DINOv2-small official Kaggle model",
    "backbone_license": "Apache 2.0",
    "unfrozen_blocks": n_unfrozen,
    "slots": [list(slot) for slot in SLOTS],
    "positions": POSITIONS.tolist(),
    "image_size": SIZE,
    "crop_mm": CROP_MM,
    "adapt_epochs": ADAPT_EPOCHS,
    "adaptation_gold_rows": 0 if FULL_RUN else None,
    "teacher_origin": TEACHER_ORIGIN,
    "direct_gold_result": DIRECT_RESULT,
    "fold_head_cv_result": CV_RESULT,
    "token_feature_shape": list(token_features.shape),
    "run_context": {
        "data_root": str(DATA_ROOT), "dicom_root": str(DICOM_ROOT),
        "dino_root": str(DINO_ROOT), "output_root": str(OUTPUT_ROOT),
    },
}
(OUTPUT_ROOT / "dino_adaptation_manifest.json").write_text(json.dumps(manifest, indent=2))

del pixel_cache
gc.collect()
if FULL_RUN and pixel_path.is_file():
    pixel_path.unlink()
assert not FULL_RUN or not pixel_path.exists()
assert (OUTPUT_ROOT / "dino_pseudo_adapted_model.pt").is_file()
assert (OUTPUT_ROOT / "dino_token_features.npy").is_file()
if FULL_RUN:
    assert (OUTPUT_ROOT / "dino_oof_metrics.csv").is_file()
    assert all((OUTPUT_ROOT / f"dino_fold_{fold}_head.pt").is_file() for fold in range(N_FOLDS))
else:
    assert (OUTPUT_ROOT / "dino_local_smoke.json").is_file()

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
