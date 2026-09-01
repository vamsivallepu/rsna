#!/usr/bin/env python3
"""Build the Kaggle-ready frozen-backbone 2.5D MRI baseline notebook."""

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
        default=Path("output/jupyter-notebook/rsna-2p5d-gpu-baseline.ipynb"),
    )
    args = parser.parse_args()

    cells = [
        markdown(
            """
# RSNA Knee: plane-aware 2.5D RadImageNet baseline

**Experiment.** Extract frozen RadImageNet ResNet-50 features from five physically
ordered 2.5D windows in each of six plane × sequence slots, then train target-specific
attention heads against the report-teacher soft targets.

**Primary validation.** Five fixed folds of the 58 expert-labelled studies. All 4,349
pseudo-labelled studies are available to each fold, while the held-out expert studies
are excluded completely from that fold's optimization. The final score is one OOF
prediction per expert study and macro ROC-AUC over all 58 rows.

The backbone is frozen deliberately: feature extraction scans the 247 GB Kaggle mount
once, produces a compact reusable cache, and makes head ablations cheap. Local execution
runs the identical DICOM/backbone path on the representative sample and performs a
one-step head smoke test.
"""
        ),
        code(
            """
from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

for candidate in (Path.cwd() / ".deps", Path.cwd().parent / ".deps", Path.cwd().parent.parent / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

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
from torchvision.models import resnet50

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
    (plane, fluid, fluid)
    for plane in ("Axial", "Coronal", "Sagittal")
    for fluid in (0, 1)
]
SIZE = 224
N_SLICES = 5
CROP_MM = 150.0
FEATURE_DIM = 2048
N_FOLDS = 5
HEAD_EPOCHS = 60
HEAD_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_KAGGLE = Path("/kaggle/working").is_dir()
FULL_RUN = ON_KAGGLE and DEVICE.type == "cuda"

print({
    "python": sys.version.split()[0], "torch": torch.__version__, "pydicom": pydicom.__version__,
    "device": str(DEVICE), "on_kaggle": ON_KAGGLE, "full_run": FULL_RUN,
})
if ON_KAGGLE:
    assert DEVICE.type == "cuda", "Kaggle run requested without a visible GPU"
    print(torch.cuda.get_device_name(0))
"""
        ),
        markdown("## Resolve the competition, teacher output, and medical backbone"),
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
    direct = [
        Path("/kaggle/input/rsna-knee-report-weak-supervision") / filename,
        Path("/kaggle/input/kernels/vallepuvamsikrishna/rsna-knee-report-weak-supervision") / filename,
        Path("/kaggle/input/resnet-50-radimagenet-marwan") / filename,
        Path("/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan") / filename,
    ]
    for root in candidate_roots():
        direct.extend([
            root / filename,
            root / "data/sample/models" / filename,
            root / "outputs/report_weak_supervision" / filename,
            root / "output/jupyter-notebook/outputs/report_weak_supervision" / filename,
        ])
    for path in direct:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    search_roots = [root for root in candidate_roots() if root != Path("/kaggle/input")]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        search_roots.extend(path for path in kaggle_input.iterdir() if path.name != "competitions")
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.glob(f"**/{filename}"):
            if path.is_file() and path.stat().st_size >= minimum_bytes:
                return path.resolve()
    raise FileNotFoundError(f"Could not locate {filename}")


DATA_ROOT = find_metadata_root()
local_dicom = next(
    (root / "data/sample/files/train_series" for root in candidate_roots()
     if (root / "data/sample/files/train_series").is_dir()),
    None,
)
DICOM_ROOT = (DATA_ROOT / "train_series") if (DATA_ROOT / "train_series").is_dir() else local_dicom
try:
    TARGET_PATH = find_named_file("training_targets.csv", 100_000)
except FileNotFoundError:
    TARGET_PATH = None
BACKBONE_PATH = find_named_file("ResNet50.pt", 80_000_000)
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/rsna_2p5d_smoke")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

assert DICOM_ROOT is not None and DICOM_ROOT.is_dir()
RUN_CONTEXT = {
    "data_root": str(DATA_ROOT), "dicom_root": str(DICOM_ROOT),
    "target_path": str(TARGET_PATH) if TARGET_PATH else None, "backbone_path": str(BACKBONE_PATH),
    "output_root": str(OUTPUT_ROOT.resolve()), "full_run": FULL_RUN,
}
RUN_CONTEXT
"""
        ),
        markdown("## Metadata and study selection"),
        code(
            """
train = pd.read_csv(DATA_ROOT / "train.csv")
series = pd.read_csv(DATA_ROOT / "train_series.csv")


def build_teacher_targets() -> tuple[pd.DataFrame, dict[str, str]]:
    source_files = {
        "pilkwang": "report_labels_v2.csv",
        "steven": "llm_labels_v4_blend.csv",
        "lixin": "report_labels_gpt56sol.csv",
    }
    source_paths = {name: find_named_file(filename, 100_000) for name, filename in source_files.items()}
    merged = train[[ID, *TARGETS]].copy()
    for name, path in source_paths.items():
        source = pd.read_csv(path, usecols=[ID, *TARGETS]).rename(
            columns={target: f"{name}__{target}" for target in TARGETS}
        )
        assert source[ID].is_unique
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
    return result, {name: str(path) for name, path in source_paths.items()}


if TARGET_PATH is not None:
    targets = pd.read_csv(TARGET_PATH)
    TEACHER_ORIGIN = {"type": "kernel_output", "path": str(TARGET_PATH)}
else:
    targets, public_source_paths = build_teacher_targets()
    TEACHER_ORIGIN = {"type": "in_notebook_fixed_consensus", "paths": public_source_paths}
    targets.to_csv(OUTPUT_ROOT / "training_targets.csv", index=False)

assert len(train) == 4_407 and train[ID].is_unique
assert len(series) == 24_371 and series["SeriesInstanceUID"].is_unique
assert len(targets) == 4_407 and targets[ID].is_unique
assert targets[TARGETS].notna().all().all()
assert set(targets[ID]) == set(train[ID])
print({"teacher_origin": TEACHER_ORIGIN})

if FULL_RUN:
    study_ids = targets[ID].tolist()
else:
    available = [path.name for path in DICOM_ROOT.iterdir() if path.is_dir()]
    assert available, "No representative DICOM study directories found"
    study_ids = [max(
        available,
        key=lambda uid: sum(1 for _ in (DICOM_ROOT / uid).glob("*/*.dcm")),
    )]

series = series[series[ID].isin(study_ids)].copy()
series["slot"] = list(zip(
    series["Anatomical_Plane"],
    series["Fluid_Sensitive"].astype(int),
    series["Fat_Suppression"].astype(int),
))
series_by_study = {
    uid: group.to_dict("records") for uid, group in series.groupby(ID, sort=False)
}

print({
    "selected_studies": len(study_ids), "selected_series": len(series),
    "gold_selected": int(targets.set_index(ID).loc[study_ids, "is_gold"].sum()),
})
"""
        ),
        markdown(
            """
## Physical ordering, 150 mm crop, and 2.5D construction

Each populated slot chooses its longest available series. Slices are ordered by the
projection of `ImagePositionPatient` onto the acquisition-plane normal; `InstanceNumber`
is only a fallback. Five center positions cover 10% through 90% of the volume. Each
network input stacks the preceding, center, and following slice as its three channels.
"""
        ),
        code(
            """
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
    files = list(series_dir.glob("*.dcm"))
    return sorted(files, key=spatial_key)


def resize_square(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array, mode="L")
    return np.asarray(image.resize((SIZE, SIZE), resample=Image.Resampling.BILINEAR))


def read_slice(path: Path) -> np.ndarray | None:
    try:
        ds = pydicom.dcmread(path)
        array = ds.pixel_array.astype(np.float32)
        array = array * float(ds.get("RescaleSlope", 1.0)) + float(ds.get("RescaleIntercept", 0.0))
        if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            array = array.max() + array.min() - array
        spacing = float(ds.get("PixelSpacing", [CROP_MM / max(array.shape)])[0])
        crop_side = max(16, int(round(CROP_MM / max(spacing, 1e-6))))
        crop_side = min(crop_side, array.shape[0], array.shape[1])
        cy, cx = array.shape[0] // 2, array.shape[1] // 2
        half = crop_side // 2
        y0 = min(max(0, cy - half), array.shape[0] - crop_side)
        x0 = min(max(0, cx - half), array.shape[1] - crop_side)
        crop = array[y0:y0 + crop_side, x0:x0 + crop_side]
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


def choose_series(study_uid: str, records: list[dict], slot: tuple[str, int, int]) -> Path | None:
    candidates = []
    for record in records:
        if tuple(record["slot"]) != slot:
            continue
        path = DICOM_ROOT / study_uid / str(record["SeriesInstanceUID"])
        count = sum(1 for _ in path.glob("*.dcm")) if path.is_dir() else 0
        if count:
            candidates.append((count, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def build_slot(series_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = ordered_files(series_dir)
    images = np.zeros((N_SLICES, 3, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros(N_SLICES, dtype=np.uint8)
    if not files:
        return images, valid
    centers = np.round(np.linspace(0.10, 0.90, N_SLICES) * (len(files) - 1)).astype(int)
    needed = sorted({int(np.clip(center + delta, 0, len(files) - 1)) for center in centers for delta in (-1, 0, 1)})
    decoded = {index: read_slice(files[index]) for index in needed}
    for out_index, center in enumerate(centers):
        channels = [decoded[int(np.clip(center + delta, 0, len(files) - 1))] for delta in (-1, 0, 1)]
        if channels[1] is None:
            continue
        channels = [channel if channel is not None else channels[1] for channel in channels]
        images[out_index] = np.stack(channels)
        valid[out_index] = 1
    return images, valid


def build_study(study_uid: str) -> tuple[np.ndarray, np.ndarray]:
    images = np.zeros((len(SLOTS), N_SLICES, 3, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros((len(SLOTS), N_SLICES), dtype=np.uint8)
    records = series_by_study.get(study_uid, [])
    for slot_index, slot in enumerate(SLOTS):
        path = choose_series(study_uid, records, slot)
        if path is not None:
            images[slot_index], valid[slot_index] = build_slot(path)
    return images, valid
"""
        ),
        code(
            """
class StudyDataset(Dataset):
    def __init__(self, uids: list[str]):
        self.uids = uids

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int) -> dict[str, object]:
        uid = self.uids[index]
        images, valid = build_study(uid)
        return {"uid": uid, "images": torch.from_numpy(images), "valid": torch.from_numpy(valid)}


sample_images, sample_valid = build_study(study_ids[0])
assert sample_images.shape == (6, 5, 3, 224, 224)
assert sample_valid.sum() > 0
print({
    "sample_uid": study_ids[0], "valid_2p5d_windows": int(sample_valid.sum()),
    "populated_slots": int((sample_valid.sum(axis=1) > 0).sum()),
})
"""
        ),
        markdown("## Frozen RadImageNet feature extraction"),
        code(
            """
class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).mean(dim=(2, 3))


encoder = RadEncoder()
state = torch.load(BACKBONE_PATH, map_location="cpu", weights_only=True)
encoder.load_state_dict(state, strict=True)
encoder.eval().to(DEVICE)
for parameter in encoder.parameters():
    parameter.requires_grad = False

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def encode_study(images: torch.Tensor, valid: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    flat = images.reshape(-1, 3, SIZE, SIZE)
    flat_valid = valid.reshape(-1).bool()
    encoded = torch.zeros((len(flat), FEATURE_DIM), device=DEVICE)
    indices = torch.where(flat_valid)[0]
    chunk_size = 64 if DEVICE.type == "cuda" else 4
    with torch.inference_mode():
        for start in range(0, len(indices), chunk_size):
            selected = indices[start:start + chunk_size]
            batch = flat[selected].to(DEVICE, non_blocking=True).float().div_(255.0)
            batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                encoded[selected] = encoder(batch).float()
    encoded = encoded.view(len(SLOTS), N_SLICES, FEATURE_DIM)
    slice_mask = valid.to(DEVICE).float().unsqueeze(-1)
    slot_count = slice_mask.sum(dim=1).clamp_min(1.0)
    slot_features = (encoded * slice_mask).sum(dim=1) / slot_count
    slot_mask = valid.sum(dim=1).gt(0).cpu().numpy().astype(np.uint8)
    return slot_features.cpu().numpy().astype(np.float16), slot_mask


loader = DataLoader(
    StudyDataset(study_ids), batch_size=1, shuffle=False,
    num_workers=4 if FULL_RUN else 0, pin_memory=DEVICE.type == "cuda",
    persistent_workers=FULL_RUN, prefetch_factor=2 if FULL_RUN else None,
)

feature_path = OUTPUT_ROOT / "study_features.npy"
slot_mask_path = OUTPUT_ROOT / "study_slot_mask.npy"
feature_cache = np.lib.format.open_memmap(
    feature_path, mode="w+", dtype=np.float16, shape=(len(study_ids), len(SLOTS), FEATURE_DIM)
)
slot_mask_cache = np.zeros((len(study_ids), len(SLOTS)), dtype=np.uint8)
failures = []
started = time.time()

for row_index, batch in enumerate(loader):
    uid = batch["uid"][0]
    try:
        feature_cache[row_index], slot_mask_cache[row_index] = encode_study(
            batch["images"][0], batch["valid"][0]
        )
        if slot_mask_cache[row_index].sum() == 0:
            failures.append({"study_uid": uid, "reason": "no_decodable_slots"})
    except Exception as exc:
        feature_cache[row_index] = 0
        failures.append({"study_uid": uid, "reason": f"{type(exc).__name__}: {exc}"})
    if (row_index + 1) % (100 if FULL_RUN else 1) == 0 or row_index + 1 == len(study_ids):
        feature_cache.flush()
        elapsed = time.time() - started
        print(f"encoded {row_index + 1:,}/{len(study_ids):,} studies in {elapsed / 60:.1f} min; failures={len(failures)}")

np.save(slot_mask_path, slot_mask_cache)
pd.DataFrame({ID: study_ids}).to_csv(OUTPUT_ROOT / "feature_studies.csv", index=False)
pd.DataFrame(failures, columns=["study_uid", "reason"]).to_csv(OUTPUT_ROOT / "feature_failures.csv", index=False)

usable = slot_mask_cache.sum(axis=1) > 0
print({
    "usable_studies": int(usable.sum()), "failed_studies": int((~usable).sum()),
    "mean_populated_slots": float(slot_mask_cache[usable].sum(axis=1).mean()),
    "feature_cache_mib": round(feature_path.stat().st_size / 2**20, 1),
})
del encoder, loader, sample_images
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
"""
        ),
        markdown(
            """
## Target-specific slot attention

The head learns a separate attention distribution over the six acquisition slots for
each finding. Expert targets receive weight 2.0; pseudo targets receive the fixed
agreement weights exported by the teacher notebook. Hyperparameters are fixed before
OOF evaluation and no fold uses early stopping on its held-out expert rows.
"""
        ),
        code(
            """
class TargetAttentionHead(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden), nn.GELU(), nn.Dropout(0.20)
        )
        self.slot_embedding = nn.Parameter(torch.zeros(len(SLOTS), hidden))
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.attention = nn.Linear(hidden, len(TARGETS))
        self.target_weight = nn.Parameter(torch.empty(len(TARGETS), hidden))
        self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.normal_(self.target_weight, std=0.02)

    def forward(self, features: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(self.norm(features)) + self.slot_embedding.unsqueeze(0)
        attention_logits = self.attention(hidden)
        attention_logits = attention_logits.masked_fill(~slot_mask.unsqueeze(-1), -1e4)
        attention = attention_logits.softmax(dim=1)
        pooled = torch.einsum("bst,bsd->btd", attention, hidden)
        return (pooled * self.target_weight.unsqueeze(0)).sum(dim=-1) + self.target_bias


feature_tensor = torch.from_numpy(np.asarray(feature_cache).astype(np.float32)).to(DEVICE)
slot_mask_tensor = torch.from_numpy(slot_mask_cache.astype(bool)).to(DEVICE)
target_rows = targets.set_index(ID).loc[study_ids].reset_index()
y_tensor = torch.from_numpy(target_rows[TARGETS].to_numpy(np.float32)).to(DEVICE)
weight_tensor = torch.from_numpy(
    target_rows[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
).to(DEVICE)
is_gold = target_rows["is_gold"].astype(bool).to_numpy()

assert feature_tensor.shape == (len(study_ids), 6, FEATURE_DIM)
assert slot_mask_tensor.any(dim=1).cpu().numpy().sum() == usable.sum()
"""
        ),
        code(
            """
def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def train_fold(train_index: np.ndarray, seed: int) -> TargetAttentionHead:
    torch.manual_seed(seed)
    model = TargetAttentionHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_index_cpu = torch.as_tensor(train_index, dtype=torch.long)
    for epoch in range(HEAD_EPOCHS):
        model.train()
        permutation = train_index_cpu[torch.randperm(len(train_index_cpu), generator=generator)]
        losses = []
        for start in range(0, len(permutation), HEAD_BATCH_SIZE):
            index = permutation[start:start + HEAD_BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_tensor[index], slot_mask_tensor[index])
            loss = weighted_bce(logits, y_tensor[index], weight_tensor[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, HEAD_EPOCHS - 1):
            print(f"  epoch {epoch + 1:02d}/{HEAD_EPOCHS}: loss={np.mean(losses):.5f}")
    return model


if not FULL_RUN:
    smoke_model = TargetAttentionHead().to(DEVICE)
    smoke_optimizer = torch.optim.AdamW(smoke_model.parameters(), lr=3e-4)
    smoke_optimizer.zero_grad(set_to_none=True)
    smoke_logits = smoke_model(feature_tensor, slot_mask_tensor)
    smoke_loss = weighted_bce(smoke_logits, y_tensor, weight_tensor)
    smoke_loss.backward()
    smoke_optimizer.step()
    assert smoke_logits.shape == (len(study_ids), len(TARGETS))
    SMOKE_RESULT = {
        "study_uid": study_ids[0], "loss": float(smoke_loss.detach()),
        "populated_slots": int(slot_mask_cache[0].sum()), "backward_pass": True,
    }
    (OUTPUT_ROOT / "local_smoke.json").write_text(json.dumps(SMOKE_RESULT, indent=2))
    SMOKE_RESULT
"""
        ),
        markdown("## Five-fold expert OOF evaluation (Kaggle GPU run only)"),
        code(
            """
CV_RESULT = None
if FULL_RUN:
    assert len(study_ids) == 4_407 and int(is_gold.sum()) == 58
    gold_indices = np.where(is_gold & usable)[0]
    pseudo_indices = np.where(~is_gold & usable)[0]
    assert len(gold_indices) == 58
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full((len(study_ids), len(TARGETS)), np.nan, dtype=np.float32)
    all_predictions = np.zeros((len(study_ids), len(TARGETS)), dtype=np.float32)
    fold_rows = []

    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        train_index = np.concatenate([pseudo_indices, gold_indices[gold_train_pos]])
        valid_index = gold_indices[gold_valid_pos]
        print(f"fold {fold}: train={len(train_index):,}, expert-valid={len(valid_index)}")
        model = train_fold(train_index, SEED + fold)
        model.eval()
        with torch.inference_mode():
            valid_prediction = torch.sigmoid(
                model(feature_tensor[valid_index], slot_mask_tensor[valid_index])
            ).cpu().numpy()
            full_prediction = torch.sigmoid(
                model(feature_tensor, slot_mask_tensor)
            ).cpu().numpy()
        oof[valid_index] = valid_prediction
        all_predictions += full_prediction / N_FOLDS
        torch.save(model.state_dict(), OUTPUT_ROOT / f"fold_{fold}_attention_head.pt")
        fold_rows.append({"fold": fold, "train_rows": len(train_index), "expert_valid_rows": len(valid_index)})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    assert np.isfinite(oof[gold_indices]).all()
    metric_rows = []
    for target_index, target in enumerate(TARGETS):
        auc = roc_auc_score(
            y_tensor[gold_indices, target_index].cpu().numpy(),
            oof[gold_indices, target_index],
        )
        metric_rows.append({"target": target, "auc": float(auc)})
    macro_auc = float(np.mean([row["auc"] for row in metric_rows]))
    metric_rows.append({"target": "MACRO", "auc": macro_auc})
    metrics = pd.DataFrame(metric_rows)

    oof_frame = target_rows.loc[is_gold, [ID, *TARGETS]].reset_index(drop=True)
    for target_index, target in enumerate(TARGETS):
        oof_frame[f"pred__{target}"] = oof[gold_indices, target_index]
    image_predictions = pd.DataFrame({ID: study_ids})
    for target_index, target in enumerate(TARGETS):
        image_predictions[target] = all_predictions[:, target_index]

    metrics.to_csv(OUTPUT_ROOT / "cv_metrics.csv", index=False)
    oof_frame.to_csv(OUTPUT_ROOT / "oof_predictions.csv", index=False)
    image_predictions.to_csv(OUTPUT_ROOT / "train_image_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "fold_summary.csv", index=False)
    CV_RESULT = {"macro_auc": macro_auc, "target_auc": metrics.set_index("target")["auc"].to_dict()}
    display(metrics)
    print(CV_RESULT)
else:
    print("Local smoke mode: five-fold training is intentionally skipped.")
"""
        ),
        markdown("## Save manifest and verify the handoff"),
        code(
            """
manifest = {
    "schema_version": 1,
    "seed": SEED,
    "full_run": FULL_RUN,
    "studies": len(study_ids),
    "usable_studies": int(usable.sum()),
    "failed_studies": int((~usable).sum()),
    "slots": [list(slot) for slot in SLOTS],
    "slices_per_slot": N_SLICES,
    "image_size": SIZE,
    "crop_mm": CROP_MM,
    "backbone": "RadImageNet ResNet50 frozen",
    "feature_dim": FEATURE_DIM,
    "head": "target-specific slot attention",
    "teacher_origin": TEACHER_ORIGIN,
    "folds": N_FOLDS if FULL_RUN else 0,
    "cv_result": CV_RESULT,
    "run_context": RUN_CONTEXT,
}
(OUTPUT_ROOT / "image_baseline_manifest.json").write_text(json.dumps(manifest, indent=2))

assert feature_path.is_file() and slot_mask_path.is_file()
assert feature_cache.shape == (len(study_ids), 6, FEATURE_DIM)
assert int(usable.sum()) >= 1
if FULL_RUN:
    assert (OUTPUT_ROOT / "cv_metrics.csv").is_file()
    assert all((OUTPUT_ROOT / f"fold_{fold}_attention_head.pt").is_file() for fold in range(N_FOLDS))
else:
    assert (OUTPUT_ROOT / "local_smoke.json").is_file()

FINAL_CHECK = {
    "all_cells_executed": True,
    "mode": "full_gpu" if FULL_RUN else "local_smoke",
    "studies": len(study_ids),
    "usable_studies": int(usable.sum()),
    "populated_slots_mean": round(float(slot_mask_cache[usable].sum(axis=1).mean()), 3),
    "cv_macro_auc": CV_RESULT["macro_auc"] if CV_RESULT else None,
    "output_files": sorted(path.name for path in OUTPUT_ROOT.glob("*")),
}
FINAL_CHECK
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
