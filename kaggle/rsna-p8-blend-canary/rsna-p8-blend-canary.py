#!/usr/bin/env python3
"""Create the P2 CoAtNet fold-head ensemble submission on Kaggle.

The visible competition mount contains only a tiny canary test.  Kaggle reruns this
same internet-disabled script against the hidden test set when the notebook version
is submitted to the competition, so no test-study count is hard coded here.
"""

# ruff: noqa: E402 - local offline dependencies must precede ML imports.

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for dependency_root in (
    Path.cwd() / ".deps",
    Path.cwd().parent / ".deps",
    Path.cwd().parent.parent / ".deps",
):
    if dependency_root.is_dir():
        sys.path.insert(0, str(dependency_root))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


COMPETITION_SLUG = "rsna-knee-abnormality-detection"
ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ARCH = "coatnet_rmlp_2_rw_384"
FEATURE_DIM = 1024
N_TARGETS = len(TARGETS)
CACHE_SIZE = 336
MODEL_SIZE = 384
K_EVAL = 32
CROP_MM = 140.0
LOW_PERCENTILE = 2.0
HIGH_PERCENTILE = 98.0
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
EXPECTED_P2_MANIFEST_SHA256 = (
    "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"
)
EXPECTED_CACHE_SIGNATURE = (
    "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
)
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(
    "outputs/p2_submission_smoke"
)

SLOT_SPECS = [
    {"name": "sagittal_fluid", "plane": "Sagittal", "kind": "fluid", "count": 18},
    {
        "name": "sagittal_structural",
        "plane": "Sagittal",
        "kind": "structural",
        "count": 14,
    },
    {"name": "coronal_fluid", "plane": "Coronal", "kind": "fluid", "count": 12},
    {
        "name": "coronal_structural",
        "plane": "Coronal",
        "kind": "structural",
        "count": 8,
    },
    {"name": "axial_any", "plane": "Axial", "kind": "any", "count": 12},
]
assert sum(slot["count"] for slot in SLOT_SPECS) == 64
SLOT_BOUNDS: list[tuple[int, int]] = []
cursor = 0
for slot in SLOT_SPECS:
    SLOT_BOUNDS.append((cursor, cursor + int(slot["count"])))
    cursor += int(slot["count"])


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_roots() -> list[Path]:
    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    if Path("/kaggle/input").is_dir():
        roots.append(Path("/kaggle/input"))
    return roots


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
        Path("/kaggle/input") / COMPETITION_SLUG,
    ]
    for root in candidate_roots():
        candidates.extend([root / "data/raw", root])
    for candidate in candidates:
        if (candidate / "test.csv").is_file() and (
            candidate / "test_series.csv"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError("competition test metadata")


def find_p2_root() -> Path:
    candidates = [
        Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p2-coatnet-rmlp2"
        ),
        Path("outputs/kaggle/rsna-p2-coatnet-rmlp2-v2"),
    ]
    for candidate in candidates:
        if (candidate / "p2_manifest.json").is_file():
            return candidate.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = list(kaggle_input.glob("**/p2_manifest.json"))
        if len(matches) == 1:
            return matches[0].parent.resolve()
    raise FileNotFoundError("validated P2 output")


HEADER_TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber"]


def spatial_key(path: Path) -> tuple[float, int, str]:
    try:
        dataset = pydicom.dcmread(
            path, stop_before_pixels=True, specific_tags=HEADER_TAGS
        )
        orientation = np.asarray(dataset.get("ImageOrientationPatient", []), dtype=float)
        position = np.asarray(dataset.get("ImagePositionPatient", []), dtype=float)
        instance = int(dataset.get("InstanceNumber", 0) or 0)
        if orientation.size == 6 and position.size == 3:
            location = float(
                np.dot(position, np.cross(orientation[:3], orientation[3:]))
            )
            return (location, instance, path.name)
        return (float(instance), instance, path.name)
    except Exception:
        return (0.0, 0, path.name)


def ordered_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.glob("*.dcm"), key=spatial_key)


def sequence_score(record: dict[str, object], kind: str) -> int:
    if kind == "any":
        return 0
    fluid = int(record["Fluid_Sensitive"])
    fat = int(record["Fat_Suppression"])
    if kind == "fluid":
        return 3 if fluid == 1 and fat == 1 else 2 if fluid == 1 else 1 if fat == 1 else 0
    return 3 if fluid == 0 and fat == 0 else 2 if fluid == 0 else 1 if fat == 0 else 0


def choose_series(
    dicom_root: Path,
    study_uid: str,
    records: list[dict[str, object]],
    slot: dict[str, object],
    used: set[str],
) -> tuple[Path | None, str | None]:
    candidates: list[tuple[int, int, str, Path]] = []
    for record in records:
        series_uid = str(record["SeriesInstanceUID"])
        if series_uid in used or str(record["Anatomical_Plane"]) != slot["plane"]:
            continue
        path = dicom_root / study_uid / series_uid
        count = sum(1 for _ in path.glob("*.dcm")) if path.is_dir() else 0
        if count:
            candidates.append(
                (sequence_score(record, str(slot["kind"])), count, series_uid, path)
            )
    if not candidates:
        return None, None
    _, _, series_uid, path = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    used.add(series_uid)
    return path, series_uid


def decode_raw(path: Path) -> tuple[np.ndarray, tuple[float, float]] | None:
    try:
        dataset = pydicom.dcmread(path)
        array = dataset.pixel_array.astype(np.float32)
        array = array * float(dataset.get("RescaleSlope", 1.0)) + float(
            dataset.get("RescaleIntercept", 0.0)
        )
        if str(dataset.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            array = array.max() + array.min() - array
        spacing = dataset.get(
            "PixelSpacing", [CROP_MM / max(array.shape)] * 2
        )
        sy, sx = float(spacing[0]), float(spacing[1])
        if not np.isfinite(array).any() or sy <= 0 or sx <= 0:
            return None
        return array, (sy, sx)
    except Exception:
        return None


def crop_resize_uint8(
    array: np.ndarray, spacing: tuple[float, float], lo: float, hi: float
) -> np.ndarray | None:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    sy, sx = spacing
    crop_h = min(array.shape[0], max(16, int(round(CROP_MM / sy))))
    crop_w = min(array.shape[1], max(16, int(round(CROP_MM / sx))))
    cy, cx = array.shape[0] // 2, array.shape[1] // 2
    y0 = min(max(0, cy - crop_h // 2), array.shape[0] - crop_h)
    x0 = min(max(0, cx - crop_w // 2), array.shape[1] - crop_w)
    crop = array[y0:y0 + crop_h, x0:x0 + crop_w]
    scaled = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    image = Image.fromarray(np.round(scaled * 255.0).astype(np.uint8))
    return np.asarray(
        image.resize((CACHE_SIZE, CACHE_SIZE), resample=Image.Resampling.BILINEAR)
    )


def build_slot(series_dir: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    files = ordered_files(series_dir)
    images = np.zeros((count, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    valid = np.zeros(count, dtype=np.uint8)
    if not files:
        return images, valid
    centers = np.round(np.linspace(0.02, 0.98, count) * (len(files) - 1)).astype(int)
    decoded = [decode_raw(files[int(index)]) for index in centers]
    finite_chunks = [
        item[0][np.isfinite(item[0])] for item in decoded if item is not None
    ]
    if not finite_chunks:
        return images, valid
    finite = np.concatenate(finite_chunks)
    lo, hi = np.percentile(finite, [LOW_PERCENTILE, HIGH_PERCENTILE])
    for output_index, item in enumerate(decoded):
        if item is None:
            continue
        processed = crop_resize_uint8(item[0], item[1], float(lo), float(hi))
        if processed is not None:
            images[output_index] = processed
            valid[output_index] = 1
    return images, valid


def valid_centers(mask: np.ndarray) -> np.ndarray:
    centers: list[int] = []
    for start, stop in SLOT_BOUNDS:
        for center in range(start + 1, stop - 1):
            if mask[center - 1:center + 2].all():
                centers.append(center)
    return np.asarray(centers, dtype=np.int16)


def choose_centers(mask: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    available = valid_centers(mask)
    if len(available) == 0:
        return np.zeros(count, dtype=np.int16), np.zeros(count, dtype=bool)
    if len(available) <= count:
        selected = np.zeros(count, dtype=np.int16)
        selected[:len(available)] = available
        valid = np.zeros(count, dtype=bool)
        valid[:len(available)] = True
        return selected, valid
    positions = np.round(np.linspace(0, len(available) - 1, count)).astype(int)
    return available[positions], np.ones(count, dtype=bool)


class TestStudyDataset(Dataset):
    def __init__(
        self,
        study_ids: list[str],
        series_by_study: dict[str, list[dict[str, object]]],
        dicom_root: Path,
    ) -> None:
        self.study_ids = study_ids
        self.series_by_study = series_by_study
        self.dicom_root = dicom_root

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int):
        study_uid = self.study_ids[index]
        volume = np.zeros((64, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        slice_mask = np.zeros(64, dtype=np.uint8)
        selections: list[dict[str, object]] = []
        used: set[str] = set()
        cursor = 0
        for slot in SLOT_SPECS:
            path, series_uid = choose_series(
                self.dicom_root,
                study_uid,
                self.series_by_study.get(study_uid, []),
                slot,
                used,
            )
            stop = cursor + int(slot["count"])
            if path is not None:
                volume[cursor:stop], slice_mask[cursor:stop] = build_slot(
                    path, int(slot["count"])
                )
            selections.append(
                {
                    "slot": slot["name"],
                    "series_uid": series_uid,
                    "valid_slices": int(slice_mask[cursor:stop].sum()),
                }
            )
            cursor = stop
        centers, window_mask = choose_centers(slice_mask, K_EVAL)
        if not window_mask.any():
            raise RuntimeError(f"{study_uid} has no valid 2.5D windows")
        windows = np.zeros(
            (K_EVAL, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8
        )
        for output_index, center in enumerate(centers):
            if window_mask[output_index]:
                windows[output_index] = volume[
                    int(center) - 1:int(center) + 2
                ]
        return (
            index,
            windows,
            window_mask.astype(np.uint8),
            centers,
            json.dumps(selections, separators=(",", ":")),
        )


class TargetAttentionHead(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, dropout: float = 0.20) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, N_TARGETS),
        )
        self.target_weight = nn.Parameter(
            torch.randn(N_TARGETS, feature_dim) * 0.02
        )
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(features)
        scores = self.attention(hidden).transpose(1, 2)
        scores = scores.masked_fill(~mask[:, None, :].bool(), -1e4)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("btk,bkf->btf", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


class CoAtNetMIL(nn.Module):
    def __init__(self, *, channels_last: bool) -> None:
        super().__init__()
        backbone = timm.create_model(ARCH, pretrained=False)
        backbone.reset_classifier(0, global_pool="avg")
        assert backbone.num_features == FEATURE_DIM
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(enable=False)
        self.backbone = backbone
        self.head = TargetAttentionHead()
        self.channels_last = channels_last
        self.register_buffer(
            "mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(MODEL_STD).view(1, 3, 1, 1)
        )

    def encode(self, windows: torch.Tensor) -> torch.Tensor:
        batch, count = windows.shape[:2]
        pixels = windows.reshape(
            batch * count, 3, CACHE_SIZE, CACHE_SIZE
        ).float().div(255.0)
        pixels = F.interpolate(
            pixels,
            size=(MODEL_SIZE, MODEL_SIZE),
            mode="bicubic",
            align_corners=False,
        )
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        pixels = (pixels - self.mean) / self.std
        return self.backbone(pixels).reshape(batch, count, FEATURE_DIM)


def head_predictions(
    head: TargetAttentionHead,
    features: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 64):
            feature = torch.from_numpy(
                np.asarray(features[start:start + 64]).astype(np.float32)
            ).to(device)
            valid = torch.from_numpy(
                np.asarray(mask[start:start + 64]).astype(bool)
            ).to(device)
            predictions.append(torch.sigmoid(head(feature, valid)).cpu().numpy())
    return np.concatenate(predictions)


def gpu_snapshot() -> list[list[str]] | None:
    if not torch.cuda.is_available():
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.split(", ") for line in result.stdout.splitlines()]

#!/usr/bin/env python3
"""Create the promoted equal-rank P2+P8 submission on Kaggle.

The deployment builder prepends the proven P2 preprocessing/model definitions to
this file. Local execution imports those definitions directly from p2_submission.
"""

# ruff: noqa: E402 - the deployment builder supplies common definitions in place.

from concurrent.futures import ThreadPoolExecutor
RUN_ID = "P8-BLEND-SUBMISSION-v1"
ENABLE_P6 = True
EXPECTED_P8_TEACHER_MANIFEST_SHA256 = "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
EXPECTED_P2_MANIFEST_SHA256 = (
    "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"
)
EXPECTED_P8_MANIFEST_SHA256 = {
    0: "f242c61269bff702fe601e35f6bc1bd7267f7d5f94c265ce684e665a4b37baf6",
    1: "1120c6173b9044ca84c035e47070cb5e79e40484eddea158cdd114f809eb10eb",
    2: "78925624ab4cebbe7044da362e1015c4d39b0ab514f04625f3d031f2a952c04d",
    3: "4d7f4a4090b7eb00517f29b8ea88822a0b8871519a593b64fb464fdaa71cd095",
    4: "94bacb27bc1ffa637ba3adbad6be0c960f19f69a0b59563a2b1fd6efe9f7ac2b",
}
EXPECTED_P8_CHECKPOINT_SHA256 = {
    0: "5883762c8f4161e68eea33766aa737b34b772b180e18e75572937eeabf366c5f",
    1: "68df88b583cca94d6f777476f0b2696c9fcd850591d863cc0255a85f248003d0",
    2: "b2b42a0c44f83986839f90ec4547038ac0ca9705a30923be3da1e209c5196498",
    3: "a2c1600b173a4abc3b94e339e92de0db92e93a0d3a5a456db1f415e0c0db3639",
    4: "d0a154d4ab0844793c195ac92beba5fb036c9ea51dfba20bdef99194fdcb868f",
}
P2_OOF_MACRO_AUC = 0.8911253543643616
P8_OOF_MACRO_AUC = 0.9029118074637345
BLEND_OOF_MACRO_AUC = 0.9022377777850243
EXPECTED_P6_MANIFEST_SHA256 = (
    "885b58c5a9af1bffaac1f52c9703db3538d25f8d9660b036b563b351b1067bce"
)
P6_OOF_MACRO_AUC = 0.9098782108453899
P6_BLEND_OOF_MACRO_AUC = 0.9097982866758175
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(
    "outputs/p8_blend_submission_smoke"
)


def find_p8_roots() -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for fold in range(5):
        direct = [
            Path(
                "/kaggle/input/notebooks/vallepuvamsikrishna/"
                f"rsna-knee-p8-noisy-student-fold-{fold}"
            ),
            Path("outputs/kaggle/rsna-p8-noisy-student-v1") / f"fold-{fold}",
        ]
        selected = next(
            (
                candidate.resolve()
                for candidate in direct
                if (candidate / f"p8_fold_{fold}_manifest.json").is_file()
            ),
            None,
        )
        if selected is None and Path("/kaggle/input").is_dir():
            matches = list(
                Path("/kaggle/input").glob(
                    f"**/p8_fold_{fold}_manifest.json"
                )
            )
            if len(matches) == 1:
                selected = matches[0].parent.resolve()
        if selected is None:
            raise FileNotFoundError(f"validated P8 fold {fold} output")
        roots[fold] = selected
    return roots


def find_p6_root() -> Path:
    direct = [
        Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p6-p2-transfer"
        ),
        Path("outputs/kaggle/rsna-p6-p2-transfer-v1"),
    ]
    selected = next(
        (
            candidate.resolve()
            for candidate in direct
            if (candidate / "p6_transfer_manifest.json").is_file()
        ),
        None,
    )
    if selected is None and Path("/kaggle/input").is_dir():
        matches = list(
            Path("/kaggle/input").glob("**/p6_transfer_manifest.json")
        )
        if len(matches) == 1:
            selected = matches[0].parent.resolve()
    if selected is None:
        raise FileNotFoundError("validated P6 transfer output")
    return selected


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index]).rank(
                method="average", pct=True
            ).to_numpy(np.float64)
            for index in range(predictions.shape[1])
        ]
    )


def load_full_model(checkpoint: Path, device: torch.device) -> CoAtNetMIL:
    channels_last = device.type == "cuda"
    model = CoAtNetMIL(channels_last=channels_last)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
        model = model.half()
    else:
        model = model.to(device)
    model.eval()
    return model


def load_fold_heads(
    model_root: Path, device: torch.device
) -> list[TargetAttentionHead]:
    heads: list[TargetAttentionHead] = []
    for fold in range(5):
        head = TargetAttentionHead().to(device)
        state = torch.load(
            model_root / f"p2_fold_{fold}_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        head.load_state_dict(state, strict=True)
        if device.type == "cuda":
            head = head.half()
        head.eval()
        heads.append(head)
        del state
    return heads


def infer_device_group(
    device: torch.device,
    windows_cpu: torch.Tensor,
    mask_cpu: torch.Tensor,
    p2_model: CoAtNetMIL | None,
    p2_heads: list[TargetAttentionHead],
    p8_models: list[tuple[int, CoAtNetMIL]],
    p6_model: CoAtNetMIL | None = None,
    p6_heads: list[TargetAttentionHead] | None = None,
) -> tuple[np.ndarray | None, dict[int, np.ndarray], np.ndarray | None]:
    windows = windows_cpu.to(device, non_blocking=device.type == "cuda")
    mask = mask_cpu.to(device, non_blocking=device.type == "cuda").bool()
    reference = (
        p2_model
        if p2_model is not None
        else (p8_models[0][1] if p8_models else p6_model)
    )
    assert reference is not None
    batch, count = windows.shape[:2]
    p2_result: np.ndarray | None = None
    p6_result: np.ndarray | None = None
    p8_result: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        pixels = windows.reshape(
            batch * count, 3, CACHE_SIZE, CACHE_SIZE
        ).float().div(255.0)
        pixels = F.interpolate(
            pixels,
            size=(MODEL_SIZE, MODEL_SIZE),
            mode="bicubic",
            align_corners=False,
        )
        if device.type == "cuda":
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        pixels = (pixels - reference.mean) / reference.std
        if device.type == "cuda":
            # Every resident backbone consumes the same normalized tensor. Cast
            # once to the dtype autocast would use for the first convolution,
            # instead of repeating that large input conversion for 3-4 models.
            pixels = pixels.to(dtype=torch.float16)
        if p2_model is not None:
            features = p2_model.backbone(pixels).reshape(
                batch, count, -1
            )
            probabilities = torch.stack(
                [torch.sigmoid(head(features, mask)) for head in p2_heads]
            ).mean(dim=0)
            p2_result = probabilities.float().cpu().numpy()
            del features, probabilities
        for fold, model in p8_models:
            features = model.backbone(pixels).reshape(batch, count, -1)
            probabilities = torch.sigmoid(model.head(features, mask))
            p8_result[fold] = probabilities.float().cpu().numpy()
            del features, probabilities
        if p6_model is not None:
            assert p6_heads
            features = p6_model.backbone(pixels).reshape(batch, count, -1)
            probabilities = torch.stack(
                [torch.sigmoid(head(features, mask)) for head in p6_heads]
            ).mean(dim=0)
            p6_result = probabilities.float().cpu().numpy()
            del features, probabilities
    return p2_result, p8_result, p6_result


def prediction_frame(study_ids: list[str], values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID: study_ids,
            **{
                target: values[:, target_index]
                for target_index, target in enumerate(TARGETS)
            },
        }
    )


def main() -> int:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not ON_KAGGLE:
        raise RuntimeError(
            "Full seven-backbone inference is intentionally Kaggle-only; "
            "the local host does not have a safe model-loading memory margin."
        )

    p2_root = find_p2_root()
    p2_manifest_path = p2_root / "p2_manifest.json"
    assert sha256_file(p2_manifest_path) == EXPECTED_P2_MANIFEST_SHA256
    p2_manifest = json.loads(p2_manifest_path.read_text())
    assert p2_manifest["run_id"] == "P2-COATNET-v1"
    assert p2_manifest["best_epoch"] == 8
    assert p2_manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
    p2_names = [
        "p2_coatnet_pseudo_best.pt",
        *[f"p2_fold_{fold}_head.pt" for fold in range(5)],
    ]
    for name in p2_names:
        path = p2_root / name
        assert path.is_file()
        assert sha256_file(path) == p2_manifest["artifacts_sha256"][name]

    p8_roots = find_p8_roots()
    p8_manifests: dict[int, dict[str, object]] = {}
    for fold, root in p8_roots.items():
        manifest_path = root / f"p8_fold_{fold}_manifest.json"
        assert sha256_file(manifest_path) == EXPECTED_P8_MANIFEST_SHA256[fold]
        manifest = json.loads(manifest_path.read_text())
        checkpoint_name = f"p8_fold_{fold}_best.pt"
        checkpoint = root / checkpoint_name
        assert manifest["run_id"] == f"P8-NOISY-STUDENT-FOLD-{fold}-v1"
        assert manifest["fold"] == fold and manifest["best_epoch"] == 8
        assert manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
        assert manifest["teacher_manifest_sha256"] == (
            EXPECTED_P8_TEACHER_MANIFEST_SHA256
        )
        assert manifest["checkpoint_selection_gold_rows"] == 0
        assert manifest["expert_valid_gradient_rows"] == 0
        assert manifest["expert_valid_report_overlap"] == 0
        assert manifest["selection_report_overlap"] == 0
        assert checkpoint.is_file()
        assert sha256_file(checkpoint) == EXPECTED_P8_CHECKPOINT_SHA256[fold]
        assert (
            manifest["artifacts_sha256"][checkpoint_name]
            == EXPECTED_P8_CHECKPOINT_SHA256[fold]
        )
        p8_manifests[fold] = manifest

    p6_root: Path | None = None
    p6_manifest: dict[str, object] | None = None
    if ENABLE_P6:
        p6_root = find_p6_root()
        p6_manifest_path = p6_root / "p6_transfer_manifest.json"
        assert sha256_file(p6_manifest_path) == EXPECTED_P6_MANIFEST_SHA256
        p6_manifest = json.loads(p6_manifest_path.read_text())
        assert p6_manifest["run_id"] == "P6-P2-TRANSFER-v1"
        assert p6_manifest["best_epoch"] == 8
        assert p6_manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
        assert p6_manifest["initialization"]["arm"] == (
            "p6_fastmri_localization_transfer"
        )
        for name in [
            "p2_coatnet_pseudo_best.pt",
            *[f"p2_fold_{fold}_head.pt" for fold in range(5)],
        ]:
            path = p6_root / name
            assert path.is_file()
            assert sha256_file(path) == p6_manifest["artifacts_sha256"][name]

    if ON_KAGGLE:
        data_root = find_competition_root()
        dicom_root = data_root / "test_series"
        test = pd.read_csv(data_root / "test.csv", dtype={ID: str})
        series = pd.read_csv(data_root / "test_series.csv", dtype={ID: str})
        sample = pd.read_csv(
            data_root / "sample_submission.csv", dtype={ID: str}
        )
        workers = 2
    else:
        data_root = Path("data/raw").resolve()
        dicom_root = Path("data/sample/files/train_series").resolve()
        metadata = pd.read_csv(data_root / "train.csv", dtype={ID: str})
        series_all = pd.read_csv(
            data_root / "train_series.csv", dtype={ID: str}
        )
        available = {
            path.name for path in dicom_root.iterdir() if path.is_dir()
        }
        local_ids = [uid for uid in metadata[ID] if uid in available]
        assert local_ids
        richest = max(
            local_ids,
            key=lambda uid: sum(
                1 for _ in (dicom_root / uid).glob("*/*.dcm")
            ),
        )
        test = pd.DataFrame({ID: [richest]})
        series = series_all[series_all[ID].isin(test[ID])].copy()
        sample = pd.DataFrame(
            {ID: test[ID], **{target: 0.5 for target in TARGETS}}
        )
        workers = 0

    assert dicom_root.is_dir()
    assert test[ID].is_unique and len(test) > 0
    assert sample.columns.tolist() == [ID, *TARGETS]
    assert sample[ID].tolist() == test[ID].tolist()
    assert set(series[ID].unique()) == set(test[ID])
    study_ids = test[ID].astype(str).tolist()
    series_by_study = {
        str(uid): group.to_dict("records")
        for uid, group in series.groupby(ID, sort=False)
    }

    visible_gpus = torch.cuda.device_count()
    devices = (
        [torch.device("cuda:0"), torch.device("cuda:1")]
        if visible_gpus >= 2
        else [torch.device("cuda:0" if visible_gpus else "cpu")]
    )
    for device in devices:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    print(
        {
            (
                "P8_BLEND_SUBMISSION_START"
                if ENABLE_P6
                else "P8_BLEND_SUBMISSION_START"
            ): True,
            "on_kaggle": ON_KAGGLE,
            "studies": len(study_ids),
            "series": len(series),
            "p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
            "p8_manifest_sha256": EXPECTED_P8_MANIFEST_SHA256,
            "p6_enabled": ENABLE_P6,
            "p6_manifest_sha256": (
                EXPECTED_P6_MANIFEST_SHA256 if ENABLE_P6 else None
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "visible_gpus": visible_gpus,
        },
        flush=True,
    )

    model_load_started = time.time()
    p2_device = devices[0]
    p2_model = load_full_model(
        p2_root / "p2_coatnet_pseudo_best.pt", p2_device
    )
    p2_heads = load_fold_heads(p2_root, p2_device)
    if len(devices) == 2:
        assignments = {0: [0, 2], 1: [1, 3, 4]}
    else:
        assignments = {0: list(range(5))}
    p8_by_device: dict[int, list[tuple[int, CoAtNetMIL]]] = {}
    for device_index, folds in assignments.items():
        device = devices[device_index]
        p8_by_device[device_index] = [
            (
                fold,
                load_full_model(
                    p8_roots[fold] / f"p8_fold_{fold}_best.pt", device
                ),
            )
            for fold in folds
        ]
    p6_models: dict[int, CoAtNetMIL] = {}
    p6_heads_by_device: dict[int, list[TargetAttentionHead]] = {}
    if ENABLE_P6:
        assert p6_root is not None
        # Duplicate the relatively small resident weights so alternating batches
        # can distribute the seventh backbone evenly across both GPUs. Each
        # device executor remains single-threaded, so CUDA work never overlaps
        # unsafely on the same device.
        for device_index, device in enumerate(devices):
            p6_models[device_index] = load_full_model(
                p6_root / "p2_coatnet_pseudo_best.pt", device
            )
            p6_heads_by_device[device_index] = load_fold_heads(p6_root, device)
    print(
        {
            (
                "P8_BLEND_MODELS_READY"
                if ENABLE_P6
                else "P8_BLEND_MODELS_READY"
            ): True,
            "execution_mode": "resident dual-device inference",
            "device_assignments": {
                str(devices[index]): [
                    *(["P2"] if index == 0 else []),
                    *[f"P8-fold-{fold}" for fold in assignments[index]],
                    *(["P6-alternating-batches"] if ENABLE_P6 else []),
                ]
                for index in assignments
            },
            "elapsed_minutes": (time.time() - model_load_started) / 60.0,
            "gpu": gpu_snapshot(),
        },
        flush=True,
    )

    batch_size = 2 if visible_gpus else 1
    dataset = TestStudyDataset(study_ids, series_by_study, dicom_root)
    loader_options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(visible_gpus),
    }
    if workers:
        loader_options.update(
            {
                "prefetch_factor": 2,
                "persistent_workers": True,
                "multiprocessing_context": "spawn",
            }
        )
    loader = DataLoader(dataset, **loader_options)

    p2_predictions = np.zeros((len(test), len(TARGETS)), dtype=np.float32)
    p8_fold_predictions = np.zeros(
        (5, len(test), len(TARGETS)), dtype=np.float32
    )
    p6_predictions = (
        np.zeros((len(test), len(TARGETS)), dtype=np.float32)
        if ENABLE_P6
        else None
    )
    feature_mask = np.zeros((len(test), K_EVAL), dtype=np.uint8)
    positions = np.full((len(test), K_EVAL), -1, dtype=np.int16)
    selection_rows: list[dict[str, object]] = []
    inference_started = time.time()
    executors = [ThreadPoolExecutor(max_workers=1) for _ in devices]

    def resolve_batch(
        pending_batch: tuple[
            np.ndarray,
            torch.Tensor,
            torch.Tensor,
            list[str],
            list[object],
        ]
    ) -> int:
        (
            batch_indices,
            window_mask,
            centers,
            selections_json,
            futures,
        ) = pending_batch
        batch_results = [future.result() for future in futures]
        p6_batches = []
        for p2_batch, p8_batches, p6_batch in batch_results:
            if p2_batch is not None:
                p2_predictions[batch_indices] = p2_batch
            for fold, values in p8_batches.items():
                p8_fold_predictions[fold, batch_indices] = values
            if p6_batch is not None:
                p6_batches.append(p6_batch)
        if ENABLE_P6:
            assert p6_predictions is not None and len(p6_batches) == 1
            p6_predictions[batch_indices] = p6_batches[0]

        masks = window_mask.numpy().astype(bool)
        center_values = centers.numpy()
        feature_mask[batch_indices] = masks.astype(np.uint8)
        for batch_offset, index in enumerate(batch_indices):
            positions[index, masks[batch_offset]] = center_values[
                batch_offset, masks[batch_offset]
            ]
            for selection in json.loads(selections_json[batch_offset]):
                selection_rows.append({ID: study_ids[index], **selection})
        return len(batch_indices)

    try:
        processed = 0
        pending: list[tuple[object, ...]] = []
        for batch_number, batch in enumerate(loader):
            indices, windows, window_mask, centers, selections_json = batch
            batch_indices = indices.numpy().astype(np.int64)
            futures = []
            for device_index, device in enumerate(devices):
                futures.append(
                    executors[device_index].submit(
                        infer_device_group,
                        device,
                        windows,
                        window_mask,
                        p2_model if device_index == 0 else None,
                        p2_heads if device_index == 0 else [],
                        p8_by_device[device_index],
                        (
                            p6_models[device_index]
                            if ENABLE_P6
                            and device_index == batch_number % len(devices)
                            else None
                        ),
                        (
                            p6_heads_by_device[device_index]
                            if ENABLE_P6
                            and device_index == batch_number % len(devices)
                            else None
                        ),
                    )
                )
            pending.append(
                (
                    batch_indices,
                    window_mask,
                    centers,
                    list(selections_json),
                    futures,
                )
            )
            # Bounded pipelining lets each device advance independently while
            # capping host RAM to a few decoded study batches.
            if len(pending) >= 4:
                processed += resolve_batch(pending.pop(0))
            if processed % 50 == 0 or processed == len(test):
                print(
                    {
                        "P8_BLEND_SUBMISSION_PROGRESS": (
                            f"{processed}/{len(test)}"
                        ),
                        "elapsed_minutes": (
                            time.time() - inference_started
                        ) / 60.0,
                        "gpu": gpu_snapshot(),
                    },
                    flush=True,
                )
        while pending:
            processed += resolve_batch(pending.pop(0))
        print(
            {
                "P8_BLEND_SUBMISSION_PROGRESS": f"{processed}/{len(test)}",
                "elapsed_minutes": (time.time() - inference_started) / 60.0,
                "gpu": gpu_snapshot(),
            },
            flush=True,
        )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    assert np.isfinite(p2_predictions).all()
    assert np.isfinite(p8_fold_predictions).all()
    if ENABLE_P6:
        assert p6_predictions is not None
        assert np.isfinite(p6_predictions).all()
    assert feature_mask.any(axis=1).all()
    assert (positions[feature_mask.astype(bool)] >= 1).all()
    p8_predictions = p8_fold_predictions.mean(axis=0)
    p2_p8_predictions = np.clip(
        (rank_columns(p2_predictions) + rank_columns(p8_predictions)) / 2.0,
        1e-6,
        1.0 - 1e-6,
    )
    blend_predictions = (
        np.clip(
            0.75 * p2_p8_predictions + 0.25 * rank_columns(p6_predictions),
            1e-6,
            1.0 - 1e-6,
        )
        if ENABLE_P6
        else p2_p8_predictions
    )
    assert blend_predictions.shape == (len(test), len(TARGETS))
    assert np.isfinite(blend_predictions).all()
    assert ((blend_predictions > 0) & (blend_predictions < 1)).all()
    if len(test) > 1:
        assert (np.ptp(p2_predictions, axis=0) > 0).all()
        assert (np.ptp(p8_predictions, axis=0) > 0).all()
        if ENABLE_P6:
            assert p6_predictions is not None
            assert (np.ptp(p6_predictions, axis=0) > 0).all()

    submission = sample.copy()
    submission.loc[:, TARGETS] = blend_predictions
    assert submission.columns.tolist() == [ID, *TARGETS]
    assert submission[ID].tolist() == study_ids
    submission_path = OUTPUT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    output_prefix = "p8_blend"
    prediction_frame(study_ids, p2_predictions).to_csv(
        OUTPUT_ROOT / f"{output_prefix}_p2_predictions.csv", index=False
    )
    prediction_frame(study_ids, p8_predictions).to_csv(
        OUTPUT_ROOT / f"{output_prefix}_p8_predictions.csv", index=False
    )
    if ENABLE_P6:
        assert p6_predictions is not None
        prediction_frame(study_ids, p6_predictions).to_csv(
            OUTPUT_ROOT / "p8_blend_p6_predictions.csv", index=False
        )
    np.save(
        OUTPUT_ROOT / f"{output_prefix}_p8_predictions_by_fold.npy",
        p8_fold_predictions,
    )
    np.save(OUTPUT_ROOT / f"{output_prefix}_feature_mask.npy", feature_mask)
    np.save(OUTPUT_ROOT / f"{output_prefix}_feature_positions.npy", positions)
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_ROOT / f"{output_prefix}_test_selections.csv", index=False
    )

    artifacts = [
        "submission.csv",
        f"{output_prefix}_p2_predictions.csv",
        f"{output_prefix}_p8_predictions.csv",
        f"{output_prefix}_p8_predictions_by_fold.npy",
        f"{output_prefix}_feature_mask.npy",
        f"{output_prefix}_feature_positions.npy",
        f"{output_prefix}_test_selections.csv",
    ]
    if ENABLE_P6:
        artifacts.append("p8_blend_p6_predictions.csv")
    effective_run_id = RUN_ID
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "studies": len(test),
        "series": len(series),
        "targets": TARGETS,
        "source_p2": {
            "run_id": p2_manifest["run_id"],
            "manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
            "checkpoint_sha256": p2_manifest["artifacts_sha256"][
                "p2_coatnet_pseudo_best.pt"
            ],
            "oof_macro_auc": P2_OOF_MACRO_AUC,
        },
        "source_p8": {
            str(fold): {
                "run_id": p8_manifests[fold]["run_id"],
                "manifest_sha256": EXPECTED_P8_MANIFEST_SHA256[fold],
                "checkpoint_sha256": EXPECTED_P8_CHECKPOINT_SHA256[fold],
                "teacher_manifest_sha256": p8_manifests[fold][
                    "teacher_manifest_sha256"
                ],
                "teacher_target_sha256": p8_manifests[fold][
                    "teacher_target_sha256"
                ],
            }
            for fold in range(5)
        },
        "source_p6": (
            {
                "run_id": p6_manifest["run_id"],
                "manifest_sha256": EXPECTED_P6_MANIFEST_SHA256,
                "checkpoint_sha256": p6_manifest["artifacts_sha256"][
                    "p2_coatnet_pseudo_best.pt"
                ],
                "oof_macro_auc": P6_OOF_MACRO_AUC,
            }
            if ENABLE_P6 and p6_manifest is not None
            else None
        ),
        "oof": {
            "p8_macro_auc": P8_OOF_MACRO_AUC,
            "equal_rank_p2_p8_macro_auc": BLEND_OOF_MACRO_AUC,
            "fixed_75pct_p2_p8_25pct_p6_macro_auc": (
                P6_BLEND_OOF_MACRO_AUC if ENABLE_P6 else None
            ),
        },
        "geometry": {
            "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
            "slot_specs": SLOT_SPECS,
            "slot_bounds": SLOT_BOUNDS,
            "relative_slice_range": [0.02, 0.98],
            "physical_crop_mm": 140.0,
            "cache_size": CACHE_SIZE,
            "model_size": MODEL_SIZE,
            "k_eval": K_EVAL,
        },
        "ensemble": {
            "p2": "mean probabilities from five report-purged P2 heads",
            "p8": "mean probabilities from five fold-safe noisy-student folds",
            "p6": (
                "mean probabilities from five P6-transfer report-purged heads"
                if ENABLE_P6
                else None
            ),
            "final": (
                "75% P2+P8 equal-rank mean plus 25% P6 target rank, "
                "clipped to [1e-6, 1-1e-6]"
                if ENABLE_P6
                else "equal-rank mean of P2 and P8 by target, clipped to "
                "[1e-6, 1-1e-6]"
            ),
            "formula": "0.375 rank(P2) + 0.375 rank(P8) + 0.25 rank(P6)",
            "weights": (
                {"p2_p8_equal_rank": 0.75, "p6_rank": 0.25}
                if ENABLE_P6
                else {"p2": 0.5, "p8": 0.5}
            ),
        },
        "inference_optimization": (
            "two-study batches; resident FP16 inference weights; bounded "
            "asynchronous per-device pipeline; one shared resize and "
            "normalization and FP16 input cast per device group; alternating "
            "P6 batches"
            if ENABLE_P6
            else "two-study batches; resident FP16 inference weights; one "
            "shared resize and normalization per device group"
        ),
        "submission_file": "submission.csv",
        "test_metadata_sha256": {
            "test.csv": sha256_file(
                data_root / ("test.csv" if ON_KAGGLE else "train.csv")
            ),
            "test_series.csv": sha256_file(
                data_root
                / ("test_series.csv" if ON_KAGGLE else "train_series.csv")
            ),
        },
        "artifacts_sha256": {
            name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
        },
        "elapsed": {
            "model_load_minutes": (
                inference_started - model_load_started
            ) / 60.0,
            "dicom_and_seven_model_inference_minutes": (
                time.time() - inference_started
            ) / 60.0,
            "dicom_and_model_inference_minutes": (
                time.time() - inference_started
            ) / 60.0,
            "total_minutes": (time.time() - started) / 60.0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "pydicom": pydicom.__version__,
            "visible_gpus": visible_gpus,
            "devices": [str(device) for device in devices],
            "batch_size": batch_size,
            "loader_workers": workers,
            "device_assignments": {
                str(devices[index]): [
                    *(["P2"] if index == 0 else []),
                    *[f"P8-fold-{fold}" for fold in assignments[index]],
                    *(["P6-alternating-batches"] if ENABLE_P6 else []),
                ]
                for index in assignments
            },
        },
    }
    manifest_path = OUTPUT_ROOT / "p8_blend_submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        {
            (
                "P8_BLEND_SUBMISSION_COMPLETE"
                if ENABLE_P6
                else "P8_BLEND_SUBMISSION_COMPLETE"
            ): True,
            "studies": len(submission),
            "submission_sha256": sha256_file(submission_path),
            "manifest_sha256": sha256_file(manifest_path),
            "prediction_ranges": {
                target: [
                    float(submission[target].min()),
                    float(submission[target].max()),
                ]
                for target in TARGETS
            },
            "elapsed_minutes": manifest["elapsed"]["total_minutes"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
