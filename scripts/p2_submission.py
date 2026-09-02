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


def main() -> int:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    p2_root = find_p2_root()
    p2_manifest_path = p2_root / "p2_manifest.json"
    assert sha256_file(p2_manifest_path) == EXPECTED_P2_MANIFEST_SHA256
    p2_manifest = json.loads(p2_manifest_path.read_text())
    assert p2_manifest["run_id"] == "P2-COATNET-v1"
    assert p2_manifest["best_epoch"] == 8
    assert p2_manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
    required_model_artifacts = [
        "p2_coatnet_pseudo_best.pt",
        *[f"p2_fold_{fold}_head.pt" for fold in range(5)],
    ]
    for name in required_model_artifacts:
        path = p2_root / name
        assert path.is_file() and sha256_file(path) == p2_manifest[
            "artifacts_sha256"
        ][name]

    if ON_KAGGLE:
        data_root = find_competition_root()
        dicom_root = data_root / "test_series"
        test = pd.read_csv(data_root / "test.csv", dtype={ID: str})
        series = pd.read_csv(data_root / "test_series.csv", dtype={ID: str})
        sample = pd.read_csv(data_root / "sample_submission.csv", dtype={ID: str})
        workers = 2
    else:
        data_root = Path("data/raw").resolve()
        dicom_root = Path("data/sample/files/train_series").resolve()
        metadata = pd.read_csv(data_root / "train.csv", dtype={ID: str})
        series_all = pd.read_csv(data_root / "train_series.csv", dtype={ID: str})
        available = {path.name for path in dicom_root.iterdir() if path.is_dir()}
        local_ids = [uid for uid in metadata[ID] if uid in available]
        assert local_ids
        richest = max(
            local_ids,
            key=lambda uid: sum(1 for _ in (dicom_root / uid).glob("*/*.dcm")),
        )
        test = pd.DataFrame({ID: [richest]})
        series = series_all[series_all[ID].isin(test[ID])].copy()
        sample = pd.DataFrame({ID: test[ID], **{target: 0.5 for target in TARGETS}})
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

    print(
        {
            "P2_SUBMISSION_START": True,
            "on_kaggle": ON_KAGGLE,
            "studies": len(study_ids),
            "series": len(series),
            "p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "visible_gpus": torch.cuda.device_count(),
        },
        flush=True,
    )

    dataset = TestStudyDataset(study_ids, series_by_study, dicom_root)
    loader_options: dict[str, object] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channels_last = device.type == "cuda"
    if channels_last:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    model = CoAtNetMIL(channels_last=channels_last)
    state = torch.load(
        p2_root / "p2_coatnet_pseudo_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model.eval()

    features = np.zeros((len(test), K_EVAL, FEATURE_DIM), dtype=np.float16)
    feature_mask = np.zeros((len(test), K_EVAL), dtype=np.uint8)
    positions = np.full((len(test), K_EVAL), -1, dtype=np.int16)
    selection_rows: list[dict[str, object]] = []
    inference_started = time.time()
    with torch.inference_mode():
        for processed, batch in enumerate(loader, start=1):
            indices, windows, window_mask, centers, selections_json = batch
            index = int(indices.item())
            windows = windows.to(device, non_blocking=True)
            window_mask = window_mask.to(device, non_blocking=True).bool()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                encoded = model.encode(windows)
            features[index] = encoded.float().cpu().numpy()[0].astype(np.float16)
            selected_mask = window_mask.cpu().numpy()[0]
            feature_mask[index] = selected_mask.astype(np.uint8)
            selected_centers = centers.numpy()[0]
            positions[index, selected_mask] = selected_centers[selected_mask]
            for selection in json.loads(selections_json[0]):
                selection_rows.append({ID: study_ids[index], **selection})
            if processed % 50 == 0 or processed == len(test):
                print(
                    {
                        "P2_SUBMISSION_PROGRESS": f"{processed}/{len(test)}",
                        "elapsed_minutes": (time.time() - inference_started) / 60.0,
                        "gpu": gpu_snapshot(),
                    },
                    flush=True,
                )

    assert np.isfinite(features).all()
    assert feature_mask.any(axis=1).all()
    assert (positions[feature_mask.astype(bool)] >= 1).all()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fold_predictions = np.zeros((5, len(test), N_TARGETS), dtype=np.float32)
    for fold in range(5):
        head = TargetAttentionHead().to(device)
        fold_state = torch.load(
            p2_root / f"p2_fold_{fold}_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        head.load_state_dict(fold_state, strict=True)
        fold_predictions[fold] = head_predictions(
            head, features, feature_mask.astype(bool), device
        )
        del head, fold_state

    direct_state = torch.load(
        p2_root / "p2_coatnet_pseudo_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    direct_head = TargetAttentionHead().to(device)
    direct_head.load_state_dict(
        {
            key.removeprefix("head."): value
            for key, value in direct_state.items()
            if key.startswith("head.")
        },
        strict=True,
    )
    direct_predictions = head_predictions(
        direct_head, features, feature_mask.astype(bool), device
    )
    submission_predictions = np.clip(
        fold_predictions.mean(axis=0), 1e-6, 1.0 - 1e-6
    )
    assert submission_predictions.shape == (len(test), N_TARGETS)
    assert np.isfinite(submission_predictions).all()
    assert ((submission_predictions > 0) & (submission_predictions < 1)).all()
    if len(test) > 1:
        assert (np.ptp(submission_predictions, axis=0) > 0).all()

    submission = sample.copy()
    submission.loc[:, TARGETS] = submission_predictions
    assert submission.columns.tolist() == [ID, *TARGETS]
    assert submission[ID].tolist() == study_ids
    submission_path = OUTPUT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    np.save(OUTPUT_ROOT / "p2_test_predictions_by_fold.npy", fold_predictions)
    np.save(OUTPUT_ROOT / "p2_test_feature_mask.npy", feature_mask)
    np.save(OUTPUT_ROOT / "p2_test_feature_positions.npy", positions)
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_ROOT / "p2_test_selections.csv", index=False
    )
    pd.DataFrame({ID: study_ids, **{
        target: direct_predictions[:, target_index]
        for target_index, target in enumerate(TARGETS)
    }}).to_csv(OUTPUT_ROOT / "p2_test_direct_predictions.csv", index=False)

    artifacts = [
        "submission.csv",
        "p2_test_predictions_by_fold.npy",
        "p2_test_feature_mask.npy",
        "p2_test_feature_positions.npy",
        "p2_test_selections.csv",
        "p2_test_direct_predictions.csv",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": "P2-SUBMISSION-v1",
        "studies": len(test),
        "series": len(series),
        "targets": TARGETS,
        "source_p2_run_id": p2_manifest["run_id"],
        "source_p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
        "source_p2_best_epoch": p2_manifest["best_epoch"],
        "source_p2_oof_macro_auc": p2_manifest["fold_oof"]["macro_auc"],
        "geometry": {
            "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
            "slot_specs": SLOT_SPECS,
            "slot_bounds": SLOT_BOUNDS,
            "relative_slice_range": [0.02, 0.98],
            "physical_crop_mm": CROP_MM,
            "cache_size": CACHE_SIZE,
            "model_size": MODEL_SIZE,
            "k_eval": K_EVAL,
        },
        "ensemble": "mean probabilities from five report-purged fold heads",
        "submission_file": "submission.csv",
        "test_metadata_sha256": {
            "test.csv": sha256_file(data_root / (
                "test.csv" if ON_KAGGLE else "train.csv"
            )),
            "test_series.csv": sha256_file(data_root / (
                "test_series.csv" if ON_KAGGLE else "train_series.csv"
            )),
        },
        "artifacts_sha256": {
            name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
        },
        "elapsed": {
            "dicom_and_feature_minutes": (time.time() - inference_started) / 60.0,
            "total_minutes": (time.time() - started) / 60.0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "pydicom": pydicom.__version__,
            "device": str(device),
            "visible_gpus": torch.cuda.device_count(),
            "loader_workers": workers,
        },
    }
    manifest_path = OUTPUT_ROOT / "p2_submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        {
            "P2_SUBMISSION_COMPLETE": True,
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
