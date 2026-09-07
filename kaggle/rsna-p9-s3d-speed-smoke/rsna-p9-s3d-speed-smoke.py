#!/usr/bin/env python3
"""Train the frozen P9 compact true-3D S3D specialist on Kaggle T4 x2."""

from __future__ import annotations

import os as _p9_bootstrap_os

_p9_bootstrap_os.environ[
    "P9_TEACHER_MANIFEST_SHA256"
] = "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
_p9_bootstrap_os.environ["P9_SPEED_SMOKE"] = "1"

import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import s3d


ID = "StudyInstanceUID"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
N_TARGETS = len(TARGETS)
N_FOLDS = 5
N_SLOTS = 5
FEATURE_DIM = 1_024
CACHE_SIZE = 336
MODEL_SIZE = 160
FRAMES = 16
REP_EPOCHS = 6
HEAD_EPOCHS = 30
GLOBAL_STUDY_BATCH = 2
GRAD_ACCUM = 4
WORKERS_PER_RANK = 2
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
REP_WEIGHT_DECAY = 0.02
HEAD_FOLD_LR = 2e-4
HEAD_FOLD_WEIGHT_DECAY = 1e-3
P9_SEED = 20260909
FOLD_SEED = 20260831
CACHE_ID = "P2-CACHE-v1"
CACHE_SIGNATURE_SHA256 = (
    "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
)
WEIGHT_FILENAME = "s3d-d76dad2f.pth"
WEIGHT_SHA256 = "d76dad2f4b2a6701ad7f64e85b294b2d1989f1ea0d3f8b240446ed2152850bbb"
TEACHER_MANIFEST_SHA256 = os.environ.get("P9_TEACHER_MANIFEST_SHA256", "")
MODEL_MEAN = (0.43216, 0.394666, 0.37645)
MODEL_STD = (0.22803, 0.22145, 0.216989)
SPEED_SMOKE = os.environ.get("P9_SPEED_SMOKE") == "1"
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/p9_s3d_local")
RUN_ID = "P9-S3D-SPEED-SMOKE-v1" if SPEED_SMOKE else "P9-S3D-SPECIALIST-v1"


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
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


def find_named_file(filename: str, minimum_bytes: int = 1) -> Path:
    direct_map = {
        "train.csv": Path(
            "/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"
        ),
        "training_targets.csv": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-report-weak-supervision/training_targets.csv"
        ),
        WEIGHT_FILENAME: Path(
            "/kaggle/input/datasets/vallepuvamsikrishna/"
            "rsna-p9-s3d-init/s3d-d76dad2f.pth"
        ),
        "p2_cache_manifest_part_a.json": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p2-wide-cache-a/p2_cache_manifest_part_a.json"
        ),
        "p2_cache_manifest_part_b.json": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p2-wide-cache-b/p2_cache_manifest_part_b.json"
        ),
    }
    candidates = [direct_map[filename]] if filename in direct_map else []
    for root in candidate_roots():
        candidates.extend(
            [
                root / filename,
                root / "data/raw" / filename,
                root / "outputs/kaggle/rsna-report-weak-supervision" / filename,
                root / "kaggle/rsna-p9-s3d-init" / filename,
            ]
        )
    for path in candidates:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = [
            path
            for path in kaggle_input.glob(f"**/{filename}")
            if path.is_file() and path.stat().st_size >= minimum_bytes
        ]
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(filename)


def find_teacher_file(filename: str, minimum_bytes: int = 1) -> Path:
    direct = [
        Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p8-teacher-labels"
        )
        / filename,
        Path("outputs/kaggle/rsna-p8-teacher-labels-v2") / filename,
    ]
    for path in direct:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = [
            path
            for path in kaggle_input.glob(f"**/{filename}")
            if path.is_file() and path.stat().st_size >= minimum_bytes
        ]
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(filename)


def prepare_cache_part(part: str) -> dict[str, object]:
    manifest_path = find_named_file(f"p2_cache_manifest_part_{part}.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_id"] == CACHE_ID
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE_SHA256
    root = manifest_path.parent
    names = {
        "pixels": f"p2_pixels_part_{part}.npy",
        "mask": f"p2_valid_mask_part_{part}.npy",
        "studies": f"p2_studies_part_{part}.csv",
    }
    paths = {key: root / name for key, name in names.items()}
    for path in paths.values():
        assert path.is_file(), path
    return {"manifest": manifest, "root": root, "names": names, **paths}


def validate_inputs(
    weight_path: Path,
    parts: list[dict[str, object]],
    teacher_manifest_path: Path,
    teacher_paths: dict[str, Path],
) -> dict[str, object]:
    assert TEACHER_MANIFEST_SHA256 and len(TEACHER_MANIFEST_SHA256) == 64
    assert sha256_file(weight_path) == WEIGHT_SHA256
    assert sha256_file(teacher_manifest_path) == TEACHER_MANIFEST_SHA256
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    assert teacher_manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert teacher_manifest["studies"] == 4_407
    assert teacher_manifest["fold_safety"]["p2_direct_expert_gradient_rows"] == 0
    assert teacher_manifest["fold_safety"]["p6_direct_expert_gradient_rows"] == 0
    for name, path in teacher_paths.items():
        assert sha256_file(path) == teacher_manifest["artifacts_sha256"][name]
    jobs: list[tuple[Path, str]] = []
    for cache_part in parts:
        manifest = cache_part["manifest"]
        names = cache_part["names"]
        assert isinstance(manifest, dict) and isinstance(names, dict)
        for key, name in names.items():
            path = cache_part[key]
            assert isinstance(path, Path)
            jobs.append((path, str(manifest["files_sha256"][name])))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(sha256_file, path): (path, expected) for path, expected in jobs
        }
        for future in as_completed(futures):
            path, expected = futures[future]
            assert future.result() == expected, f"{path.name} hash mismatch"
            print({"P9_INPUT_HASH_OK": path.name}, flush=True)
    return teacher_manifest


def normalize_report(report: object) -> str:
    text = unicodedata.normalize("NFKC", str(report)).casefold()
    return "".join(character for character in text if character.isalnum())


def report_hash(report: object) -> str:
    normalized = normalize_report(report)
    if not normalized:
        raise ValueError("empty report after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


def stable_group_split(
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups, dtype=str)
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_by_group = dict(zip(unique_groups.tolist(), counts.tolist(), strict=True))
    ordered_groups = sorted(
        unique_groups.tolist(),
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    target_rows = round(len(indices) * valid_fraction)
    valid_groups: set[str] = set()
    valid_rows = 0
    for group in ordered_groups:
        candidate_rows = valid_rows + count_by_group[group]
        if abs(candidate_rows - target_rows) <= abs(valid_rows - target_rows):
            valid_groups.add(group)
            valid_rows = candidate_rows
        if valid_rows == target_rows:
            break
    valid_mask = np.asarray([group in valid_groups for group in groups], dtype=bool)
    assert valid_mask.any() and (~valid_mask).any()
    assert not set(groups[valid_mask]) & set(groups[~valid_mask])
    return indices[~valid_mask], indices[valid_mask]


class GpuTelemetry:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.output_path.write_text(
            "unix_time,timestamp,index,name,utilization_gpu_percent,"
            "utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,"
            "temperature_c\n"
        )
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        query = (
            "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,"
            "memory.total,power.draw,temperature.gpu"
        )
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
                now = time.time()
                with self.output_path.open("a", newline="") as handle:
                    writer = csv.writer(handle)
                    for row in rows:
                        writer.writerow([now, *row])
                print({"P9_GPU": rows}, flush=True)
            except Exception as error:
                print({"P9_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)
            self.stop.wait(30)

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


class TargetSlotAttentionHead(nn.Module):
    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        self.slot_embedding = nn.Parameter(torch.randn(N_SLOTS, FEATURE_DIM) * 0.01)
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.attention = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, N_TARGETS),
        )
        self.target_weight = nn.Parameter(torch.randn(N_TARGETS, FEATURE_DIM) * 0.02)
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(features + self.slot_embedding[None])
        scores = self.attention(hidden).transpose(1, 2)
        scores = scores.masked_fill(~mask[:, None, :].bool(), -1e4)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("bts,bsf->btf", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


class S3DMIL(nn.Module):
    def __init__(self, weight_path: Path) -> None:
        super().__init__()
        base = s3d(weights=None)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        incompatible = base.load_state_dict(state, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        self.backbone = base.features
        self.head = TargetSlotAttentionHead()
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1, 1))

    def encode(
        self, clips: torch.Tensor, slot_mask: torch.Tensor, augment: bool
    ) -> torch.Tensor:
        batch, slots, frames, height, width = clips.shape
        assert slots == N_SLOTS and frames == FRAMES
        pixels = clips.reshape(batch * slots, frames, height, width)
        pixels = pixels[:, None].expand(-1, 3, -1, -1, -1).float().div_(255.0)
        pixels = F.interpolate(
            pixels,
            size=(FRAMES, MODEL_SIZE, MODEL_SIZE),
            mode="trilinear",
            align_corners=False,
        )
        if augment:
            gain = torch.empty(
                (len(pixels), 1, 1, 1, 1), device=pixels.device
            ).uniform_(0.90, 1.10)
            offset = torch.empty(
                (len(pixels), 1, 1, 1, 1), device=pixels.device
            ).uniform_(-0.05, 0.05)
            pixels = (pixels * gain + offset).clamp_(0.0, 1.0)
        pixels = ((pixels - self.mean) / self.std).contiguous(
            memory_format=torch.channels_last_3d
        )
        encoded = self.backbone(pixels)
        encoded = F.adaptive_avg_pool3d(encoded, 1).flatten(1)
        encoded = encoded.reshape(batch, slots, FEATURE_DIM)
        return encoded * slot_mask[:, :, None].to(encoded.dtype)

    def forward(
        self,
        clips: torch.Tensor,
        slot_mask: torch.Tensor,
        augment: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        features = self.encode(clips, slot_mask, augment)
        if return_features:
            return features
        return self.head(features, slot_mask)


def weighted_terms(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * weights).sum(), weights.sum().clamp_min(1.0)


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    numerator, denominator = weighted_terms(logits, labels, weights)
    return numerator / denominator


def calculate_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[pd.DataFrame, float]:
    rows = [
        {"target": target, "auc": float(roc_auc_score(labels[:, i], predictions[:, i]))}
        for i, target in enumerate(TARGETS)
    ]
    macro = float(np.mean([row["auc"] for row in rows]))
    rows.append({"target": "MACRO", "auc": macro})
    return pd.DataFrame(rows), macro


def launch_ddp() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P9 requires exactly two Kaggle GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    print({"P9_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P9_DDP_PROCESS_COMPLETE": True}, flush=True)
    return 0


def worker_main() -> int:
    distributed = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if ON_KAGGLE:
        assert distributed and world_size == 2
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_primary = rank == 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    telemetry = GpuTelemetry(OUTPUT_ROOT / "p9_gpu_telemetry.csv") if is_primary else None
    if telemetry is not None:
        telemetry.start()

    random.seed(P9_SEED)
    np.random.seed(P9_SEED)
    torch.manual_seed(P9_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(P9_SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_path = find_named_file("train.csv", 100_000)
    target_path = find_named_file("training_targets.csv", 100_000)
    weight_path = find_named_file(WEIGHT_FILENAME, 30_000_000)
    teacher_manifest_path = find_teacher_file("p8_teacher_manifest.json", 1_000)
    teacher_paths = {
        name: find_teacher_file(name, minimum)
        for name, minimum in {
            "p8_teacher_studies.csv": 100_000,
            "p8_p2_direct_predictions.npy": 100_000,
            "p8_p6_direct_predictions.npy": 100_000,
        }.items()
    }
    part_a = prepare_cache_part("a")
    part_b = prepare_cache_part("b")
    parts = [part_a, part_b]
    if is_primary:
        if SPEED_SMOKE:
            assert sha256_file(weight_path) == WEIGHT_SHA256
            assert sha256_file(teacher_manifest_path) == TEACHER_MANIFEST_SHA256
            print({"P9_SMOKE_INPUT_MANIFESTS_VALIDATED": True}, flush=True)
        else:
            validate_inputs(
                weight_path, parts, teacher_manifest_path, teacher_paths
            )
            print({"P9_IMMUTABLE_INPUTS_VALIDATED": True}, flush=True)
    if distributed:
        dist.barrier()

    assert part_a["manifest"]["row_range"] == [0, 2204]
    assert part_b["manifest"]["row_range"] == [2204, 4407]
    assert part_a["manifest"]["contract"] == part_b["manifest"]["contract"]
    train_all = pd.read_csv(train_path, dtype={ID: str})
    original = pd.read_csv(target_path, dtype={ID: str})
    ids_a = pd.read_csv(part_a["studies"], dtype={ID: str})[ID].tolist()
    ids_b = pd.read_csv(part_b["studies"], dtype={ID: str})[ID].tolist()
    study_ids = ids_a + ids_b
    assert len(study_ids) == 4_407
    assert study_ids == train_all[ID].tolist() == original[ID].tolist()
    teacher_studies = pd.read_csv(teacher_paths["p8_teacher_studies.csv"], dtype={ID: str})
    assert teacher_studies[ID].tolist() == study_ids

    p2_teacher = np.load(teacher_paths["p8_p2_direct_predictions.npy"], mmap_mode="r")
    p6_teacher = np.load(teacher_paths["p8_p6_direct_predictions.npy"], mmap_mode="r")
    assert p2_teacher.shape == p6_teacher.shape == (4_407, N_TARGETS)
    report_y = original[TARGETS].to_numpy(np.float32)
    y = report_y.copy()
    label_weights = original[
        [f"weight__{target}" for target in TARGETS]
    ].to_numpy(np.float32)
    is_gold = original["is_gold"].astype(bool).to_numpy()
    pseudo_indices = np.where(~is_gold)[0]
    gold_indices = np.where(is_gold)[0]
    assert len(pseudo_indices) == 4_349 and len(gold_indices) == 58
    y[pseudo_indices] = (
        0.50 * report_y[pseudo_indices]
        + 0.25 * np.asarray(p2_teacher[pseudo_indices], dtype=np.float32)
        + 0.25 * np.asarray(p6_teacher[pseudo_indices], dtype=np.float32)
    )
    assert np.array_equal(y[gold_indices], report_y[gold_indices])
    assert np.isfinite(y).all() and 0 <= y.min() <= y.max() <= 1

    metadata = train_all.set_index(ID).loc[study_ids].reset_index()
    report_hashes = metadata["Report"].map(report_hash).to_numpy(str)
    adapt_indices, pseudo_valid_indices = stable_group_split(
        pseudo_indices,
        report_hashes[pseudo_indices],
        valid_fraction=0.10,
        seed=P9_SEED,
    )
    assert len(adapt_indices) == 3_914 and len(pseudo_valid_indices) == 435
    assert len(adapt_indices) % world_size == 0
    assert not (
        set(report_hashes[adapt_indices]) & set(report_hashes[pseudo_valid_indices])
    )
    if SPEED_SMOKE:
        adapt_indices = adapt_indices[:64]
        pseudo_valid_indices = pseudo_valid_indices[:16]

    pixels_a = np.load(part_a["pixels"], mmap_mode="r")
    pixels_b = np.load(part_b["pixels"], mmap_mode="r")
    mask_a = np.load(part_a["mask"], mmap_mode="r")
    mask_b = np.load(part_b["mask"], mmap_mode="r")
    slot_bounds = np.load(Path(part_a["root"]) / "p2_slot_bounds.npy").astype(int)
    assert slot_bounds.tolist() == [[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]]

    def read_study(index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < len(pixels_a):
            return np.asarray(pixels_a[index]), np.asarray(mask_a[index]).astype(bool)
        local = index - len(pixels_a)
        return np.asarray(pixels_b[local]), np.asarray(mask_b[local]).astype(bool)

    def build_clips(
        index: int, *, stochastic: bool, epoch: int
    ) -> tuple[np.ndarray, np.ndarray]:
        volume, valid_mask = read_study(index)
        clips = np.zeros((N_SLOTS, FRAMES, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        slot_mask = np.zeros(N_SLOTS, dtype=bool)
        rng = np.random.default_rng(P9_SEED + 10_007 * epoch + index)
        for slot, (start, stop) in enumerate(slot_bounds):
            valid = np.flatnonzero(valid_mask[start:stop]) + start
            if len(valid) == 0:
                continue
            if stochastic and len(valid) > FRAMES:
                offset = int(rng.integers(0, len(valid) - FRAMES + 1))
                selected = valid[offset : offset + FRAMES]
            else:
                positions = np.rint(np.linspace(0, len(valid) - 1, FRAMES)).astype(int)
                selected = valid[positions]
            clips[slot] = volume[selected]
            slot_mask[slot] = True
        assert slot_mask.any()
        return clips, slot_mask

    class StudyDataset(Dataset):
        def __init__(self, indices: np.ndarray, *, stochastic: bool, epoch: int = 0):
            self.indices = np.asarray(indices, dtype=int)
            self.stochastic = stochastic
            self.epoch = epoch

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, position: int):
            index = int(self.indices[position])
            clips, slot_mask = build_clips(
                index, stochastic=self.stochastic, epoch=self.epoch
            )
            return index, clips, slot_mask, y[index], label_weights[index]

    workers = WORKERS_PER_RANK if ON_KAGGLE else 0

    def make_loader(dataset: Dataset) -> DataLoader:
        options: dict[str, object] = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
        }
        if workers:
            options.update({"prefetch_factor": 2, "persistent_workers": False})
        return DataLoader(dataset, **options)

    def move_batch(batch):
        index, clips, slot_mask, labels, weights = batch
        return (
            index.numpy(),
            clips.to(device, non_blocking=True),
            slot_mask.to(device, non_blocking=True).bool(),
            labels.to(device, non_blocking=True),
            weights.to(device, non_blocking=True),
        )

    if is_primary:
        print(
            {
                "P9_START": True,
                "run_id": RUN_ID,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "world_size": world_size,
                "visible_gpus": torch.cuda.device_count(),
                "speed_smoke": SPEED_SMOKE,
                "pseudo_train": len(adapt_indices),
                "pseudo_valid": len(pseudo_valid_indices),
                "clips_per_study": N_SLOTS,
                "frames_per_clip": FRAMES,
                "image_size": MODEL_SIZE,
                "pseudo_formula": "0.50 report + 0.25 P2 direct + 0.25 P6 direct",
            },
            flush=True,
        )

    model = S3DMIL(weight_path).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 7_500_000 < parameter_count < 8_500_000
    train_model: nn.Module = model
    if distributed:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
    print(
        {
            "P9_S3D_DDP_READY": True,
            "rank": rank,
            "device": str(device),
            "parameters": parameter_count,
        },
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=REP_WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    rep_epochs = 1 if SPEED_SMOKE else REP_EPOCHS
    global_steps_per_epoch = math.ceil(len(adapt_indices) / GLOBAL_STUDY_BATCH)
    updates_per_epoch = math.ceil(global_steps_per_epoch / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_LR, HEAD_LR],
        total_steps=rep_epochs * updates_per_epoch,
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    def sync_buffers() -> None:
        if distributed:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

    @torch.inference_mode()
    def validation_loss(indices: np.ndarray) -> float:
        sync_buffers()
        local_indices = indices[rank::world_size]
        loader = make_loader(StudyDataset(local_indices, stochastic=False))
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in loader:
            _, clips, slot_mask, labels, weights = move_batch(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(clips, slot_mask)
                numerator, denominator = weighted_terms(logits, labels, weights)
            totals += torch.stack([numerator.double(), denominator.double()])
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return float((totals[0] / totals[1].clamp_min(1.0)).item())

    training_started = time.time()
    for epoch in range(rep_epochs):
        epoch_started = time.time()
        generator = torch.Generator().manual_seed(P9_SEED + epoch)
        permutation = torch.randperm(len(adapt_indices), generator=generator).numpy()
        ordered = adapt_indices[permutation]
        local_indices = ordered[rank::world_size]
        loader = make_loader(
            StudyDataset(local_indices, stochastic=True, epoch=epoch)
        )
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for step, batch in enumerate(loader):
            should_update = (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader)
            sync_context = (
                contextlib.nullcontext()
                if not distributed or should_update
                else train_model.no_sync()  # type: ignore[union-attr]
            )
            with sync_context:
                _, clips, slot_mask, labels, weights = move_batch(batch)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = train_model(clips, slot_mask, augment=True)
                    numerator, denominator = weighted_terms(logits, labels, weights)
                    global_denominator = denominator.detach().clone()
                    if distributed:
                        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
                    loss = numerator * world_size / global_denominator.clamp_min(1.0)
                scaler.scale(loss / GRAD_ACCUM).backward()
            totals += torch.stack([numerator.detach().double(), denominator.detach().double()])
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            if is_primary and (
                (SPEED_SMOKE and (step + 1) % 8 == 0)
                or (not SPEED_SMOKE and (step + 1) % 200 == 0)
            ):
                print(
                    {
                        "P9_TRAIN_PROGRESS": {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "local_steps": len(loader),
                            "elapsed_minutes": (time.time() - epoch_started) / 60.0,
                        }
                    },
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.time() - epoch_started
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_loss = float((totals[0] / totals[1].clamp_min(1.0)).item())
        validation_started = time.time()
        valid_loss = validation_loss(pseudo_valid_indices)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds = time.time() - validation_started
        row = {
            "epoch": epoch + 1,
            "train_weighted_bce": train_loss,
            "pseudo_valid_weighted_bce": valid_loss,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "elapsed_minutes": (time.time() - training_started) / 60.0,
        }
        history.append(row)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if is_primary:
            print(
                {"P9_EPOCH_COMPLETE": row, "best_epoch": best_epoch}, flush=True
            )

    assert best_state is not None and best_epoch > 0
    model.load_state_dict(best_state, strict=True)
    if distributed:
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
        dist.barrier()

    if SPEED_SMOKE:
        memory = torch.tensor(
            [torch.cuda.max_memory_allocated(device) / (1024**2)],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.zeros_like(memory) for _ in range(world_size)]
        if distributed:
            dist.all_gather(gathered, memory)
        if is_primary:
            seconds_per_global_step = history[0]["training_seconds"] / math.ceil(
                len(adapt_indices) / GLOBAL_STUDY_BATCH
            )
            validation_seconds_per_study = history[0]["validation_seconds"] / len(
                pseudo_valid_indices
            )
            full_epoch_seconds = (
                seconds_per_global_step * math.ceil(3_914 / GLOBAL_STUDY_BATCH)
                + validation_seconds_per_study * 435
            )
            result = {
                "run_id": RUN_ID,
                "world_size": world_size,
                "global_study_batch": GLOBAL_STUDY_BATCH,
                "gradient_accumulation": GRAD_ACCUM,
                "clips_per_study": N_SLOTS,
                "frames_per_clip": FRAMES,
                "image_size": MODEL_SIZE,
                "train_studies": len(adapt_indices),
                "valid_studies": len(pseudo_valid_indices),
                "global_train_steps": math.ceil(
                    len(adapt_indices) / GLOBAL_STUDY_BATCH
                ),
                "seconds_per_global_step": seconds_per_global_step,
                "validation_seconds_per_study": validation_seconds_per_study,
                "train_weighted_bce": history[0]["train_weighted_bce"],
                "pseudo_valid_weighted_bce": history[0][
                    "pseudo_valid_weighted_bce"
                ],
                "projected_six_epoch_training_hours": (
                    full_epoch_seconds * REP_EPOCHS / 3_600
                ),
                "peak_memory_mib_by_rank": [float(value.item()) for value in gathered],
                "weight_sha256": WEIGHT_SHA256,
                "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
            }
            (OUTPUT_ROOT / "p9_speed_smoke.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print({"P9_SPEED_SMOKE_COMPLETE": result}, flush=True)
        if telemetry is not None:
            telemetry.close()
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return 0

    if is_primary:
        torch.save(best_state, OUTPUT_ROOT / "p9_s3d_representation.pt")
        pd.DataFrame(history).to_csv(
            OUTPUT_ROOT / "p9_representation_history.csv", index=False
        )
        safe_targets = original.copy()
        safe_targets.loc[pseudo_indices, TARGETS] = y[pseudo_indices]
        safe_targets.to_csv(OUTPUT_ROOT / "p9_safe_training_targets.csv", index=False)
    if distributed:
        dist.barrier()

    extraction_started = time.time()
    local_feature_indices = np.arange(rank, len(study_ids), world_size, dtype=np.int64)
    feature_shard_path = OUTPUT_ROOT / f"p9_feature_shard_rank_{rank}.npy"
    mask_shard_path = OUTPUT_ROOT / f"p9_mask_shard_rank_{rank}.npy"
    index_shard_path = OUTPUT_ROOT / f"p9_index_shard_rank_{rank}.npy"
    feature_shard = np.lib.format.open_memmap(
        feature_shard_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(local_feature_indices), N_SLOTS, FEATURE_DIM),
    )
    mask_shard = np.zeros((len(local_feature_indices), N_SLOTS), dtype=np.uint8)
    feature_loader = make_loader(
        StudyDataset(local_feature_indices, stochastic=False)
    )
    model.eval()
    processed = 0
    with torch.inference_mode():
        for batch in feature_loader:
            _, clips, slot_mask, _, _ = move_batch(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = model(
                    clips, slot_mask, augment=False, return_features=True
                )
            feature_shard[processed] = features.float().cpu().numpy()[0].astype(
                np.float16
            )
            mask_shard[processed] = slot_mask.cpu().numpy()[0].astype(np.uint8)
            processed += 1
            if is_primary and (processed % 250 == 0 or processed == len(local_feature_indices)):
                print(
                    f"P9 features ~{min(processed * world_size, len(study_ids)):,}/"
                    f"{len(study_ids):,} in {(time.time() - extraction_started) / 60:.1f} min",
                    flush=True,
                )
    feature_shard.flush()
    np.save(mask_shard_path, mask_shard)
    np.save(index_shard_path, local_feature_indices)
    if distributed:
        dist.barrier()

    feature_path = OUTPUT_ROOT / "p9_s3d_features.npy"
    mask_path = OUTPUT_ROOT / "p9_s3d_feature_mask.npy"
    if is_primary:
        feature_cache = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(study_ids), N_SLOTS, FEATURE_DIM),
        )
        feature_mask = np.zeros((len(study_ids), N_SLOTS), dtype=np.uint8)
        for shard_rank in range(world_size):
            shard_indices = np.load(OUTPUT_ROOT / f"p9_index_shard_rank_{shard_rank}.npy")
            feature_cache[shard_indices] = np.load(
                OUTPUT_ROOT / f"p9_feature_shard_rank_{shard_rank}.npy", mmap_mode="r"
            )
            feature_mask[shard_indices] = np.load(
                OUTPUT_ROOT / f"p9_mask_shard_rank_{shard_rank}.npy"
            )
        feature_cache.flush()
        assert np.isfinite(feature_cache).all() and feature_mask.any(axis=1).all()
        np.save(mask_path, feature_mask)
        pd.DataFrame({ID: study_ids}).to_csv(
            OUTPUT_ROOT / "p9_s3d_feature_studies.csv", index=False
        )
        for shard_rank in range(world_size):
            for kind in ("feature", "mask", "index"):
                (OUTPUT_ROOT / f"p9_{kind}_shard_rank_{shard_rank}.npy").unlink()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if not is_primary:
        return 0

    features_tensor = torch.from_numpy(
        np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
    ).to(device)
    masks_tensor = torch.from_numpy(np.load(mask_path).astype(bool)).to(device)
    labels_tensor = torch.from_numpy(y).to(device)
    weights_tensor = torch.from_numpy(label_weights).to(device)
    initial_head_state = {
        key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()
    }
    model.head.eval()
    with torch.inference_mode():
        direct_predictions = torch.sigmoid(
            model.head(features_tensor[gold_indices], masks_tensor[gold_indices])
        ).cpu().numpy()
    direct_metrics, direct_macro = calculate_metrics(
        y[gold_indices], direct_predictions
    )
    direct_metrics.to_csv(OUTPUT_ROOT / "p9_direct_expert_metrics.csv", index=False)
    direct_frame = pd.DataFrame({ID: np.asarray(study_ids)[gold_indices]})
    for target_index, target in enumerate(TARGETS):
        direct_frame[f"label__{target}"] = y[gold_indices, target_index]
        direct_frame[f"pred__{target}"] = direct_predictions[:, target_index]
    direct_frame.to_csv(
        OUTPUT_ROOT / "p9_direct_expert_predictions.csv", index=False
    )
    print({"P9_DIRECT_EXPERT_MACRO_AUC": direct_macro}, flush=True)

    oof = np.full((len(study_ids), N_TARGETS), np.nan, dtype=np.float32)
    gold_fold = np.full(len(gold_indices), -1, dtype=int)
    fold_rows: list[dict[str, object]] = []
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
        valid_index = gold_indices[gold_valid_pos]
        valid_hashes = set(report_hashes[valid_index])
        keep_pseudo = np.asarray(
            [
                index
                for index in pseudo_indices
                if report_hashes[index] not in valid_hashes
            ],
            dtype=int,
        )
        train_index = np.concatenate([keep_pseudo, gold_indices[gold_train_pos]])
        assert not (set(report_hashes[train_index]) & valid_hashes)
        gold_fold[gold_valid_pos] = fold
        fold_seed = FOLD_SEED + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        head = TargetSlotAttentionHead().to(device)
        head.load_state_dict(initial_head_state, strict=True)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=HEAD_FOLD_LR,
            weight_decay=HEAD_FOLD_WEIGHT_DECAY,
            fused=device.type == "cuda",
        )
        generator = torch.Generator().manual_seed(fold_seed)
        train_cpu = torch.as_tensor(train_index, dtype=torch.long)
        final_loss = float("nan")
        for _ in range(HEAD_EPOCHS):
            head.train()
            permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
            losses: list[float] = []
            for start in range(0, len(permutation), 256):
                index = permutation[start : start + 256].to(device)
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
        torch.save(head.state_dict(), OUTPUT_ROOT / f"p9_fold_{fold}_head.pt")
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "pseudo_rows": len(keep_pseudo),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": len(pseudo_indices) - len(keep_pseudo),
                "report_hash_overlap": 0,
                "final_loss": final_loss,
            }
        )
        print({"P9_HEAD_FOLD_COMPLETE": fold_rows[-1]}, flush=True)
    assert (gold_fold >= 0).all() and np.isfinite(oof[gold_indices]).all()
    oof_metrics, oof_macro = calculate_metrics(y[gold_indices], oof[gold_indices])
    oof_metrics.to_csv(OUTPUT_ROOT / "p9_oof_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "p9_fold_summary.csv", index=False)
    oof_frame = pd.DataFrame(
        {ID: np.asarray(study_ids)[gold_indices], "fold": gold_fold}
    )
    for target_index, target in enumerate(TARGETS):
        oof_frame[f"label__{target}"] = y[gold_indices, target_index]
        oof_frame[f"pred__{target}"] = oof[gold_indices, target_index]
    oof_frame.to_csv(OUTPUT_ROOT / "p9_oof_predictions.csv", index=False)
    print({"P9_OOF_MACRO_AUC": oof_macro}, flush=True)

    if telemetry is not None:
        telemetry.close()
    artifacts = [
        "p9_s3d_representation.pt",
        "p9_representation_history.csv",
        "p9_safe_training_targets.csv",
        "p9_s3d_features.npy",
        "p9_s3d_feature_mask.npy",
        "p9_s3d_feature_studies.csv",
        "p9_direct_expert_metrics.csv",
        "p9_direct_expert_predictions.csv",
        "p9_oof_metrics.csv",
        "p9_oof_predictions.csv",
        "p9_fold_summary.csv",
        "p9_gpu_telemetry.csv",
        *[f"p9_fold_{fold}_head.pt" for fold in range(N_FOLDS)],
    ]
    for name in artifacts:
        assert (OUTPUT_ROOT / name).is_file(), name
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "complete",
        "seeds": {"representation": P9_SEED, "expert_folds": FOLD_SEED},
        "studies": len(study_ids),
        "pseudo_studies": len(pseudo_indices),
        "expert_studies": len(gold_indices),
        "cache_id": CACHE_ID,
        "cache_signature_sha256": CACHE_SIGNATURE_SHA256,
        "model": {
            "architecture": "torchvision.models.video.s3d",
            "initialization": "S3D_Weights.KINETICS400_V1",
            "checkpoint_filename": WEIGHT_FILENAME,
            "checkpoint_sha256": WEIGHT_SHA256,
            "parameters_with_head": parameter_count,
            "feature_dim": FEATURE_DIM,
            "clips_per_study": N_SLOTS,
            "frames_per_clip": FRAMES,
            "image_size": MODEL_SIZE,
            "mean": MODEL_MEAN,
            "std": MODEL_STD,
        },
        "pseudo_target_formula": "0.50 report + 0.25 P2 direct + 0.25 P6 direct",
        "p4_teacher_excluded": True,
        "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
        "representation_expert_gradient_rows": 0,
        "pseudo_report_group_overlap": 0,
        "representation_epochs": REP_EPOCHS,
        "best_epoch": best_epoch,
        "best_pseudo_valid_weighted_bce": best_loss,
        "direct_expert_macro_auc": direct_macro,
        "oof_macro_auc": oof_macro,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "visible_gpus": torch.cuda.device_count(),
            "world_size": world_size,
            "distributed_backend": "nccl",
            "global_study_batch": GLOBAL_STUDY_BATCH,
            "gradient_accumulation": GRAD_ACCUM,
            "mixed_precision": "float16",
            "channels_last_3d": True,
        },
        "elapsed": {
            "representation_minutes": (
                extraction_started - training_started
            )
            / 60.0,
            "feature_and_head_minutes": (time.time() - extraction_started) / 60.0,
            "total_minutes": (time.time() - training_started) / 60.0,
        },
        "artifacts_sha256": {
            name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = OUTPUT_ROOT / "p9_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        {
            "P9_FULL_RUN_COMPLETE": True,
            "best_epoch": best_epoch,
            "direct_macro": direct_macro,
            "oof_macro": oof_macro,
            "manifest_sha256": sha256_file(manifest_path),
            "total_minutes": manifest["elapsed"]["total_minutes"],
        },
        flush=True,
    )
    return 0


def main() -> int:
    if ON_KAGGLE and torch.cuda.is_available() and "LOCAL_RANK" not in os.environ:
        return launch_ddp()
    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
