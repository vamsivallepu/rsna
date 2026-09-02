#!/usr/bin/env python3
"""Run the frozen P2 CoAtNet experiment with two-process Kaggle DDP.

The outer Kaggle process relaunches this file through torchrun.  Each worker owns
one T4, preserving the frozen global study batch of two while avoiding
``nn.DataParallel`` model replication on every forward pass.
"""

# ruff: noqa: E402 - local offline dependencies must be added before ML imports.

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import math
import os
os.environ["P2_SPEED_SMOKE"] = "1"
import platform
import random
import subprocess
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch.nn.parallel import DistributedDataParallel as DDP
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
GLOBAL_STUDY_BATCH = 2
GRAD_ACCUM = 4
GLOBAL_EVAL_STUDY_BATCH = 2
WORKERS_PER_RANK = 2
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3
REP_WEIGHT_DECAY = 0.02
HEAD_FOLD_LR = 2e-4
HEAD_FOLD_WEIGHT_DECAY = 1e-3
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
EXECUTION_REVISION = "ddp_t4_v3"

ON_KAGGLE = Path("/kaggle/working").is_dir()
FULL_RUN = ON_KAGGLE and torch.cuda.is_available()
SPEED_SMOKE = os.environ.get("P2_SPEED_SMOKE") == "1"
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/p2_coatnet_smoke")


def launch_ddp() -> int:
    """Relaunch the Kaggle script with one process per visible GPU."""
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P2 requires exactly two Kaggle GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    print({"P2_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P2_DDP_PROCESS_COMPLETE": True}, flush=True)
    return 0


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
    roots.append(Path("/tmp/rsna-coatnet-rmlp2-init"))
    return roots


def find_named_file(filename: str, minimum_bytes: int = 1) -> Path:
    direct: list[Path] = []
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


def prepare_cache_part(part: str) -> dict[str, object]:
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
    paths = {key: root / name for key, name in names.items()}
    for path in paths.values():
        assert path.is_file(), path
    return {"manifest": manifest, "root": root, "names": names, **paths}


def validate_immutable_inputs(model_path: Path, parts: list[dict[str, object]]) -> None:
    """Fully hash immutable inputs, reading the independent cache parts concurrently."""
    started = time.time()
    assert sha256_file(model_path) == MODEL_SHA256, "official model hash mismatch"
    jobs: list[tuple[Path, str]] = []
    for part in parts:
        manifest = part["manifest"]
        names = part["names"]
        assert isinstance(manifest, dict) and isinstance(names, dict)
        for key, name in names.items():
            path = part[key]
            assert isinstance(path, Path)
            jobs.append((path, str(manifest["files_sha256"][name])))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(sha256_file, path): (path, expected) for path, expected in jobs}
        for future in as_completed(futures):
            path, expected = futures[future]
            observed = future.result()
            assert observed == expected, f"{path.name} hash mismatch"
            print({"P2_INPUT_HASH_OK": path.name}, flush=True)
    print({"P2_INPUT_VALIDATION_MINUTES": (time.time() - started) / 60.0}, flush=True)


def normalize_report(report: object) -> str:
    text = unicodedata.normalize("NFKC", str(report)).casefold()
    return "".join(character for character in text if character.isalnum())


def report_hash(report: object) -> str:
    normalized = normalize_report(report)
    if not normalized:
        raise ValueError("empty report after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


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
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
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
                print({"P2_GPU": rows}, flush=True)
            except Exception as error:  # telemetry must never terminate training
                print({"P2_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)
            self.stop.wait(60)

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


class TargetAttentionHead(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, dropout: float = 0.20) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(256, N_TARGETS),
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
    def __init__(self, weight_path: Path, *, channels_last: bool) -> None:
        super().__init__()
        backbone = timm.create_model(ARCH, pretrained=False)
        state = load_file(str(weight_path), device="cpu")
        incompatible = backbone.load_state_dict(state, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        backbone.reset_classifier(0, global_pool="avg")
        assert backbone.num_features == FEATURE_DIM
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(enable=False)
        self.backbone = backbone
        self.head = TargetAttentionHead()
        self.channels_last = channels_last
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1))

    def encode(self, windows: torch.Tensor, augment: bool = False) -> torch.Tensor:
        batch, count = windows.shape[:2]
        pixels = windows.reshape(batch * count, 3, CACHE_SIZE, CACHE_SIZE).float().div(255.0)
        pixels = F.interpolate(
            pixels, size=(MODEL_SIZE, MODEL_SIZE), mode="bicubic", align_corners=False
        )
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        if augment:
            gain = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(0.90, 1.10)
            offset = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(-0.05, 0.05)
            pixels = (pixels * gain + offset).clamp(0.0, 1.0)
        pixels = (pixels - self.mean) / self.std
        return self.backbone(pixels).reshape(batch, count, FEATURE_DIM)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        augment: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        features = self.encode(windows, augment=augment)
        return features if return_features else self.head(features, mask)


def weighted_terms(
    logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * label_weights).sum(), label_weights.sum().clamp_min(1.0)


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor
) -> torch.Tensor:
    numerator, denominator = weighted_terms(logits, labels, label_weights)
    return numerator / denominator


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows = [
        {"target": target, "auc": float(roc_auc_score(labels[:, i], predictions[:, i]))}
        for i, target in enumerate(TARGETS)
    ]
    macro = float(np.mean([row["auc"] for row in rows]))
    rows.append({"target": "MACRO", "auc": macro})
    return pd.DataFrame(rows), macro


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repeats: int = 5_000,
    seed: int = FOLD_SEED + 100_000,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), len(labels))
        target_deltas = []
        for target_index in range(labels.shape[1]):
            sampled_labels = labels[sampled, target_index]
            if np.unique(sampled_labels).size < 2:
                break
            target_deltas.append(
                roc_auc_score(sampled_labels, candidate[sampled, target_index])
                - roc_auc_score(sampled_labels, baseline[sampled, target_index])
            )
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


def worker_main() -> int:
    distributed = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if FULL_RUN:
        assert distributed and world_size == 2
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_primary = rank == 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    telemetry = GpuTelemetry(OUTPUT_ROOT / "p2_gpu_telemetry.csv") if FULL_RUN and is_primary else None
    if telemetry is not None:
        telemetry.start()

    random.seed(P2_SEED)
    np.random.seed(P2_SEED)
    torch.manual_seed(P2_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(P2_SEED)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    rep_epochs = REP_EPOCHS if FULL_RUN and not SPEED_SMOKE else 1
    head_epochs = HEAD_EPOCHS if FULL_RUN else 1
    k_train = K_TRAIN if FULL_RUN else 1
    k_eval = K_EVAL if FULL_RUN else 1
    grad_accum = GRAD_ACCUM if FULL_RUN else 1
    workers = WORKERS_PER_RANK if FULL_RUN else 0

    train_path = find_named_file("train.csv", 100_000)
    target_path = find_named_file("training_targets.csv", 100_000)
    model_path = find_named_file("model.safetensors", 250_000_000)
    part_a = prepare_cache_part("a")
    part_b = prepare_cache_part("b") if FULL_RUN else None
    parts = [part_a, *([part_b] if part_b is not None else [])]
    if is_primary:
        if SPEED_SMOKE:
            assert sha256_file(model_path) == MODEL_SHA256
            print({"P2_SPEED_SMOKE_INPUTS": "manifests_and_model_validated"}, flush=True)
        else:
            validate_immutable_inputs(model_path, parts)
    if distributed:
        dist.barrier()

    if FULL_RUN:
        assert part_b is not None
        assert part_a["manifest"]["row_range"] == [0, 2204]
        assert part_b["manifest"]["row_range"] == [2204, 4407]
        assert part_a["manifest"]["contract"] == part_b["manifest"]["contract"]

    train_all = pd.read_csv(train_path, dtype={ID: str})
    targets_all = pd.read_csv(target_path, dtype={ID: str})
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

    report_hashes = metadata["Report"].map(report_hash).to_numpy(str)
    y = targets[TARGETS].to_numpy(np.float32)
    weights = targets[[f"weight__{target}" for target in TARGETS]].to_numpy(np.float32)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    if FULL_RUN:
        assert len(gold_indices) == 58 and len(pseudo_indices) == 4_349
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=P2_SEED)
        train_pos, valid_pos = next(
            splitter.split(pseudo_indices, groups=report_hashes[pseudo_indices])
        )
        adapt_indices = pseudo_indices[train_pos]
        pseudo_valid_indices = pseudo_indices[valid_pos]
        assert not (
            set(report_hashes[adapt_indices]) & set(report_hashes[pseudo_valid_indices])
        )
        if SPEED_SMOKE:
            adapt_indices = adapt_indices[:256]
            pseudo_valid_indices = pseudo_valid_indices[:16]
        assert len(adapt_indices) % world_size == 0
    else:
        adapt_indices = pseudo_valid_indices = np.arange(len(study_ids))

    def read_study(index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < len(pixels_a):
            return np.asarray(pixels_a[index]), np.asarray(mask_a[index])
        assert pixels_b is not None and mask_b is not None
        local = index - len(pixels_a)
        return np.asarray(pixels_b[local]), np.asarray(mask_b[local])

    def read_mask(index: int) -> np.ndarray:
        if index < len(mask_a):
            return np.asarray(mask_a[index])
        assert mask_b is not None
        return np.asarray(mask_b[index - len(mask_a)])

    def valid_centers(mask: np.ndarray) -> np.ndarray:
        centers: list[int] = []
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

    def build_windows(
        index: int, count: int, stochastic: bool, epoch: int
    ) -> tuple[np.ndarray, np.ndarray]:
        volume, mask = read_study(index)
        centers, window_mask = choose_centers(
            mask,
            count,
            stochastic=stochastic,
            seed=P2_SEED + 10_007 * epoch + index,
        )
        windows = np.zeros((count, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        for out_index, center in enumerate(centers):
            if window_mask[out_index]:
                windows[out_index] = volume[int(center) - 1:int(center) + 2]
        return windows, window_mask

    class StudyDataset(Dataset):
        def __init__(
            self, indices: np.ndarray, count: int, stochastic: bool, epoch: int = 0
        ) -> None:
            self.indices = np.asarray(indices, dtype=int)
            self.count = count
            self.stochastic = stochastic
            self.epoch = epoch

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, position: int):
            index = int(self.indices[position])
            windows, window_mask = build_windows(
                index, self.count, self.stochastic, self.epoch
            )
            return index, windows, window_mask, y[index], weights[index]

    def make_loader(dataset: Dataset, *, batch_size: int = 1) -> DataLoader:
        options: dict[str, object] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
        }
        if workers:
            options.update({"prefetch_factor": 2, "persistent_workers": False})
        return DataLoader(dataset, **options)

    def move_batch(batch):
        index, windows, window_mask, labels, label_weights = batch
        return (
            index.numpy(),
            windows.to(device, non_blocking=True),
            window_mask.to(device, non_blocking=True).bool(),
            labels.to(device, non_blocking=True),
            label_weights.to(device, non_blocking=True),
        )

    if is_primary:
        print({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "device": str(device),
            "full_run": FULL_RUN,
            "visible_gpus": torch.cuda.device_count(),
            "world_size": world_size,
            "execution_revision": EXECUTION_REVISION,
            "speed_smoke": SPEED_SMOKE,
            "gradient_checkpointing": False,
            "channels_last": device.type == "cuda",
            "cudnn_benchmark": device.type == "cuda",
            "studies": len(study_ids),
            "pseudo_train": len(adapt_indices),
            "pseudo_valid": len(pseudo_valid_indices),
            "gold": len(gold_indices),
        }, flush=True)

    channels_last = device.type == "cuda"
    model = CoAtNetMIL(model_path, channels_last=channels_last)
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 73_000_000 < parameter_count < 76_000_000
    train_model: nn.Module
    if distributed:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    else:
        train_model = model

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=REP_WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(adapt_indices) / GLOBAL_STUDY_BATCH) / grad_accum
    )
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

    def sync_model_buffers() -> None:
        if distributed:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

    @torch.inference_mode()
    def validation_loss(indices: np.ndarray) -> float:
        sync_model_buffers()
        local_indices = indices[rank::world_size]
        loader = make_loader(StudyDataset(local_indices, k_eval, False))
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in loader:
            _, windows, window_mask, labels, label_weights = move_batch(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(windows, window_mask, augment=False)
                numerator, denominator = weighted_terms(logits, labels, label_weights)
            totals[0] += numerator.double()
            totals[1] += denominator.double()
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return float((totals[0] / totals[1].clamp_min(1.0)).item())

    training_started = time.time()
    for epoch in range(rep_epochs):
        generator = torch.Generator().manual_seed(P2_SEED + epoch)
        permutation = torch.randperm(len(adapt_indices), generator=generator).numpy()
        ordered = adapt_indices[permutation]
        local_indices = ordered[rank::world_size]
        loader = make_loader(StudyDataset(local_indices, k_train, True, epoch))
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = torch.zeros(2, dtype=torch.float64, device=device)
        window_totals = torch.zeros(2, dtype=torch.float64, device=device)
        for step, batch in enumerate(loader):
            should_update = (step + 1) % grad_accum == 0 or step + 1 == len(loader)
            sync_context = (
                contextlib.nullcontext()
                if not distributed or should_update
                else train_model.no_sync()  # type: ignore[union-attr]
            )
            with sync_context:
                _, windows, window_mask, labels, label_weights = move_batch(batch)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = train_model(windows, window_mask, augment=True)
                    local_num, local_den = weighted_terms(logits, labels, label_weights)
                    global_den = local_den.detach().clone()
                    if distributed:
                        dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
                    # DDP averages gradients. Multiplying by world_size makes this
                    # exactly the old global weighted loss across the two studies.
                    loss = local_num * world_size / global_den.clamp_min(1.0)
                scaler.scale(loss / grad_accum).backward()
            epoch_totals += torch.stack([local_num.detach().double(), local_den.detach().double()])
            window_totals += torch.stack([local_num.detach().double(), local_den.detach().double()])
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            if FULL_RUN and (step + 1) % 250 == 0:
                logged = window_totals.clone()
                if distributed:
                    dist.all_reduce(logged, op=dist.ReduceOp.SUM)
                if is_primary:
                    print(
                        f"P2 train epoch {epoch + 1}/{rep_epochs} "
                        f"step {step + 1}/{len(loader)} "
                        f"loss={(logged[0] / logged[1].clamp_min(1.0)).item():.5f}",
                        flush=True,
                    )
                window_totals.zero_()

        global_epoch_totals = epoch_totals.clone()
        if distributed:
            dist.all_reduce(global_epoch_totals, op=dist.ReduceOp.SUM)
        train_loss = float(
            (global_epoch_totals[0] / global_epoch_totals[1].clamp_min(1.0)).item()
        )
        valid_loss = validation_loss(pseudo_valid_indices)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "pseudo_valid_weighted_bce": valid_loss,
            "elapsed_minutes": (time.time() - training_started) / 60.0,
        }
        if is_primary:
            history.append(row)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print({"P2_EPOCH_COMPLETE": row, "best_epoch": best_epoch}, flush=True)

    if SPEED_SMOKE:
        memory_mib = torch.tensor(
            [torch.cuda.max_memory_allocated(device) / (1024 ** 2)],
            dtype=torch.float64,
            device=device,
        )
        gathered_memory = (
            [torch.zeros_like(memory_mib) for _ in range(world_size)]
            if is_primary
            else None
        )
        if distributed:
            dist.gather(memory_mib, gather_list=gathered_memory, dst=0)
            dist.barrier()
        if is_primary:
            result = {
                "execution_revision": EXECUTION_REVISION,
                "world_size": world_size,
                "global_study_batch": GLOBAL_STUDY_BATCH,
                "gradient_accumulation": grad_accum,
                "global_steps": len(adapt_indices) // GLOBAL_STUDY_BATCH,
                "elapsed_minutes": history[-1]["elapsed_minutes"],
                "seconds_per_global_step": (
                    history[-1]["elapsed_minutes"] * 60
                    / (len(adapt_indices) // GLOBAL_STUDY_BATCH)
                ),
                "pseudo_valid_weighted_bce": history[-1]["pseudo_valid_weighted_bce"],
                "peak_memory_mib_by_rank": [
                    float(value.item()) for value in (gathered_memory or [memory_mib])
                ],
                "gradient_checkpointing": False,
                "channels_last": channels_last,
                "cudnn_benchmark": device.type == "cuda",
                "fused_adamw": device.type == "cuda",
            }
            (OUTPUT_ROOT / "p2_speed_smoke.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print({"P2_SPEED_SMOKE_COMPLETE": result}, flush=True)
            if telemetry is not None:
                telemetry.close()
        if distributed:
            dist.destroy_process_group()
        return 0

    checkpoint_path = OUTPUT_ROOT / "p2_coatnet_pseudo_best.pt"
    if is_primary:
        assert best_state is not None and best_epoch >= 1
        torch.save(best_state, checkpoint_path)
        pd.DataFrame(history).to_csv(
            OUTPUT_ROOT / "p2_representation_history.csv", index=False
        )
        del best_state
    if distributed:
        dist.barrier()
    selected_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state, strict=True)
    del selected_state
    gc.collect()
    if is_primary:
        print({
            "P2_REPRESENTATION_COMPLETE": True,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
        }, flush=True)

    # Each rank writes a disjoint inference shard. Rank zero merges the shards
    # after both GPUs have completed, avoiding DDP collectives with an odd row count.
    extract_started = time.time()
    local_feature_indices = np.arange(rank, len(study_ids), world_size, dtype=np.int64)
    feature_shard_path = OUTPUT_ROOT / f"p2_feature_shard_rank_{rank}.npy"
    mask_shard_path = OUTPUT_ROOT / f"p2_mask_shard_rank_{rank}.npy"
    position_shard_path = OUTPUT_ROOT / f"p2_position_shard_rank_{rank}.npy"
    index_shard_path = OUTPUT_ROOT / f"p2_index_shard_rank_{rank}.npy"
    feature_shard = np.lib.format.open_memmap(
        feature_shard_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(local_feature_indices), k_eval, FEATURE_DIM),
    )
    mask_shard = np.zeros((len(local_feature_indices), k_eval), dtype=np.uint8)
    position_shard = np.full((len(local_feature_indices), k_eval), -1, dtype=np.int16)
    feature_loader = make_loader(
        StudyDataset(local_feature_indices, k_eval, False), batch_size=1
    )
    model.eval()
    with torch.inference_mode():
        processed = 0
        for batch in feature_loader:
            indices, windows, window_mask, _, _ = move_batch(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                encoded = model(windows, window_mask, augment=False, return_features=True)
            selected_masks = window_mask.cpu().numpy().astype(bool)
            stop = processed + len(indices)
            feature_shard[processed:stop] = encoded.float().cpu().numpy().astype(np.float16)
            mask_shard[processed:stop] = selected_masks.astype(np.uint8)
            for offset, (index, selected_mask) in enumerate(
                zip(indices, selected_masks, strict=True)
            ):
                slice_mask = read_mask(int(index))
                centers, _ = choose_centers(
                    slice_mask, k_eval, stochastic=False, seed=0
                )
                position_shard[processed + offset, selected_mask] = centers[selected_mask]
            processed = stop
            if is_primary and (processed % 50 == 0 or processed == len(local_feature_indices)):
                print(
                    f"P2 features ~{min(processed * world_size, len(study_ids)):,}/"
                    f"{len(study_ids):,} in {(time.time() - extract_started) / 60:.1f} min",
                    flush=True,
                )
    feature_shard.flush()
    np.save(mask_shard_path, mask_shard)
    np.save(position_shard_path, position_shard)
    np.save(index_shard_path, local_feature_indices)
    if distributed:
        dist.barrier()

    feature_path = OUTPUT_ROOT / "p2_coatnet_features.npy"
    mask_path = OUTPUT_ROOT / "p2_coatnet_feature_mask.npy"
    position_path = OUTPUT_ROOT / "p2_coatnet_feature_positions.npy"
    if is_primary:
        feature_cache = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(study_ids), k_eval, FEATURE_DIM),
        )
        feature_mask = np.zeros((len(study_ids), k_eval), dtype=np.uint8)
        feature_positions = np.full((len(study_ids), k_eval), -1, dtype=np.int16)
        for shard_rank in range(world_size):
            shard_indices = np.load(
                OUTPUT_ROOT / f"p2_index_shard_rank_{shard_rank}.npy"
            )
            feature_cache[shard_indices] = np.load(
                OUTPUT_ROOT / f"p2_feature_shard_rank_{shard_rank}.npy", mmap_mode="r"
            )
            feature_mask[shard_indices] = np.load(
                OUTPUT_ROOT / f"p2_mask_shard_rank_{shard_rank}.npy"
            )
            feature_positions[shard_indices] = np.load(
                OUTPUT_ROOT / f"p2_position_shard_rank_{shard_rank}.npy"
            )
        feature_cache.flush()
        assert np.isfinite(feature_cache).all() and feature_mask.any(axis=1).all()
        np.save(mask_path, feature_mask)
        np.save(position_path, feature_positions)
        pd.DataFrame({ID: study_ids}).to_csv(
            OUTPUT_ROOT / "p2_coatnet_feature_studies.csv", index=False
        )
        for shard_rank in range(world_size):
            for prefix in ("feature", "mask", "position", "index"):
                (OUTPUT_ROOT / f"p2_{prefix}_shard_rank_{shard_rank}.npy").unlink()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if not is_primary:
        return 0

    feature_cache = np.load(feature_path, mmap_mode="r")
    feature_mask = np.load(mask_path).astype(bool)
    feature_positions = np.load(position_path)
    direct_result: dict[str, float] | None = None
    fold_result: dict[str, float] | None = None
    if FULL_RUN:
        feature_tensor_gold = torch.from_numpy(
            np.asarray(feature_cache[gold_indices]).astype(np.float32)
        ).to(device)
        mask_tensor_gold = torch.from_numpy(feature_mask[gold_indices]).to(device)
        model.head.eval()
        with torch.inference_mode():
            direct_predictions = torch.sigmoid(
                model.head(feature_tensor_gold, mask_tensor_gold)
            ).cpu().numpy()
        direct_metrics, direct_macro = calculate_metrics(
            y[gold_indices], direct_predictions
        )
        direct_metrics.to_csv(
            OUTPUT_ROOT / "p2_direct_expert_metrics.csv", index=False
        )
        direct_frame = pd.DataFrame({ID: np.asarray(study_ids)[gold_indices]})
        for i, target in enumerate(TARGETS):
            direct_frame[f"label__{target}"] = y[gold_indices, i]
            direct_frame[f"pred__{target}"] = direct_predictions[:, i]
        direct_frame.to_csv(
            OUTPUT_ROOT / "p2_direct_expert_predictions.csv", index=False
        )
        direct_result = {"macro_auc": direct_macro}
        print({"P2_DIRECT_EXPERT_MACRO_AUC": direct_macro}, flush=True)

        features_tensor = torch.from_numpy(
            np.asarray(feature_cache).astype(np.float32)
        ).to(device)
        masks_tensor = torch.from_numpy(feature_mask).to(device)
        labels_tensor = torch.from_numpy(y).to(device)
        weights_tensor = torch.from_numpy(weights).to(device)
        initial_head_state = {
            key: value.detach().cpu().clone()
            for key, value in model.head.state_dict().items()
        }
        oof = np.full((len(study_ids), N_TARGETS), np.nan, dtype=np.float32)
        fold_rows: list[dict[str, object]] = []
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        gold_fold = np.full(len(gold_indices), -1, dtype=int)
        for fold, (gold_train_pos, gold_valid_pos) in enumerate(
            splitter.split(gold_indices)
        ):
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
            train_index = np.concatenate(
                [keep_pseudo, gold_indices[gold_train_pos]]
            )
            assert not (set(report_hashes[train_index]) & valid_hashes)
            gold_fold[gold_valid_pos] = fold
            fold_seed = FOLD_SEED + fold
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            head = TargetAttentionHead().to(device)
            head.load_state_dict(initial_head_state, strict=True)
            head_optimizer = torch.optim.AdamW(
                head.parameters(),
                lr=HEAD_FOLD_LR,
                weight_decay=HEAD_FOLD_WEIGHT_DECAY,
                fused=device.type == "cuda",
            )
            generator = torch.Generator().manual_seed(fold_seed)
            train_cpu = torch.as_tensor(train_index, dtype=torch.long)
            final_loss = float("nan")
            for _ in range(head_epochs):
                head.train()
                permutation = train_cpu[
                    torch.randperm(len(train_cpu), generator=generator)
                ]
                losses: list[float] = []
                for start in range(0, len(permutation), 256):
                    index = permutation[start:start + 256].to(device)
                    head_optimizer.zero_grad(set_to_none=True)
                    logits = head(features_tensor[index], masks_tensor[index])
                    loss = weighted_bce(
                        logits, labels_tensor[index], weights_tensor[index]
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                    head_optimizer.step()
                    losses.append(float(loss.detach()))
                final_loss = float(np.mean(losses))
            head.eval()
            with torch.inference_mode():
                oof[valid_index] = torch.sigmoid(
                    head(features_tensor[valid_index], masks_tensor[valid_index])
                ).cpu().numpy()
            torch.save(
                head.state_dict(), OUTPUT_ROOT / f"p2_fold_{fold}_head.pt"
            )
            excluded = len(pseudo_indices) - len(keep_pseudo)
            fold_rows.append({
                "fold": fold,
                "train_rows": len(train_index),
                "pseudo_rows": len(keep_pseudo),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": excluded,
                "report_hash_overlap": 0,
                "final_loss": final_loss,
            })
            print({"P2_HEAD_FOLD_COMPLETE": fold_rows[-1]}, flush=True)
        assert (gold_fold >= 0).all() and np.isfinite(oof[gold_indices]).all()
        oof_metrics, oof_macro = calculate_metrics(
            y[gold_indices], oof[gold_indices]
        )
        oof_metrics.to_csv(OUTPUT_ROOT / "p2_oof_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(
            OUTPUT_ROOT / "p2_fold_summary.csv", index=False
        )
        oof_frame = pd.DataFrame({
            ID: np.asarray(study_ids)[gold_indices],
            "fold": gold_fold,
        })
        for i, target in enumerate(TARGETS):
            oof_frame[f"label__{target}"] = y[gold_indices, i]
            oof_frame[f"pred__{target}"] = oof[gold_indices, i]
        oof_frame.to_csv(
            OUTPUT_ROOT / "p2_oof_predictions.csv", index=False
        )
        fold_result = {"macro_auc": oof_macro}
        print({"P2_OOF_MACRO_AUC": oof_macro}, flush=True)
    else:
        (OUTPUT_ROOT / "p2_local_smoke.json").write_text(json.dumps({
            "study_uid": study_ids[0],
            "features_shape": list(feature_cache.shape),
            "training_backward": True,
        }, indent=2) + "\n")

    if telemetry is not None:
        telemetry.close()
    artifacts = [
        "p2_coatnet_pseudo_best.pt",
        "p2_representation_history.csv",
        "p2_coatnet_features.npy",
        "p2_coatnet_feature_mask.npy",
        "p2_coatnet_feature_positions.npy",
        "p2_coatnet_feature_studies.csv",
    ]
    if FULL_RUN:
        artifacts.extend([
            "p2_direct_expert_metrics.csv",
            "p2_direct_expert_predictions.csv",
            "p2_oof_metrics.csv",
            "p2_oof_predictions.csv",
            "p2_fold_summary.csv",
            *[f"p2_fold_{fold}_head.pt" for fold in range(N_FOLDS)],
            "p2_gpu_telemetry.csv",
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
        "k_train": k_train,
        "k_eval": k_eval,
        "representation_epochs": rep_epochs,
        "execution_revision": EXECUTION_REVISION,
        "visible_gpus": torch.cuda.device_count(),
        "data_parallel": distributed,
        "data_parallel_backend": "DistributedDataParallel" if distributed else None,
        "distributed_backend": "nccl" if distributed else None,
        "world_size": world_size,
        "study_batch": GLOBAL_STUDY_BATCH if FULL_RUN else 1,
        "gradient_accumulation": grad_accum,
        "eval_study_batch": GLOBAL_EVAL_STUDY_BATCH if FULL_RUN else 1,
        "loader_workers_per_rank": workers,
        "gradient_checkpointing": False,
        "channels_last": channels_last,
        "cudnn_benchmark": device.type == "cuda",
        "fused_adamw": device.type == "cuda",
        "best_epoch": best_epoch,
        "best_pseudo_valid_weighted_bce": best_loss,
        "direct_expert": direct_result,
        "fold_oof": fold_result,
        "representation_gold_rows": 0 if FULL_RUN else None,
        "pseudo_report_group_overlap": 0 if FULL_RUN else None,
        "artifacts_sha256": {
            name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
        },
        "elapsed": {
            "representation_minutes": (extract_started - training_started) / 60.0,
            "feature_extraction_minutes": (time.time() - extract_started) / 60.0,
        },
        "run_context": {
            "device": str(device),
            "model_path": str(model_path),
            "cache_a": str(part_a["root"]),
            "cache_b": str(part_b["root"]) if part_b else None,
        },
    }
    manifest_path = OUTPUT_ROOT / "p2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print({
        "P2_COMPLETE": True,
        "best_epoch": best_epoch,
        "direct_macro": None if direct_result is None else direct_result["macro_auc"],
        "oof_macro": None if fold_result is None else fold_result["macro_auc"],
        "manifest_sha256": sha256_file(manifest_path),
    }, flush=True)
    return 0


def main() -> int:
    if FULL_RUN and "--ddp-worker" not in sys.argv:
        return launch_ddp()
    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
