#!/usr/bin/env python3
"""Run frozen P3 DINOv2-small hierarchical MIL with two-process Kaggle DDP."""

# ruff: noqa: E402 - offline dependency paths must precede ML imports.

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import math
import os
os.environ["P3_SPEED_SMOKE"] = "1"
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
SLOT_NAMES = [
    "sagittal_fluid",
    "sagittal_structural",
    "coronal_fluid",
    "coronal_structural",
    "axial_any",
]
RUN_ID = "P3-DINOV2-HMIL-v1"
CACHE_ID = "P2-CACHE-v1"
EXPECTED_CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
MODEL_ID = "metaresearch/dinov2/PyTorch/small/1"
MODEL_SHA256 = "1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050"
CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
P3_SEED = 20260903
FOLD_SEED = 20260831
N_FOLDS = 5
N_TARGETS = len(TARGETS)
N_SLOTS = len(SLOT_NAMES)
CACHE_SIZE = 336
PATCH_SIZE = 14
PATCH_GRID = CACHE_SIZE // PATCH_SIZE
BACKBONE_DIM = 384
REGION_GRID = 2
N_REGIONS = 1 + REGION_GRID * REGION_GRID
HEAD_DIM = 256
K_TRAIN = 8
K_EVAL = 32
REP_EPOCHS = 8
HEAD_EPOCHS = 30
GLOBAL_STUDY_BATCH = 2
GRAD_ACCUM = 4
EVAL_BATCH_PER_RANK = 1
WORKERS_PER_RANK = 2
BACKBONE_LR = 1e-5
HEAD_LR = 5e-4
REP_WEIGHT_DECAY = 0.02
HEAD_FOLD_LR = 2e-4
HEAD_FOLD_WEIGHT_DECAY = 1e-3
MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)
UNFROZEN_BLOCKS = 2
EXECUTION_REVISION = "p3_ddp_t4_v1"

ON_KAGGLE = Path("/kaggle/working").is_dir()
FULL_RUN = ON_KAGGLE and torch.cuda.is_available()
SPEED_SMOKE = os.environ.get("P3_SPEED_SMOKE") == "1"
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/p3_dinov2_hmil_smoke")


def launch_ddp() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P3 requires exactly two Kaggle GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    print({"P3_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P3_DDP_PROCESS_COMPLETE": True}, flush=True)
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
    return roots


def find_named_file(filename: str, minimum_bytes: int = 1) -> Path:
    kaggle_direct = {
        "train.csv": Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"),
        "training_targets.csv": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-report-weak-supervision/training_targets.csv"
        ),
        "p2_cache_manifest_part_a.json": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p2-wide-cache-a/p2_cache_manifest_part_a.json"
        ),
        "p2_cache_manifest_part_b.json": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p2-wide-cache-b/p2_cache_manifest_part_b.json"
        ),
        "p2_oof_predictions.csv": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p2-coatnet-rmlp2/p2_oof_predictions.csv"
        ),
    }
    direct = [kaggle_direct[filename]] if filename in kaggle_direct else []
    for root in candidate_roots():
        direct.extend([
            root / filename,
            root / "data/raw" / filename,
            root / "outputs/kaggle/rsna-report-weak-supervision" / filename,
            root / "outputs/kaggle/rsna-p2-coatnet-rmlp2-v2" / filename,
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


def find_dino_root() -> Path:
    candidates = [
        Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1"),
        Path("/kaggle/input/dinov2/pytorch/small/1"),
    ]
    for root in candidate_roots():
        candidates.append(root / "data/sample/models/dinov2-small")
    for path in candidates:
        if (path / "config.json").is_file() and (path / "pytorch_model.bin").is_file():
            return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for config in kaggle_input.glob("**/config.json"):
            if (config.parent / "pytorch_model.bin").is_file():
                return config.parent.resolve()
    raise FileNotFoundError("official DINOv2-small model root")


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


def validate_immutable_inputs(model_root: Path, parts: list[dict[str, object]]) -> None:
    started = time.time()
    assert sha256_file(model_root / "pytorch_model.bin") == MODEL_SHA256
    assert sha256_file(model_root / "config.json") == CONFIG_SHA256
    jobs: list[tuple[Path, str]] = []
    for part in parts:
        manifest = part["manifest"]
        assert isinstance(manifest, dict)
        for name, expected in manifest["files_sha256"].items():
            jobs.append((part["root"] / name, str(expected)))  # type: ignore[operator]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(sha256_file, path): (path, expected) for path, expected in jobs}
        for future in as_completed(futures):
            path, expected = futures[future]
            assert future.result() == expected, f"{path.name} hash mismatch"
            print({"P3_INPUT_HASH_OK": path.name}, flush=True)
    print({"P3_INPUT_VALIDATION_MINUTES": (time.time() - started) / 60.0}, flush=True)


def normalize_report(report: object) -> str:
    text = unicodedata.normalize("NFKC", str(report)).casefold()
    return "".join(character for character in text if character.isalnum())


def report_hash(report: object) -> str:
    normalized = normalize_report(report)
    if not normalized:
        raise ValueError("empty report after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


class GpuTelemetry:
    def __init__(self, output_path: Path, interval_seconds: int) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
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
                print({"P3_GPU": rows}, flush=True)
            except Exception as error:
                print({"P3_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)
            self.stop.wait(self.interval_seconds)

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


class HierarchicalDiagnosisHead(nn.Module):
    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(BACKBONE_DIM)
        self.projection = nn.Sequential(
            nn.Linear(BACKBONE_DIM, HEAD_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.region_embedding = nn.Parameter(torch.randn(N_REGIONS, HEAD_DIM) * 0.02)
        self.slot_embedding = nn.Parameter(torch.randn(N_SLOTS, HEAD_DIM) * 0.02)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, HEAD_DIM),
            nn.GELU(),
            nn.Linear(HEAD_DIM, HEAD_DIM),
        )
        self.region_query = nn.Parameter(torch.randn(N_TARGETS, HEAD_DIM) * 0.02)
        self.window_query = nn.Parameter(torch.randn(N_TARGETS, HEAD_DIM) * 0.02)
        self.target_weight = nn.Parameter(torch.randn(N_TARGETS, HEAD_DIM) * 0.02)
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        slot_ids: torch.Tensor,
        relative_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.projection(self.input_norm(features))
        hidden = hidden + self.region_embedding[None, None]
        region_scores = torch.einsum("bkrh,th->bkrt", hidden, self.region_query)
        region_scores = region_scores / math.sqrt(HEAD_DIM)
        region_attention = region_scores.softmax(dim=2)
        local = torch.einsum("bkrt,bkrh->bkth", region_attention, hidden)
        context = self.slot_embedding[slot_ids.clamp(0, N_SLOTS - 1)]
        context = context + self.position_mlp(relative_positions[..., None])
        local = local + context[:, :, None, :]
        window_scores = torch.einsum("bkth,th->bkt", torch.tanh(local), self.window_query)
        window_scores = window_scores / math.sqrt(HEAD_DIM)
        window_scores = window_scores.masked_fill(~mask[:, :, None].bool(), -1e4)
        window_attention = window_scores.softmax(dim=1)
        pooled = torch.einsum("bkt,bkth->bth", window_attention, local)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


class DinoHierarchicalMIL(nn.Module):
    def __init__(self, model_root: Path) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(str(model_root), local_files_only=True)
        assert self.backbone.config.hidden_size == BACKBONE_DIM
        assert self.backbone.config.patch_size == PATCH_SIZE
        assert self.backbone.config.num_hidden_layers == 12
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in self.backbone.encoder.layer[-UNFROZEN_BLOCKS:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = True
        self.head = HierarchicalDiagnosisHead()
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1))

    def encode(self, windows: torch.Tensor, *, augment: bool = False) -> torch.Tensor:
        batch, count = windows.shape[:2]
        pixels = windows.reshape(batch * count, 3, CACHE_SIZE, CACHE_SIZE).float().div(255.0)
        if augment:
            gain = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(0.90, 1.10)
            offset = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(-0.05, 0.05)
            pixels = (pixels * gain + offset).clamp(0.0, 1.0)
        pixels = (pixels - self.mean) / self.std
        sequence = self.backbone(pixel_values=pixels).last_hidden_state
        assert sequence.shape[1:] == (1 + PATCH_GRID * PATCH_GRID, BACKBONE_DIM)
        cls = sequence[:, :1]
        grid = sequence[:, 1:].transpose(1, 2).reshape(-1, BACKBONE_DIM, PATCH_GRID, PATCH_GRID)
        regions = F.adaptive_avg_pool2d(grid, (REGION_GRID, REGION_GRID))
        regions = regions.flatten(2).transpose(1, 2)
        return torch.cat([cls, regions], dim=1).reshape(
            batch, count, N_REGIONS, BACKBONE_DIM
        )

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        slot_ids: torch.Tensor,
        relative_positions: torch.Tensor,
        *,
        augment: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        features = self.encode(windows, augment=augment)
        if return_features:
            return features
        return self.head(features, mask, slot_ids, relative_positions)


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
    seed: int = FOLD_SEED + 300_000,
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


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack([
        pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
        for index in range(predictions.shape[1])
    ]).astype(np.float32)


def worker_main() -> int:
    distributed = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if FULL_RUN:
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
    telemetry = (
        GpuTelemetry(
            OUTPUT_ROOT / "p3_gpu_telemetry.csv",
            interval_seconds=5 if SPEED_SMOKE else 60,
        )
        if FULL_RUN and is_primary
        else None
    )
    if telemetry is not None:
        telemetry.start()

    random.seed(P3_SEED)
    np.random.seed(P3_SEED)
    torch.manual_seed(P3_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(P3_SEED)
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
    model_root = find_dino_root()
    part_a = prepare_cache_part("a")
    part_b = prepare_cache_part("b") if FULL_RUN else None
    parts = [part_a, *([part_b] if part_b is not None else [])]
    if is_primary:
        if SPEED_SMOKE:
            assert sha256_file(model_root / "pytorch_model.bin") == MODEL_SHA256
            assert sha256_file(model_root / "config.json") == CONFIG_SHA256
            print({"P3_SPEED_SMOKE_INPUTS": "manifests_and_model_validated"}, flush=True)
        else:
            validate_immutable_inputs(model_root, parts)
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
    dropped_for_ddp = np.asarray([], dtype=int)
    full_adapt_count = len(study_ids)
    full_valid_count = len(study_ids)
    if FULL_RUN:
        assert len(gold_indices) == 58 and len(pseudo_indices) == 4_349
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=P3_SEED)
        train_pos, valid_pos = next(splitter.split(pseudo_indices, groups=report_hashes[pseudo_indices]))
        adapt_indices = pseudo_indices[train_pos]
        pseudo_valid_indices = pseudo_indices[valid_pos]
        assert not (set(report_hashes[adapt_indices]) & set(report_hashes[pseudo_valid_indices]))
        full_adapt_count = len(adapt_indices)
        full_valid_count = len(pseudo_valid_indices)
        remainder = len(adapt_indices) % world_size
        if remainder:
            dropped_for_ddp = adapt_indices[-remainder:]
            adapt_indices = adapt_indices[:-remainder]
        if SPEED_SMOKE:
            adapt_indices = adapt_indices[:64]
            pseudo_valid_indices = pseudo_valid_indices[:8]
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

    def center_geometry(centers: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slots = np.zeros(len(centers), dtype=np.int64)
        relative = np.zeros(len(centers), dtype=np.float32)
        for position, center in enumerate(centers):
            if not valid[position]:
                continue
            for slot, (start, stop) in enumerate(slot_bounds):
                if start <= center < stop:
                    slots[position] = slot
                    relative[position] = 2.0 * (float(center - start) / float(stop - start - 1)) - 1.0
                    break
            else:
                raise AssertionError(f"center outside slot bounds: {center}")
        return slots, relative

    def build_windows(
        index: int, count: int, stochastic: bool, epoch: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        volume, mask = read_study(index)
        centers, window_mask = choose_centers(
            mask,
            count,
            stochastic=stochastic,
            seed=P3_SEED + 10_007 * epoch + index,
        )
        windows = np.zeros((count, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        for out_index, center in enumerate(centers):
            if window_mask[out_index]:
                windows[out_index] = volume[int(center) - 1:int(center) + 2]
        slots, relative = center_geometry(centers, window_mask)
        return windows, window_mask, slots, relative

    class StudyDataset(Dataset):
        def __init__(self, indices: np.ndarray, count: int, stochastic: bool, epoch: int = 0) -> None:
            self.indices = np.asarray(indices, dtype=int)
            self.count = count
            self.stochastic = stochastic
            self.epoch = epoch

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, position: int):
            index = int(self.indices[position])
            windows, window_mask, slots, relative = build_windows(
                index, self.count, self.stochastic, self.epoch
            )
            return index, windows, window_mask, slots, relative, y[index], weights[index]

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
        index, windows, window_mask, slots, relative, labels, label_weights = batch
        return (
            index.numpy(),
            windows.to(device, non_blocking=True),
            window_mask.to(device, non_blocking=True).bool(),
            slots.to(device, non_blocking=True).long(),
            relative.to(device, non_blocking=True).float(),
            labels.to(device, non_blocking=True),
            label_weights.to(device, non_blocking=True),
        )

    if is_primary:
        print({
            "P3_START": True,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "full_run": FULL_RUN,
            "speed_smoke": SPEED_SMOKE,
            "visible_gpus": torch.cuda.device_count(),
            "world_size": world_size,
            "execution_revision": EXECUTION_REVISION,
            "studies": len(study_ids),
            "pseudo_train": len(adapt_indices),
            "pseudo_valid": len(pseudo_valid_indices),
            "gold": len(gold_indices),
            "k_train": k_train,
            "k_eval": k_eval,
            "region_tokens": N_REGIONS,
            "dropped_for_ddp": dropped_for_ddp.tolist(),
        }, flush=True)

    model = DinoHierarchicalMIL(model_root).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert 22_000_000 < total_parameters < 24_000_000
    assert 3_000_000 < trainable_parameters < 6_000_000
    train_model: nn.Module
    if distributed:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            static_graph=False,
        )
        print({
            "P3_DDP_READY": True,
            "rank": rank,
            "device": str(device),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        }, flush=True)
    else:
        train_model = model

    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": BACKBONE_LR},
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
        loader = make_loader(
            StudyDataset(local_indices, k_eval, False),
            batch_size=EVAL_BATCH_PER_RANK,
        )
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in loader:
            _, windows, window_mask, slots, relative, labels, label_weights = move_batch(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(windows, window_mask, slots, relative, augment=False)
                numerator, denominator = weighted_terms(logits, labels, label_weights)
            totals[0] += numerator.double()
            totals[1] += denominator.double()
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return float((totals[0] / totals[1].clamp_min(1.0)).item())

    training_started = time.time()
    for epoch in range(rep_epochs):
        epoch_started = time.time()
        generator = torch.Generator().manual_seed(P3_SEED + epoch)
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
                _, windows, window_mask, slots, relative, labels, label_weights = move_batch(batch)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = train_model(
                        windows,
                        window_mask,
                        slots,
                        relative,
                        augment=True,
                    )
                    local_num, local_den = weighted_terms(logits, labels, label_weights)
                    global_den = local_den.detach().clone()
                    if distributed:
                        dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
                    loss = local_num * world_size / global_den.clamp_min(1.0)
                scaler.scale(loss / grad_accum).backward()
            epoch_totals += torch.stack([local_num.detach().double(), local_den.detach().double()])
            window_totals += torch.stack([local_num.detach().double(), local_den.detach().double()])
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    3.0,
                )
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
                        f"P3 train epoch {epoch + 1}/{rep_epochs} "
                        f"step {step + 1}/{len(loader)} "
                        f"loss={(logged[0] / logged[1].clamp_min(1.0)).item():.5f}",
                        flush=True,
                    )
                window_totals.zero_()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.time() - epoch_started
        global_epoch_totals = epoch_totals.clone()
        if distributed:
            dist.all_reduce(global_epoch_totals, op=dist.ReduceOp.SUM)
        train_loss = float(
            (global_epoch_totals[0] / global_epoch_totals[1].clamp_min(1.0)).item()
        )
        validation_started = time.time()
        valid_loss = validation_loss(pseudo_valid_indices)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds = time.time() - validation_started
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "pseudo_valid_weighted_bce": valid_loss,
            "elapsed_minutes": (time.time() - training_started) / 60.0,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
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
            print({"P3_EPOCH_COMPLETE": row, "best_epoch": best_epoch}, flush=True)

    if SPEED_SMOKE:
        memory_mib = torch.tensor(
            [torch.cuda.max_memory_allocated(device) / (1024 ** 2)],
            dtype=torch.float64,
            device=device,
        )
        gathered_memory = [torch.zeros_like(memory_mib) for _ in range(world_size)]
        if distributed:
            dist.all_gather(gathered_memory, memory_mib)
        if is_primary:
            projected_training = history[-1]["training_seconds"] * full_adapt_count / len(adapt_indices)
            projected_validation = history[-1]["validation_seconds"] * full_valid_count / len(pseudo_valid_indices)
            projected_feature = history[-1]["validation_seconds"] * len(study_ids) / len(pseudo_valid_indices)
            result = {
                "run_id": RUN_ID,
                "execution_revision": EXECUTION_REVISION,
                "world_size": world_size,
                "global_study_batch": GLOBAL_STUDY_BATCH,
                "gradient_accumulation": grad_accum,
                "smoke_train_studies": len(adapt_indices),
                "smoke_valid_studies": len(pseudo_valid_indices),
                "training_seconds": history[-1]["training_seconds"],
                "validation_seconds": history[-1]["validation_seconds"],
                "seconds_per_global_study": history[-1]["training_seconds"] / len(adapt_indices),
                "pseudo_valid_weighted_bce": history[-1]["pseudo_valid_weighted_bce"],
                "peak_memory_mib_by_rank": [float(value.item()) for value in gathered_memory],
                "projected_full_minutes": (
                    REP_EPOCHS * (projected_training + projected_validation) + projected_feature
                ) / 60.0,
                "k_train": k_train,
                "k_eval": k_eval,
                "region_tokens": N_REGIONS,
                "trainable_parameters": trainable_parameters,
            }
            (OUTPUT_ROOT / "p3_speed_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
            print({"P3_SPEED_SMOKE_COMPLETE": result}, flush=True)
            if telemetry is not None:
                telemetry.close()
        if distributed:
            dist.destroy_process_group()
        return 0

    checkpoint_path = OUTPUT_ROOT / "p3_dinov2_pseudo_best.pt"
    if is_primary:
        assert best_state is not None and best_epoch >= 1
        torch.save(best_state, checkpoint_path)
        pd.DataFrame(history).to_csv(OUTPUT_ROOT / "p3_representation_history.csv", index=False)
        del best_state
    if distributed:
        dist.barrier()
    selected_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state, strict=True)
    del selected_state
    gc.collect()
    if is_primary:
        print({"P3_REPRESENTATION_COMPLETE": True, "best_epoch": best_epoch, "best_loss": best_loss}, flush=True)

    extract_started = time.time()
    local_feature_indices = np.arange(rank, len(study_ids), world_size, dtype=np.int64)
    feature_shard_path = OUTPUT_ROOT / f"p3_feature_shard_rank_{rank}.npy"
    mask_shard_path = OUTPUT_ROOT / f"p3_mask_shard_rank_{rank}.npy"
    slot_shard_path = OUTPUT_ROOT / f"p3_slot_shard_rank_{rank}.npy"
    position_shard_path = OUTPUT_ROOT / f"p3_position_shard_rank_{rank}.npy"
    index_shard_path = OUTPUT_ROOT / f"p3_index_shard_rank_{rank}.npy"
    feature_shard = np.lib.format.open_memmap(
        feature_shard_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(local_feature_indices), k_eval, N_REGIONS, BACKBONE_DIM),
    )
    mask_shard = np.zeros((len(local_feature_indices), k_eval), dtype=np.uint8)
    slot_shard = np.full((len(local_feature_indices), k_eval), -1, dtype=np.int8)
    position_shard = np.zeros((len(local_feature_indices), k_eval), dtype=np.float16)
    feature_loader = make_loader(
        StudyDataset(local_feature_indices, k_eval, False),
        batch_size=EVAL_BATCH_PER_RANK,
    )
    model.eval()
    with torch.inference_mode():
        processed = 0
        for batch in feature_loader:
            indices, windows, window_mask, slots, relative, _, _ = move_batch(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                encoded = model(
                    windows,
                    window_mask,
                    slots,
                    relative,
                    augment=False,
                    return_features=True,
                )
            stop = processed + len(indices)
            feature_shard[processed:stop] = encoded.float().cpu().numpy().astype(np.float16)
            selected_masks = window_mask.cpu().numpy().astype(bool)
            mask_shard[processed:stop] = selected_masks.astype(np.uint8)
            selected_slots = slots.cpu().numpy().astype(np.int8)
            selected_slots[~selected_masks] = -1
            slot_shard[processed:stop] = selected_slots
            position_shard[processed:stop] = relative.cpu().numpy().astype(np.float16)
            processed = stop
            if is_primary and (processed % 50 == 0 or processed == len(local_feature_indices)):
                print(
                    f"P3 features ~{min(processed * world_size, len(study_ids)):,}/"
                    f"{len(study_ids):,} in {(time.time() - extract_started) / 60:.1f} min",
                    flush=True,
                )
    feature_shard.flush()
    np.save(mask_shard_path, mask_shard)
    np.save(slot_shard_path, slot_shard)
    np.save(position_shard_path, position_shard)
    np.save(index_shard_path, local_feature_indices)
    if distributed:
        dist.barrier()

    feature_path = OUTPUT_ROOT / "p3_dinov2_region_features.npy"
    mask_path = OUTPUT_ROOT / "p3_feature_mask.npy"
    slot_path = OUTPUT_ROOT / "p3_feature_slots.npy"
    position_path = OUTPUT_ROOT / "p3_feature_relative_positions.npy"
    if is_primary:
        feature_cache = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(study_ids), k_eval, N_REGIONS, BACKBONE_DIM),
        )
        feature_mask = np.zeros((len(study_ids), k_eval), dtype=np.uint8)
        feature_slots = np.full((len(study_ids), k_eval), -1, dtype=np.int8)
        feature_positions = np.zeros((len(study_ids), k_eval), dtype=np.float16)
        for shard_rank in range(world_size):
            shard_indices = np.load(OUTPUT_ROOT / f"p3_index_shard_rank_{shard_rank}.npy")
            feature_cache[shard_indices] = np.load(
                OUTPUT_ROOT / f"p3_feature_shard_rank_{shard_rank}.npy", mmap_mode="r"
            )
            feature_mask[shard_indices] = np.load(OUTPUT_ROOT / f"p3_mask_shard_rank_{shard_rank}.npy")
            feature_slots[shard_indices] = np.load(OUTPUT_ROOT / f"p3_slot_shard_rank_{shard_rank}.npy")
            feature_positions[shard_indices] = np.load(
                OUTPUT_ROOT / f"p3_position_shard_rank_{shard_rank}.npy"
            )
        feature_cache.flush()
        assert np.isfinite(feature_cache).all() and feature_mask.any(axis=1).all()
        assert (feature_slots[feature_mask.astype(bool)] >= 0).all()
        np.save(mask_path, feature_mask)
        np.save(slot_path, feature_slots)
        np.save(position_path, feature_positions)
        pd.DataFrame({ID: study_ids}).to_csv(OUTPUT_ROOT / "p3_feature_studies.csv", index=False)
        for shard_rank in range(world_size):
            for prefix in ("feature", "mask", "slot", "position", "index"):
                (OUTPUT_ROOT / f"p3_{prefix}_shard_rank_{shard_rank}.npy").unlink()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if not is_primary:
        return 0

    feature_cache = np.load(feature_path, mmap_mode="r")
    feature_mask = np.load(mask_path).astype(bool)
    feature_slots = np.load(slot_path).astype(np.int64)
    feature_positions = np.load(position_path).astype(np.float32)
    direct_result: dict[str, float] | None = None
    fold_result: dict[str, object] | None = None
    comparison_result: dict[str, object] | None = None
    if FULL_RUN:
        features_tensor = torch.from_numpy(np.asarray(feature_cache).astype(np.float32)).to(device)
        masks_tensor = torch.from_numpy(feature_mask).to(device)
        slots_tensor = torch.from_numpy(feature_slots).to(device)
        positions_tensor = torch.from_numpy(feature_positions).to(device)
        labels_tensor = torch.from_numpy(y).to(device)
        weights_tensor = torch.from_numpy(weights).to(device)

        model.head.eval()
        with torch.inference_mode():
            direct_predictions = torch.sigmoid(
                model.head(
                    features_tensor[gold_indices],
                    masks_tensor[gold_indices],
                    slots_tensor[gold_indices],
                    positions_tensor[gold_indices],
                )
            ).cpu().numpy()
        direct_metrics, direct_macro = calculate_metrics(y[gold_indices], direct_predictions)
        direct_metrics.to_csv(OUTPUT_ROOT / "p3_direct_expert_metrics.csv", index=False)
        direct_frame = pd.DataFrame({ID: np.asarray(study_ids)[gold_indices]})
        for i, target in enumerate(TARGETS):
            direct_frame[f"label__{target}"] = y[gold_indices, i]
            direct_frame[f"pred__{target}"] = direct_predictions[:, i]
        direct_frame.to_csv(OUTPUT_ROOT / "p3_direct_expert_predictions.csv", index=False)
        direct_result = {"macro_auc": direct_macro}
        print({"P3_DIRECT_EXPERT_MACRO_AUC": direct_macro}, flush=True)

        initial_head_state = {
            key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()
        }
        oof = np.full((len(study_ids), N_TARGETS), np.nan, dtype=np.float32)
        fold_rows: list[dict[str, object]] = []
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        gold_fold = np.full(len(gold_indices), -1, dtype=int)
        for fold, (gold_train_pos, gold_valid_pos) in enumerate(splitter.split(gold_indices)):
            valid_index = gold_indices[gold_valid_pos]
            valid_hashes = set(report_hashes[valid_index])
            keep_pseudo = np.asarray(
                [index for index in pseudo_indices if report_hashes[index] not in valid_hashes],
                dtype=int,
            )
            train_index = np.concatenate([keep_pseudo, gold_indices[gold_train_pos]])
            assert not (set(report_hashes[train_index]) & valid_hashes)
            gold_fold[gold_valid_pos] = fold
            fold_seed = FOLD_SEED + fold
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            head = HierarchicalDiagnosisHead().to(device)
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
                permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
                losses: list[float] = []
                for start in range(0, len(permutation), 256):
                    index = permutation[start:start + 256].to(device)
                    head_optimizer.zero_grad(set_to_none=True)
                    logits = head(
                        features_tensor[index],
                        masks_tensor[index],
                        slots_tensor[index],
                        positions_tensor[index],
                    )
                    loss = weighted_bce(logits, labels_tensor[index], weights_tensor[index])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                    head_optimizer.step()
                    losses.append(float(loss.detach()))
                final_loss = float(np.mean(losses))
            head.eval()
            with torch.inference_mode():
                oof[valid_index] = torch.sigmoid(
                    head(
                        features_tensor[valid_index],
                        masks_tensor[valid_index],
                        slots_tensor[valid_index],
                        positions_tensor[valid_index],
                    )
                ).cpu().numpy()
            torch.save(head.state_dict(), OUTPUT_ROOT / f"p3_fold_{fold}_head.pt")
            fold_rows.append({
                "fold": fold,
                "train_rows": len(train_index),
                "pseudo_rows": len(keep_pseudo),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": len(pseudo_indices) - len(keep_pseudo),
                "report_hash_overlap": 0,
                "final_loss": final_loss,
            })
            print({"P3_HEAD_FOLD_COMPLETE": fold_rows[-1]}, flush=True)

        assert (gold_fold >= 0).all() and np.isfinite(oof[gold_indices]).all()
        oof_metrics, oof_macro = calculate_metrics(y[gold_indices], oof[gold_indices])
        oof_metrics.to_csv(OUTPUT_ROOT / "p3_oof_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "p3_fold_summary.csv", index=False)
        oof_frame = pd.DataFrame({ID: np.asarray(study_ids)[gold_indices], "fold": gold_fold})
        for i, target in enumerate(TARGETS):
            oof_frame[f"label__{target}"] = y[gold_indices, i]
            oof_frame[f"pred__{target}"] = oof[gold_indices, i]
        oof_frame.to_csv(OUTPUT_ROOT / "p3_oof_predictions.csv", index=False)
        fold_result = {"macro_auc": oof_macro}
        print({"P3_OOF_MACRO_AUC": oof_macro}, flush=True)

        p2 = pd.read_csv(find_named_file("p2_oof_predictions.csv", 10_000), dtype={ID: str})
        p2 = p2.set_index(ID).loc[oof_frame[ID]].reset_index()
        p2_labels = p2[[f"label__{target}" for target in TARGETS]].to_numpy(np.float32)
        p2_predictions = p2[[f"pred__{target}" for target in TARGETS]].to_numpy(np.float32)
        labels_gold = y[gold_indices]
        p3_predictions = oof[gold_indices]
        assert np.array_equal(p2_labels, labels_gold)
        p2_metrics, p2_macro = calculate_metrics(labels_gold, p2_predictions)
        equal_rank_predictions = (rank_columns(p2_predictions) + rank_columns(p3_predictions)) / 2.0
        rank_metrics, rank_macro = calculate_metrics(labels_gold, equal_rank_predictions)
        rank_metrics.to_csv(OUTPUT_ROOT / "p3_p2_equal_rank_metrics.csv", index=False)
        comparison = pd.DataFrame({"target": [*TARGETS, "MACRO"]})
        comparison["p2_auc"] = p2_metrics["auc"]
        comparison["p3_auc"] = oof_metrics["auc"]
        comparison["equal_rank_auc"] = rank_metrics["auc"]
        comparison["p3_minus_p2"] = comparison["p3_auc"] - comparison["p2_auc"]
        comparison["rank_minus_p2"] = comparison["equal_rank_auc"] - comparison["p2_auc"]
        comparison.to_csv(OUTPUT_ROOT / "p3_p2_target_comparison.csv", index=False)
        rank_frame = pd.DataFrame({ID: oof_frame[ID], "fold": gold_fold})
        for i, target in enumerate(TARGETS):
            rank_frame[f"label__{target}"] = labels_gold[:, i]
            rank_frame[f"pred__{target}"] = equal_rank_predictions[:, i]
        rank_frame.to_csv(OUTPUT_ROOT / "p3_p2_equal_rank_oof_predictions.csv", index=False)
        comparison_result = {
            "p2_macro_auc": p2_macro,
            "p3_macro_auc": oof_macro,
            "p3_minus_p2": oof_macro - p2_macro,
            "p3_vs_p2_bootstrap": paired_bootstrap(labels_gold, p2_predictions, p3_predictions),
            "equal_rank_macro_auc": rank_macro,
            "equal_rank_minus_p2": rank_macro - p2_macro,
            "equal_rank_vs_p2_bootstrap": paired_bootstrap(
                labels_gold, p2_predictions, equal_rank_predictions, seed=FOLD_SEED + 300_001
            ),
        }
        (OUTPUT_ROOT / "p3_p2_comparison.json").write_text(
            json.dumps(comparison_result, indent=2) + "\n"
        )
        print({"P3_P2_COMPARISON": comparison_result}, flush=True)
    else:
        (OUTPUT_ROOT / "p3_local_smoke.json").write_text(json.dumps({
            "study_uid": study_ids[0],
            "features_shape": list(feature_cache.shape),
            "training_backward": True,
            "model_sha256": MODEL_SHA256,
        }, indent=2) + "\n")

    if telemetry is not None:
        telemetry.close()
    artifacts = [
        "p3_dinov2_pseudo_best.pt",
        "p3_representation_history.csv",
        "p3_dinov2_region_features.npy",
        "p3_feature_mask.npy",
        "p3_feature_slots.npy",
        "p3_feature_relative_positions.npy",
        "p3_feature_studies.csv",
    ]
    if FULL_RUN:
        artifacts.extend([
            "p3_direct_expert_metrics.csv",
            "p3_direct_expert_predictions.csv",
            "p3_oof_metrics.csv",
            "p3_oof_predictions.csv",
            "p3_fold_summary.csv",
            "p3_p2_equal_rank_metrics.csv",
            "p3_p2_target_comparison.csv",
            "p3_p2_equal_rank_oof_predictions.csv",
            "p3_p2_comparison.json",
            *[f"p3_fold_{fold}_head.pt" for fold in range(N_FOLDS)],
            "p3_gpu_telemetry.csv",
        ])
    else:
        artifacts.append("p3_local_smoke.json")
    for name in artifacts:
        assert (OUTPUT_ROOT / name).is_file(), name

    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "full_run": FULL_RUN,
        "seeds": {"representation": P3_SEED, "folds": FOLD_SEED},
        "studies": len(study_ids),
        "pseudo_studies": int((~is_gold).sum()),
        "expert_studies": int(is_gold.sum()),
        "cache_id": CACHE_ID,
        "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "model_license": "Apache-2.0",
        "cache_size": CACHE_SIZE,
        "patch_size": PATCH_SIZE,
        "patch_grid": PATCH_GRID,
        "backbone_dim": BACKBONE_DIM,
        "region_grid": REGION_GRID,
        "region_tokens": N_REGIONS,
        "head_dim": HEAD_DIM,
        "unfrozen_blocks": UNFROZEN_BLOCKS,
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
        "loader_workers_per_rank": workers,
        "best_epoch": best_epoch,
        "best_pseudo_valid_weighted_bce": best_loss,
        "direct_expert": direct_result,
        "fold_oof": fold_result,
        "p2_comparison": comparison_result,
        "representation_gold_rows": 0 if FULL_RUN else None,
        "pseudo_report_group_overlap": 0 if FULL_RUN else None,
        "dropped_for_ddp": dropped_for_ddp.tolist(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "artifacts_sha256": {name: sha256_file(OUTPUT_ROOT / name) for name in artifacts},
        "elapsed": {
            "representation_minutes": (extract_started - training_started) / 60.0,
            "feature_head_evaluation_minutes": (time.time() - extract_started) / 60.0,
        },
        "run_context": {
            "device": str(device),
            "model_root": str(model_root),
            "cache_a": str(part_a["root"]),
            "cache_b": str(part_b["root"]) if part_b else None,
        },
    }
    manifest_path = OUTPUT_ROOT / "p3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print({
        "P3_COMPLETE": True,
        "best_epoch": best_epoch,
        "direct_macro": None if direct_result is None else direct_result["macro_auc"],
        "oof_macro": None if fold_result is None else fold_result["macro_auc"],
        "p2_comparison": comparison_result,
        "manifest_sha256": sha256_file(manifest_path),
    }, flush=True)
    return 0


def main() -> int:
    if FULL_RUN and "--ddp-worker" not in sys.argv:
        return launch_ddp()
    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
