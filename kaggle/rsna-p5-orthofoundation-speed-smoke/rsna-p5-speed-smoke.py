#!/usr/bin/env python3
"""Benchmark the frozen P5 OrthoFoundation feature path on two Kaggle T4s."""

# ruff: noqa: E402 - runtime paths and offline flags precede third-party imports.

from __future__ import annotations

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
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", "/kaggle/working/torch-cache")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
RUN_ID = "P5A-ORTHOFOUNDATION-FROZEN-v1"
EXECUTION_REVISION = "p5a_frozen_t4x2_smoke_v1"
CACHE_ID = "P2-CACHE-v1"
CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
MODEL_SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"
MODEL_BYTES = 1_213_056_638
ORTHO_COMMIT = "4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
SEED = 20260904
CACHE_SIZE = 336
MODEL_SIZE = 224
BACKBONE_DIM = 1024
N_REGIONS = 2
HEAD_DIM = 256
K_SLICES = 32
SLOT_QUOTAS = (9, 7, 6, 4, 6)
SMOKE_STUDIES = 128
STUDY_BATCH_PER_RANK = 2
WORKERS_PER_RANK = 2
HEAD_SMOKE_STEPS = 5
MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)
OUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(
    "outputs/p5_orthofoundation_speed_smoke"
)


def launch() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P5 speed smoke requires exactly two GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    print({"P5_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P5_DDP_PROCESS_COMPLETE": True}, flush=True)
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
    direct_map = {
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
    }
    candidates = [direct_map[filename]] if filename in direct_map else []
    for root in candidate_roots():
        candidates.extend(
            [
                root / filename,
                root / "data/raw" / filename,
                root / "outputs/kaggle/rsna-report-weak-supervision" / filename,
                root / "kaggle/rsna-p2-wide-cache-a/outputs/p2_cache_smoke_a" / filename,
            ]
        )
    for path in candidates:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for path in kaggle_input.glob(f"**/{filename}"):
            if path.is_file() and path.stat().st_size >= minimum_bytes:
                return path.resolve()
    raise FileNotFoundError(filename)


def find_runtime() -> tuple[Path, Path]:
    for root in candidate_roots():
        for checkpoint in root.glob("**/OrthoFoudation-L.pth"):
            runtime = checkpoint.parent
            if (runtime / "dinov3/hubconf.py").is_file():
                return runtime.resolve(), checkpoint.resolve()
    raise FileNotFoundError("OrthoFoundation-L checkpoint plus bundled DINOv3 runtime")


def cache_part(part: str) -> dict[str, object]:
    manifest_path = find_named_file(f"p2_cache_manifest_part_{part}.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_id"] == CACHE_ID
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    root = manifest_path.parent
    result: dict[str, object] = {"manifest": manifest, "root": root}
    for key, name in {
        "pixels": f"p2_pixels_part_{part}.npy",
        "mask": f"p2_valid_mask_part_{part}.npy",
        "studies": f"p2_studies_part_{part}.csv",
    }.items():
        path = root / name
        assert path.is_file(), path
        result[key] = path
    return result


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
                print({"P5_GPU": rows}, flush=True)
            except Exception as error:
                print({"P5_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)
            self.stop.wait(1)

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


class HierarchicalDiagnosisHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(BACKBONE_DIM)
        self.projection = nn.Sequential(
            nn.Linear(BACKBONE_DIM, HEAD_DIM),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.region_embedding = nn.Parameter(torch.randn(N_REGIONS, HEAD_DIM) * 0.02)
        self.slot_embedding = nn.Parameter(torch.randn(len(SLOT_QUOTAS), HEAD_DIM) * 0.02)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, HEAD_DIM), nn.GELU(), nn.Linear(HEAD_DIM, HEAD_DIM)
        )
        self.region_query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.slice_query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.target_weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.projection(self.input_norm(features.float()))
        hidden = hidden + self.region_embedding[None, None]
        region_scores = torch.einsum("bkrh,th->bkrt", hidden, self.region_query)
        region_attention = (region_scores / math.sqrt(HEAD_DIM)).softmax(dim=2)
        local = torch.einsum("bkrt,bkrh->bkth", region_attention, hidden)
        context = self.slot_embedding[slots.clamp(0, len(SLOT_QUOTAS) - 1)]
        context = context + self.position_mlp(positions[..., None])
        local = local + context[:, :, None]
        slice_scores = torch.einsum("bkth,th->bkt", torch.tanh(local), self.slice_query)
        slice_scores = (slice_scores / math.sqrt(HEAD_DIM)).masked_fill(
            ~mask[:, :, None], -1e4
        )
        slice_attention = slice_scores.softmax(dim=1)
        pooled = torch.einsum("bkt,bkth->bth", slice_attention, local)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


def worker() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    is_primary = rank == 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    telemetry = GpuTelemetry(OUTPUT_ROOT / "p5_speed_gpu_telemetry.csv") if is_primary else None
    if telemetry is not None:
        telemetry.start()

    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    runtime, checkpoint = find_runtime()
    part_a, part_b = cache_part("a"), cache_part("b")
    if is_primary:
        assert checkpoint.stat().st_size == MODEL_BYTES
        assert sha256_file(checkpoint) == MODEL_SHA256
        provenance = (runtime / "PROVENANCE.md").read_text()
        assert ORTHO_COMMIT in provenance and DINOV3_COMMIT in provenance
        assert (runtime / "ORTHOFOUNDATION_LICENSE").is_file()
        assert (runtime / "dinov3/LICENSE.md").is_file()
        print(
            {
                "P5_SPEED_INPUTS_OK": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": MODEL_SHA256,
                "runtime": str(runtime),
            },
            flush=True,
        )
    dist.barrier()

    train_path = find_named_file("train.csv", 100_000)
    target_path = find_named_file("training_targets.csv", 100_000)
    train = pd.read_csv(train_path, dtype={ID: str})
    targets = pd.read_csv(target_path, dtype={ID: str}).set_index(ID).loc[train[ID]].reset_index()
    assert len(train) == len(targets) == 4_407
    pixels_a = np.load(part_a["pixels"], mmap_mode="r")
    pixels_b = np.load(part_b["pixels"], mmap_mode="r")
    masks_a = np.load(part_a["mask"], mmap_mode="r")
    masks_b = np.load(part_b["mask"], mmap_mode="r")
    assert len(pixels_a) == 2_204 and len(pixels_b) == 2_203
    slot_bounds = np.load(Path(part_a["root"]) / "p2_slot_bounds.npy").astype(int)
    assert slot_bounds.tolist() == [[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]]

    def read_study(index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < len(pixels_a):
            return np.asarray(pixels_a[index]), np.asarray(masks_a[index]).astype(bool)
        local = index - len(pixels_a)
        return np.asarray(pixels_b[local]), np.asarray(masks_b[local]).astype(bool)

    def select_slices(index: int):
        volume, valid_mask = read_study(index)
        slices = np.zeros((K_SLICES, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        selected_mask = np.zeros(K_SLICES, dtype=bool)
        slots = np.zeros(K_SLICES, dtype=np.int64)
        positions = np.zeros(K_SLICES, dtype=np.float32)
        cursor = 0
        for slot, ((start, stop), quota) in enumerate(zip(slot_bounds, SLOT_QUOTAS)):
            available = np.where(valid_mask[start:stop])[0] + start
            count = min(len(available), quota)
            if count:
                if len(available) <= quota:
                    chosen = available
                else:
                    chosen = available[np.round(np.linspace(0, len(available) - 1, quota)).astype(int)]
                chosen = np.asarray(chosen[:count], dtype=int)
                out = slice(cursor, cursor + count)
                slices[out] = volume[chosen]
                selected_mask[out] = True
                slots[out] = slot
                positions[out] = 2.0 * ((chosen - start) / max(stop - start - 1, 1)) - 1.0
            slots[cursor:cursor + quota] = slot
            cursor += quota
        assert cursor == K_SLICES and selected_mask.any()
        labels = targets.loc[index, TARGETS].to_numpy(np.float32)
        weights = targets.loc[index, [f"weight__{target}" for target in TARGETS]].to_numpy(
            np.float32
        )
        return index, slices, selected_mask, slots, positions, labels, weights

    smoke_indices = np.round(np.linspace(0, len(train) - 1, SMOKE_STUDIES)).astype(int)
    assert len(np.unique(smoke_indices)) == SMOKE_STUDIES
    local_indices = smoke_indices[rank::world_size]

    class SmokeDataset(Dataset):
        def __len__(self) -> int:
            return len(local_indices)

        def __getitem__(self, position: int):
            return select_slices(int(local_indices[position]))

    loader = DataLoader(
        SmokeDataset(),
        batch_size=STUDY_BATCH_PER_RANK,
        shuffle=False,
        num_workers=WORKERS_PER_RANK,
        pin_memory=True,
        prefetch_factor=2,
    )

    load_started = time.time()
    model = torch.hub.load(
        str(runtime / "dinov3"),
        "dinov3_vitl16",
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 300_000_000 < total_parameters < 310_000_000
    model_load_seconds = time.time() - load_started
    dist.barrier()
    if is_primary:
        print(
            {
                "P5_MODEL_READY": True,
                "parameters": total_parameters,
                "load_seconds": model_load_seconds,
                "world_size": world_size,
            },
            flush=True,
        )

    mean = torch.tensor(MODEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(MODEL_STD, device=device).view(1, 3, 1, 1)
    feature_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    slot_batches: list[torch.Tensor] = []
    position_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    weight_batches: list[torch.Tensor] = []
    torch.cuda.reset_peak_memory_stats(device)
    extract_started = time.time()
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            _, slices, slice_mask, slots, positions, labels, weights = batch
            slices = slices.to(device, non_blocking=True)
            batch_size, count = slices.shape[:2]
            pixels = slices.reshape(-1, 1, CACHE_SIZE, CACHE_SIZE).float().div_(255.0)
            pixels = F.interpolate(
                pixels, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False
            )
            pixels = pixels.expand(-1, 3, -1, -1)
            pixels = ((pixels - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                encoded = model.forward_features(pixels)
                cls = encoded["x_norm_clstoken"]
                patch_mean = encoded["x_norm_patchtokens"].mean(dim=1)
                features = torch.stack((cls, patch_mean), dim=1)
            features = features.reshape(batch_size, count, N_REGIONS, BACKBONE_DIM)
            assert torch.isfinite(features).all()
            feature_batches.append(features.half())
            mask_batches.append(slice_mask.to(device, non_blocking=True).bool())
            slot_batches.append(slots.to(device, non_blocking=True).long())
            position_batches.append(positions.to(device, non_blocking=True).float())
            label_batches.append(labels.to(device, non_blocking=True).float())
            weight_batches.append(weights.to(device, non_blocking=True).float())
            if step == 1 or step == len(loader):
                print(
                    {
                        "P5_FEATURE_PROGRESS": f"rank={rank} step={step}/{len(loader)}",
                        "feature_shape": list(features.shape),
                    },
                    flush=True,
                )
    torch.cuda.synchronize(device)
    extraction_seconds = time.time() - extract_started

    features_all = torch.cat(feature_batches)
    masks_all = torch.cat(mask_batches)
    slots_all = torch.cat(slot_batches)
    positions_all = torch.cat(position_batches)
    labels_all = torch.cat(label_batches)
    weights_all = torch.cat(weight_batches)
    head = HierarchicalDiagnosisHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3, fused=True)
    head_started = time.time()
    final_loss = float("nan")
    for step in range(HEAD_SMOKE_STEPS):
        generator = torch.Generator(device=device).manual_seed(SEED + rank + step)
        chosen = torch.randperm(len(features_all), generator=generator, device=device)[:64]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head(
                features_all[chosen],
                masks_all[chosen],
                slots_all[chosen],
                positions_all[chosen],
            )
            raw = F.binary_cross_entropy_with_logits(logits, labels_all[chosen], reduction="none")
            loss = (raw * weights_all[chosen]).sum() / weights_all[chosen].sum().clamp_min(1.0)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        optimizer.step()
        final_loss = float(loss.detach())
    torch.cuda.synchronize(device)
    head_seconds = time.time() - head_started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    local_stats = torch.tensor(
        [
            model_load_seconds,
            extraction_seconds,
            head_seconds,
            peak_memory_mib,
            float(len(local_indices)),
            final_loss,
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]
    dist.all_gather(gathered, local_stats)
    dist.barrier()
    if is_primary:
        rows = [item.cpu().tolist() for item in gathered]
        max_load = max(row[0] for row in rows)
        max_extract = max(row[1] for row in rows)
        global_images = sum(row[4] for row in rows) * K_SLICES
        images_per_second = global_images / max_extract
        projected_full_feature_minutes = (4_407 * K_SLICES / images_per_second) / 60.0
        projected_hidden_feature_minutes = (1_300 * K_SLICES / images_per_second) / 60.0
        result = {
            "run_id": RUN_ID,
            "execution_revision": EXECUTION_REVISION,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "world_size": world_size,
            "smoke_studies": SMOKE_STUDIES,
            "slices_per_study": K_SLICES,
            "slot_quotas": list(SLOT_QUOTAS),
            "study_batch_per_rank": STUDY_BATCH_PER_RANK,
            "image_batch_per_rank": STUDY_BATCH_PER_RANK * K_SLICES,
            "feature_shape_per_study": [K_SLICES, N_REGIONS, BACKBONE_DIM],
            "checkpoint_sha256": MODEL_SHA256,
            "checkpoint_bytes": MODEL_BYTES,
            "orthofoundation_commit": ORTHO_COMMIT,
            "dinov3_commit": DINOV3_COMMIT,
            "cache_id": CACHE_ID,
            "cache_signature_sha256": CACHE_SIGNATURE,
            "model_parameters": total_parameters,
            "model_load_seconds_by_rank": [row[0] for row in rows],
            "extraction_seconds_by_rank": [row[1] for row in rows],
            "images_per_second_two_gpu": images_per_second,
            "projected_full_feature_minutes": projected_full_feature_minutes,
            "projected_hidden_feature_minutes": projected_hidden_feature_minutes,
            "head_smoke_steps": HEAD_SMOKE_STEPS,
            "head_seconds_by_rank": [row[2] for row in rows],
            "head_final_loss_by_rank": [row[5] for row in rows],
            "peak_memory_mib_by_rank": [row[3] for row in rows],
            "licenses": ["MIT", "DINOv3 License"],
            "finite": True,
            "estimated_full_fixed_model_load_minutes": max_load / 60.0,
        }
        (OUTPUT_ROOT / "p5_speed_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print({"P5_SPEED_SMOKE_COMPLETE": result}, flush=True)
        if telemetry is not None:
            telemetry.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main() -> int:
    if "--worker" not in sys.argv:
        return launch()
    return worker()


if __name__ == "__main__":
    raise SystemExit(main())
