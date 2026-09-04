#!/usr/bin/env python3
"""Run the gated P6 fastMRI+ CoAtNet localization GPU smoke.

The smoke uses the hash-pinned P6-B cohort and sanitation artifacts. It keeps
volume-level train/validation isolation, reads the fastMRI+ labeling frame,
trains the exact P2 CoAtNet backbone with a lightweight class-agnostic feature
pyramid head, and records loss, utilization, memory, throughput, and projected
full-source runtime. It intentionally does not produce a promotable checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset


ARCH = "coatnet_rmlp_2_rw_384"
MODEL_SHA256 = "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
COHORT_SHA256 = "cbd03d737395aa23f3ba3fd182a7572fb407ba4c614bedf51f2622ee52955598"
INDEX_SHA256 = "e04fbae294b85e214441b94f1b807c68ab4828407e59402f97f5d8fa6af0a333"
SANITATION_SHA256 = "02eb96b91bdbf19c748155608d961ea5901247212f2597e70322c8d88e308fd1"
IMAGE_SIZE = 384
OUTPUT_SIZE = 96
FPN_CHANNELS = 128
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
FILE_RE = re.compile(r"^file\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--hdf5-index", type=Path, required=True)
    parser.add_argument("--sanitation", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--max-validation-samples", type=int, default=128)
    parser.add_argument("--negative-slices-per-volume", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--projected-epochs", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--background-weight", type=float, default=0.10)
    parser.add_argument("--size-loss-weight", type=float, default=0.10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str, seed: int, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode()).hexdigest()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SliceSample:
    file: str
    path: str
    slice: int
    split: str
    boxes: tuple[tuple[float, float, float, float], ...]

    @property
    def identity(self) -> str:
        return f"{self.file}:{self.slice:04d}"


def cap_samples(
    samples: list[SliceSample], maximum: int, seed: int, namespace: str
) -> list[SliceSample]:
    positives = [sample for sample in samples if sample.boxes]
    negatives = [sample for sample in samples if not sample.boxes]
    positives.sort(key=lambda sample: stable_key(sample.identity, seed, namespace + "-pos"))
    negatives.sort(key=lambda sample: stable_key(sample.identity, seed, namespace + "-neg"))
    if len(samples) <= maximum:
        selected = samples
    else:
        wanted_negative = min(len(negatives), max(1, round(maximum * 0.20)))
        wanted_positive = min(len(positives), maximum - wanted_negative)
        if wanted_positive + wanted_negative < maximum:
            wanted_negative = min(len(negatives), maximum - wanted_positive)
        selected = positives[:wanted_positive] + negatives[:wanted_negative]
    return sorted(selected, key=lambda sample: sample.identity)


def build_samples(args: argparse.Namespace) -> tuple[list[SliceSample], list[SliceSample]]:
    expected_hashes = {
        args.cohort: COHORT_SHA256,
        args.hdf5_index: INDEX_SHA256,
        args.sanitation: SANITATION_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}: {observed}")

    cohort = pd.read_csv(args.cohort, dtype={"file": str})
    index = pd.read_csv(args.hdf5_index, dtype={"file": str, "path": str})
    boxes = pd.read_csv(args.sanitation, dtype={"file": str})
    if set(cohort.columns) != {"file", "has_annotation", "split"}:
        raise ValueError(f"Unexpected cohort columns: {cohort.columns.tolist()}")
    if cohort["file"].duplicated().any() or not cohort["file"].map(FILE_RE.fullmatch).all():
        raise ValueError("Invalid or duplicated cohort volume identity")
    if set(cohort["split"]) != {"train", "validation"}:
        raise ValueError(f"Unexpected splits: {sorted(set(cohort['split']))}")
    train_volumes = set(cohort.loc[cohort["split"].eq("train"), "file"])
    validation_volumes = set(cohort.loc[cohort["split"].eq("validation"), "file"])
    if train_volumes & validation_volumes:
        raise ValueError("Volume leakage across train and validation")

    index = index.merge(cohort[["file", "split", "has_annotation"]], on="file", how="inner")
    if len(index) != len(cohort):
        raise ValueError("HDF5 index does not cover the full smoke cohort exactly once")
    path_by_file = dict(zip(index["file"], index["path"], strict=True))
    depth_by_file = dict(zip(index["file"], index["slices"].astype(int), strict=True))

    kept = boxes.loc[boxes["kept"].astype(str).str.lower().eq("true")].copy()
    grouped_boxes: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for row in kept.itertuples(index=False):
        grouped_boxes.setdefault((row.file, int(row.slice)), []).append(
            (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
        )

    samples: list[SliceSample] = []
    for row in cohort.itertuples(index=False):
        volume_boxes = sorted(
            (key, value) for key, value in grouped_boxes.items() if key[0] == row.file
        )
        for (_, slice_index), slice_boxes in volume_boxes:
            samples.append(
                SliceSample(
                    file=row.file,
                    path=path_by_file[row.file],
                    slice=slice_index,
                    split=row.split,
                    boxes=tuple(slice_boxes),
                )
            )
        if not bool(row.has_annotation):
            depth = depth_by_file[row.file]
            fractions = np.linspace(0.20, 0.80, args.negative_slices_per_volume)
            for slice_index in sorted({min(depth - 1, round((depth - 1) * x)) for x in fractions}):
                samples.append(
                    SliceSample(
                        file=row.file,
                        path=path_by_file[row.file],
                        slice=int(slice_index),
                        split=row.split,
                        boxes=(),
                    )
                )

    train = cap_samples(
        [sample for sample in samples if sample.split == "train"],
        args.max_train_samples,
        args.seed,
        "train",
    )
    validation = cap_samples(
        [sample for sample in samples if sample.split == "validation"],
        args.max_validation_samples,
        args.seed,
        "validation",
    )
    if not train or not validation or not any(sample.boxes for sample in validation):
        raise ValueError("Training or validation sample construction failed")
    if {sample.file for sample in train} & {sample.file for sample in validation}:
        raise ValueError("Constructed slice samples leak volumes")
    return train, validation


def gaussian_radius(width: float, height: float) -> int:
    sigma = max(1.0, min(width, height) / 6.0)
    return max(1, int(math.ceil(3.0 * sigma)))


def draw_gaussian(target: np.ndarray, center_x: float, center_y: float, radius: int) -> None:
    x0 = int(round(center_x))
    y0 = int(round(center_y))
    left = max(0, x0 - radius)
    right = min(target.shape[1], x0 + radius + 1)
    top = max(0, y0 - radius)
    bottom = min(target.shape[0], y0 + radius + 1)
    if left >= right or top >= bottom:
        return
    yy, xx = np.ogrid[top:bottom, left:right]
    sigma = max(radius / 3.0, 1.0 / 3.0)
    gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
    target[top:bottom, left:right] = np.maximum(target[top:bottom, left:right], gaussian)


class LocalizationDataset(Dataset):
    def __init__(self, root: Path, samples: list[SliceSample]) -> None:
        self.root = root
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with h5py.File(self.root / sample.path, "r") as handle:
            pixels = np.asarray(
                handle["reconstruction_rss"][sample.slice, :, :], dtype=np.float32
            )[::-1, :].copy()
        height, width = pixels.shape
        low, high = np.percentile(pixels, [0.5, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid pixels for {sample.identity}")
        pixels = np.clip((pixels - low) / (high - low), 0.0, 1.0)
        pixels_u8 = np.asarray(
            Image.fromarray((pixels * 255.0).round().astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
            )
        ).copy()
        image = torch.from_numpy(pixels_u8).unsqueeze(0).repeat(3, 1, 1)

        heatmap = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        size = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        center_mask = np.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        for x1, y1, x2, y2 in sample.boxes:
            box_width = (x2 - x1) * OUTPUT_SIZE / width
            box_height = (y2 - y1) * OUTPUT_SIZE / height
            center_x = ((x1 + x2) * 0.5) * OUTPUT_SIZE / width
            center_y = ((y1 + y2) * 0.5) * OUTPUT_SIZE / height
            center_x = min(max(center_x, 0.0), OUTPUT_SIZE - 1.0)
            center_y = min(max(center_y, 0.0), OUTPUT_SIZE - 1.0)
            draw_gaussian(
                heatmap, center_x, center_y, gaussian_radius(box_width, box_height)
            )
            cell_x = int(round(center_x))
            cell_y = int(round(center_y))
            size[:, cell_y, cell_x] = (box_width, box_height)
            center_mask[0, cell_y, cell_x] = 1.0
            heatmap[cell_y, cell_x] = 1.0
        return (
            image,
            torch.from_numpy(heatmap).unsqueeze(0),
            torch.from_numpy(size),
            torch.from_numpy(center_mask),
            sample.identity,
        )


class FeaturePyramidHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (128, 256, 512, 1024)
        self.lateral = nn.ModuleList(
            nn.Conv2d(value, FPN_CHANNELS, kernel_size=1) for value in channels
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(FPN_CHANNELS, FPN_CHANNELS, kernel_size=3, padding=1),
                nn.GroupNorm(16, FPN_CHANNELS),
                nn.SiLU(inplace=True),
            )
            for _ in channels
        )
        self.refine = nn.Sequential(
            nn.Conv2d(FPN_CHANNELS, FPN_CHANNELS, kernel_size=3, padding=1),
            nn.GroupNorm(16, FPN_CHANNELS),
            nn.SiLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(FPN_CHANNELS, 1, kernel_size=1)
        self.box_size = nn.Conv2d(FPN_CHANNELS, 2, kernel_size=1)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pyramid = [layer(value) for layer, value in zip(self.lateral, features, strict=True)]
        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1], size=pyramid[index].shape[-2:], mode="nearest"
            )
        target_size = pyramid[0].shape[-2:]
        fused = sum(
            F.interpolate(layer(value), size=target_size, mode="bilinear", align_corners=False)
            for layer, value in zip(self.smooth, pyramid, strict=True)
        ) / len(pyramid)
        fused = self.refine(fused)
        return self.heatmap(fused), self.box_size(fused)


class CoAtNetLocalizer(nn.Module):
    def __init__(self, weight_path: Path) -> None:
        super().__init__()
        if sha256_file(weight_path) != MODEL_SHA256:
            raise ValueError("Official CoAtNet weight hash mismatch")
        self.backbone = timm.create_model(ARCH, pretrained=False)
        state = load_file(str(weight_path), device="cpu")
        incompatible = self.backbone.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"Strict CoAtNet load failed: {incompatible}")
        self.head = FeaturePyramidHead()
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1))

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = pixels.float().div(255.0)
        pixels = (pixels - self.mean) / self.std
        _, features = self.backbone.forward_intermediates(pixels, indices=(1, 2, 3, 4))
        return self.head(features)


def localization_loss(
    heatmap_logits: torch.Tensor,
    size_logits: torch.Tensor,
    heatmap_target: torch.Tensor,
    size_target: torch.Tensor,
    center_mask: torch.Tensor,
    background_weight: float,
    size_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probability = heatmap_logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    positive = heatmap_target.eq(1.0)
    negative = heatmap_target.lt(1.0)
    negative_weight = (1.0 - heatmap_target).pow(4)
    positive_loss = -(probability.log()) * (1.0 - probability).pow(2) * positive
    negative_loss = (
        -((1.0 - probability).log())
        * probability.pow(2)
        * negative_weight
        * negative
    )
    positive_count = positive.sum().clamp_min(1)
    heatmap_loss = (positive_loss.sum() + background_weight * negative_loss.sum()) / positive_count
    predicted_size = F.softplus(size_logits)
    size_denominator = (center_mask.sum() * 2.0).clamp_min(1.0)
    size_loss = (torch.abs(predicted_size - size_target) * center_mask).sum() / size_denominator
    total = heatmap_loss + size_loss_weight * size_loss
    return total, {"heatmap": heatmap_loss.detach(), "size": size_loss.detach()}


def infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def move_batch(batch, device: torch.device):
    images, heatmap, size, mask, identities = batch
    return (
        images.to(device, non_blocking=True, memory_format=torch.channels_last),
        heatmap.to(device, non_blocking=True),
        size.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        identities,
    )


def forward_loss(
    model: nn.Module,
    batch,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images, heatmap, size, mask, _ = move_batch(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        heatmap_logits, size_logits = model(images)
        return localization_loss(
            heatmap_logits,
            size_logits,
            heatmap,
            size,
            mask,
            args.background_weight,
            args.size_loss_weight,
        )


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    centers_found = 0
    centers_total = 0
    for batch in loader:
        images, heatmap, size, mask, _ = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            heatmap_logits, size_logits = model(images)
            loss, _ = localization_loss(
                heatmap_logits,
                size_logits,
                heatmap,
                size,
                mask,
                args.background_weight,
                args.size_loss_weight,
            )
        total_loss += float(loss)
        total_batches += 1
        probability = heatmap_logits.sigmoid()
        for sample_index in range(len(images)):
            centers = torch.nonzero(mask[sample_index, 0] > 0.5, as_tuple=False)
            centers_total += len(centers)
            for yx in centers:
                y, x = map(int, yx)
                y1, y2 = max(0, y - 2), min(OUTPUT_SIZE, y + 3)
                x1, x2 = max(0, x - 2), min(OUTPUT_SIZE, x + 3)
                centers_found += int(float(probability[sample_index, 0, y1:y2, x1:x2].max()) >= 0.10)
    return {
        "loss": total_loss / max(total_batches, 1),
        "lesion_center_recall_at_0.10_radius_2": centers_found / max(centers_total, 1),
        "lesion_centers": centers_total,
    }


def start_nvidia_monitor(path: Path) -> subprocess.Popen:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
        "-lms",
        "500",
        "-f",
        str(path),
    ]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop_nvidia_monitor(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def parse_nvidia_monitor(path: Path) -> dict[str, float | int]:
    rows: list[tuple[float, float, float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                rows.append(tuple(float(value) for value in parts[1:4]))
            except ValueError:
                continue
    if not rows:
        raise ValueError("nvidia-smi utilization monitor produced no samples")
    values = np.asarray(rows)
    return {
        "samples": len(rows),
        "gpu_utilization_mean_percent": float(values[:, 0].mean()),
        "gpu_utilization_p10_percent": float(np.percentile(values[:, 0], 10)),
        "gpu_utilization_max_percent": float(values[:, 0].max()),
        "nvidia_memory_used_max_mib": float(values[:, 1].max()),
        "power_draw_mean_w": float(values[:, 2].mean()),
        "power_draw_max_w": float(values[:, 2].max()),
    }


def count_projected_samples(args: argparse.Namespace) -> dict[str, int]:
    annotations = pd.read_csv(args.annotations)
    boxes = annotations.loc[annotations["study_level"].eq("No")].copy()
    unique_positive_slices = int(boxes[["file", "slice"]].drop_duplicates().shape[0])
    annotated_volumes = int(boxes["file"].nunique())
    reviewed_negative_volumes = 198
    per_epoch = unique_positive_slices + reviewed_negative_volumes * args.negative_slices_per_volume
    return {
        "unique_positive_slices": unique_positive_slices,
        "annotated_volumes": annotated_volumes,
        "reviewed_negative_volumes": reviewed_negative_volumes,
        "samples_per_epoch": per_epoch,
        "epochs": args.projected_epochs,
        "images": per_epoch * args.projected_epochs,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"P6 smoke requires exactly one CUDA GPU; found {torch.cuda.device_count()}")
    if args.batch_size <= 0 or min(args.overfit_steps, args.warmup_steps, args.timed_steps) <= 0:
        raise ValueError("Batch size and all step counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    train_samples, validation_samples = build_samples(args)
    train_dataset = LocalizationDataset(args.hdf5_root, train_samples)
    validation_dataset = LocalizationDataset(args.hdf5_root, validation_samples)
    loader_options: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_options
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has no complete batch")

    # Start persistent HDF5 workers before creating CUDA state. Forking workers
    # after a large CUDA model is resident caused a one-time idle gap that does
    # not represent steady-state training throughput.
    train_iterator = infinite_loader(train_loader)
    fixed_batch = next(train_iterator)

    model = CoAtNetLocalizer(args.model_weights).to(
        device=device, memory_format=torch.channels_last
    )
    backbone_parameters = sum(value.numel() for value in model.backbone.parameters())
    head_parameters = sum(value.numel() for value in model.head.parameters())
    if not 73_000_000 < backbone_parameters < 76_000_000:
        raise ValueError(f"Unexpected P2 backbone parameter count: {backbone_parameters}")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        initial_overfit_loss = float(forward_loss(model, fixed_batch, device, args)[0])
    model.train()
    overfit_losses: list[float] = []
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, fixed_batch, device, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        overfit_losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final_overfit_loss = float(forward_loss(model, fixed_batch, device, args)[0])

    model.train()
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, next(train_iterator), device, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    monitor_path = args.output_dir / "p6_gpu_samples.csv"
    monitor = start_nvidia_monitor(monitor_path)
    started = time.perf_counter()
    timed_losses: list[float] = []
    for _ in range(args.timed_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, next(train_iterator), device, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        timed_losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop_nvidia_monitor(monitor)

    validation = evaluate(model, validation_loader, device, args)
    utilization = parse_nvidia_monitor(monitor_path)
    images = args.timed_steps * args.batch_size
    images_per_second = images / elapsed
    projected = count_projected_samples(args)
    projected_seconds = projected["images"] / images_per_second
    overfit_reduction = (initial_overfit_loss - final_overfit_loss) / initial_overfit_loss
    gates = {
        "finite_losses": bool(
            np.isfinite(overfit_losses + timed_losses + [validation["loss"]]).all()
        ),
        "overfit_loss_reduction_at_least_5_percent": overfit_reduction >= 0.05,
        "mean_gpu_utilization_at_least_70_percent": utilization[
            "gpu_utilization_mean_percent"
        ]
        >= 70.0,
        "peak_torch_allocation_below_22_gib": torch.cuda.max_memory_allocated(device)
        < 22 * 1024**3,
        "projected_eight_epoch_runtime_below_six_hours": projected_seconds < 6 * 3600,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "P6B-FASTMRI-PLUS-LOCALIZATION-GPU-SMOKE-v1",
        "status": "validated" if all(gates.values()) else "gate_failed",
        "source": {
            "model_architecture": ARCH,
            "model_sha256": MODEL_SHA256,
            "cohort_sha256": COHORT_SHA256,
            "hdf5_index_sha256": INDEX_SHA256,
            "sanitation_sha256": SANITATION_SHA256,
            "image_field": "reconstruction_rss",
            "labeling_frame_transform": "pixels[:, ::-1, :]",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "h5py": h5py.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "visible_gpus": torch.cuda.device_count(),
            "cpu_count": os.cpu_count(),
        },
        "model": {
            "backbone_parameters": backbone_parameters,
            "localization_head_parameters": head_parameters,
            "head": "four-level 128-channel FPN with class-agnostic center and box-size outputs",
            "background_weight": args.background_weight,
            "size_loss_weight": args.size_loss_weight,
        },
        "data": {
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "train_volumes": len({sample.file for sample in train_samples}),
            "validation_volumes": len({sample.file for sample in validation_samples}),
            "volume_overlap": len(
                {sample.file for sample in train_samples}
                & {sample.file for sample in validation_samples}
            ),
            "train_positive_samples": sum(bool(sample.boxes) for sample in train_samples),
            "train_negative_samples": sum(not sample.boxes for sample in train_samples),
            "validation_positive_samples": sum(
                bool(sample.boxes) for sample in validation_samples
            ),
            "validation_negative_samples": sum(
                not sample.boxes for sample in validation_samples
            ),
        },
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "amp": "float16",
            "channels_last": True,
            "optimizer": "fused AdamW",
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "overfit_steps": args.overfit_steps,
            "initial_overfit_loss": initial_overfit_loss,
            "final_overfit_loss": final_overfit_loss,
            "overfit_loss_reduction_fraction": overfit_reduction,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "timed_mean_loss": float(np.mean(timed_losses)),
            "timed_elapsed_seconds": elapsed,
            "images_per_second": images_per_second,
            "updates_per_second": args.timed_steps / elapsed,
        },
        "validation": validation,
        "memory": {
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
        "utilization": utilization,
        "full_source_projection": {
            **projected,
            "seconds": projected_seconds,
            "hours": projected_seconds / 3600.0,
            "excludes_one_time_archive_download_and_cache_build": True,
        },
        "gates": gates,
        "artifacts": {
            "gpu_samples": monitor_path.name,
            "checkpoint_saved": False,
        },
    }
    manifest_path = args.output_dir / "p6_localization_gpu_smoke.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if manifest["status"] != "validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
