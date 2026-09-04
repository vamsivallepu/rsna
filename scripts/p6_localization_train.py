#!/usr/bin/env python3
"""Train the full P6 fastMRI+ class-agnostic localization source model.

The official fastMRI single-coil train and validation archives define the
volume-isolated split. Checkpoint selection uses validation localization loss,
with lesion-center recall as the deterministic tie-breaker. The selected output
is a full CoAtNet backbone state dict that can strictly initialize the unchanged
P2 image model after its classification head is reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from p6_fastmri_hdf5_smoke import (
    ANNOTATION_SHA256,
    REVIEWED_SHA256,
    index_hdf5_paths,
    load_release,
    sanitize_boxes,
    sha256_file,
)
from p6_localization_speed_smoke import (
    CoAtNetLocalizer,
    LocalizationDataset,
    SliceSample,
    localization_loss,
    move_batch,
    parse_nvidia_monitor,
    set_reproducibility,
    start_nvidia_monitor,
    stop_nvidia_monitor,
)


TRAIN_ARCHIVE_SHA256 = (
    "1e8ab7f602bb0fea649736020aa9317ebf3ba9c417c211b286b85c81a28bce5f"
)
VAL_ARCHIVE_SHA256 = "cd65a27f25d11369ad73fd329ff0c68ea8d2510f98e9f46aaeff7fcaf0dba43b"
SOURCE_COMMIT = "67ed9a68ee33d3197f93893e8125688470c4a577"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hdf5-root", type=Path, required=True)
    parser.add_argument("--validation-hdf5-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reviewed-files", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--negative-slices-per-volume", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--background-weight", type=float, default=0.10)
    parser.add_argument("--size-loss-weight", type=float, default=0.10)
    return parser.parse_args()


def sha256_jsonable(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def inspect_geometry(
    root: Path, index: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[int, int, int]]]:
    rows: list[dict[str, object]] = []
    geometry: dict[str, tuple[int, int, int]] = {}
    for row in index.itertuples(index=False):
        with h5py.File(root / row.path, "r") as handle:
            required = {
                "kspace",
                "reconstruction_rss",
                "reconstruction_esc",
                "ismrmrd_header",
            }
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise ValueError(f"Missing HDF5 fields in {row.path}: {missing}")
            reconstruction = handle["reconstruction_rss"]
            if reconstruction.ndim != 3:
                raise ValueError(f"Unexpected reconstruction shape in {row.path}")
            depth, height, width = map(int, reconstruction.shape)
            if min(depth, height, width) <= 0:
                raise ValueError(f"Invalid reconstruction shape in {row.path}")
        geometry[row.file] = (depth, height, width)
        rows.append(
            {
                "file": row.file,
                "path": row.path,
                "file_bytes": int(row.file_bytes),
                "slices": depth,
                "rows": height,
                "columns": width,
            }
        )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True), geometry


def build_split_samples(
    split: str,
    index: pd.DataFrame,
    geometry: dict[str, tuple[int, int, int]],
    sanitation: pd.DataFrame,
    negative_slices_per_volume: int,
    excluded_negative_files: set[str],
) -> list[SliceSample]:
    path_by_file = dict(zip(index["file"], index["path"], strict=True))
    kept = sanitation.loc[sanitation["kept"]].copy()
    grouped: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for row in kept.itertuples(index=False):
        grouped.setdefault((row.file, int(row.slice)), []).append(
            (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
        )
    samples: list[SliceSample] = []
    positive_files = {key[0] for key in grouped}
    for (file_id, slice_index), boxes in sorted(grouped.items()):
        samples.append(
            SliceSample(
                file=file_id,
                path=path_by_file[file_id],
                slice=slice_index,
                split=split,
                boxes=tuple(boxes),
            )
        )
    for file_id in sorted(
        set(index["file"]) - positive_files - excluded_negative_files
    ):
        depth = geometry[file_id][0]
        fractions = np.linspace(0.20, 0.80, negative_slices_per_volume)
        slice_indices = sorted(
            {min(depth - 1, round((depth - 1) * x)) for x in fractions}
        )
        for slice_index in slice_indices:
            samples.append(
                SliceSample(
                    file=file_id,
                    path=path_by_file[file_id],
                    slice=int(slice_index),
                    split=split,
                    boxes=(),
                )
            )
    return sorted(samples, key=lambda sample: sample.identity)


def sample_records(samples: list[SliceSample]) -> list[dict[str, object]]:
    return [
        {
            "file": sample.file,
            "slice": sample.slice,
            "split": sample.split,
            "box_count": len(sample.boxes),
            "path": sample.path,
        }
        for sample in samples
    ]


def prepare_data(args: argparse.Namespace):
    annotations, reviewed = load_release(args.annotations, args.reviewed_files)
    study_level_artifact_files = set(
        annotations.loc[annotations["study_level"].eq("Yes"), "file"].astype(str)
    )
    slice_level_box_files = set(
        annotations.loc[annotations["study_level"].eq("No"), "file"].astype(str)
    )
    artifact_only_files = study_level_artifact_files - slice_level_box_files
    if len(study_level_artifact_files) != 13:
        raise ValueError(
            "Unexpected study-level artifact volume count: "
            f"{len(study_level_artifact_files)}"
        )
    if len(artifact_only_files) != 5:
        raise ValueError(
            f"Unexpected artifact-only volume count: {len(artifact_only_files)}"
        )
    train_raw, train_malformed = index_hdf5_paths(args.train_hdf5_root)
    validation_raw, validation_malformed = index_hdf5_paths(args.validation_hdf5_root)
    if train_malformed or validation_malformed:
        raise ValueError("Malformed HDF5 filenames are not allowed")
    train_files = set(train_raw["file"])
    validation_files = set(validation_raw["file"])
    if len(train_files) != 973 or len(validation_files) != 199:
        raise ValueError(
            f"Unexpected official archive counts: train={len(train_files)} "
            f"validation={len(validation_files)}"
        )
    if train_files & validation_files:
        raise ValueError("Official train and validation archives overlap")
    available = train_files | validation_files
    if available != reviewed:
        raise ValueError(
            f"Official HDF5 identities differ from the reviewed list: "
            f"missing={len(reviewed - available)} extra={len(available - reviewed)}"
        )

    train_index, train_geometry = inspect_geometry(args.train_hdf5_root, train_raw)
    validation_index, validation_geometry = inspect_geometry(
        args.validation_hdf5_root, validation_raw
    )
    geometry = {**train_geometry, **validation_geometry}
    sanitation = sanitize_boxes(annotations, reviewed, geometry)
    kept = sanitation.loc[sanitation["kept"]]
    if len(kept) != 16_153:
        raise ValueError(f"Unexpected kept box count: {len(kept)}")

    train_sanitation = sanitation.loc[sanitation["file"].isin(train_files)].copy()
    validation_sanitation = sanitation.loc[
        sanitation["file"].isin(validation_files)
    ].copy()
    train_samples = build_split_samples(
        "train",
        train_index,
        train_geometry,
        train_sanitation,
        args.negative_slices_per_volume,
        artifact_only_files,
    )
    validation_samples = build_split_samples(
        "validation",
        validation_index,
        validation_geometry,
        validation_sanitation,
        args.negative_slices_per_volume,
        artifact_only_files,
    )
    if {sample.file for sample in train_samples} & {
        sample.file for sample in validation_samples
    }:
        raise ValueError("Constructed source samples leak volumes")
    return (
        train_samples,
        validation_samples,
        train_index,
        validation_index,
        sanitation,
        study_level_artifact_files,
        artifact_only_files,
    )


def make_loaders(
    args: argparse.Namespace,
    train_samples: list[SliceSample],
    validation_samples: list[SliceSample],
) -> tuple[DataLoader, DataLoader]:
    options: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        options["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        LocalizationDataset(args.train_hdf5_root, train_samples),
        shuffle=True,
        drop_last=True,
        generator=generator,
        **options,
    )
    validation_loader = DataLoader(
        LocalizationDataset(args.validation_hdf5_root, validation_samples),
        shuffle=False,
        drop_last=False,
        **options,
    )
    return train_loader, validation_loader


def augment_intensity(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    gain = torch.empty((len(images), 1, 1, 1), device=images.device).uniform_(
        0.90, 1.10
    )
    offset = torch.empty((len(images), 1, 1, 1), device=images.device).uniform_(
        -12.75, 12.75
    )
    return (images * gain + offset).clamp(0.0, 255.0)


@torch.inference_mode()
def evaluate_localization(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
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
        local_maximum = probability.eq(
            F.max_pool2d(probability, 3, stride=1, padding=1)
        )
        for sample_index in range(len(images)):
            centers = torch.nonzero(mask[sample_index, 0] > 0.5, as_tuple=False)
            centers_total += len(centers)
            if len(centers) == 0:
                continue
            scores = probability[sample_index, 0].masked_fill(
                ~local_maximum[sample_index, 0], -1.0
            )
            topk = torch.topk(scores.flatten(), k=len(centers)).indices
            predicted = torch.stack(
                (topk // scores.shape[1], topk % scores.shape[1]), dim=1
            )
            for center in centers:
                distances = torch.linalg.vector_norm(
                    predicted.float() - center[None].float(), dim=1
                )
                centers_found += int(float(distances.min()) <= 2.0)
    return {
        "loss": total_loss / max(total_batches, 1),
        "lesion_center_recall_topk_radius_2": centers_found / max(centers_total, 1),
        "lesion_centers": centers_total,
    }


def save_state(path: Path, module: torch.nn.Module) -> str:
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in module.state_dict().items()
    }
    save_file(state, str(path))
    return sha256_file(path)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"P6 full training requires one CUDA GPU; found {torch.cuda.device_count()}"
        )
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epoch and batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    (
        train_samples,
        validation_samples,
        train_index,
        validation_index,
        sanitation,
        study_level_artifact_files,
        artifact_only_files,
    ) = prepare_data(args)
    train_records = sample_records(train_samples)
    validation_records = sample_records(validation_samples)
    train_index_path = args.output_dir / "p6_train_hdf5_index.csv"
    validation_index_path = args.output_dir / "p6_validation_hdf5_index.csv"
    sanitation_path = args.output_dir / "p6_full_box_sanitation.csv"
    samples_path = args.output_dir / "p6_source_samples.csv"
    train_index.to_csv(train_index_path, index=False)
    validation_index.to_csv(validation_index_path, index=False)
    sanitation.to_csv(sanitation_path, index=False)
    pd.DataFrame(train_records + validation_records).to_csv(samples_path, index=False)

    train_loader, validation_loader = make_loaders(
        args, train_samples, validation_samples
    )
    # Start and prime all workers before CUDA/model construction.
    train_prime = iter(train_loader)
    next(train_prime)
    validation_prime = iter(validation_loader)
    next(validation_prime)

    device = torch.device("cuda:0")
    model = CoAtNetLocalizer(args.model_weights).to(
        device=device, memory_format=torch.channels_last
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        fused=True,
    )
    updates_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.backbone_lr, args.head_lr],
        total_steps=args.epochs * updates_per_epoch,
        pct_start=0.10,
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    baseline_validation = evaluate_localization(model, validation_loader, device, args)
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "validation_loss": baseline_validation["loss"],
            "validation_lesion_center_recall_topk_radius_2": baseline_validation[
                "lesion_center_recall_topk_radius_2"
            ],
            "validation_lesion_centers": int(baseline_validation["lesion_centers"]),
        }
    ]
    monitor_path = args.output_dir / "p6_full_gpu_samples.csv"
    monitor = start_nvidia_monitor(monitor_path)
    best_loss = float("inf")
    best_recall = -1.0
    best_epoch = -1
    best_backbone_sha = ""
    best_head_sha = ""
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        epoch_images = 0
        for batch in train_loader:
            images, heatmap, size, mask, _ = move_batch(batch, device)
            images = augment_intensity(images)
            optimizer.zero_grad(set_to_none=True)
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            epoch_images += len(images)
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        validation = evaluate_localization(model, validation_loader, device, args)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation["loss"],
            "validation_lesion_center_recall_topk_radius_2": validation[
                "lesion_center_recall_topk_radius_2"
            ],
            "validation_lesion_centers": int(validation["lesion_centers"]),
            "epoch_seconds": epoch_seconds,
            "train_images_per_second": epoch_images / epoch_seconds,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        recall = validation["lesion_center_recall_topk_radius_2"]
        improved = validation["loss"] < best_loss or (
            validation["loss"] == best_loss and recall > best_recall
        )
        if improved:
            best_loss = validation["loss"]
            best_recall = recall
            best_epoch = epoch
            best_backbone_sha = save_state(
                args.output_dir / "p6_localization_backbone_best.safetensors",
                model.backbone,
            )
            best_head_sha = save_state(
                args.output_dir / "p6_localization_head_best.safetensors", model.head
            )
        write_json(args.output_dir / "p6_training_history.json", history)
        print(json.dumps(record, sort_keys=True), flush=True)

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    stop_nvidia_monitor(monitor)
    utilization = parse_nvidia_monitor(monitor_path)
    if best_epoch < 1 or not best_backbone_sha or not best_head_sha:
        raise ValueError("No source checkpoint was selected")

    manifest = {
        "schema_version": 1,
        "experiment_id": "P6B-FASTMRI-PLUS-LOCALIZATION-v1",
        "status": "validated",
        "code_commit": args.code_commit,
        "source": {
            "fastmri_plus_commit": SOURCE_COMMIT,
            "annotations_sha256": ANNOTATION_SHA256,
            "reviewed_files_sha256": REVIEWED_SHA256,
            "train_archive_sha256": TRAIN_ARCHIVE_SHA256,
            "validation_archive_sha256": VAL_ARCHIVE_SHA256,
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
        },
        "data": {
            "train_volumes": len(train_index),
            "validation_volumes": len(validation_index),
            "volume_overlap": 0,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "train_positive_samples": sum(
                bool(sample.boxes) for sample in train_samples
            ),
            "train_negative_samples": sum(not sample.boxes for sample in train_samples),
            "validation_positive_samples": sum(
                bool(sample.boxes) for sample in validation_samples
            ),
            "validation_negative_samples": sum(
                not sample.boxes for sample in validation_samples
            ),
            "kept_boxes": int(sanitation["kept"].sum()),
            "discarded_boxes": int((~sanitation["kept"]).sum()),
            "study_level_artifact_volumes": len(study_level_artifact_files),
            "excluded_study_level_artifact_volumes": len(artifact_only_files),
            "sample_signature": sha256_jsonable(train_records + validation_records),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "optimizer": "fused AdamW",
            "scheduler": "OneCycleLR",
            "amp": "float16",
            "channels_last": True,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "background_weight": args.background_weight,
            "size_loss_weight": args.size_loss_weight,
            "intensity_augmentation": "gain 0.90-1.10 plus offset -12.75 to 12.75",
            "geometric_augmentation": None,
            "training_seconds": training_seconds,
            "training_hours": training_seconds / 3600.0,
        },
        "selection": {
            "primary": "minimum held-out-volume localization loss",
            "tie_breaker": "maximum lesion-center top-K recall at radius 2",
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "best_validation_lesion_center_recall_topk_radius_2": best_recall,
            "baseline_validation_loss": baseline_validation["loss"],
            "baseline_validation_lesion_center_recall_topk_radius_2": baseline_validation[
                "lesion_center_recall_topk_radius_2"
            ],
        },
        "memory": {
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device)
            / 1024**2,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
        "utilization": utilization,
        "artifacts": {
            "train_index_sha256": sha256_file(train_index_path),
            "validation_index_sha256": sha256_file(validation_index_path),
            "sanitation_sha256": sha256_file(sanitation_path),
            "samples_sha256": sha256_file(samples_path),
            "backbone_sha256": best_backbone_sha,
            "head_sha256": best_head_sha,
            "history_sha256": sha256_file(args.output_dir / "p6_training_history.json"),
            "gpu_samples": monitor_path.name,
        },
        "history": history,
    }
    write_json(args.output_dir / "p6_localization_training.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
