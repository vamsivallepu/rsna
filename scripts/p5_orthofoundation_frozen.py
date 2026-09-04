#!/usr/bin/env python3
"""Run the full frozen P5-A OrthoFoundation feature and OOF experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch.utils.data import DataLoader, Dataset

import p5_orthofoundation_speed_smoke as common


RUN_ID = common.RUN_ID
EXECUTION_REVISION = "p5a_frozen_t4x2_full_v2"
FOLD_SEED = 20260831
N_FOLDS = 5
PSEUDO_EPOCHS = 40
FOLD_EPOCHS = 30
HEAD_BATCH = 128
PSEUDO_LR = 3e-4
FOLD_LR = 2e-4
WEIGHT_DECAY = 1e-3
OUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(
    "outputs/p5_orthofoundation_frozen"
)


def launch() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P5-A requires exactly two GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    print({"P5_FULL_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P5_FULL_DDP_PROCESS_COMPLETE": True}, flush=True)
    return 0


def normalize_report(report: object) -> str:
    text = unicodedata.normalize("NFKC", str(report)).casefold()
    return "".join(character for character in text if character.isalnum())


def report_hash(report: object) -> str:
    normalized = normalize_report(report)
    if not normalized:
        raise ValueError("empty report after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


def calculate_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[pd.DataFrame, float]:
    rows = [
        {
            "target": target,
            "auc": float(roc_auc_score(labels[:, index], predictions[:, index])),
        }
        for index, target in enumerate(common.TARGETS)
    ]
    macro = float(np.mean([row["auc"] for row in rows]))
    rows.append({"target": "MACRO", "auc": macro})
    return pd.DataFrame(rows), macro


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index]).rank(method="average", pct=True).to_numpy()
            for index in range(predictions.shape[1])
        ]
    ).astype(np.float32)


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    repeats: int = 5_000,
    seed: int = common.SEED + 300_000,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.integers(0, len(labels), len(labels))
        target_deltas: list[float] = []
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


def load_model(runtime: Path, checkpoint: Path, device: torch.device):
    model = torch.hub.load(
        str(runtime / "dinov3"),
        "dinov3_vitl16",
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and state
    keys = list(state)
    if all(key.startswith("backbone.") for key in keys):
        state = {key.removeprefix("backbone."): value for key, value in state.items()}
        namespace = "backbone."
    elif all(key.startswith("module.backbone.") for key in keys):
        state = {key.removeprefix("module.backbone."): value for key, value in state.items()}
        namespace = "module.backbone."
    else:
        namespace = "unprefixed"
    assert set(state) == set(model.state_dict())
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to(device)
    return model, namespace


def validate_inputs(
    runtime: Path,
    checkpoint: Path,
    parts: list[dict[str, object]],
) -> None:
    started = time.time()
    assert checkpoint.stat().st_size == common.MODEL_BYTES
    assert common.sha256_file(checkpoint) == common.MODEL_SHA256
    provenance = (runtime / "PROVENANCE.md").read_text()
    assert common.ORTHO_COMMIT in provenance and common.DINOV3_COMMIT in provenance
    assert (runtime / "ORTHOFOUNDATION_LICENSE").is_file()
    assert (runtime / "dinov3/LICENSE.md").is_file()
    jobs: list[tuple[Path, str]] = []
    for part in parts:
        manifest = part["manifest"]
        assert isinstance(manifest, dict)
        root = Path(part["root"])
        for name, expected in manifest["files_sha256"].items():
            jobs.append((root / name, str(expected)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(common.sha256_file, path): (path, expected) for path, expected in jobs}
        for future in as_completed(futures):
            path, expected = futures[future]
            assert future.result() == expected, f"{path.name} hash mismatch"
            print({"P5_FULL_INPUT_HASH_OK": path.name}, flush=True)
    print({"P5_FULL_INPUT_VALIDATION_MINUTES": (time.time() - started) / 60.0}, flush=True)


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
    telemetry = (
        common.GpuTelemetry(OUTPUT_ROOT / "p5_gpu_telemetry.csv", interval_seconds=30)
        if is_primary
        else None
    )
    if telemetry is not None:
        telemetry.start()

    random.seed(common.SEED + rank)
    np.random.seed(common.SEED + rank)
    torch.manual_seed(common.SEED + rank)
    torch.cuda.manual_seed_all(common.SEED + rank)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    runtime, checkpoint = common.find_runtime()
    part_a, part_b = common.cache_part("a"), common.cache_part("b")
    parts = [part_a, part_b]
    if is_primary:
        validate_inputs(runtime, checkpoint, parts)
    dist.barrier()

    train_path = common.find_named_file("train.csv", 100_000)
    target_path = common.find_named_file("training_targets.csv", 100_000)
    train = pd.read_csv(train_path, dtype={common.ID: str})
    targets = (
        pd.read_csv(target_path, dtype={common.ID: str})
        .set_index(common.ID)
        .loc[train[common.ID]]
        .reset_index()
    )
    assert len(train) == len(targets) == 4_407
    assert targets[common.TARGETS].notna().all().all()
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
        slices = np.zeros(
            (common.K_SLICES, common.CACHE_SIZE, common.CACHE_SIZE), dtype=np.uint8
        )
        selected_mask = np.zeros(common.K_SLICES, dtype=bool)
        slots = np.zeros(common.K_SLICES, dtype=np.int64)
        positions = np.zeros(common.K_SLICES, dtype=np.float32)
        cursor = 0
        for slot, ((start, stop), quota) in enumerate(zip(slot_bounds, common.SLOT_QUOTAS)):
            available = np.where(valid_mask[start:stop])[0] + start
            count = min(len(available), quota)
            if count:
                if len(available) <= quota:
                    chosen = available
                else:
                    chosen = available[
                        np.round(np.linspace(0, len(available) - 1, quota)).astype(int)
                    ]
                chosen = np.asarray(chosen[:count], dtype=int)
                out = slice(cursor, cursor + count)
                slices[out] = volume[chosen]
                selected_mask[out] = True
                slots[out] = slot
                positions[out] = 2.0 * ((chosen - start) / max(stop - start - 1, 1)) - 1.0
            slots[cursor:cursor + quota] = slot
            cursor += quota
        assert cursor == common.K_SLICES and selected_mask.any()
        return index, slices, selected_mask, slots, positions

    local_indices = np.arange(rank, len(train), world_size, dtype=np.int64)

    class FeatureDataset(Dataset):
        def __len__(self) -> int:
            return len(local_indices)

        def __getitem__(self, position: int):
            return select_slices(int(local_indices[position]))

    loader = DataLoader(
        FeatureDataset(),
        batch_size=common.STUDY_BATCH_PER_RANK,
        shuffle=False,
        num_workers=common.WORKERS_PER_RANK,
        pin_memory=True,
        prefetch_factor=2,
    )

    load_started = time.time()
    model, checkpoint_namespace = load_model(runtime, checkpoint, device)
    model_load_seconds = time.time() - load_started
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 300_000_000 < total_parameters < 310_000_000
    dist.barrier()
    if is_primary:
        print(
            {
                "P5_FULL_START": True,
                "run_id": RUN_ID,
                "revision": EXECUTION_REVISION,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "studies": len(train),
                "world_size": world_size,
                "model_parameters": total_parameters,
                "checkpoint_namespace": checkpoint_namespace,
            },
            flush=True,
        )

    feature_shard_path = OUTPUT_ROOT / f"p5_feature_shard_rank_{rank}.npy"
    mask_shard_path = OUTPUT_ROOT / f"p5_mask_shard_rank_{rank}.npy"
    slot_shard_path = OUTPUT_ROOT / f"p5_slot_shard_rank_{rank}.npy"
    position_shard_path = OUTPUT_ROOT / f"p5_position_shard_rank_{rank}.npy"
    index_shard_path = OUTPUT_ROOT / f"p5_index_shard_rank_{rank}.npy"
    feature_shard = np.lib.format.open_memmap(
        feature_shard_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(local_indices),
            common.K_SLICES,
            common.N_REGIONS,
            common.BACKBONE_DIM,
        ),
    )
    mask_shard = np.zeros((len(local_indices), common.K_SLICES), dtype=np.uint8)
    slot_shard = np.zeros((len(local_indices), common.K_SLICES), dtype=np.int8)
    position_shard = np.zeros((len(local_indices), common.K_SLICES), dtype=np.float16)
    mean = torch.tensor(common.MODEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(common.MODEL_STD, device=device).view(1, 3, 1, 1)
    torch.cuda.reset_peak_memory_stats(device)
    extract_started = time.time()
    offset = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            _, slices, slice_mask, slots, positions = batch
            slices = slices.to(device, non_blocking=True)
            batch_size, count = slices.shape[:2]
            pixels = slices.reshape(-1, 1, common.CACHE_SIZE, common.CACHE_SIZE)
            pixels = pixels.float().div_(255.0)
            pixels = F.interpolate(
                pixels,
                size=(common.MODEL_SIZE, common.MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            pixels = pixels.expand(-1, 3, -1, -1)
            pixels = ((pixels - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                encoded = model.forward_features(pixels)
                features = torch.stack(
                    (
                        encoded["x_norm_clstoken"],
                        encoded["x_norm_patchtokens"].mean(dim=1),
                    ),
                    dim=1,
                )
            features = features.reshape(
                batch_size,
                count,
                common.N_REGIONS,
                common.BACKBONE_DIM,
            )
            assert torch.isfinite(features).all()
            stop = offset + batch_size
            feature_shard[offset:stop] = features.cpu().numpy().astype(np.float16)
            mask_shard[offset:stop] = slice_mask.numpy().astype(np.uint8)
            slot_shard[offset:stop] = slots.numpy().astype(np.int8)
            position_shard[offset:stop] = positions.numpy().astype(np.float16)
            offset = stop
            if step % 100 == 0 or step == len(loader):
                print(
                    {
                        "P5_FULL_FEATURE_PROGRESS": f"rank={rank} {offset}/{len(local_indices)}",
                        "elapsed_minutes": (time.time() - extract_started) / 60.0,
                    },
                    flush=True,
                )
    assert offset == len(local_indices)
    feature_shard.flush()
    np.save(mask_shard_path, mask_shard)
    np.save(slot_shard_path, slot_shard)
    np.save(position_shard_path, position_shard)
    np.save(index_shard_path, local_indices)
    torch.cuda.synchronize(device)
    extraction_seconds = time.time() - extract_started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    local_stats = torch.tensor(
        [model_load_seconds, extraction_seconds, peak_memory_mib],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]
    dist.all_gather(gathered, local_stats)
    dist.barrier()

    feature_path = OUTPUT_ROOT / "p5_orthofoundation_features.npy"
    mask_path = OUTPUT_ROOT / "p5_feature_mask.npy"
    slot_path = OUTPUT_ROOT / "p5_feature_slots.npy"
    position_path = OUTPUT_ROOT / "p5_feature_positions.npy"
    if is_primary:
        merged_features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(train),
                common.K_SLICES,
                common.N_REGIONS,
                common.BACKBONE_DIM,
            ),
        )
        merged_mask = np.zeros((len(train), common.K_SLICES), dtype=np.uint8)
        merged_slots = np.zeros((len(train), common.K_SLICES), dtype=np.int8)
        merged_positions = np.zeros((len(train), common.K_SLICES), dtype=np.float16)
        for shard_rank in range(world_size):
            indices = np.load(OUTPUT_ROOT / f"p5_index_shard_rank_{shard_rank}.npy")
            merged_features[indices] = np.load(
                OUTPUT_ROOT / f"p5_feature_shard_rank_{shard_rank}.npy", mmap_mode="r"
            )
            merged_mask[indices] = np.load(OUTPUT_ROOT / f"p5_mask_shard_rank_{shard_rank}.npy")
            merged_slots[indices] = np.load(OUTPUT_ROOT / f"p5_slot_shard_rank_{shard_rank}.npy")
            merged_positions[indices] = np.load(
                OUTPUT_ROOT / f"p5_position_shard_rank_{shard_rank}.npy"
            )
        merged_features.flush()
        assert np.isfinite(merged_features).all() and merged_mask.any(axis=1).all()
        np.save(mask_path, merged_mask)
        np.save(slot_path, merged_slots)
        np.save(position_path, merged_positions)
        pd.DataFrame({common.ID: train[common.ID]}).to_csv(
            OUTPUT_ROOT / "p5_feature_studies.csv", index=False
        )
        for shard_rank in range(world_size):
            for name in ("feature", "mask", "slot", "position", "index"):
                (OUTPUT_ROOT / f"p5_{name}_shard_rank_{shard_rank}.npy").unlink()
        print({"P5_FULL_FEATURE_EXTRACTION_COMPLETE": True}, flush=True)
    dist.barrier()
    del model
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    if not is_primary:
        return 0

    features = torch.from_numpy(np.asarray(np.load(feature_path, mmap_mode="r"))).to(device)
    feature_mask = torch.from_numpy(np.load(mask_path).astype(bool)).to(device)
    feature_slots = torch.from_numpy(np.load(slot_path).astype(np.int64)).to(device)
    feature_positions = torch.from_numpy(np.load(position_path).astype(np.float32)).to(device)
    labels_np = targets[common.TARGETS].to_numpy(np.float32)
    weights_np = targets[
        [f"weight__{target}" for target in common.TARGETS]
    ].to_numpy(np.float32)
    labels = torch.from_numpy(labels_np).to(device)
    weights = torch.from_numpy(weights_np).to(device)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    pseudo_indices = np.where(~is_gold)[0]
    assert len(gold_indices) == 58 and len(pseudo_indices) == 4_349
    report_hashes = train["Report"].map(report_hash).to_numpy(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=common.SEED)
    pseudo_train_pos, pseudo_valid_pos = next(
        splitter.split(pseudo_indices, groups=report_hashes[pseudo_indices])
    )
    pseudo_train = pseudo_indices[pseudo_train_pos]
    pseudo_valid = pseudo_indices[pseudo_valid_pos]
    assert not (set(report_hashes[pseudo_train]) & set(report_hashes[pseudo_valid]))

    def evaluate_loss(head, indices: np.ndarray) -> float:
        head.eval()
        numerator = 0.0
        denominator = 0.0
        with torch.inference_mode():
            for start in range(0, len(indices), HEAD_BATCH):
                index = torch.as_tensor(indices[start:start + HEAD_BATCH], device=device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = head(
                        features[index],
                        feature_mask[index],
                        feature_slots[index],
                        feature_positions[index],
                    )
                    raw = F.binary_cross_entropy_with_logits(
                        logits, labels[index], reduction="none"
                    )
                numerator += float((raw * weights[index]).sum())
                denominator += float(weights[index].sum())
        return numerator / max(denominator, 1.0)

    def fit_head(
        train_indices: np.ndarray,
        *,
        epochs: int,
        lr: float,
        seed: int,
        initial_state: dict[str, torch.Tensor] | None,
        valid_indices: np.ndarray | None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        head = common.HierarchicalDiagnosisHead().to(device)
        if initial_state is not None:
            head.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True
        )
        generator = torch.Generator().manual_seed(seed)
        train_cpu = torch.as_tensor(train_indices, dtype=torch.long)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            head.train()
            permutation = train_cpu[torch.randperm(len(train_cpu), generator=generator)]
            losses: list[float] = []
            for start in range(0, len(permutation), HEAD_BATCH):
                index = permutation[start:start + HEAD_BATCH].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = head(
                        features[index],
                        feature_mask[index],
                        feature_slots[index],
                        feature_positions[index],
                    )
                    raw = F.binary_cross_entropy_with_logits(
                        logits, labels[index], reduction="none"
                    )
                    loss = (raw * weights[index]).sum() / weights[index].sum().clamp_min(1.0)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            validation_loss = (
                evaluate_loss(head, valid_indices)
                if valid_indices is not None
                else float(np.mean(losses))
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                }
            )
            if valid_indices is None or validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in head.state_dict().items()
                }
        assert best_state is not None
        return best_state, history

    head_started = time.time()
    pseudo_state, pseudo_history = fit_head(
        pseudo_train,
        epochs=PSEUDO_EPOCHS,
        lr=PSEUDO_LR,
        seed=common.SEED,
        initial_state=None,
        valid_indices=pseudo_valid,
    )
    torch.save(pseudo_state, OUTPUT_ROOT / "p5_pseudo_head.pt")
    pd.DataFrame(pseudo_history).to_csv(OUTPUT_ROOT / "p5_pseudo_history.csv", index=False)
    pseudo_head = common.HierarchicalDiagnosisHead().to(device)
    pseudo_head.load_state_dict(pseudo_state, strict=True)
    pseudo_head.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        direct_predictions = torch.sigmoid(
            pseudo_head(
                features[gold_indices],
                feature_mask[gold_indices],
                feature_slots[gold_indices],
                feature_positions[gold_indices],
            )
        ).cpu().numpy()
    direct_metrics, direct_macro = calculate_metrics(labels_np[gold_indices], direct_predictions)
    direct_metrics.to_csv(OUTPUT_ROOT / "p5_direct_expert_metrics.csv", index=False)
    direct_frame = pd.DataFrame({common.ID: train.loc[gold_indices, common.ID].to_numpy()})
    for index, target in enumerate(common.TARGETS):
        direct_frame[f"label__{target}"] = labels_np[gold_indices, index]
        direct_frame[f"pred__{target}"] = direct_predictions[:, index]
    direct_frame.to_csv(OUTPUT_ROOT / "p5_direct_expert_predictions.csv", index=False)
    print({"P5_DIRECT_EXPERT_MACRO_AUC": direct_macro}, flush=True)

    oof = np.full((len(train), len(common.TARGETS)), np.nan, dtype=np.float32)
    gold_fold = np.full(len(gold_indices), -1, dtype=int)
    fold_rows: list[dict[str, object]] = []
    fold_splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (gold_train_pos, gold_valid_pos) in enumerate(fold_splitter.split(gold_indices)):
        valid_index = gold_indices[gold_valid_pos]
        valid_hashes = set(report_hashes[valid_index])
        keep_pseudo = np.asarray(
            [index for index in pseudo_indices if report_hashes[index] not in valid_hashes],
            dtype=int,
        )
        train_index = np.concatenate((keep_pseudo, gold_indices[gold_train_pos]))
        assert not (set(report_hashes[train_index]) & valid_hashes)
        state, history = fit_head(
            train_index,
            epochs=FOLD_EPOCHS,
            lr=FOLD_LR,
            seed=FOLD_SEED + fold,
            initial_state=pseudo_state,
            valid_indices=None,
        )
        head = common.HierarchicalDiagnosisHead().to(device)
        head.load_state_dict(state, strict=True)
        head.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            oof[valid_index] = torch.sigmoid(
                head(
                    features[valid_index],
                    feature_mask[valid_index],
                    feature_slots[valid_index],
                    feature_positions[valid_index],
                )
            ).cpu().numpy()
        torch.save(state, OUTPUT_ROOT / f"p5_fold_{fold}_head.pt")
        pd.DataFrame(history).to_csv(OUTPUT_ROOT / f"p5_fold_{fold}_history.csv", index=False)
        gold_fold[gold_valid_pos] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "pseudo_rows": len(keep_pseudo),
                "expert_train_rows": len(gold_train_pos),
                "expert_valid_rows": len(valid_index),
                "excluded_pseudo_rows": len(pseudo_indices) - len(keep_pseudo),
                "report_hash_overlap": 0,
                "final_loss": history[-1]["train_loss"],
            }
        )
        print({"P5_FOLD_COMPLETE": fold_rows[-1]}, flush=True)
    assert (gold_fold >= 0).all() and np.isfinite(oof[gold_indices]).all()
    pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "p5_fold_summary.csv", index=False)
    oof_metrics, oof_macro = calculate_metrics(labels_np[gold_indices], oof[gold_indices])
    oof_metrics.to_csv(OUTPUT_ROOT / "p5_oof_metrics.csv", index=False)
    oof_frame = pd.DataFrame(
        {common.ID: train.loc[gold_indices, common.ID].to_numpy(), "fold": gold_fold}
    )
    for index, target in enumerate(common.TARGETS):
        oof_frame[f"label__{target}"] = labels_np[gold_indices, index]
        oof_frame[f"pred__{target}"] = oof[gold_indices, index]
    oof_frame.to_csv(OUTPUT_ROOT / "p5_oof_predictions.csv", index=False)
    print({"P5_OOF_MACRO_AUC": oof_macro}, flush=True)

    p2_path = common.find_named_file("p2_oof_predictions.csv", 10_000)
    p2 = pd.read_csv(p2_path, dtype={common.ID: str}).set_index(common.ID).loc[
        oof_frame[common.ID]
    ].reset_index()
    p4_paths = sorted(Path("/kaggle/input").glob("**/p4_fold_*_oof_predictions.csv"))
    assert len(p4_paths) == N_FOLDS, p4_paths
    p4 = pd.concat(
        [pd.read_csv(path, dtype={common.ID: str}) for path in p4_paths], ignore_index=True
    )
    assert p4[common.ID].is_unique and len(p4) == len(gold_indices)
    p4 = p4.set_index(common.ID).loc[oof_frame[common.ID]].reset_index()
    label_columns = [f"label__{target}" for target in common.TARGETS]
    prediction_columns = [f"pred__{target}" for target in common.TARGETS]
    labels_gold = labels_np[gold_indices]
    assert np.array_equal(p2[label_columns].to_numpy(np.float32), labels_gold)
    assert np.array_equal(p4[label_columns].to_numpy(np.float32), labels_gold)
    p2_predictions = p2[prediction_columns].to_numpy(np.float32)
    p4_predictions = p4[prediction_columns].to_numpy(np.float32)
    p5_predictions = oof[gold_indices]
    current_baseline = (rank_columns(p2_predictions) + rank_columns(p4_predictions)) / 2.0
    primary_candidate = 0.75 * current_baseline + 0.25 * rank_columns(p5_predictions)
    diagnostic_three_way = (
        rank_columns(p2_predictions) + rank_columns(p4_predictions) + rank_columns(p5_predictions)
    ) / 3.0
    baseline_metrics, baseline_macro = calculate_metrics(labels_gold, current_baseline)
    primary_metrics, primary_macro = calculate_metrics(labels_gold, primary_candidate)
    diagnostic_metrics, diagnostic_macro = calculate_metrics(labels_gold, diagnostic_three_way)
    baseline_metrics.to_csv(OUTPUT_ROOT / "p5_current_baseline_metrics.csv", index=False)
    primary_metrics.to_csv(OUTPUT_ROOT / "p5_primary_blend_metrics.csv", index=False)
    diagnostic_metrics.to_csv(OUTPUT_ROOT / "p5_three_way_diagnostic_metrics.csv", index=False)
    target_delta = primary_metrics["auc"].to_numpy()[:-1] - baseline_metrics["auc"].to_numpy()[:-1]
    bootstrap = paired_bootstrap(labels_gold, current_baseline, primary_candidate)
    promotion_gate = {
        "macro_delta_at_least_0_003": bool(primary_macro - baseline_macro >= 0.003),
        "bootstrap_probability_at_least_0_80": bool(bootstrap["probability_positive"] >= 0.80),
        "at_least_seven_nonnegative_targets": bool((target_delta >= 0).sum() >= 7),
        "worst_target_above_minus_0_05": bool(target_delta.min() > -0.05),
    }
    promotion_gate["passed"] = all(promotion_gate.values())
    lora_gate = {
        "standalone_macro_at_least_0_84": bool(oof_macro >= 0.84),
        "primary_delta_above_0_001": bool(primary_macro - baseline_macro > 0.001),
        "bootstrap_probability_at_least_0_65": bool(bootstrap["probability_positive"] >= 0.65),
        "runtime_gate": True,
    }
    lora_gate["passed"] = all(lora_gate.values())
    comparison = {
        "p5_standalone_macro_auc": oof_macro,
        "current_p2_p4_macro_auc": baseline_macro,
        "primary_p5_25pct_macro_auc": primary_macro,
        "primary_delta": primary_macro - baseline_macro,
        "diagnostic_equal_three_way_macro_auc": diagnostic_macro,
        "primary_vs_current_bootstrap": bootstrap,
        "nonnegative_target_count": int((target_delta >= 0).sum()),
        "worst_target_delta": float(target_delta.min()),
        "promotion_gate": promotion_gate,
        "lora_gate": lora_gate,
    }
    (OUTPUT_ROOT / "p5_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    comparison_frame = pd.DataFrame({"target": [*common.TARGETS, "MACRO"]})
    comparison_frame["p5_auc"] = oof_metrics["auc"]
    comparison_frame["current_p2_p4_auc"] = baseline_metrics["auc"]
    comparison_frame["primary_p5_25pct_auc"] = primary_metrics["auc"]
    comparison_frame["diagnostic_three_way_auc"] = diagnostic_metrics["auc"]
    comparison_frame["primary_minus_current"] = (
        comparison_frame["primary_p5_25pct_auc"] - comparison_frame["current_p2_p4_auc"]
    )
    comparison_frame.to_csv(OUTPUT_ROOT / "p5_target_comparison.csv", index=False)
    blend_frame = pd.DataFrame({common.ID: oof_frame[common.ID], "fold": gold_fold})
    for index, target in enumerate(common.TARGETS):
        blend_frame[f"label__{target}"] = labels_gold[:, index]
        blend_frame[f"pred__{target}"] = primary_candidate[:, index]
    blend_frame.to_csv(OUTPUT_ROOT / "p5_primary_blend_oof_predictions.csv", index=False)
    print({"P5_COMPARISON": comparison}, flush=True)

    if telemetry is not None:
        telemetry.close()
    stats = [item.cpu().tolist() for item in gathered]
    artifact_names = [
        "p5_orthofoundation_features.npy",
        "p5_feature_mask.npy",
        "p5_feature_slots.npy",
        "p5_feature_positions.npy",
        "p5_feature_studies.csv",
        "p5_pseudo_head.pt",
        "p5_pseudo_history.csv",
        "p5_direct_expert_metrics.csv",
        "p5_direct_expert_predictions.csv",
        "p5_fold_summary.csv",
        "p5_oof_metrics.csv",
        "p5_oof_predictions.csv",
        "p5_current_baseline_metrics.csv",
        "p5_primary_blend_metrics.csv",
        "p5_three_way_diagnostic_metrics.csv",
        "p5_primary_blend_oof_predictions.csv",
        "p5_target_comparison.csv",
        "p5_comparison.json",
        "p5_gpu_telemetry.csv",
        *[f"p5_fold_{fold}_head.pt" for fold in range(N_FOLDS)],
        *[f"p5_fold_{fold}_history.csv" for fold in range(N_FOLDS)],
    ]
    for name in artifact_names:
        assert (OUTPUT_ROOT / name).is_file(), name
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "execution_revision": EXECUTION_REVISION,
        "seeds": {"head": common.SEED, "folds": FOLD_SEED},
        "studies": len(train),
        "pseudo_studies": len(pseudo_indices),
        "expert_studies": len(gold_indices),
        "cache_id": common.CACHE_ID,
        "cache_signature_sha256": common.CACHE_SIGNATURE,
        "checkpoint_sha256": common.MODEL_SHA256,
        "checkpoint_bytes": common.MODEL_BYTES,
        "checkpoint_namespace": checkpoint_namespace,
        "orthofoundation_commit": common.ORTHO_COMMIT,
        "dinov3_commit": common.DINOV3_COMMIT,
        "licenses": ["MIT", "DINOv3 License"],
        "model_parameters": total_parameters,
        "input_size": common.MODEL_SIZE,
        "slices_per_study": common.K_SLICES,
        "slot_quotas": list(common.SLOT_QUOTAS),
        "feature_shape": [len(train), common.K_SLICES, common.N_REGIONS, common.BACKBONE_DIM],
        "head_dim": common.HEAD_DIM,
        "pseudo_epochs": PSEUDO_EPOCHS,
        "fold_epochs": FOLD_EPOCHS,
        "head_batch": HEAD_BATCH,
        "representation_gold_rows": 0,
        "pseudo_report_group_overlap": 0,
        "fold_report_hash_overlap": 0,
        "world_size": world_size,
        "model_load_seconds_by_rank": [row[0] for row in stats],
        "feature_extraction_seconds_by_rank": [row[1] for row in stats],
        "peak_memory_mib_by_rank": [row[2] for row in stats],
        "head_training_minutes": (time.time() - head_started) / 60.0,
        "direct_expert_macro_auc": direct_macro,
        "fold_oof_macro_auc": oof_macro,
        "comparison": comparison,
        "artifacts_sha256": {
            name: common.sha256_file(OUTPUT_ROOT / name) for name in artifact_names
        },
    }
    manifest_path = OUTPUT_ROOT / "p5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        {
            "P5_COMPLETE": True,
            "direct_macro": direct_macro,
            "oof_macro": oof_macro,
            "primary_blend_macro": primary_macro,
            "promotion_gate": promotion_gate,
            "lora_gate": lora_gate,
            "manifest_sha256": common.sha256_file(manifest_path),
        },
        flush=True,
    )
    return 0


def main() -> int:
    if "--worker" not in sys.argv:
        return launch()
    return worker()


if __name__ == "__main__":
    raise SystemExit(main())
