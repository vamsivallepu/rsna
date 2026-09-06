#!/usr/bin/env python3
"""Generate frozen fold-safe P8 noisy-student targets on two Kaggle GPUs."""

from __future__ import annotations

# ruff: noqa: E402 - local offline dependencies must precede ML imports.

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

for dependency_root in (
    Path.cwd() / ".deps",
    Path.cwd().parent / ".deps",
    Path.cwd().parent.parent / ".deps",
):
    if dependency_root.is_dir():
        sys.path.insert(0, str(dependency_root))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


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
FOLD_SEED = 20260831
RUN_ID = "P8-FOLD-SAFE-TEACHER-v1"
ARCH = "coatnet_rmlp_2_rw_384"
FEATURE_DIM = 1024
CACHE_SIZE = 336
MODEL_SIZE = 384
K_EVAL = 32
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
CACHE_ID = "P2-CACHE-v1"
CACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
P2_MANIFEST_SHA256 = "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"
P6_MANIFEST_SHA256 = "885b58c5a9af1bffaac1f52c9703db3538d25f8d9660b036b563b351b1067bce"
WEAK_MANIFEST_SHA256 = "561a8f6a8d645a05ef6f193a2f1dd93db3f146a7ccaa4d16dd109e24b6ec2816"
TRAINING_TARGETS_SHA256 = "ca63cfff863c87286e07a8059fee88f86ff8b3f4b640a98078d62113f6b93f2a"
CONSENSUS_SHA256 = "c152263597c8588894b6b70ef54c4bd3fd517b610ae7177e4f78443b8ed0dfc7"
P4_MANIFEST_SHA256 = {
    0: "6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932",
    1: "9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5",
    2: "93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac",
    3: "9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c",
    4: "e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228",
}
DIRECT_REPLAY_ATOL = 1e-5
P4_REPLAY_MAX_ATOL = 5e-3
P4_REPLAY_MEAN_ATOL = 5e-4
REPORT_WEIGHT = 0.50
IMAGE_WEIGHT = 0.50
IMAGE_COMPONENT_WEIGHT = 1.0 / 3.0
EXPECTED_REPORT_MACRO = 0.8948112944772516
EXPECTED_IMAGE_MACRO = 0.9034874698692111
EXPECTED_RELABEL_MACRO = 0.9209204490507412
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(
    "outputs/p8_teacher_labels_smoke"
)


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def launch_ddp() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P8-A requires exactly two GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    print({"P8_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P8_DDP_PROCESS_COMPLETE": True}, flush=True)
    return 0


def find_root(marker: str, slug: str, local: Path | None = None) -> Path:
    candidates = [
        Path("/kaggle/input/notebooks/vallepuvamsikrishna") / slug,
        *([local] if local is not None else []),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / marker).is_file():
            return candidate.resolve()
    if Path("/kaggle/input").is_dir():
        matches = list(Path("/kaggle/input").glob(f"**/{marker}"))
        if len(matches) == 1:
            return matches[0].parent.resolve()
    raise FileNotFoundError(marker)


def find_train_csv() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"),
        Path("data/raw/train.csv"),
    ]
    return next(path.resolve() for path in candidates if path.is_file())


def prepare_cache_part(part: str) -> dict[str, object]:
    root = find_root(
        f"p2_cache_manifest_part_{part}.json",
        f"rsna-knee-p2-wide-cache-{part}",
        Path(f"kaggle/rsna-p2-wide-cache-{part}/outputs/p2_cache_smoke_{part}"),
    )
    manifest_path = root / f"p2_cache_manifest_part_{part}.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_id"] == CACHE_ID
    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE
    names = {
        "pixels": f"p2_pixels_part_{part}.npy",
        "mask": f"p2_valid_mask_part_{part}.npy",
        "studies": f"p2_studies_part_{part}.csv",
    }
    paths = {key: root / name for key, name in names.items()}
    for path in paths.values():
        assert path.is_file()
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "names": names,
        **paths,
    }


def validate_manifest_artifact(
    root: Path, manifest: dict[str, object], name: str
) -> None:
    expected = manifest["artifacts_sha256"][name]
    assert sha256_file(root / name) == expected, name


class GpuTelemetry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.write_text(
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
                with self.path.open("a", newline="") as handle:
                    writer = csv.writer(handle)
                    for row in rows:
                        writer.writerow([time.time(), *row])
                print({"P8_GPU": rows}, flush=True)
            except Exception as error:
                print({"P8_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)
            self.stop.wait(60)

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


class TargetAttentionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.attention = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.Tanh(),
            nn.Dropout(0.20),
            nn.Linear(256, N_TARGETS),
        )
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, FEATURE_DIM))
        self.target_bias = nn.Parameter(torch.empty(N_TARGETS))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(features)
        scores = self.attention(hidden).transpose(1, 2)
        scores = scores.masked_fill(~mask[:, None, :].bool(), -1e4)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("btk,bkf->btf", attention, hidden)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias


class CoAtNetMIL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(ARCH, pretrained=False)
        self.backbone.reset_classifier(0, global_pool="avg")
        assert self.backbone.num_features == FEATURE_DIM
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=False)
        self.head = TargetAttentionHead()
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1))


class CacheDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        pixels_a: np.ndarray,
        mask_a: np.ndarray,
        pixels_b: np.ndarray,
        mask_b: np.ndarray,
        slot_bounds: np.ndarray,
    ) -> None:
        self.indices = indices
        self.pixels_a = pixels_a
        self.mask_a = mask_a
        self.pixels_b = pixels_b
        self.mask_b = mask_b
        self.slot_bounds = slot_bounds

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        if index < len(self.pixels_a):
            volume = np.asarray(self.pixels_a[index])
            mask = np.asarray(self.mask_a[index]).astype(bool)
        else:
            local = index - len(self.pixels_a)
            volume = np.asarray(self.pixels_b[local])
            mask = np.asarray(self.mask_b[local]).astype(bool)
        available = []
        for start, stop in self.slot_bounds:
            for center in range(int(start) + 1, int(stop) - 1):
                if mask[center - 1 : center + 2].all():
                    available.append(center)
        available_array = np.asarray(available, dtype=np.int16)
        windows = np.zeros((K_EVAL, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        window_mask = np.zeros(K_EVAL, dtype=bool)
        if len(available_array) <= K_EVAL:
            centers = available_array
        else:
            positions = np.round(
                np.linspace(0, len(available_array) - 1, K_EVAL)
            ).astype(int)
            centers = available_array[positions]
        window_mask[: len(centers)] = True
        for output_index, center in enumerate(centers):
            windows[output_index] = volume[int(center) - 1 : int(center) + 2]
        assert window_mask.any()
        return index, windows, window_mask


def load_direct_head(root: Path, device: torch.device) -> TargetAttentionHead:
    state = torch.load(
        root / "p2_coatnet_pseudo_best.pt", map_location="cpu", weights_only=True
    )
    head_state = {
        key.removeprefix("head."): value
        for key, value in state.items()
        if key.startswith("head.")
    }
    del state
    head = TargetAttentionHead().to(device)
    head.load_state_dict(head_state, strict=True)
    head.eval()
    return head


def predict_direct(
    root: Path, local_indices: np.ndarray, device: torch.device
) -> np.ndarray:
    features = np.load(root / "p2_coatnet_features.npy", mmap_mode="r")
    masks = np.load(root / "p2_coatnet_feature_mask.npy", mmap_mode="r")
    head = load_direct_head(root, device)
    output = np.empty((len(local_indices), N_TARGETS), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(local_indices), 128):
            stop = min(start + 128, len(local_indices))
            selected = local_indices[start:stop]
            feature = torch.from_numpy(
                np.asarray(features[selected]).astype(np.float32)
            ).to(device)
            mask = torch.from_numpy(np.asarray(masks[selected]).astype(bool)).to(device)
            output[start:stop] = torch.sigmoid(head(feature, mask)).cpu().numpy()
    del head
    torch.cuda.empty_cache()
    return output


def load_p4_models(roots: dict[int, Path], device: torch.device) -> list[CoAtNetMIL]:
    models = []
    for fold in range(N_FOLDS):
        model = CoAtNetMIL()
        state = torch.load(
            roots[fold] / f"p4_fold_{fold}_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        del state
        model = model.to(device=device, memory_format=torch.channels_last)
        model.eval()
        models.append(model)
    return models


def predict_p4(
    models: list[CoAtNetMIL],
    loader: DataLoader,
    local_count: int,
    device: torch.device,
    phase: str,
) -> np.ndarray:
    output = np.empty((N_FOLDS, local_count, N_TARGETS), dtype=np.float32)
    processed = 0
    started = time.time()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for indices, windows, mask in loader:
            windows = windows.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).bool()
            batch, count = windows.shape[:2]
            pixels = windows.reshape(
                batch * count, 3, CACHE_SIZE, CACHE_SIZE
            ).float().div(255.0)
            pixels = F.interpolate(
                pixels,
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bicubic",
                align_corners=False,
            ).contiguous(memory_format=torch.channels_last)
            pixels = (pixels - models[0].mean) / models[0].std
            stop = processed + batch
            for fold, model in enumerate(models):
                features = model.backbone(pixels).reshape(batch, count, FEATURE_DIM)
                output[fold, processed:stop] = (
                    torch.sigmoid(model.head(features, mask)).float().cpu().numpy()
                )
            processed = stop
            if processed % 50 == 0 or processed == local_count:
                print(
                    {
                        "P8_P4_PROGRESS": {
                            "phase": phase,
                            "device": str(device),
                            "local_studies": processed,
                            "local_total": local_count,
                            "elapsed_minutes": (time.time() - started) / 60.0,
                        }
                    },
                    flush=True,
                )
    assert processed == local_count and np.isfinite(output).all()
    return output


def difference_statistics(
    computed: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    absolute = np.abs(computed.astype(np.float64) - reference.astype(np.float64))
    return {
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
    }


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
    }
    result["MACRO"] = float(np.mean(list(result.values())))
    return result


def validate_sources(
    p2_root: Path,
    p6_root: Path,
    p4_roots: dict[int, Path],
    weak_root: Path,
    cache_parts: list[dict[str, object]],
) -> dict[str, object]:
    p2_manifest_path = p2_root / "p2_manifest.json"
    p6_manifest_path = p6_root / "p6_transfer_manifest.json"
    weak_manifest_path = weak_root / "weak_label_manifest.json"
    assert sha256_file(p2_manifest_path) == P2_MANIFEST_SHA256
    assert sha256_file(p6_manifest_path) == P6_MANIFEST_SHA256
    assert sha256_file(weak_manifest_path) == WEAK_MANIFEST_SHA256
    assert sha256_file(weak_root / "training_targets.csv") == TRAINING_TARGETS_SHA256
    assert sha256_file(weak_root / "consensus_weak_labels.csv") == CONSENSUS_SHA256
    p2_manifest = json.loads(p2_manifest_path.read_text())
    p6_manifest = json.loads(p6_manifest_path.read_text())
    assert p2_manifest["run_id"] == "P2-COATNET-v1"
    assert p6_manifest["run_id"] == "P6-P2-TRANSFER-v1"
    for root, manifest in ((p2_root, p2_manifest), (p6_root, p6_manifest)):
        for name in (
            "p2_coatnet_pseudo_best.pt",
            "p2_coatnet_features.npy",
            "p2_coatnet_feature_mask.npy",
            "p2_coatnet_feature_studies.csv",
            "p2_direct_expert_predictions.csv",
        ):
            validate_manifest_artifact(root, manifest, name)
    p4_manifests = {}
    for fold, root in p4_roots.items():
        manifest_name = f"p4_fold_{fold}_manifest.json"
        manifest_path = root / manifest_name
        assert sha256_file(manifest_path) == P4_MANIFEST_SHA256[fold]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["run_id"] == f"P4-COATNET-E2E-FOLD-{fold}-v1"
        assert manifest["fold"] == fold
        assert manifest["expert_valid_gradient_rows"] == 0
        assert manifest["selection_report_overlap"] == 0
        for name in (
            f"p4_fold_{fold}_best.pt",
            f"p4_fold_{fold}_oof_predictions.csv",
            f"p4_fold_{fold}_split.csv",
        ):
            validate_manifest_artifact(root, manifest, name)
        p4_manifests[str(fold)] = P4_MANIFEST_SHA256[fold]
    for part in cache_parts:
        manifest = part["manifest"]
        names = part["names"]
        assert isinstance(manifest, dict) and isinstance(names, dict)
        for key, name in names.items():
            path = part[key]
            assert isinstance(path, Path)
            assert sha256_file(path) == manifest["files_sha256"][name]
    return {
        "p2_manifest_sha256": P2_MANIFEST_SHA256,
        "p6_manifest_sha256": P6_MANIFEST_SHA256,
        "p4_manifest_sha256": p4_manifests,
        "weak_manifest_sha256": WEAK_MANIFEST_SHA256,
        "training_targets_sha256": TRAINING_TARGETS_SHA256,
        "consensus_sha256": CONSENSUS_SHA256,
        "cache_manifest_sha256": {
            str(part["manifest"]["part"]): sha256_file(part["manifest_path"])
            for part in cache_parts
        },
    }


def main_worker() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2 and torch.cuda.device_count() == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    torch.cuda.reset_peak_memory_stats(device)
    is_primary = rank == 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    telemetry = GpuTelemetry(OUTPUT_ROOT / "p8_teacher_gpu_telemetry.csv") if is_primary else None
    if telemetry is not None:
        telemetry.start()
    started = time.time()

    p2_root = find_root(
        "p2_manifest.json",
        "rsna-knee-p2-coatnet-rmlp2",
        Path("outputs/kaggle/rsna-p2-coatnet-rmlp2-v2"),
    )
    p6_root = find_root(
        "p6_transfer_manifest.json",
        "rsna-knee-p6-p2-transfer",
        Path("outputs/kaggle/rsna-p6-p2-transfer-v1"),
    )
    weak_root = find_root(
        "training_targets.csv",
        "rsna-knee-report-weak-supervision",
        Path("outputs/kaggle/rsna-report-weak-supervision"),
    )
    p4_roots = {
        fold: find_root(
            f"p4_fold_{fold}_manifest.json",
            f"rsna-knee-p4-coatnet-e2e-fold-{fold}",
            Path("outputs/kaggle/rsna-p4-coatnet-e2e-v1") / f"fold-{fold}",
        )
        for fold in range(N_FOLDS)
    }
    part_a = prepare_cache_part("a")
    part_b = prepare_cache_part("b")
    source_hashes = None
    if is_primary:
        source_hashes = validate_sources(
            p2_root, p6_root, p4_roots, weak_root, [part_a, part_b]
        )
        print({"P8_SOURCES_VALIDATED": source_hashes}, flush=True)
    dist.barrier()

    train = pd.read_csv(find_train_csv(), dtype={ID: str})
    targets = pd.read_csv(weak_root / "training_targets.csv", dtype={ID: str})
    consensus = pd.read_csv(
        weak_root / "consensus_weak_labels.csv", dtype={ID: str}
    )
    ids_a = pd.read_csv(part_a["studies"], dtype={ID: str})[ID].tolist()
    ids_b = pd.read_csv(part_b["studies"], dtype={ID: str})[ID].tolist()
    study_ids = ids_a + ids_b
    assert len(study_ids) == 4_407
    assert study_ids == train[ID].tolist() == targets[ID].tolist() == consensus[ID].tolist()
    for root in (p2_root, p6_root):
        assert pd.read_csv(root / "p2_coatnet_feature_studies.csv", dtype={ID: str})[
            ID
        ].tolist() == study_ids

    labels = targets[TARGETS].to_numpy(np.float64)
    is_gold = targets["is_gold"].astype(bool).to_numpy()
    gold_indices = np.where(is_gold)[0]
    assert len(gold_indices) == 58
    expert_fold = np.full(len(study_ids), -1, dtype=np.int8)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, valid_position) in enumerate(splitter.split(gold_indices)):
        expert_fold[gold_indices[valid_position]] = fold
    assert (expert_fold[gold_indices] >= 0).all()

    local_indices = np.arange(rank, len(study_ids), world_size, dtype=np.int64)
    p2_direct = predict_direct(p2_root, local_indices, device)
    p6_direct = predict_direct(p6_root, local_indices, device)
    np.save(OUTPUT_ROOT / f"p8_indices_rank_{rank}.npy", local_indices)
    np.save(OUTPUT_ROOT / f"p8_p2_rank_{rank}.npy", p2_direct)
    np.save(OUTPUT_ROOT / f"p8_p6_rank_{rank}.npy", p6_direct)
    pixels_a = np.load(part_a["pixels"], mmap_mode="r")
    mask_a = np.load(part_a["mask"], mmap_mode="r")
    pixels_b = np.load(part_b["pixels"], mmap_mode="r")
    mask_b = np.load(part_b["mask"], mmap_mode="r")
    slot_bounds = np.load(part_a["root"] / "p2_slot_bounds.npy")
    assert slot_bounds.tolist() == [[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]]
    p4_models = load_p4_models(p4_roots, device)

    # Validate the numerically equivalent batch-of-two replay before paying for
    # all 4,407 studies. The original P4 OOF inference used a per-rank batch of
    # one, so FP16 kernel selection can move low-order probability digits.
    local_preflight_indices = gold_indices[gold_indices % world_size == rank]
    preflight_dataset = CacheDataset(
        local_preflight_indices,
        pixels_a,
        mask_a,
        pixels_b,
        mask_b,
        slot_bounds,
    )
    preflight_loader = DataLoader(
        preflight_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    preflight_p4 = predict_p4(
        p4_models,
        preflight_loader,
        len(local_preflight_indices),
        device,
        "expert_replay_preflight",
    )
    np.save(
        OUTPUT_ROOT / f"p8_preflight_indices_rank_{rank}.npy",
        local_preflight_indices,
    )
    np.save(
        OUTPUT_ROOT / f"p8_preflight_p4_rank_{rank}.npy", preflight_p4
    )
    dist.barrier()

    if is_primary:
        preflight_full_p2 = np.full(
            (len(study_ids), N_TARGETS), np.nan, dtype=np.float32
        )
        preflight_full_p6 = np.full_like(preflight_full_p2, np.nan)
        preflight_full_p4 = np.full(
            (N_FOLDS, len(study_ids), N_TARGETS), np.nan, dtype=np.float32
        )
        for shard_rank in range(world_size):
            indices = np.load(OUTPUT_ROOT / f"p8_indices_rank_{shard_rank}.npy")
            preflight_full_p2[indices] = np.load(
                OUTPUT_ROOT / f"p8_p2_rank_{shard_rank}.npy"
            )
            preflight_full_p6[indices] = np.load(
                OUTPUT_ROOT / f"p8_p6_rank_{shard_rank}.npy"
            )
            preflight_indices = np.load(
                OUTPUT_ROOT / f"p8_preflight_indices_rank_{shard_rank}.npy"
            )
            preflight_full_p4[:, preflight_indices] = np.load(
                OUTPUT_ROOT / f"p8_preflight_p4_rank_{shard_rank}.npy"
            )
        assert np.isfinite(preflight_full_p2[gold_indices]).all()
        assert np.isfinite(preflight_full_p6[gold_indices]).all()
        assert np.isfinite(preflight_full_p4[:, gold_indices]).all()

        p4_differences: dict[str, dict[str, float]] = {}
        for fold in range(N_FOLDS):
            valid_indices = gold_indices[expert_fold[gold_indices] == fold]
            saved = pd.read_csv(
                p4_roots[fold] / f"p4_fold_{fold}_oof_predictions.csv",
                dtype={ID: str},
            ).set_index(ID).loc[np.asarray(study_ids)[valid_indices]]
            reference = saved[
                [f"pred__{target}" for target in TARGETS]
            ].to_numpy(np.float32)
            statistics = difference_statistics(
                preflight_full_p4[fold, valid_indices], reference
            )
            assert statistics["maximum_absolute_error"] <= P4_REPLAY_MAX_ATOL
            assert statistics["mean_absolute_error"] <= P4_REPLAY_MEAN_ATOL
            p4_differences[str(fold)] = statistics
            split = pd.read_csv(
                p4_roots[fold] / f"p4_fold_{fold}_split.csv", dtype={ID: str}
            )
            assert split[ID].tolist() == study_ids
            assert (split.loc[valid_indices, "role"] == "expert_valid").all()

        direct_differences: dict[str, dict[str, float]] = {}
        for source, root, computed in (
            ("P2 direct", p2_root, preflight_full_p2),
            ("P6 direct", p6_root, preflight_full_p6),
        ):
            saved = pd.read_csv(
                root / "p2_direct_expert_predictions.csv", dtype={ID: str}
            ).set_index(ID).loc[np.asarray(study_ids)[gold_indices]]
            reference = saved[
                [f"pred__{target}" for target in TARGETS]
            ].to_numpy(np.float32)
            statistics = difference_statistics(computed[gold_indices], reference)
            assert statistics["maximum_absolute_error"] <= DIRECT_REPLAY_ATOL
            direct_differences[source] = statistics

        replay_preflight = {
            "passed": True,
            "studies": len(gold_indices),
            "inference_batch_per_rank": 2,
            "direct_maximum_absolute_tolerance": DIRECT_REPLAY_ATOL,
            "p4_maximum_absolute_tolerance": P4_REPLAY_MAX_ATOL,
            "p4_mean_absolute_tolerance": P4_REPLAY_MEAN_ATOL,
            "p4_fold_differences": p4_differences,
            "direct_differences": direct_differences,
            "canonical_expert_values_used_for_audit": True,
        }
        (OUTPUT_ROOT / "p8_replay_preflight.json").write_text(
            json.dumps(replay_preflight, indent=2) + "\n"
        )
        print({"P8_REPLAY_PREFLIGHT": replay_preflight}, flush=True)
    dist.barrier()

    dataset = CacheDataset(
        local_indices, pixels_a, mask_a, pixels_b, mask_b, slot_bounds
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    p4_predictions = predict_p4(
        p4_models, loader, len(local_indices), device, "full_teacher_generation"
    )
    np.save(OUTPUT_ROOT / f"p8_p4_rank_{rank}.npy", p4_predictions)
    del p4_models
    torch.cuda.empty_cache()
    memory_mib = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / (1024**2)],
        dtype=torch.float64,
        device=device,
    )
    gathered_memory = [torch.zeros_like(memory_mib) for _ in range(world_size)]
    dist.all_gather(gathered_memory, memory_mib)
    peak_memory_mib_by_rank = [float(value.item()) for value in gathered_memory]
    dist.barrier()

    if is_primary:
        merged_p2 = np.empty((len(study_ids), N_TARGETS), dtype=np.float32)
        merged_p6 = np.empty_like(merged_p2)
        merged_p4 = np.empty((N_FOLDS, len(study_ids), N_TARGETS), dtype=np.float32)
        for shard_rank in range(world_size):
            indices = np.load(OUTPUT_ROOT / f"p8_indices_rank_{shard_rank}.npy")
            merged_p2[indices] = np.load(
                OUTPUT_ROOT / f"p8_p2_rank_{shard_rank}.npy"
            )
            merged_p6[indices] = np.load(
                OUTPUT_ROOT / f"p8_p6_rank_{shard_rank}.npy"
            )
            merged_p4[:, indices] = np.load(
                OUTPUT_ROOT / f"p8_p4_rank_{shard_rank}.npy"
            )
        assert np.isfinite(merged_p2).all()
        assert np.isfinite(merged_p6).all()
        assert np.isfinite(merged_p4).all()

        # The preflight proved that this batch-of-two replay is numerically
        # equivalent to the original batch-of-one OOF inference. Preserve the
        # already-frozen canonical values on the 58 audit rows so teacher audit
        # metrics remain byte-for-byte independent of CUDA kernel selection.
        for fold in range(N_FOLDS):
            valid_indices = gold_indices[expert_fold[gold_indices] == fold]
            saved = pd.read_csv(
                p4_roots[fold] / f"p4_fold_{fold}_oof_predictions.csv",
                dtype={ID: str},
            ).set_index(ID).loc[np.asarray(study_ids)[valid_indices]]
            saved_predictions = saved[
                [f"pred__{target}" for target in TARGETS]
            ].to_numpy(np.float32)
            merged_p4[fold, valid_indices] = saved_predictions

        for root, computed in ((p2_root, merged_p2), (p6_root, merged_p6)):
            saved = pd.read_csv(
                root / "p2_direct_expert_predictions.csv", dtype={ID: str}
            ).set_index(ID).loc[np.asarray(study_ids)[gold_indices]]
            saved_predictions = saved[
                [f"pred__{target}" for target in TARGETS]
            ].to_numpy(np.float32)
            computed[gold_indices] = saved_predictions

        report_predictions = consensus[
            [f"mean__{target}" for target in TARGETS]
        ].to_numpy(np.float64)
        expert_p4 = np.stack(
            [merged_p4[expert_fold[index], index] for index in gold_indices]
        )
        expert_image = IMAGE_COMPONENT_WEIGHT * (
            merged_p2[gold_indices] + expert_p4 + merged_p6[gold_indices]
        )
        expert_relabel = (
            REPORT_WEIGHT * report_predictions[gold_indices]
            + IMAGE_WEIGHT * expert_image
        )
        report_metrics = calculate_metrics(
            labels[gold_indices], report_predictions[gold_indices]
        )
        image_metrics = calculate_metrics(labels[gold_indices], expert_image)
        relabel_metrics = calculate_metrics(labels[gold_indices], expert_relabel)
        assert math.isclose(
            report_metrics["MACRO"], EXPECTED_REPORT_MACRO, abs_tol=1e-12
        )
        assert math.isclose(
            image_metrics["MACRO"], EXPECTED_IMAGE_MACRO, abs_tol=1e-12
        )
        assert math.isclose(
            relabel_metrics["MACRO"], EXPECTED_RELABEL_MACRO, abs_tol=1e-12
        )

        np.save(OUTPUT_ROOT / "p8_p2_direct_predictions.npy", merged_p2)
        np.save(OUTPUT_ROOT / "p8_p6_direct_predictions.npy", merged_p6)
        np.save(OUTPUT_ROOT / "p8_p4_fold_predictions.npy", merged_p4)
        np.save(OUTPUT_ROOT / "p8_expert_fold.npy", expert_fold)
        pd.DataFrame({ID: study_ids}).to_csv(
            OUTPUT_ROOT / "p8_teacher_studies.csv", index=False
        )
        pseudo = ~is_gold
        for fold in range(N_FOLDS):
            image_teacher = IMAGE_COMPONENT_WEIGHT * (
                merged_p2 + merged_p4[fold] + merged_p6
            )
            relabel = REPORT_WEIGHT * report_predictions + IMAGE_WEIGHT * image_teacher
            output_targets = targets.copy()
            output_targets.loc[pseudo, TARGETS] = relabel[pseudo]
            assert np.array_equal(
                output_targets.loc[is_gold, TARGETS].to_numpy(np.float64),
                labels[is_gold],
            )
            assert output_targets.filter(like="weight__").equals(
                targets.filter(like="weight__")
            )
            output_targets.to_csv(
                OUTPUT_ROOT / f"p8_fold_{fold}_training_targets.csv", index=False
            )

        audit = pd.DataFrame(
            {ID: np.asarray(study_ids)[gold_indices], "fold": expert_fold[gold_indices]}
        )
        for target_index, target in enumerate(TARGETS):
            audit[f"label__{target}"] = labels[gold_indices, target_index]
            audit[f"report__{target}"] = report_predictions[gold_indices, target_index]
            audit[f"p2__{target}"] = merged_p2[gold_indices, target_index]
            audit[f"p4__{target}"] = expert_p4[:, target_index]
            audit[f"p6__{target}"] = merged_p6[gold_indices, target_index]
            audit[f"image__{target}"] = expert_image[:, target_index]
            audit[f"relabel__{target}"] = expert_relabel[:, target_index]
        audit.to_csv(OUTPUT_ROOT / "p8_teacher_expert_audit.csv", index=False)

        for shard_rank in range(world_size):
            for prefix in ("indices", "p2", "p6", "p4"):
                (OUTPUT_ROOT / f"p8_{prefix}_rank_{shard_rank}.npy").unlink()
            for prefix in ("indices", "p4"):
                (
                    OUTPUT_ROOT
                    / f"p8_preflight_{prefix}_rank_{shard_rank}.npy"
                ).unlink()
        if telemetry is not None:
            telemetry.close()
        artifact_names = [
            "p8_p2_direct_predictions.npy",
            "p8_p6_direct_predictions.npy",
            "p8_p4_fold_predictions.npy",
            "p8_expert_fold.npy",
            "p8_teacher_studies.csv",
            "p8_teacher_expert_audit.csv",
            "p8_teacher_gpu_telemetry.csv",
            "p8_replay_preflight.json",
            *[f"p8_fold_{fold}_training_targets.csv" for fold in range(N_FOLDS)],
        ]
        assert source_hashes is not None
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "studies": len(study_ids),
            "pseudo_studies": int(pseudo.sum()),
            "expert_studies": int(is_gold.sum()),
            "folds": N_FOLDS,
            "formula": {
                "report_weight": REPORT_WEIGHT,
                "image_weight": IMAGE_WEIGHT,
                "image_component_weight": IMAGE_COMPONENT_WEIGHT,
                "image_components": ["P2 direct", "P4 fold", "P6 direct"],
                "target_specific_adjustments": False,
                "pseudo_weights_changed": False,
                "expert_targets_changed": False,
            },
            "fold_safety": {
                "p2_direct_expert_gradient_rows": 0,
                "p6_direct_expert_gradient_rows": 0,
                "p4_teacher_matches_student_fold": True,
                "expert_validation_teacher_gradient_rows": 0,
                "expert_fold_seed": FOLD_SEED,
            },
            "replay_preflight": replay_preflight,
            "expert_audit_macro_auc": {
                "report": report_metrics["MACRO"],
                "image": image_metrics["MACRO"],
                "relabel": relabel_metrics["MACRO"],
            },
            "source_hashes": source_hashes,
            "cache_id": CACHE_ID,
            "cache_signature_sha256": CACHE_SIGNATURE,
            "execution": {
                "world_size": world_size,
                "visible_gpus": torch.cuda.device_count(),
                "backend": "nccl",
                "model_dtype": "float32 with float16 autocast",
                "elapsed_minutes": (time.time() - started) / 60.0,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "timm": timm.__version__,
                "peak_allocated_mib_by_rank": peak_memory_mib_by_rank,
            },
            "artifacts_sha256": {
                name: sha256_file(OUTPUT_ROOT / name) for name in artifact_names
            },
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        manifest_path = OUTPUT_ROOT / "p8_teacher_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            {
                "P8_TEACHER_COMPLETE": True,
                "elapsed_minutes": manifest["execution"]["elapsed_minutes"],
                "relabel_macro_auc": relabel_metrics["MACRO"],
                "manifest_sha256": sha256_file(manifest_path),
            },
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main() -> int:
    if ON_KAGGLE and "--ddp-worker" not in sys.argv:
        return launch_ddp()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("P8 teacher generation requires the Kaggle two-GPU run")
    return main_worker()


if __name__ == "__main__":
    raise SystemExit(main())
