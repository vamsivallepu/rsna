#!/usr/bin/env python3
"""Run the frozen P2 CoAtNet experiment with two-process Kaggle DDP.

The outer Kaggle process relaunches this file through torchrun.  Each worker owns
one T4, preserving the frozen global study batch of two while avoiding
``nn.DataParallel`` model replication on every forward pass.
"""

# ruff: noqa: E402 - local offline dependencies must be added before ML imports.

from __future__ import annotations

import os as _p8_bootstrap_os

_p8_bootstrap_os.environ["P8_FOLD"] = "3"
_p8_bootstrap_os.environ[
    "P8_TEACHER_MANIFEST_SHA256"
] = "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"

import contextlib
import csv
import gc
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
ARCH = "coatnet_rmlp_2_rw_384"
IMAGENET_MODEL_ID = "timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
IMAGENET_MODEL_SHA256 = (
    "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
)
P6_MODEL_ID = "P6B-FASTMRI-PLUS-LOCALIZATION-v1/e2be22d"
P6_MODEL_SHA256 = "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195"
P6_SOURCE_MANIFEST_SHA256 = (
    "f6fb96ee2434ee8ead9e9e586be7d85bb8cae3fd6c71e059f552a67fb02452ef"
)
CACHE_ID = "P2-CACHE-v1"
EXPECTED_CACHE_SIGNATURE = (
    "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"
)
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
EVAL_BATCH_PER_RANK = 1
GLOBAL_EVAL_STUDY_BATCH = 2
WORKERS_PER_RANK = 2
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3
REP_WEIGHT_DECAY = 0.02
HEAD_FOLD_LR = 2e-4
HEAD_FOLD_WEIGHT_DECAY = 1e-3
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
P6_TRANSFER = os.environ.get("P6_TRANSFER") == "1"
P6_MATCHED_CONTROL = os.environ.get("P6_MATCHED_CONTROL") == "1"
P7_REPORT_AUX = os.environ.get("P7_REPORT_AUX") == "1"
if sum((P6_TRANSFER, P6_MATCHED_CONTROL, P7_REPORT_AUX)) > 1:
    raise ValueError("P6 and P7 execution modes are mutually exclusive")
P4_FOLD = int(os.environ["P4_FOLD"]) if "P4_FOLD" in os.environ else None
P8_FOLD = int(os.environ["P8_FOLD"]) if "P8_FOLD" in os.environ else None
if P4_FOLD is not None and P4_FOLD not in range(N_FOLDS):
    raise ValueError(f"P4_FOLD must be in [0, {N_FOLDS - 1}], got {P4_FOLD}")
if P8_FOLD is not None and P8_FOLD not in range(N_FOLDS):
    raise ValueError(f"P8_FOLD must be in [0, {N_FOLDS - 1}], got {P8_FOLD}")
if P4_FOLD is not None and P8_FOLD is not None:
    raise ValueError("P4_FOLD and P8_FOLD are mutually exclusive")
if P4_FOLD is not None and (P6_TRANSFER or P6_MATCHED_CONTROL or P7_REPORT_AUX):
    raise ValueError("P4 mode cannot be combined with a P6/P7 matched arm")
if P8_FOLD is not None and (P6_TRANSFER or P6_MATCHED_CONTROL or P7_REPORT_AUX):
    raise ValueError("P8 mode cannot be combined with a P6/P7 matched arm")
P4_MODE = P4_FOLD is not None
P8_MODE = P8_FOLD is not None
FOLD_MODE = P4_MODE or P8_MODE
FOLD_INDEX = P4_FOLD if P4_MODE else P8_FOLD
FOLD_PREFIX = "p8" if P8_MODE else "p4"
FOLD_RUN_LABEL = "P8" if P8_MODE else "P4"
FOLD_RUN_ID_PREFIX = "P8-NOISY-STUDENT" if P8_MODE else "P4-COATNET-E2E"
P4_SEED_BASE = 20260904
P8_SEED_BASE = 20260908
RUN_SEED = (
    P8_SEED_BASE + P8_FOLD
    if P8_MODE
    else P4_SEED_BASE + P4_FOLD
    if P4_MODE
    else P2_SEED
)
RUN_LABEL = (
    FOLD_RUN_LABEL
    if FOLD_MODE
    else "P7_REPORT_AUX"
    if P7_REPORT_AUX
    else "P6_TRANSFER"
    if P6_TRANSFER
    else "P6_CONTROL"
    if P6_MATCHED_CONTROL
    else "P2"
)
RUN_ID = (
    "P8-NOISY-STUDENT-v1"
    if P8_MODE
    else "P7-REPORT-IMAGE-AUX-v1"
    if P7_REPORT_AUX
    else "P6-P2-TRANSFER-v1"
    if P6_TRANSFER
    else "P6-P2-MATCHED-CONTROL-v1"
    if P6_MATCHED_CONTROL
    else "P2-COATNET-v1"
)
USES_P6_INITIALIZATION = P6_TRANSFER or P7_REPORT_AUX or P8_MODE
MODEL_ID = P6_MODEL_ID if USES_P6_INITIALIZATION else IMAGENET_MODEL_ID
MODEL_SHA256 = (
    P6_MODEL_SHA256 if USES_P6_INITIALIZATION else IMAGENET_MODEL_SHA256
)
MODEL_FILENAME = (
    "p6_localization_backbone_best.safetensors"
    if USES_P6_INITIALIZATION
    else "model.safetensors"
)
MODEL_LICENSE = (
    "Apache-2.0 backbone; MIT annotations; fastMRI data-use agreement"
    if USES_P6_INITIALIZATION
    else "Apache-2.0"
)
P7_REPORT_EMBEDDING_DIM = 1024
P7_REPORT_TEMPERATURE = 0.07
P7_REPORT_AUX_WEIGHT = 0.10
P7_REPORT_EMBEDDING_RUN_ID = "P7-REPORT-EMBEDDINGS-v1"
P7_REPORT_EMBEDDINGS_MANIFEST_SHA256 = os.environ.get(
    "P7_REPORT_EMBEDDINGS_MANIFEST_SHA256", ""
)
P8_TEACHER_MANIFEST_SHA256 = os.environ.get("P8_TEACHER_MANIFEST_SHA256", "")
EXECUTION_REVISION = (
    "p8_noisy_student_ddp_t4_v1"
    if P8_MODE
    else "p7_report_aux_ddp_t4_v1"
    if P7_REPORT_AUX
    else "ddp_t4_v3"
)
P4_SPEED_SMOKE = os.environ.get("P4_SPEED_SMOKE") == "1"
P8_SPEED_SMOKE = os.environ.get("P8_SPEED_SMOKE") == "1"
if P4_SPEED_SMOKE and not P4_MODE:
    raise ValueError("P4_SPEED_SMOKE requires P4_FOLD")
if P8_SPEED_SMOKE and not P8_MODE:
    raise ValueError("P8_SPEED_SMOKE requires P8_FOLD")
if P4_SPEED_SMOKE and P8_SPEED_SMOKE:
    raise ValueError("P4 and P8 speed-smoke modes are mutually exclusive")

ON_KAGGLE = Path("/kaggle/working").is_dir()
FULL_RUN = ON_KAGGLE and torch.cuda.is_available()
SPEED_SMOKE = os.environ.get("P2_SPEED_SMOKE") == "1"
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/p2_coatnet_smoke")


def launch_ddp() -> int:
    """Relaunch the Kaggle script with one process per visible GPU."""
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(
            f"{RUN_LABEL} requires exactly two Kaggle GPUs; observed {visible}"
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    print({f"{RUN_LABEL}_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({f"{RUN_LABEL}_DDP_PROCESS_COMPLETE": True}, flush=True)
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
    kaggle_direct = {
        "train.csv": Path(
            "/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"
        ),
        "training_targets.csv": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-report-weak-supervision/training_targets.csv"
        ),
        "model.safetensors": Path(
            "/kaggle/input/datasets/vallepuvamsikrishna/"
            "rsna-coatnet-rmlp2-imagenet-init/model.safetensors"
        ),
        "p6_localization_backbone_best.safetensors": Path(
            "/kaggle/input/datasets/vallepuvamsikrishna/"
            "rsna-p6-fastmri-localization-init/"
            "p6_localization_backbone_best.safetensors"
        ),
        "p6_localization_training.json": Path(
            "/kaggle/input/datasets/vallepuvamsikrishna/"
            "rsna-p6-fastmri-localization-init/p6_localization_training.json"
        ),
        "p7_report_embeddings_manifest.json": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p7-report-embeddings/p7_report_embeddings_manifest.json"
        ),
        "p7_bge_m3_report_embeddings.npy": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p7-report-embeddings/p7_bge_m3_report_embeddings.npy"
        ),
        "p7_bge_m3_report_studies.csv": Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p7-report-embeddings/p7_bge_m3_report_studies.csv"
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
    direct: list[Path] = [kaggle_direct[filename]] if filename in kaggle_direct else []
    for root in candidate_roots():
        direct.extend(
            [
                root / filename,
                root / "data/raw" / filename,
                root / "outputs/kaggle/rsna-report-weak-supervision" / filename,
                root
                / "kaggle/rsna-p2-wide-cache-a/outputs/p2_cache_smoke_a"
                / filename,
            ]
        )
    for path in direct:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for path in kaggle_input.glob(f"**/{filename}"):
            if path.is_file() and path.stat().st_size >= minimum_bytes:
                return path.resolve()
    raise FileNotFoundError(filename)


def find_p8_teacher_file(filename: str, minimum_bytes: int = 1) -> Path:
    direct = [
        Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p8-teacher-labels"
        )
        / filename,
        Path("outputs/kaggle/rsna-p8-teacher-labels-v1") / filename,
    ]
    for path in direct:
        if path.is_file() and path.stat().st_size >= minimum_bytes:
            return path.resolve()
    if Path("/kaggle/input").is_dir():
        matches = [
            path
            for path in Path("/kaggle/input").glob(f"**/{filename}")
            if path.is_file() and path.stat().st_size >= minimum_bytes
        ]
        if len(matches) == 1:
            return matches[0].resolve()
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
        futures = {
            pool.submit(sha256_file, path): (path, expected) for path, expected in jobs
        }
        for future in as_completed(futures):
            path, expected = futures[future]
            observed = future.result()
            assert observed == expected, f"{path.name} hash mismatch"
            print({"P2_INPUT_HASH_OK": path.name}, flush=True)
    print({"P2_INPUT_VALIDATION_MINUTES": (time.time() - started) / 60.0}, flush=True)


def validate_p6_source_manifest(path: Path) -> dict[str, object]:
    assert sha256_file(path) == P6_SOURCE_MANIFEST_SHA256
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "validated"
    assert manifest["experiment_id"] == "P6B-FASTMRI-PLUS-LOCALIZATION-v1"
    assert manifest["code_commit"] == "e2be22d"
    assert manifest["selection"]["best_epoch"] == 2
    assert manifest["artifacts"]["backbone_sha256"] == P6_MODEL_SHA256
    assert manifest["data"]["train_volumes"] == 973
    assert manifest["data"]["validation_volumes"] == 199
    assert manifest["data"]["volume_overlap"] == 0
    return manifest


def validate_p8_teacher_manifest(
    manifest_path: Path, target_path: Path, fold: int
) -> dict[str, object]:
    assert P8_TEACHER_MANIFEST_SHA256
    assert sha256_file(manifest_path) == P8_TEACHER_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "P8-FOLD-SAFE-TEACHER-v1"
    assert manifest["studies"] == 4_407
    assert manifest["pseudo_studies"] == 4_349
    assert manifest["expert_studies"] == 58
    assert manifest["folds"] == N_FOLDS
    assert manifest["formula"] == {
        "report_weight": 0.5,
        "image_weight": 0.5,
        "image_component_weight": 1.0 / 3.0,
        "image_components": ["P2 direct", "P4 fold", "P6 direct"],
        "target_specific_adjustments": False,
        "pseudo_weights_changed": False,
        "expert_targets_changed": False,
    }
    assert manifest["fold_safety"] == {
        "p2_direct_expert_gradient_rows": 0,
        "p6_direct_expert_gradient_rows": 0,
        "p4_teacher_matches_student_fold": True,
        "expert_validation_teacher_gradient_rows": 0,
        "expert_fold_seed": FOLD_SEED,
    }
    assert manifest["expert_audit_macro_auc"]["relabel"] == 0.9209204490507412
    artifact = f"p8_fold_{fold}_training_targets.csv"
    assert target_path.name == artifact
    assert sha256_file(target_path) == manifest["artifacts_sha256"][artifact]
    return manifest


def validate_p7_report_embeddings(
    manifest_path: Path,
    embeddings_path: Path,
    studies_path: Path,
    study_ids: list[str],
) -> tuple[dict[str, object], np.ndarray]:
    assert P7_REPORT_EMBEDDINGS_MANIFEST_SHA256
    assert sha256_file(manifest_path) == P7_REPORT_EMBEDDINGS_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == P7_REPORT_EMBEDDING_RUN_ID
    assert manifest["studies"] == manifest["unique_studies"] == 4_407
    assert manifest["nonempty_reports"] == 4_407
    assert manifest["model"]["source"] == (
        "yethukmutt/bge-m3/Transformers/m3/1"
    )
    assert manifest["model"]["upstream"] == "BAAI/bge-m3"
    assert manifest["model"]["license"] == "MIT"
    assert manifest["model"]["embedding_dimension"] == P7_REPORT_EMBEDDING_DIM
    assert manifest["tokens"]["truncated_reports"] == 0
    expected_artifacts = manifest["artifacts_sha256"]
    assert sha256_file(embeddings_path) == expected_artifacts[embeddings_path.name]
    assert sha256_file(studies_path) == expected_artifacts[studies_path.name]
    embedding_studies = pd.read_csv(studies_path, dtype={ID: str})
    assert embedding_studies.columns.tolist() == [ID]
    assert embedding_studies[ID].tolist() == study_ids
    embeddings = np.load(embeddings_path, mmap_mode="r")
    assert embeddings.shape == (4_407, P7_REPORT_EMBEDDING_DIM)
    assert embeddings.dtype == np.float16 and np.isfinite(embeddings).all()
    norms = np.linalg.norm(np.asarray(embeddings).astype(np.float32), axis=1)
    assert np.max(np.abs(norms - 1.0)) < 0.002
    return manifest, embeddings


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
    """Return a version-independent, group-pure split in original row order."""
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups, dtype=str)
    assert len(indices) == len(groups) and 0.0 < valid_fraction < 1.0
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
                rows = list(
                    csv.reader(result.stdout.splitlines(), skipinitialspace=True)
                )
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
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
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
        if P7_REPORT_AUX:
            self.report_projection = nn.Sequential(
                nn.LayerNorm(FEATURE_DIM),
                nn.Linear(FEATURE_DIM, P7_REPORT_EMBEDDING_DIM),
                nn.GELU(),
                nn.Linear(P7_REPORT_EMBEDDING_DIM, P7_REPORT_EMBEDDING_DIM),
            )
        self.channels_last = channels_last
        self.register_buffer("mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(MODEL_STD).view(1, 3, 1, 1))

    def encode(self, windows: torch.Tensor, augment: bool = False) -> torch.Tensor:
        batch, count = windows.shape[:2]
        pixels = (
            windows.reshape(batch * count, 3, CACHE_SIZE, CACHE_SIZE).float().div(255.0)
        )
        pixels = F.interpolate(
            pixels, size=(MODEL_SIZE, MODEL_SIZE), mode="bicubic", align_corners=False
        )
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        if augment:
            gain = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(
                0.90, 1.10
            )
            offset = torch.empty((len(pixels), 1, 1, 1), device=pixels.device).uniform_(
                -0.05, 0.05
            )
            pixels = (pixels * gain + offset).clamp(0.0, 1.0)
        pixels = (pixels - self.mean) / self.std
        return self.backbone(pixels).reshape(batch, count, FEATURE_DIM)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        augment: bool = False,
        return_features: bool = False,
        return_report_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(windows, augment=augment)
        if return_features:
            return features
        logits = self.head(features, mask)
        if not return_report_embedding:
            return logits
        if not P7_REPORT_AUX:
            raise RuntimeError("report embedding requested outside P7 mode")
        valid = mask[:, :, None].to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        embedding = F.normalize(self.report_projection(pooled).float(), dim=-1)
        return logits, embedding


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


def p7_multi_positive_infonce(
    image_embeddings: torch.Tensor,
    study_indices: np.ndarray,
    report_bank: torch.Tensor,
    bank_group_ids: torch.Tensor,
    all_group_ids: torch.Tensor,
) -> torch.Tensor:
    """Duplicate-aware image-to-report InfoNCE over the frozen report bank."""
    assert P7_REPORT_AUX and len(image_embeddings) == len(study_indices)
    indices = torch.as_tensor(study_indices, dtype=torch.long, device=report_bank.device)
    query_groups = all_group_ids[indices]
    positive = query_groups[:, None] == bank_group_ids[None, :]
    assert positive.any(dim=1).all()
    logits = (
        image_embeddings.to(report_bank.dtype) @ report_bank.transpose(0, 1)
    ).float() / P7_REPORT_TEMPERATURE
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    raw = (denominator - numerator).mean()
    return raw / math.log(report_bank.shape[0])


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
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_primary = rank == 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    telemetry_name = (
        f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_gpu_telemetry.csv"
        if FOLD_MODE
        else "p2_gpu_telemetry.csv"
    )
    telemetry = (
        GpuTelemetry(OUTPUT_ROOT / telemetry_name) if FULL_RUN and is_primary else None
    )
    if telemetry is not None:
        telemetry.start()

    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RUN_SEED)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    fold_speed_smoke = P4_SPEED_SMOKE or P8_SPEED_SMOKE
    rep_epochs = (
        REP_EPOCHS if FULL_RUN and not SPEED_SMOKE and not fold_speed_smoke else 1
    )
    head_epochs = HEAD_EPOCHS if FULL_RUN else 1
    k_train = K_TRAIN if FULL_RUN else 1
    k_eval = K_EVAL if FULL_RUN else 1
    grad_accum = GRAD_ACCUM if FULL_RUN else 1
    eval_batch_per_rank = EVAL_BATCH_PER_RANK if FULL_RUN else 1
    workers = WORKERS_PER_RANK if FULL_RUN else 0

    train_path = find_named_file("train.csv", 100_000)
    target_path = (
        find_p8_teacher_file(f"p8_fold_{P8_FOLD}_training_targets.csv", 100_000)
        if P8_MODE
        else find_named_file("training_targets.csv", 100_000)
    )
    model_path = find_named_file(MODEL_FILENAME, 250_000_000)
    p6_source_manifest_path = (
        find_named_file("p6_localization_training.json", 1_000)
        if USES_P6_INITIALIZATION
        else None
    )
    p8_teacher_manifest_path = (
        find_p8_teacher_file("p8_teacher_manifest.json", 1_000)
        if P8_MODE
        else None
    )
    p7_report_manifest_path = (
        find_named_file("p7_report_embeddings_manifest.json", 1_000)
        if P7_REPORT_AUX
        else None
    )
    p7_report_embeddings_path = (
        find_named_file("p7_bge_m3_report_embeddings.npy", 1_000_000)
        if P7_REPORT_AUX
        else None
    )
    p7_report_studies_path = (
        find_named_file("p7_bge_m3_report_studies.csv", 100_000)
        if P7_REPORT_AUX
        else None
    )
    part_a = prepare_cache_part("a")
    part_b = prepare_cache_part("b") if FULL_RUN else None
    parts = [part_a, *([part_b] if part_b is not None else [])]
    if is_primary:
        if SPEED_SMOKE:
            assert sha256_file(model_path) == MODEL_SHA256
            print(
                {"P2_SPEED_SMOKE_INPUTS": "manifests_and_model_validated"}, flush=True
            )
        else:
            validate_immutable_inputs(model_path, parts)
            if p6_source_manifest_path is not None:
                validate_p6_source_manifest(p6_source_manifest_path)
                print(
                    {
                        (
                            "P7_P6_SOURCE_VALIDATED"
                            if P7_REPORT_AUX
                            else "P8_P6_SOURCE_VALIDATED"
                            if P8_MODE
                            else "P6_TRANSFER_SOURCE_VALIDATED"
                        ): True,
                        "manifest_sha256": P6_SOURCE_MANIFEST_SHA256,
                        "checkpoint_sha256": P6_MODEL_SHA256,
                    },
                    flush=True,
                )
            if P8_MODE:
                assert P8_FOLD is not None and p8_teacher_manifest_path is not None
                validate_p8_teacher_manifest(
                    p8_teacher_manifest_path, target_path, P8_FOLD
                )
                print(
                    {
                        "P8_TEACHER_TARGET_VALIDATED": True,
                        "fold": P8_FOLD,
                        "manifest_sha256": P8_TEACHER_MANIFEST_SHA256,
                        "target_sha256": sha256_file(target_path),
                    },
                    flush=True,
                )
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
    ids_b = (
        pd.read_csv(part_b["studies"], dtype={ID: str})[ID].tolist() if part_b else []
    )
    study_ids = ids_a + ids_b
    if FULL_RUN:
        assert len(study_ids) == 4_407 and study_ids == train_all[ID].tolist()
    else:
        assert len(study_ids) == 1
    metadata = train_all.set_index(ID).loc[study_ids].reset_index()
    targets = targets_all.set_index(ID).loc[study_ids].reset_index()
    assert metadata[ID].tolist() == targets[ID].tolist() == study_ids
    assert targets[TARGETS].notna().all().all()

    p7_report_manifest: dict[str, object] | None = None
    p7_report_embeddings: np.ndarray | None = None
    if P7_REPORT_AUX:
        assert FULL_RUN
        assert p7_report_manifest_path is not None
        assert p7_report_embeddings_path is not None
        assert p7_report_studies_path is not None
        if is_primary:
            p7_report_manifest, p7_report_embeddings = (
                validate_p7_report_embeddings(
                    p7_report_manifest_path,
                    p7_report_embeddings_path,
                    p7_report_studies_path,
                    study_ids,
                )
            )
            print(
                {
                    "P7_REPORT_EMBEDDINGS_VALIDATED": True,
                    "manifest_sha256": P7_REPORT_EMBEDDINGS_MANIFEST_SHA256,
                },
                flush=True,
            )
        if distributed:
            dist.barrier()
        if p7_report_embeddings is None:
            p7_report_embeddings = np.load(
                p7_report_embeddings_path, mmap_mode="r"
            )

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
    p4_expert_train_indices = np.empty(0, dtype=int)
    p4_expert_valid_indices = np.empty(0, dtype=int)
    p4_dropped_indices = np.empty(0, dtype=int)
    p4_split_roles = np.full(len(study_ids), "", dtype=object)
    p7_report_bank_indices = np.empty(0, dtype=int)
    if FULL_RUN:
        assert len(gold_indices) == 58 and len(pseudo_indices) == 4_349
        if FOLD_MODE:
            adapt_indices, pseudo_valid_indices = stable_group_split(
                pseudo_indices,
                report_hashes[pseudo_indices],
                valid_fraction=0.10,
                seed=P2_SEED,
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=0.10, random_state=P2_SEED
            )
            train_pos, valid_pos = next(
                splitter.split(pseudo_indices, groups=report_hashes[pseudo_indices])
            )
            adapt_indices = pseudo_indices[train_pos]
            pseudo_valid_indices = pseudo_indices[valid_pos]
        assert not (
            set(report_hashes[adapt_indices]) & set(report_hashes[pseudo_valid_indices])
        )
        if FOLD_MODE:
            assert FOLD_INDEX is not None
            folds = list(
                KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(
                    gold_indices
                )
            )
            expert_train_pos, expert_valid_pos = folds[FOLD_INDEX]
            p4_expert_train_indices = gold_indices[expert_train_pos]
            p4_expert_valid_indices = gold_indices[expert_valid_pos]
            expert_valid_hashes = set(report_hashes[p4_expert_valid_indices])
            expert_train_hashes = set(report_hashes[p4_expert_train_indices])

            base_pseudo_train = adapt_indices.copy()
            base_pseudo_valid = pseudo_valid_indices.copy()
            base_pseudo_valid_set = set(base_pseudo_valid.tolist())
            p4_pseudo_train = np.asarray(
                [
                    index
                    for index in base_pseudo_train
                    if report_hashes[index] not in expert_valid_hashes
                ],
                dtype=int,
            )
            p4_pseudo_valid = np.asarray(
                [
                    index
                    for index in base_pseudo_valid
                    if report_hashes[index] not in expert_valid_hashes
                    and report_hashes[index] not in expert_train_hashes
                ],
                dtype=int,
            )

            p4_split_roles[base_pseudo_train] = "pseudo_train"
            p4_split_roles[base_pseudo_valid] = "pseudo_valid"
            p4_split_roles[p4_expert_train_indices] = "expert_train"
            p4_split_roles[p4_expert_valid_indices] = "expert_valid"
            for index in pseudo_indices:
                if report_hashes[index] in expert_valid_hashes:
                    p4_split_roles[index] = "excluded_expert_valid_report_twin"
                elif (
                    index in base_pseudo_valid_set
                    and report_hashes[index] in expert_train_hashes
                ):
                    p4_split_roles[index] = "excluded_pseudo_valid_expert_train_twin"

            adapt_indices = np.concatenate(
                [p4_pseudo_train, p4_expert_train_indices]
            ).astype(int)
            pseudo_valid_indices = p4_pseudo_valid
            if len(adapt_indices) % world_size:
                pseudo_positions = np.where(~is_gold[adapt_indices])[0]
                assert len(pseudo_positions) > 0
                drop_position = int(pseudo_positions[-1])
                p4_dropped_indices = adapt_indices[drop_position : drop_position + 1]
                p4_split_roles[p4_dropped_indices] = "pseudo_ddp_balance_drop"
                adapt_indices = np.delete(adapt_indices, drop_position)

            assert len(p4_expert_valid_indices) in {11, 12}
            assert len(p4_expert_train_indices) + len(p4_expert_valid_indices) == 58
            assert len(pseudo_valid_indices) > 400
            assert not (set(report_hashes[adapt_indices]) & expert_valid_hashes)
            assert not (
                set(report_hashes[pseudo_valid_indices])
                & set(report_hashes[adapt_indices])
            )
            assert not (set(report_hashes[pseudo_valid_indices]) & expert_valid_hashes)
            assert (p4_split_roles != "").all()
            if fold_speed_smoke:
                smoke_pseudo_train = adapt_indices[~is_gold[adapt_indices]][:62]
                smoke_expert_train = p4_expert_train_indices[:2]
                adapt_indices = np.concatenate(
                    [smoke_pseudo_train, smoke_expert_train]
                ).astype(int)
                pseudo_valid_indices = pseudo_valid_indices[:8]
                assert len(adapt_indices) == 64
                assert int(is_gold[adapt_indices].sum()) == 2
                assert len(pseudo_valid_indices) == 8
                assert not (set(report_hashes[adapt_indices]) & expert_valid_hashes)
                assert not (
                    set(report_hashes[pseudo_valid_indices])
                    & set(report_hashes[adapt_indices])
                )
        if P7_REPORT_AUX:
            p7_report_bank_indices = adapt_indices.copy()
            assert not is_gold[p7_report_bank_indices].any()
            assert not (
                set(report_hashes[p7_report_bank_indices])
                & set(report_hashes[pseudo_valid_indices])
            )
        if SPEED_SMOKE:
            # Preserve the full tensor shapes and accumulation path while keeping
            # the remote diagnostic short enough to stream and inspect live.
            adapt_indices = adapt_indices[:64]
            pseudo_valid_indices = pseudo_valid_indices[:8]
        assert len(adapt_indices) % world_size == 0
    else:
        adapt_indices = pseudo_valid_indices = np.arange(len(study_ids))

    p7_report_bank: torch.Tensor | None = None
    p7_bank_group_ids: torch.Tensor | None = None
    p7_all_group_ids: torch.Tensor | None = None
    p7_report_bank_unique_groups = 0
    if P7_REPORT_AUX:
        assert p7_report_embeddings is not None
        assert len(p7_report_bank_indices) > 3_800
        unique_hashes = {
            value: index
            for index, value in enumerate(sorted(set(report_hashes.tolist())))
        }
        group_ids = np.asarray(
            [unique_hashes[value] for value in report_hashes], dtype=np.int64
        )
        p7_report_bank = torch.from_numpy(
            np.asarray(p7_report_embeddings[p7_report_bank_indices]).copy()
        ).to(device=device, dtype=torch.float16)
        p7_bank_group_ids = torch.from_numpy(
            group_ids[p7_report_bank_indices]
        ).to(device=device)
        p7_all_group_ids = torch.from_numpy(group_ids).to(device=device)
        assert torch.isfinite(p7_report_bank).all()
        p7_report_bank_unique_groups = len(
            np.unique(group_ids[p7_report_bank_indices])
        )
        if is_primary:
            np.save(OUTPUT_ROOT / "p7_report_bank_indices.npy", p7_report_bank_indices)
            np.save(OUTPUT_ROOT / "p7_report_group_ids.npy", group_ids)
            print(
                {
                    "P7_REPORT_BANK_READY": True,
                    "rows": len(p7_report_bank_indices),
                    "unique_report_groups": p7_report_bank_unique_groups,
                    "expert_rows": int(is_gold[p7_report_bank_indices].sum()),
                    "selection_report_overlap": 0,
                },
                flush=True,
            )

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
                if mask[center - 1 : center + 2].all():
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
            selected[: len(available)] = available
            valid = np.zeros(count, dtype=bool)
            valid[: len(available)] = True
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
            seed=RUN_SEED + 10_007 * epoch + index,
        )
        windows = np.zeros((count, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
        for out_index, center in enumerate(centers):
            if window_mask[out_index]:
                windows[out_index] = volume[int(center) - 1 : int(center) + 2]
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
        print(
            {
                "run_label": RUN_LABEL,
                "p4_fold": P4_FOLD,
                "p8_fold": P8_FOLD,
                "run_seed": RUN_SEED,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "timm": timm.__version__,
                "device": str(device),
                "full_run": FULL_RUN,
                "visible_gpus": torch.cuda.device_count(),
                "world_size": world_size,
                "execution_revision": EXECUTION_REVISION,
                "speed_smoke": SPEED_SMOKE,
                "p4_speed_smoke": P4_SPEED_SMOKE,
                "p8_speed_smoke": P8_SPEED_SMOKE,
                "p7_report_aux": P7_REPORT_AUX,
                "p7_report_temperature": (
                    P7_REPORT_TEMPERATURE if P7_REPORT_AUX else None
                ),
                "p7_report_aux_weight": P7_REPORT_AUX_WEIGHT if P7_REPORT_AUX else None,
                "p7_report_bank_rows": (
                    len(p7_report_bank_indices) if P7_REPORT_AUX else 0
                ),
                "gradient_checkpointing": False,
                "channels_last": device.type == "cuda",
                "cudnn_benchmark": device.type == "cuda",
                "studies": len(study_ids),
                "pseudo_train": len(adapt_indices),
                "pseudo_valid": len(pseudo_valid_indices),
                "gold": len(gold_indices),
                "expert_train": len(p4_expert_train_indices) if FOLD_MODE else 0,
                "expert_valid": len(p4_expert_valid_indices) if FOLD_MODE else 0,
                "ddp_balance_drops": len(p4_dropped_indices) if FOLD_MODE else 0,
            },
            flush=True,
        )

    channels_last = device.type == "cuda"
    model = CoAtNetMIL(model_path, channels_last=channels_last)
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if P7_REPORT_AUX:
        assert 75_000_000 < parameter_count < 79_000_000
    else:
        assert 73_000_000 < parameter_count < 76_000_000
    train_model: nn.Module
    if distributed:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            # PyTorch 2.10's static reducer is incompatible with the no_sync()
            # accumulation pattern used to preserve the frozen effective batch.
            gradient_as_bucket_view=False,
            static_graph=False,
        )
        print(
            {
                f"{RUN_LABEL}_DDP_READY": True,
                "rank": rank,
                "device": str(device),
                "parameters": parameter_count,
            },
            flush=True,
        )
    else:
        train_model = model

    optimizer_groups = [
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ]
    if P7_REPORT_AUX:
        optimizer_groups.append(
            {"params": model.report_projection.parameters(), "lr": HEAD_LR}
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=REP_WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(adapt_indices) / GLOBAL_STUDY_BATCH) / grad_accum
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            BACKBONE_LR,
            HEAD_LR,
            *([HEAD_LR] if P7_REPORT_AUX else []),
        ],
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
            batch_size=eval_batch_per_rank,
        )
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
        epoch_started = time.time()
        generator = torch.Generator().manual_seed(RUN_SEED + epoch)
        permutation = torch.randperm(len(adapt_indices), generator=generator).numpy()
        ordered = adapt_indices[permutation]
        local_indices = ordered[rank::world_size]
        loader = make_loader(StudyDataset(local_indices, k_train, True, epoch))
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = torch.zeros(2, dtype=torch.float64, device=device)
        window_totals = torch.zeros(2, dtype=torch.float64, device=device)
        epoch_report_totals = torch.zeros(2, dtype=torch.float64, device=device)
        window_report_totals = torch.zeros(2, dtype=torch.float64, device=device)
        for step, batch in enumerate(loader):
            should_update = (step + 1) % grad_accum == 0 or step + 1 == len(loader)
            sync_context = (
                contextlib.nullcontext()
                if not distributed or should_update
                else train_model.no_sync()  # type: ignore[union-attr]
            )
            with sync_context:
                indices, windows, window_mask, labels, label_weights = move_batch(batch)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output = train_model(
                        windows,
                        window_mask,
                        augment=True,
                        return_report_embedding=P7_REPORT_AUX,
                    )
                    if P7_REPORT_AUX:
                        assert isinstance(output, tuple)
                        logits, image_embeddings = output
                        assert p7_report_bank is not None
                        assert p7_bank_group_ids is not None
                        assert p7_all_group_ids is not None
                        report_loss = p7_multi_positive_infonce(
                            image_embeddings,
                            indices,
                            p7_report_bank,
                            p7_bank_group_ids,
                            p7_all_group_ids,
                        )
                    else:
                        assert isinstance(output, torch.Tensor)
                        logits = output
                        report_loss = torch.zeros((), device=device)
                    local_num, local_den = weighted_terms(logits, labels, label_weights)
                    global_den = local_den.detach().clone()
                    if distributed:
                        dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
                    # DDP averages gradients. Multiplying by world_size makes this
                    # exactly the old global weighted loss across the two studies.
                    classification_loss = (
                        local_num * world_size / global_den.clamp_min(1.0)
                    )
                    loss = classification_loss + P7_REPORT_AUX_WEIGHT * report_loss
                scaler.scale(loss / grad_accum).backward()
            epoch_totals += torch.stack(
                [local_num.detach().double(), local_den.detach().double()]
            )
            window_totals += torch.stack(
                [local_num.detach().double(), local_den.detach().double()]
            )
            report_totals = torch.stack(
                [
                    report_loss.detach().double() * len(indices),
                    torch.as_tensor(len(indices), dtype=torch.float64, device=device),
                ]
            )
            epoch_report_totals += report_totals
            window_report_totals += report_totals
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
                logged_report = window_report_totals.clone()
                if P7_REPORT_AUX and distributed:
                    dist.all_reduce(logged_report, op=dist.ReduceOp.SUM)
                if is_primary:
                    message = (
                        f"{RUN_LABEL} train epoch {epoch + 1}/{rep_epochs} "
                        f"step {step + 1}/{len(loader)} "
                        f"bce={(logged[0] / logged[1].clamp_min(1.0)).item():.5f}"
                    )
                    if P7_REPORT_AUX:
                        message += (
                            " report_nce="
                            f"{(logged_report[0] / logged_report[1].clamp_min(1.0)).item():.5f}"
                        )
                    print(message, flush=True)
                window_totals.zero_()
                window_report_totals.zero_()
            if SPEED_SMOKE and is_primary and (step + 1) % 4 == 0:
                print(
                    {
                        "P2_SPEED_SMOKE_PROGRESS": {
                            "step": step + 1,
                            "steps": len(loader),
                            "elapsed_seconds": time.time() - epoch_started,
                        }
                    },
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_training_seconds = time.time() - epoch_started
        global_epoch_totals = epoch_totals.clone()
        global_report_totals = epoch_report_totals.clone()
        if distributed:
            dist.all_reduce(global_epoch_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_report_totals, op=dist.ReduceOp.SUM)
        train_loss = float(
            (global_epoch_totals[0] / global_epoch_totals[1].clamp_min(1.0)).item()
        )
        report_train_loss = float(
            (global_report_totals[0] / global_report_totals[1].clamp_min(1.0)).item()
        )
        validation_started = time.time()
        valid_loss = validation_loss(pseudo_valid_indices)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds = time.time() - validation_started
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_report_contrastive": report_train_loss if P7_REPORT_AUX else 0.0,
            "train_combined_loss": (
                train_loss + P7_REPORT_AUX_WEIGHT * report_train_loss
            ),
            "pseudo_valid_weighted_bce": valid_loss,
            "elapsed_minutes": (time.time() - training_started) / 60.0,
            "training_seconds": epoch_training_seconds,
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
            print(
                {f"{RUN_LABEL}_EPOCH_COMPLETE": row, "best_epoch": best_epoch},
                flush=True,
            )

    if SPEED_SMOKE:
        smoke_global_steps = math.ceil(len(adapt_indices) / GLOBAL_STUDY_BATCH)
        memory_mib = torch.tensor(
            [torch.cuda.max_memory_allocated(device) / (1024**2)],
            dtype=torch.float64,
            device=device,
        )
        gathered_memory = [torch.zeros_like(memory_mib) for _ in range(world_size)]
        if distributed:
            dist.all_gather(gathered_memory, memory_mib)
        if is_primary:
            result = {
                "execution_revision": EXECUTION_REVISION,
                "world_size": world_size,
                "global_study_batch": GLOBAL_STUDY_BATCH,
                "gradient_accumulation": grad_accum,
                "global_steps": smoke_global_steps,
                "elapsed_minutes": history[-1]["elapsed_minutes"],
                "seconds_per_global_step": (
                    history[-1]["training_seconds"] / smoke_global_steps
                ),
                "pseudo_valid_weighted_bce": history[-1]["pseudo_valid_weighted_bce"],
                "train_report_contrastive": history[-1][
                    "train_report_contrastive"
                ],
                "train_combined_loss": history[-1]["train_combined_loss"],
                "p7_report_aux": P7_REPORT_AUX,
                "p7_report_temperature": (
                    P7_REPORT_TEMPERATURE if P7_REPORT_AUX else None
                ),
                "p7_report_aux_weight": (
                    P7_REPORT_AUX_WEIGHT if P7_REPORT_AUX else None
                ),
                "p7_report_bank_rows": (
                    len(p7_report_bank_indices) if P7_REPORT_AUX else 0
                ),
                "validation_seconds": history[-1]["validation_seconds"],
                "eval_study_batch_per_rank": eval_batch_per_rank,
                "peak_memory_mib_by_rank": [
                    float(value.item()) for value in gathered_memory
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

    checkpoint_name = (
        f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_best.pt"
        if FOLD_MODE
        else "p2_coatnet_pseudo_best.pt"
    )
    history_name = (
        f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_history.csv"
        if FOLD_MODE
        else "p2_representation_history.csv"
    )
    checkpoint_path = OUTPUT_ROOT / checkpoint_name
    report_projection_name = "p7_report_projection_best.pt"
    report_projection_path = OUTPUT_ROOT / report_projection_name
    if is_primary:
        assert best_state is not None and best_epoch >= 1
        if P7_REPORT_AUX:
            deployable_state = {
                key: value
                for key, value in best_state.items()
                if not key.startswith("report_projection.")
            }
            projection_state = {
                key.removeprefix("report_projection."): value
                for key, value in best_state.items()
                if key.startswith("report_projection.")
            }
            assert deployable_state and projection_state
            torch.save(deployable_state, checkpoint_path)
            torch.save(projection_state, report_projection_path)
        else:
            torch.save(best_state, checkpoint_path)
        pd.DataFrame(history).to_csv(OUTPUT_ROOT / history_name, index=False)
        del best_state
    if distributed:
        dist.barrier()
    selected_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(selected_state, strict=not P7_REPORT_AUX)
    if P7_REPORT_AUX:
        expected_missing = {
            f"report_projection.{key}"
            for key in model.report_projection.state_dict()
        }
        assert set(incompatible.missing_keys) == expected_missing
        assert not incompatible.unexpected_keys
        selected_projection = torch.load(
            report_projection_path, map_location=device, weights_only=True
        )
        model.report_projection.load_state_dict(selected_projection, strict=True)
        del selected_projection
    del selected_state
    gc.collect()
    if is_primary:
        print(
            {
                f"{RUN_LABEL}_REPRESENTATION_COMPLETE": True,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
            },
            flush=True,
        )

    if FOLD_MODE and FULL_RUN:
        assert FOLD_INDEX is not None
        fold_inference_started = time.time()
        local_valid_indices = p4_expert_valid_indices[rank::world_size]
        local_predictions: list[np.ndarray] = []
        local_prediction_indices: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            valid_loader = make_loader(
                StudyDataset(local_valid_indices, k_eval, False),
                batch_size=eval_batch_per_rank,
            )
            for batch in valid_loader:
                indices, windows, window_mask, _, _ = move_batch(batch)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(windows, window_mask, augment=False)
                local_prediction_indices.append(indices.astype(np.int64))
                local_predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        rank_indices = (
            np.concatenate(local_prediction_indices)
            if local_prediction_indices
            else np.empty(0, dtype=np.int64)
        )
        rank_predictions = (
            np.concatenate(local_predictions)
            if local_predictions
            else np.empty((0, N_TARGETS), dtype=np.float32)
        )
        np.save(
            OUTPUT_ROOT
            / f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_indices_rank_{rank}.npy",
            rank_indices,
        )
        np.save(
            OUTPUT_ROOT
            / f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_predictions_rank_{rank}.npy",
            rank_predictions,
        )
        if distributed:
            dist.barrier()

        if is_primary:
            prediction_by_index: dict[int, np.ndarray] = {}
            for shard_rank in range(world_size):
                shard_indices = np.load(
                    OUTPUT_ROOT
                    / f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_indices_rank_{shard_rank}.npy"
                )
                shard_predictions = np.load(
                    OUTPUT_ROOT
                    / f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_predictions_rank_{shard_rank}.npy"
                )
                assert len(shard_indices) == len(shard_predictions)
                prediction_by_index.update(
                    {
                        int(index): prediction
                        for index, prediction in zip(
                            shard_indices, shard_predictions, strict=True
                        )
                    }
                )
            assert set(prediction_by_index) == set(p4_expert_valid_indices.tolist())
            fold_predictions = np.stack(
                [prediction_by_index[int(index)] for index in p4_expert_valid_indices]
            )
            assert fold_predictions.shape == (len(p4_expert_valid_indices), N_TARGETS)
            assert np.isfinite(fold_predictions).all()

            prediction_frame = pd.DataFrame(
                {
                    ID: np.asarray(study_ids)[p4_expert_valid_indices],
                    "fold": FOLD_INDEX,
                }
            )
            for target_index, target in enumerate(TARGETS):
                prediction_frame[f"label__{target}"] = y[
                    p4_expert_valid_indices, target_index
                ]
                prediction_frame[f"pred__{target}"] = fold_predictions[:, target_index]
            prediction_name = f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_oof_predictions.csv"
            prediction_frame.to_csv(OUTPUT_ROOT / prediction_name, index=False)

            split_name = f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_split.csv"
            pd.DataFrame(
                {
                    ID: study_ids,
                    "report_hash": report_hashes,
                    "is_gold": is_gold,
                    "role": p4_split_roles,
                }
            ).to_csv(OUTPUT_ROOT / split_name, index=False)

            for shard_rank in range(world_size):
                (
                    OUTPUT_ROOT
                    / f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_indices_rank_{shard_rank}.npy"
                ).unlink()
                (
                    OUTPUT_ROOT
                    / f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_predictions_rank_{shard_rank}.npy"
                ).unlink()

            if telemetry is not None:
                telemetry.close()
            artifacts = [
                checkpoint_name,
                history_name,
                prediction_name,
                split_name,
                telemetry_name,
            ]
            for name in artifacts:
                assert (OUTPUT_ROOT / name).is_file(), name
            manifest = {
                "schema_version": 1,
                "run_id": (
                    f"{FOLD_RUN_ID_PREFIX}-FOLD-{FOLD_INDEX}-SPEED-SMOKE-v1"
                    if fold_speed_smoke
                    else f"{FOLD_RUN_ID_PREFIX}-FOLD-{FOLD_INDEX}-v1"
                ),
                "full_run": not fold_speed_smoke,
                "speed_smoke": fold_speed_smoke,
                "fold": FOLD_INDEX,
                "seeds": {
                    "pseudo_split": P2_SEED,
                    "expert_folds": FOLD_SEED,
                    "representation": RUN_SEED,
                },
                "studies": len(study_ids),
                "pseudo_studies": int((~is_gold).sum()),
                "expert_studies": int(is_gold.sum()),
                "pseudo_train_rows": int((~is_gold[adapt_indices]).sum()),
                "pseudo_valid_rows": len(pseudo_valid_indices),
                "expert_train_rows": int(is_gold[adapt_indices].sum()),
                "expert_valid_rows": len(p4_expert_valid_indices),
                "planned_pseudo_train_rows": int(
                    np.sum(p4_split_roles == "pseudo_train")
                ),
                "planned_pseudo_valid_rows": int(
                    np.sum(p4_split_roles == "pseudo_valid")
                ),
                "planned_expert_train_rows": len(p4_expert_train_indices),
                "excluded_expert_valid_report_twins": int(
                    np.sum(p4_split_roles == "excluded_expert_valid_report_twin")
                ),
                "excluded_pseudo_valid_expert_train_twins": int(
                    np.sum(p4_split_roles == "excluded_pseudo_valid_expert_train_twin")
                ),
                "ddp_balance_dropped_rows": len(p4_dropped_indices),
                "ddp_balance_dropped_studies": [
                    study_ids[index] for index in p4_dropped_indices
                ],
                "representation_rows": len(adapt_indices),
                "representation_gold_rows": int(is_gold[adapt_indices].sum()),
                "checkpoint_selection_gold_rows": 0,
                "expert_valid_gradient_rows": 0,
                "expert_valid_report_overlap": 0,
                "selection_report_overlap": 0,
                "cache_id": CACHE_ID,
                "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
                "model_id": MODEL_ID,
                "model_sha256": MODEL_SHA256,
                "model_license": MODEL_LICENSE,
                "feature_dim": FEATURE_DIM,
                "cache_size": CACHE_SIZE,
                "model_size": MODEL_SIZE,
                "normalization": {"mean": MODEL_MEAN, "std": MODEL_STD},
                "k_train": k_train,
                "k_eval": k_eval,
                "representation_epochs": rep_epochs,
                "execution_revision": (
                    f"{FOLD_PREFIX}_e2e_fold_ddp_t4_speed_smoke_v1"
                    if fold_speed_smoke
                    else f"{FOLD_PREFIX}_e2e_fold_ddp_t4_v1"
                ),
                "visible_gpus": torch.cuda.device_count(),
                "data_parallel": distributed,
                "data_parallel_backend": "DistributedDataParallel",
                "distributed_backend": "nccl",
                "world_size": world_size,
                "study_batch": GLOBAL_STUDY_BATCH,
                "gradient_accumulation": grad_accum,
                "eval_study_batch": GLOBAL_EVAL_STUDY_BATCH,
                "loader_workers_per_rank": workers,
                "gradient_checkpointing": False,
                "channels_last": channels_last,
                "cudnn_benchmark": device.type == "cuda",
                "fused_adamw": device.type == "cuda",
                "best_epoch": best_epoch,
                "best_pseudo_valid_weighted_bce": best_loss,
                "artifacts_sha256": {
                    name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
                },
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "elapsed": {
                    "representation_minutes": (
                        fold_inference_started - training_started
                    )
                    / 60.0,
                    "fold_inference_minutes": (time.time() - fold_inference_started)
                    / 60.0,
                },
                "run_context": {
                    "device": str(device),
                    "model_path": str(model_path),
                    "cache_a": str(part_a["root"]),
                    "cache_b": str(part_b["root"]) if part_b else None,
                },
            }
            if P8_MODE:
                manifest["teacher_manifest_sha256"] = P8_TEACHER_MANIFEST_SHA256
                manifest["teacher_target_sha256"] = sha256_file(target_path)
                manifest["teacher_target_file"] = target_path.name
                manifest["pseudo_target_formula"] = (
                    "0.50 report + 0.50 mean(P2 direct, P4 fold, P6 direct)"
                )
            manifest_name = f"{FOLD_PREFIX}_fold_{FOLD_INDEX}_manifest.json"
            manifest_path = OUTPUT_ROOT / manifest_name
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                {
                    (
                        f"{FOLD_RUN_LABEL}_SPEED_SMOKE_COMPLETE"
                        if fold_speed_smoke
                        else f"{FOLD_RUN_LABEL}_FOLD_COMPLETE"
                    ): True,
                    "fold": FOLD_INDEX,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "expert_valid_rows": len(p4_expert_valid_indices),
                    "manifest_sha256": sha256_file(manifest_path),
                },
                flush=True,
            )
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return 0

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
        StudyDataset(local_feature_indices, k_eval, False),
        batch_size=eval_batch_per_rank,
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
                encoded = model(
                    windows, window_mask, augment=False, return_features=True
                )
            selected_masks = window_mask.cpu().numpy().astype(bool)
            stop = processed + len(indices)
            feature_shard[processed:stop] = (
                encoded.float().cpu().numpy().astype(np.float16)
            )
            mask_shard[processed:stop] = selected_masks.astype(np.uint8)
            for offset, (index, selected_mask) in enumerate(
                zip(indices, selected_masks, strict=True)
            ):
                slice_mask = read_mask(int(index))
                centers, _ = choose_centers(
                    slice_mask, k_eval, stochastic=False, seed=0
                )
                position_shard[processed + offset, selected_mask] = centers[
                    selected_mask
                ]
            processed = stop
            if is_primary and (
                processed % 50 == 0 or processed == len(local_feature_indices)
            ):
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
            direct_predictions = (
                torch.sigmoid(model.head(feature_tensor_gold, mask_tensor_gold))
                .cpu()
                .numpy()
            )
        direct_metrics, direct_macro = calculate_metrics(
            y[gold_indices], direct_predictions
        )
        direct_metrics.to_csv(OUTPUT_ROOT / "p2_direct_expert_metrics.csv", index=False)
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
            train_index = np.concatenate([keep_pseudo, gold_indices[gold_train_pos]])
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
                    index = permutation[start : start + 256].to(device)
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
                oof[valid_index] = (
                    torch.sigmoid(
                        head(features_tensor[valid_index], masks_tensor[valid_index])
                    )
                    .cpu()
                    .numpy()
                )
            torch.save(head.state_dict(), OUTPUT_ROOT / f"p2_fold_{fold}_head.pt")
            excluded = len(pseudo_indices) - len(keep_pseudo)
            fold_rows.append(
                {
                    "fold": fold,
                    "train_rows": len(train_index),
                    "pseudo_rows": len(keep_pseudo),
                    "expert_train_rows": len(gold_train_pos),
                    "expert_valid_rows": len(valid_index),
                    "excluded_pseudo_rows": excluded,
                    "report_hash_overlap": 0,
                    "final_loss": final_loss,
                }
            )
            print({"P2_HEAD_FOLD_COMPLETE": fold_rows[-1]}, flush=True)
        assert (gold_fold >= 0).all() and np.isfinite(oof[gold_indices]).all()
        oof_metrics, oof_macro = calculate_metrics(y[gold_indices], oof[gold_indices])
        oof_metrics.to_csv(OUTPUT_ROOT / "p2_oof_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUTPUT_ROOT / "p2_fold_summary.csv", index=False)
        oof_frame = pd.DataFrame(
            {
                ID: np.asarray(study_ids)[gold_indices],
                "fold": gold_fold,
            }
        )
        for i, target in enumerate(TARGETS):
            oof_frame[f"label__{target}"] = y[gold_indices, i]
            oof_frame[f"pred__{target}"] = oof[gold_indices, i]
        oof_frame.to_csv(OUTPUT_ROOT / "p2_oof_predictions.csv", index=False)
        fold_result = {"macro_auc": oof_macro}
        print({"P2_OOF_MACRO_AUC": oof_macro}, flush=True)
    else:
        (OUTPUT_ROOT / "p2_local_smoke.json").write_text(
            json.dumps(
                {
                    "study_uid": study_ids[0],
                    "features_shape": list(feature_cache.shape),
                    "training_backward": True,
                },
                indent=2,
            )
            + "\n"
        )

    if telemetry is not None:
        telemetry.close()
    artifacts = [
        checkpoint_name,
        history_name,
        "p2_coatnet_features.npy",
        "p2_coatnet_feature_mask.npy",
        "p2_coatnet_feature_positions.npy",
        "p2_coatnet_feature_studies.csv",
    ]
    if P7_REPORT_AUX:
        artifacts.extend(
            [
                report_projection_name,
                "p7_report_bank_indices.npy",
                "p7_report_group_ids.npy",
            ]
        )
    if FULL_RUN:
        artifacts.extend(
            [
                "p2_direct_expert_metrics.csv",
                "p2_direct_expert_predictions.csv",
                "p2_oof_metrics.csv",
                "p2_oof_predictions.csv",
                "p2_fold_summary.csv",
                *[f"p2_fold_{fold}_head.pt" for fold in range(N_FOLDS)],
                "p2_gpu_telemetry.csv",
            ]
        )
    else:
        artifacts.append("p2_local_smoke.json")
    for name in artifacts:
        assert (OUTPUT_ROOT / name).is_file(), name

    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "full_run": FULL_RUN,
        "seeds": {"representation": P2_SEED, "folds": FOLD_SEED},
        "studies": len(study_ids),
        "pseudo_studies": int((~is_gold).sum()),
        "expert_studies": int(is_gold.sum()),
        "cache_id": CACHE_ID,
        "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "model_license": MODEL_LICENSE,
        "initialization": {
            "arm": (
                "p6_fastmri_localization_plus_report_aux"
                if P7_REPORT_AUX
                else "p6_fastmri_localization_transfer"
                if P6_TRANSFER
                else "matched_imagenet_control"
                if P6_MATCHED_CONTROL
                else "canonical_p2"
            ),
            "checkpoint_filename": MODEL_FILENAME,
            "checkpoint_sha256": MODEL_SHA256,
            "p6_source_manifest_sha256": (
                P6_SOURCE_MANIFEST_SHA256
                if P6_TRANSFER or P7_REPORT_AUX
                else None
            ),
        },
        "report_auxiliary": (
            {
                "enabled": True,
                "training_only": True,
                "text_model_source": "yethukmutt/bge-m3/Transformers/m3/1",
                "text_model_upstream": "BAAI/bge-m3",
                "text_model_license": "MIT",
                "embedding_manifest_sha256": (
                    P7_REPORT_EMBEDDINGS_MANIFEST_SHA256
                ),
                "embedding_artifacts_sha256": (
                    p7_report_manifest["artifacts_sha256"]
                    if p7_report_manifest is not None
                    else None
                ),
                "embedding_dimension": P7_REPORT_EMBEDDING_DIM,
                "bank_rows": len(p7_report_bank_indices),
                "bank_unique_report_groups": p7_report_bank_unique_groups,
                "bank_expert_rows": int(is_gold[p7_report_bank_indices].sum()),
                "selection_report_overlap": 0,
                "temperature": P7_REPORT_TEMPERATURE,
                "loss_weight": P7_REPORT_AUX_WEIGHT,
                "loss": "duplicate-aware multi-positive image-to-report InfoNCE",
                "loss_normalization": "divide by log(report bank rows)",
                "projection_artifact": report_projection_name,
                "deployable_checkpoint_excludes_projection": True,
            }
            if P7_REPORT_AUX
            else {"enabled": False}
        ),
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
        "report_auxiliary_gold_rows": 0 if P7_REPORT_AUX else None,
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
            "p6_source_manifest_path": (
                str(p6_source_manifest_path)
                if p6_source_manifest_path is not None
                else None
            ),
            "p7_report_embeddings_manifest_path": (
                str(p7_report_manifest_path)
                if p7_report_manifest_path is not None
                else None
            ),
            "p7_report_embeddings_path": (
                str(p7_report_embeddings_path)
                if p7_report_embeddings_path is not None
                else None
            ),
            "cache_a": str(part_a["root"]),
            "cache_b": str(part_b["root"]) if part_b else None,
        },
    }
    manifest_path = OUTPUT_ROOT / (
        "p7_report_aux_manifest.json"
        if P7_REPORT_AUX
        else "p6_transfer_manifest.json"
        if P6_TRANSFER
        else "p6_control_manifest.json"
        if P6_MATCHED_CONTROL
        else "p2_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        {
            f"{RUN_LABEL}_COMPLETE": True,
            "best_epoch": best_epoch,
            "direct_macro": None
            if direct_result is None
            else direct_result["macro_auc"],
            "oof_macro": None if fold_result is None else fold_result["macro_auc"],
            "manifest_sha256": sha256_file(manifest_path),
        },
        flush=True,
    )
    return 0


def main() -> int:
    if FULL_RUN and "--ddp-worker" not in sys.argv:
        return launch_ddp()
    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
