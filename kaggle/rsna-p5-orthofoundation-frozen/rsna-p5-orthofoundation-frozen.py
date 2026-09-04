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

import types as _p5_types
common = _p5_types.ModuleType('p5_orthofoundation_embedded_common')
common.__file__ = '<embedded-p5-common>'
exec(compile('#!/usr/bin/env python3\n"""Benchmark the frozen P5 OrthoFoundation feature path on two Kaggle T4s."""\n\n# ruff: noqa: E402 - runtime paths and offline flags precede third-party imports.\n\nfrom __future__ import annotations\n\nimport csv\nimport hashlib\nimport json\nimport math\nimport os\nimport platform\nimport random\nimport subprocess\nimport sys\nimport threading\nimport time\nfrom pathlib import Path\n\nos.environ.setdefault("HF_HUB_OFFLINE", "1")\nos.environ.setdefault("TRANSFORMERS_OFFLINE", "1")\nos.environ.setdefault("TORCH_HOME", "/kaggle/working/torch-cache")\n\nimport numpy as np\nimport pandas as pd\nimport torch\nimport torch.distributed as dist\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader, Dataset\n\n\nID = "StudyInstanceUID"\nTARGETS = [\n    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",\n    "Medial OA", "Lateral OA", "PF OA", "Effusion",\n    "Synovitis", "Baker\'s", "Contusion", "Fracture",\n]\nRUN_ID = "P5A-ORTHOFOUNDATION-FROZEN-v1"\nEXECUTION_REVISION = "p5a_frozen_t4x2_smoke_v2"\nCACHE_ID = "P2-CACHE-v1"\nCACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"\nMODEL_SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"\nMODEL_BYTES = 1_213_056_638\nORTHO_COMMIT = "4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470"\nDINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"\nSEED = 20260904\nCACHE_SIZE = 336\nMODEL_SIZE = 224\nBACKBONE_DIM = 1024\nN_REGIONS = 2\nHEAD_DIM = 256\nK_SLICES = 32\nSLOT_QUOTAS = (9, 7, 6, 4, 6)\nSMOKE_STUDIES = 128\nSTUDY_BATCH_PER_RANK = 2\nWORKERS_PER_RANK = 2\nHEAD_SMOKE_STEPS = 5\nMODEL_MEAN = (0.485, 0.456, 0.406)\nMODEL_STD = (0.229, 0.224, 0.225)\nOUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(\n    "outputs/p5_orthofoundation_speed_smoke"\n)\n\n\ndef launch() -> int:\n    visible = torch.cuda.device_count()\n    if visible != 2:\n        raise RuntimeError(f"P5 speed smoke requires exactly two GPUs; observed {visible}")\n    command = [\n        sys.executable,\n        "-m",\n        "torch.distributed.run",\n        "--standalone",\n        "--nproc_per_node=2",\n        str(Path(__file__).resolve()),\n        "--worker",\n    ]\n    print({"P5_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)\n    subprocess.run(command, check=True)\n    print({"P5_DDP_PROCESS_COMPLETE": True}, flush=True)\n    return 0\n\n\ndef sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:\n    digest = hashlib.sha256()\n    with path.open("rb") as handle:\n        while chunk := handle.read(chunk_size):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef candidate_roots() -> list[Path]:\n    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]\n    if Path("/kaggle/input").is_dir():\n        roots.append(Path("/kaggle/input"))\n    return roots\n\n\ndef find_named_file(filename: str, minimum_bytes: int = 1) -> Path:\n    direct_map = {\n        "train.csv": Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"),\n        "training_targets.csv": Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-report-weak-supervision/training_targets.csv"\n        ),\n        "p2_cache_manifest_part_a.json": Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-p2-wide-cache-a/p2_cache_manifest_part_a.json"\n        ),\n        "p2_cache_manifest_part_b.json": Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-p2-wide-cache-b/p2_cache_manifest_part_b.json"\n        ),\n    }\n    candidates = [direct_map[filename]] if filename in direct_map else []\n    for root in candidate_roots():\n        candidates.extend(\n            [\n                root / filename,\n                root / "data/raw" / filename,\n                root / "outputs/kaggle/rsna-report-weak-supervision" / filename,\n                root / "kaggle/rsna-p2-wide-cache-a/outputs/p2_cache_smoke_a" / filename,\n            ]\n        )\n    for path in candidates:\n        if path.is_file() and path.stat().st_size >= minimum_bytes:\n            return path.resolve()\n    kaggle_input = Path("/kaggle/input")\n    if kaggle_input.is_dir():\n        for path in kaggle_input.glob(f"**/{filename}"):\n            if path.is_file() and path.stat().st_size >= minimum_bytes:\n                return path.resolve()\n    raise FileNotFoundError(filename)\n\n\ndef find_runtime() -> tuple[Path, Path]:\n    for root in candidate_roots():\n        for checkpoint in root.glob("**/OrthoFoudation-L.pth"):\n            runtime = checkpoint.parent\n            if (runtime / "dinov3/hubconf.py").is_file():\n                return runtime.resolve(), checkpoint.resolve()\n    raise FileNotFoundError("OrthoFoundation-L checkpoint plus bundled DINOv3 runtime")\n\n\ndef cache_part(part: str) -> dict[str, object]:\n    manifest_path = find_named_file(f"p2_cache_manifest_part_{part}.json")\n    manifest = json.loads(manifest_path.read_text())\n    assert manifest["cache_id"] == CACHE_ID\n    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE\n    root = manifest_path.parent\n    result: dict[str, object] = {"manifest": manifest, "root": root}\n    for key, name in {\n        "pixels": f"p2_pixels_part_{part}.npy",\n        "mask": f"p2_valid_mask_part_{part}.npy",\n        "studies": f"p2_studies_part_{part}.csv",\n    }.items():\n        path = root / name\n        assert path.is_file(), path\n        result[key] = path\n    return result\n\n\nclass GpuTelemetry:\n    def __init__(self, output_path: Path, interval_seconds: int = 1) -> None:\n        self.output_path = output_path\n        self.interval_seconds = interval_seconds\n        self.stop = threading.Event()\n        self.thread: threading.Thread | None = None\n\n    def start(self) -> None:\n        self.output_path.write_text(\n            "unix_time,timestamp,index,name,utilization_gpu_percent,"\n            "utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,"\n            "temperature_c\\n"\n        )\n        self.thread = threading.Thread(target=self._sample, daemon=True)\n        self.thread.start()\n\n    def _sample(self) -> None:\n        query = (\n            "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,"\n            "memory.total,power.draw,temperature.gpu"\n        )\n        while not self.stop.is_set():\n            try:\n                result = subprocess.run(\n                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],\n                    check=True,\n                    capture_output=True,\n                    text=True,\n                )\n                rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))\n                now = time.time()\n                with self.output_path.open("a", newline="") as handle:\n                    writer = csv.writer(handle)\n                    for row in rows:\n                        writer.writerow([now, *row])\n                print({"P5_GPU": rows}, flush=True)\n            except Exception as error:\n                print({"P5_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)\n            self.stop.wait(self.interval_seconds)\n\n    def close(self) -> None:\n        self.stop.set()\n        if self.thread is not None:\n            self.thread.join(timeout=10)\n\n\nclass HierarchicalDiagnosisHead(nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n        self.input_norm = nn.LayerNorm(BACKBONE_DIM)\n        self.projection = nn.Sequential(\n            nn.Linear(BACKBONE_DIM, HEAD_DIM),\n            nn.GELU(),\n            nn.Dropout(0.10),\n        )\n        self.region_embedding = nn.Parameter(torch.randn(N_REGIONS, HEAD_DIM) * 0.02)\n        self.slot_embedding = nn.Parameter(torch.randn(len(SLOT_QUOTAS), HEAD_DIM) * 0.02)\n        self.position_mlp = nn.Sequential(\n            nn.Linear(1, HEAD_DIM), nn.GELU(), nn.Linear(HEAD_DIM, HEAD_DIM)\n        )\n        self.region_query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)\n        self.slice_query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)\n        self.target_weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)\n        self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))\n\n    def forward(\n        self,\n        features: torch.Tensor,\n        mask: torch.Tensor,\n        slots: torch.Tensor,\n        positions: torch.Tensor,\n    ) -> torch.Tensor:\n        hidden = self.projection(self.input_norm(features.float()))\n        hidden = hidden + self.region_embedding[None, None]\n        region_scores = torch.einsum("bkrh,th->bkrt", hidden, self.region_query)\n        region_attention = (region_scores / math.sqrt(HEAD_DIM)).softmax(dim=2)\n        local = torch.einsum("bkrt,bkrh->bkth", region_attention, hidden)\n        context = self.slot_embedding[slots.clamp(0, len(SLOT_QUOTAS) - 1)]\n        context = context + self.position_mlp(positions[..., None])\n        local = local + context[:, :, None]\n        slice_scores = torch.einsum("bkth,th->bkt", torch.tanh(local), self.slice_query)\n        slice_scores = (slice_scores / math.sqrt(HEAD_DIM)).masked_fill(\n            ~mask[:, :, None], -1e4\n        )\n        slice_attention = slice_scores.softmax(dim=1)\n        pooled = torch.einsum("bkt,bkth->bth", slice_attention, local)\n        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias\n\n\ndef worker() -> int:\n    rank = int(os.environ["RANK"])\n    local_rank = int(os.environ["LOCAL_RANK"])\n    world_size = int(os.environ["WORLD_SIZE"])\n    assert world_size == 2\n    torch.cuda.set_device(local_rank)\n    device = torch.device("cuda", local_rank)\n    dist.init_process_group("nccl", device_id=device)\n    is_primary = rank == 0\n    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n    telemetry = GpuTelemetry(OUTPUT_ROOT / "p5_speed_gpu_telemetry.csv") if is_primary else None\n    if telemetry is not None:\n        telemetry.start()\n\n    random.seed(SEED + rank)\n    np.random.seed(SEED + rank)\n    torch.manual_seed(SEED + rank)\n    torch.cuda.manual_seed_all(SEED + rank)\n    torch.backends.cudnn.benchmark = True\n    torch.set_float32_matmul_precision("high")\n\n    runtime, checkpoint = find_runtime()\n    part_a, part_b = cache_part("a"), cache_part("b")\n    if is_primary:\n        assert checkpoint.stat().st_size == MODEL_BYTES\n        assert sha256_file(checkpoint) == MODEL_SHA256\n        provenance = (runtime / "PROVENANCE.md").read_text()\n        assert ORTHO_COMMIT in provenance and DINOV3_COMMIT in provenance\n        assert (runtime / "ORTHOFOUNDATION_LICENSE").is_file()\n        assert (runtime / "dinov3/LICENSE.md").is_file()\n        print(\n            {\n                "P5_SPEED_INPUTS_OK": True,\n                "checkpoint": str(checkpoint),\n                "checkpoint_sha256": MODEL_SHA256,\n                "runtime": str(runtime),\n            },\n            flush=True,\n        )\n    dist.barrier()\n\n    train_path = find_named_file("train.csv", 100_000)\n    target_path = find_named_file("training_targets.csv", 100_000)\n    train = pd.read_csv(train_path, dtype={ID: str})\n    targets = pd.read_csv(target_path, dtype={ID: str}).set_index(ID).loc[train[ID]].reset_index()\n    assert len(train) == len(targets) == 4_407\n    pixels_a = np.load(part_a["pixels"], mmap_mode="r")\n    pixels_b = np.load(part_b["pixels"], mmap_mode="r")\n    masks_a = np.load(part_a["mask"], mmap_mode="r")\n    masks_b = np.load(part_b["mask"], mmap_mode="r")\n    assert len(pixels_a) == 2_204 and len(pixels_b) == 2_203\n    slot_bounds = np.load(Path(part_a["root"]) / "p2_slot_bounds.npy").astype(int)\n    assert slot_bounds.tolist() == [[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]]\n\n    def read_study(index: int) -> tuple[np.ndarray, np.ndarray]:\n        if index < len(pixels_a):\n            return np.asarray(pixels_a[index]), np.asarray(masks_a[index]).astype(bool)\n        local = index - len(pixels_a)\n        return np.asarray(pixels_b[local]), np.asarray(masks_b[local]).astype(bool)\n\n    def select_slices(index: int):\n        volume, valid_mask = read_study(index)\n        slices = np.zeros((K_SLICES, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)\n        selected_mask = np.zeros(K_SLICES, dtype=bool)\n        slots = np.zeros(K_SLICES, dtype=np.int64)\n        positions = np.zeros(K_SLICES, dtype=np.float32)\n        cursor = 0\n        for slot, ((start, stop), quota) in enumerate(zip(slot_bounds, SLOT_QUOTAS)):\n            available = np.where(valid_mask[start:stop])[0] + start\n            count = min(len(available), quota)\n            if count:\n                if len(available) <= quota:\n                    chosen = available\n                else:\n                    chosen = available[np.round(np.linspace(0, len(available) - 1, quota)).astype(int)]\n                chosen = np.asarray(chosen[:count], dtype=int)\n                out = slice(cursor, cursor + count)\n                slices[out] = volume[chosen]\n                selected_mask[out] = True\n                slots[out] = slot\n                positions[out] = 2.0 * ((chosen - start) / max(stop - start - 1, 1)) - 1.0\n            slots[cursor:cursor + quota] = slot\n            cursor += quota\n        assert cursor == K_SLICES and selected_mask.any()\n        labels = targets.loc[index, TARGETS].to_numpy(np.float32)\n        weights = targets.loc[index, [f"weight__{target}" for target in TARGETS]].to_numpy(\n            np.float32\n        )\n        return index, slices, selected_mask, slots, positions, labels, weights\n\n    smoke_indices = np.round(np.linspace(0, len(train) - 1, SMOKE_STUDIES)).astype(int)\n    assert len(np.unique(smoke_indices)) == SMOKE_STUDIES\n    local_indices = smoke_indices[rank::world_size]\n\n    class SmokeDataset(Dataset):\n        def __len__(self) -> int:\n            return len(local_indices)\n\n        def __getitem__(self, position: int):\n            return select_slices(int(local_indices[position]))\n\n    loader = DataLoader(\n        SmokeDataset(),\n        batch_size=STUDY_BATCH_PER_RANK,\n        shuffle=False,\n        num_workers=WORKERS_PER_RANK,\n        pin_memory=True,\n        prefetch_factor=2,\n    )\n\n    load_started = time.time()\n    model = torch.hub.load(\n        str(runtime / "dinov3"),\n        "dinov3_vitl16",\n        source="local",\n        pretrained=False,\n    )\n    state = torch.load(checkpoint, map_location="cpu", weights_only=True)\n    assert isinstance(state, dict) and state\n    original_keys = list(state)\n    if all(key.startswith("backbone.") for key in original_keys):\n        state = {key.removeprefix("backbone."): value for key, value in state.items()}\n        checkpoint_namespace = "backbone."\n    elif all(key.startswith("module.backbone.") for key in original_keys):\n        state = {key.removeprefix("module.backbone."): value for key, value in state.items()}\n        checkpoint_namespace = "module.backbone."\n    else:\n        checkpoint_namespace = "unprefixed"\n    assert set(state) == set(model.state_dict()), {\n        "missing": sorted(set(model.state_dict()) - set(state))[:10],\n        "unexpected": sorted(set(state) - set(model.state_dict()))[:10],\n    }\n    model.load_state_dict(state, strict=True)\n    del state\n    model.requires_grad_(False).eval().to(device)\n    total_parameters = sum(parameter.numel() for parameter in model.parameters())\n    assert 300_000_000 < total_parameters < 310_000_000\n    model_load_seconds = time.time() - load_started\n    dist.barrier()\n    if is_primary:\n        print(\n            {\n                "P5_MODEL_READY": True,\n                "parameters": total_parameters,\n                "load_seconds": model_load_seconds,\n                "world_size": world_size,\n                "checkpoint_namespace": checkpoint_namespace,\n            },\n            flush=True,\n        )\n\n    mean = torch.tensor(MODEL_MEAN, device=device).view(1, 3, 1, 1)\n    std = torch.tensor(MODEL_STD, device=device).view(1, 3, 1, 1)\n    feature_batches: list[torch.Tensor] = []\n    mask_batches: list[torch.Tensor] = []\n    slot_batches: list[torch.Tensor] = []\n    position_batches: list[torch.Tensor] = []\n    label_batches: list[torch.Tensor] = []\n    weight_batches: list[torch.Tensor] = []\n    torch.cuda.reset_peak_memory_stats(device)\n    extract_started = time.time()\n    with torch.inference_mode():\n        for step, batch in enumerate(loader, start=1):\n            _, slices, slice_mask, slots, positions, labels, weights = batch\n            slices = slices.to(device, non_blocking=True)\n            batch_size, count = slices.shape[:2]\n            pixels = slices.reshape(-1, 1, CACHE_SIZE, CACHE_SIZE).float().div_(255.0)\n            pixels = F.interpolate(\n                pixels, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False\n            )\n            pixels = pixels.expand(-1, 3, -1, -1)\n            pixels = ((pixels - mean) / std).contiguous(memory_format=torch.channels_last)\n            with torch.autocast("cuda", dtype=torch.float16):\n                encoded = model.forward_features(pixels)\n                cls = encoded["x_norm_clstoken"]\n                patch_mean = encoded["x_norm_patchtokens"].mean(dim=1)\n                features = torch.stack((cls, patch_mean), dim=1)\n            features = features.reshape(batch_size, count, N_REGIONS, BACKBONE_DIM)\n            assert torch.isfinite(features).all()\n            feature_batches.append(features.half())\n            mask_batches.append(slice_mask.to(device, non_blocking=True).bool())\n            slot_batches.append(slots.to(device, non_blocking=True).long())\n            position_batches.append(positions.to(device, non_blocking=True).float())\n            label_batches.append(labels.to(device, non_blocking=True).float())\n            weight_batches.append(weights.to(device, non_blocking=True).float())\n            if step == 1 or step == len(loader):\n                print(\n                    {\n                        "P5_FEATURE_PROGRESS": f"rank={rank} step={step}/{len(loader)}",\n                        "feature_shape": list(features.shape),\n                    },\n                    flush=True,\n                )\n    torch.cuda.synchronize(device)\n    extraction_seconds = time.time() - extract_started\n\n    features_all = torch.cat(feature_batches)\n    masks_all = torch.cat(mask_batches)\n    slots_all = torch.cat(slot_batches)\n    positions_all = torch.cat(position_batches)\n    labels_all = torch.cat(label_batches)\n    weights_all = torch.cat(weight_batches)\n    head = HierarchicalDiagnosisHead().to(device)\n    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3, fused=True)\n    head_started = time.time()\n    final_loss = float("nan")\n    for step in range(HEAD_SMOKE_STEPS):\n        generator = torch.Generator(device=device).manual_seed(SEED + rank + step)\n        chosen = torch.randperm(len(features_all), generator=generator, device=device)[:64]\n        optimizer.zero_grad(set_to_none=True)\n        with torch.autocast("cuda", dtype=torch.float16):\n            logits = head(\n                features_all[chosen],\n                masks_all[chosen],\n                slots_all[chosen],\n                positions_all[chosen],\n            )\n            raw = F.binary_cross_entropy_with_logits(logits, labels_all[chosen], reduction="none")\n            loss = (raw * weights_all[chosen]).sum() / weights_all[chosen].sum().clamp_min(1.0)\n        assert torch.isfinite(loss)\n        loss.backward()\n        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)\n        optimizer.step()\n        final_loss = float(loss.detach())\n    torch.cuda.synchronize(device)\n    head_seconds = time.time() - head_started\n    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)\n\n    local_stats = torch.tensor(\n        [\n            model_load_seconds,\n            extraction_seconds,\n            head_seconds,\n            peak_memory_mib,\n            float(len(local_indices)),\n            final_loss,\n        ],\n        dtype=torch.float64,\n        device=device,\n    )\n    gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]\n    dist.all_gather(gathered, local_stats)\n    dist.barrier()\n    if is_primary:\n        rows = [item.cpu().tolist() for item in gathered]\n        max_load = max(row[0] for row in rows)\n        max_extract = max(row[1] for row in rows)\n        global_images = sum(row[4] for row in rows) * K_SLICES\n        images_per_second = global_images / max_extract\n        projected_full_feature_minutes = (4_407 * K_SLICES / images_per_second) / 60.0\n        projected_hidden_feature_minutes = (1_300 * K_SLICES / images_per_second) / 60.0\n        result = {\n            "run_id": RUN_ID,\n            "execution_revision": EXECUTION_REVISION,\n            "python": platform.python_version(),\n            "torch": torch.__version__,\n            "world_size": world_size,\n            "smoke_studies": SMOKE_STUDIES,\n            "slices_per_study": K_SLICES,\n            "slot_quotas": list(SLOT_QUOTAS),\n            "study_batch_per_rank": STUDY_BATCH_PER_RANK,\n            "image_batch_per_rank": STUDY_BATCH_PER_RANK * K_SLICES,\n            "feature_shape_per_study": [K_SLICES, N_REGIONS, BACKBONE_DIM],\n            "checkpoint_sha256": MODEL_SHA256,\n            "checkpoint_bytes": MODEL_BYTES,\n            "checkpoint_namespace": checkpoint_namespace,\n            "orthofoundation_commit": ORTHO_COMMIT,\n            "dinov3_commit": DINOV3_COMMIT,\n            "cache_id": CACHE_ID,\n            "cache_signature_sha256": CACHE_SIGNATURE,\n            "model_parameters": total_parameters,\n            "model_load_seconds_by_rank": [row[0] for row in rows],\n            "extraction_seconds_by_rank": [row[1] for row in rows],\n            "images_per_second_two_gpu": images_per_second,\n            "projected_full_feature_minutes": projected_full_feature_minutes,\n            "projected_hidden_feature_minutes": projected_hidden_feature_minutes,\n            "head_smoke_steps": HEAD_SMOKE_STEPS,\n            "head_seconds_by_rank": [row[2] for row in rows],\n            "head_final_loss_by_rank": [row[5] for row in rows],\n            "peak_memory_mib_by_rank": [row[3] for row in rows],\n            "licenses": ["MIT", "DINOv3 License"],\n            "finite": True,\n            "estimated_full_fixed_model_load_minutes": max_load / 60.0,\n        }\n        (OUTPUT_ROOT / "p5_speed_smoke.json").write_text(json.dumps(result, indent=2) + "\\n")\n        print({"P5_SPEED_SMOKE_COMPLETE": result}, flush=True)\n        if telemetry is not None:\n            telemetry.close()\n    dist.barrier()\n    dist.destroy_process_group()\n    return 0\n\n\ndef main() -> int:\n    if "--worker" not in sys.argv:\n        return launch()\n    return worker()\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n', common.__file__, 'exec'), common.__dict__)


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
