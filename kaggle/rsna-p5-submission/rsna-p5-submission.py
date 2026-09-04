#!/usr/bin/env python3
"""Create a diagnostic P5 OrthoFoundation submission on Kaggle."""

# ruff: noqa: E402 - the deployment builder embeds local modules before imports.

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import types as _bundle_types
prep = _bundle_types.ModuleType('p5_embedded_p2_preprocessing')
prep.__file__ = '<embedded-p2-preprocessing>'
exec(compile('#!/usr/bin/env python3\n"""Create the P2 CoAtNet fold-head ensemble submission on Kaggle.\n\nThe visible competition mount contains only a tiny canary test.  Kaggle reruns this\nsame internet-disabled script against the hidden test set when the notebook version\nis submitted to the competition, so no test-study count is hard coded here.\n"""\n\n# ruff: noqa: E402 - local offline dependencies must precede ML imports.\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nimport os\nimport platform\nimport subprocess\nimport sys\nimport time\nfrom pathlib import Path\n\nfor dependency_root in (\n    Path.cwd() / ".deps",\n    Path.cwd().parent / ".deps",\n    Path.cwd().parent.parent / ".deps",\n):\n    if dependency_root.is_dir():\n        sys.path.insert(0, str(dependency_root))\n\nos.environ.setdefault("HF_HUB_OFFLINE", "1")\nos.environ.setdefault("TRANSFORMERS_OFFLINE", "1")\nos.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")\n\nimport numpy as np\nimport pandas as pd\nimport pydicom\nimport timm\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom PIL import Image\nfrom torch.utils.data import DataLoader, Dataset\n\n\nCOMPETITION_SLUG = "rsna-knee-abnormality-detection"\nID = "StudyInstanceUID"\nTARGETS = [\n    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",\n    "Medial OA", "Lateral OA", "PF OA", "Effusion",\n    "Synovitis", "Baker\'s", "Contusion", "Fracture",\n]\nARCH = "coatnet_rmlp_2_rw_384"\nFEATURE_DIM = 1024\nN_TARGETS = len(TARGETS)\nCACHE_SIZE = 336\nMODEL_SIZE = 384\nK_EVAL = 32\nCROP_MM = 140.0\nLOW_PERCENTILE = 2.0\nHIGH_PERCENTILE = 98.0\nMODEL_MEAN = (0.5, 0.5, 0.5)\nMODEL_STD = (0.5, 0.5, 0.5)\nEXPECTED_P2_MANIFEST_SHA256 = (\n    "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"\n)\nEXPECTED_CACHE_SIGNATURE = (\n    "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"\n)\nON_KAGGLE = Path("/kaggle/working").is_dir()\nOUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(\n    "outputs/p2_submission_smoke"\n)\n\nSLOT_SPECS = [\n    {"name": "sagittal_fluid", "plane": "Sagittal", "kind": "fluid", "count": 18},\n    {\n        "name": "sagittal_structural",\n        "plane": "Sagittal",\n        "kind": "structural",\n        "count": 14,\n    },\n    {"name": "coronal_fluid", "plane": "Coronal", "kind": "fluid", "count": 12},\n    {\n        "name": "coronal_structural",\n        "plane": "Coronal",\n        "kind": "structural",\n        "count": 8,\n    },\n    {"name": "axial_any", "plane": "Axial", "kind": "any", "count": 12},\n]\nassert sum(slot["count"] for slot in SLOT_SPECS) == 64\nSLOT_BOUNDS: list[tuple[int, int]] = []\ncursor = 0\nfor slot in SLOT_SPECS:\n    SLOT_BOUNDS.append((cursor, cursor + int(slot["count"])))\n    cursor += int(slot["count"])\n\n\ndef sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:\n    digest = hashlib.sha256()\n    with path.open("rb") as handle:\n        while chunk := handle.read(chunk_size):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef candidate_roots() -> list[Path]:\n    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]\n    if Path("/kaggle/input").is_dir():\n        roots.append(Path("/kaggle/input"))\n    return roots\n\n\ndef find_competition_root() -> Path:\n    candidates = [\n        Path("/kaggle/input/competitions") / COMPETITION_SLUG,\n        Path("/kaggle/input") / COMPETITION_SLUG,\n    ]\n    for root in candidate_roots():\n        candidates.extend([root / "data/raw", root])\n    for candidate in candidates:\n        if (candidate / "test.csv").is_file() and (\n            candidate / "test_series.csv"\n        ).is_file():\n            return candidate.resolve()\n    raise FileNotFoundError("competition test metadata")\n\n\ndef find_p2_root() -> Path:\n    candidates = [\n        Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-p2-coatnet-rmlp2"\n        ),\n        Path("outputs/kaggle/rsna-p2-coatnet-rmlp2-v2"),\n    ]\n    for candidate in candidates:\n        if (candidate / "p2_manifest.json").is_file():\n            return candidate.resolve()\n    kaggle_input = Path("/kaggle/input")\n    if kaggle_input.is_dir():\n        matches = list(kaggle_input.glob("**/p2_manifest.json"))\n        if len(matches) == 1:\n            return matches[0].parent.resolve()\n    raise FileNotFoundError("validated P2 output")\n\n\nHEADER_TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber"]\n\n\ndef spatial_key(path: Path) -> tuple[float, int, str]:\n    try:\n        dataset = pydicom.dcmread(\n            path, stop_before_pixels=True, specific_tags=HEADER_TAGS\n        )\n        orientation = np.asarray(dataset.get("ImageOrientationPatient", []), dtype=float)\n        position = np.asarray(dataset.get("ImagePositionPatient", []), dtype=float)\n        instance = int(dataset.get("InstanceNumber", 0) or 0)\n        if orientation.size == 6 and position.size == 3:\n            location = float(\n                np.dot(position, np.cross(orientation[:3], orientation[3:]))\n            )\n            return (location, instance, path.name)\n        return (float(instance), instance, path.name)\n    except Exception:\n        return (0.0, 0, path.name)\n\n\ndef ordered_files(series_dir: Path) -> list[Path]:\n    return sorted(series_dir.glob("*.dcm"), key=spatial_key)\n\n\ndef sequence_score(record: dict[str, object], kind: str) -> int:\n    if kind == "any":\n        return 0\n    fluid = int(record["Fluid_Sensitive"])\n    fat = int(record["Fat_Suppression"])\n    if kind == "fluid":\n        return 3 if fluid == 1 and fat == 1 else 2 if fluid == 1 else 1 if fat == 1 else 0\n    return 3 if fluid == 0 and fat == 0 else 2 if fluid == 0 else 1 if fat == 0 else 0\n\n\ndef choose_series(\n    dicom_root: Path,\n    study_uid: str,\n    records: list[dict[str, object]],\n    slot: dict[str, object],\n    used: set[str],\n) -> tuple[Path | None, str | None]:\n    candidates: list[tuple[int, int, str, Path]] = []\n    for record in records:\n        series_uid = str(record["SeriesInstanceUID"])\n        if series_uid in used or str(record["Anatomical_Plane"]) != slot["plane"]:\n            continue\n        path = dicom_root / study_uid / series_uid\n        count = sum(1 for _ in path.glob("*.dcm")) if path.is_dir() else 0\n        if count:\n            candidates.append(\n                (sequence_score(record, str(slot["kind"])), count, series_uid, path)\n            )\n    if not candidates:\n        return None, None\n    _, _, series_uid, path = max(\n        candidates, key=lambda item: (item[0], item[1], item[2])\n    )\n    used.add(series_uid)\n    return path, series_uid\n\n\ndef decode_raw(path: Path) -> tuple[np.ndarray, tuple[float, float]] | None:\n    try:\n        dataset = pydicom.dcmread(path)\n        array = dataset.pixel_array.astype(np.float32)\n        array = array * float(dataset.get("RescaleSlope", 1.0)) + float(\n            dataset.get("RescaleIntercept", 0.0)\n        )\n        if str(dataset.get("PhotometricInterpretation", "")) == "MONOCHROME1":\n            array = array.max() + array.min() - array\n        spacing = dataset.get(\n            "PixelSpacing", [CROP_MM / max(array.shape)] * 2\n        )\n        sy, sx = float(spacing[0]), float(spacing[1])\n        if not np.isfinite(array).any() or sy <= 0 or sx <= 0:\n            return None\n        return array, (sy, sx)\n    except Exception:\n        return None\n\n\ndef crop_resize_uint8(\n    array: np.ndarray, spacing: tuple[float, float], lo: float, hi: float\n) -> np.ndarray | None:\n    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:\n        return None\n    sy, sx = spacing\n    crop_h = min(array.shape[0], max(16, int(round(CROP_MM / sy))))\n    crop_w = min(array.shape[1], max(16, int(round(CROP_MM / sx))))\n    cy, cx = array.shape[0] // 2, array.shape[1] // 2\n    y0 = min(max(0, cy - crop_h // 2), array.shape[0] - crop_h)\n    x0 = min(max(0, cx - crop_w // 2), array.shape[1] - crop_w)\n    crop = array[y0:y0 + crop_h, x0:x0 + crop_w]\n    scaled = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)\n    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)\n    image = Image.fromarray(np.round(scaled * 255.0).astype(np.uint8))\n    return np.asarray(\n        image.resize((CACHE_SIZE, CACHE_SIZE), resample=Image.Resampling.BILINEAR)\n    )\n\n\ndef build_slot(series_dir: Path, count: int) -> tuple[np.ndarray, np.ndarray]:\n    files = ordered_files(series_dir)\n    images = np.zeros((count, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)\n    valid = np.zeros(count, dtype=np.uint8)\n    if not files:\n        return images, valid\n    centers = np.round(np.linspace(0.02, 0.98, count) * (len(files) - 1)).astype(int)\n    decoded = [decode_raw(files[int(index)]) for index in centers]\n    finite_chunks = [\n        item[0][np.isfinite(item[0])] for item in decoded if item is not None\n    ]\n    if not finite_chunks:\n        return images, valid\n    finite = np.concatenate(finite_chunks)\n    lo, hi = np.percentile(finite, [LOW_PERCENTILE, HIGH_PERCENTILE])\n    for output_index, item in enumerate(decoded):\n        if item is None:\n            continue\n        processed = crop_resize_uint8(item[0], item[1], float(lo), float(hi))\n        if processed is not None:\n            images[output_index] = processed\n            valid[output_index] = 1\n    return images, valid\n\n\ndef valid_centers(mask: np.ndarray) -> np.ndarray:\n    centers: list[int] = []\n    for start, stop in SLOT_BOUNDS:\n        for center in range(start + 1, stop - 1):\n            if mask[center - 1:center + 2].all():\n                centers.append(center)\n    return np.asarray(centers, dtype=np.int16)\n\n\ndef choose_centers(mask: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:\n    available = valid_centers(mask)\n    if len(available) == 0:\n        return np.zeros(count, dtype=np.int16), np.zeros(count, dtype=bool)\n    if len(available) <= count:\n        selected = np.zeros(count, dtype=np.int16)\n        selected[:len(available)] = available\n        valid = np.zeros(count, dtype=bool)\n        valid[:len(available)] = True\n        return selected, valid\n    positions = np.round(np.linspace(0, len(available) - 1, count)).astype(int)\n    return available[positions], np.ones(count, dtype=bool)\n\n\nclass TestStudyDataset(Dataset):\n    def __init__(\n        self,\n        study_ids: list[str],\n        series_by_study: dict[str, list[dict[str, object]]],\n        dicom_root: Path,\n    ) -> None:\n        self.study_ids = study_ids\n        self.series_by_study = series_by_study\n        self.dicom_root = dicom_root\n\n    def __len__(self) -> int:\n        return len(self.study_ids)\n\n    def __getitem__(self, index: int):\n        study_uid = self.study_ids[index]\n        volume = np.zeros((64, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)\n        slice_mask = np.zeros(64, dtype=np.uint8)\n        selections: list[dict[str, object]] = []\n        used: set[str] = set()\n        cursor = 0\n        for slot in SLOT_SPECS:\n            path, series_uid = choose_series(\n                self.dicom_root,\n                study_uid,\n                self.series_by_study.get(study_uid, []),\n                slot,\n                used,\n            )\n            stop = cursor + int(slot["count"])\n            if path is not None:\n                volume[cursor:stop], slice_mask[cursor:stop] = build_slot(\n                    path, int(slot["count"])\n                )\n            selections.append(\n                {\n                    "slot": slot["name"],\n                    "series_uid": series_uid,\n                    "valid_slices": int(slice_mask[cursor:stop].sum()),\n                }\n            )\n            cursor = stop\n        centers, window_mask = choose_centers(slice_mask, K_EVAL)\n        if not window_mask.any():\n            raise RuntimeError(f"{study_uid} has no valid 2.5D windows")\n        windows = np.zeros(\n            (K_EVAL, 3, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8\n        )\n        for output_index, center in enumerate(centers):\n            if window_mask[output_index]:\n                windows[output_index] = volume[\n                    int(center) - 1:int(center) + 2\n                ]\n        return (\n            index,\n            windows,\n            window_mask.astype(np.uint8),\n            centers,\n            json.dumps(selections, separators=(",", ":")),\n        )\n\n\nclass TargetAttentionHead(nn.Module):\n    def __init__(self, feature_dim: int = FEATURE_DIM, dropout: float = 0.20) -> None:\n        super().__init__()\n        self.norm = nn.LayerNorm(feature_dim)\n        self.attention = nn.Sequential(\n            nn.Linear(feature_dim, 256),\n            nn.Tanh(),\n            nn.Dropout(dropout),\n            nn.Linear(256, N_TARGETS),\n        )\n        self.target_weight = nn.Parameter(\n            torch.randn(N_TARGETS, feature_dim) * 0.02\n        )\n        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))\n\n    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:\n        hidden = self.norm(features)\n        scores = self.attention(hidden).transpose(1, 2)\n        scores = scores.masked_fill(~mask[:, None, :].bool(), -1e4)\n        attention = scores.softmax(dim=-1)\n        pooled = torch.einsum("btk,bkf->btf", attention, hidden)\n        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias\n\n\nclass CoAtNetMIL(nn.Module):\n    def __init__(self, *, channels_last: bool) -> None:\n        super().__init__()\n        backbone = timm.create_model(ARCH, pretrained=False)\n        backbone.reset_classifier(0, global_pool="avg")\n        assert backbone.num_features == FEATURE_DIM\n        if hasattr(backbone, "set_grad_checkpointing"):\n            backbone.set_grad_checkpointing(enable=False)\n        self.backbone = backbone\n        self.head = TargetAttentionHead()\n        self.channels_last = channels_last\n        self.register_buffer(\n            "mean", torch.tensor(MODEL_MEAN).view(1, 3, 1, 1)\n        )\n        self.register_buffer(\n            "std", torch.tensor(MODEL_STD).view(1, 3, 1, 1)\n        )\n\n    def encode(self, windows: torch.Tensor) -> torch.Tensor:\n        batch, count = windows.shape[:2]\n        pixels = windows.reshape(\n            batch * count, 3, CACHE_SIZE, CACHE_SIZE\n        ).float().div(255.0)\n        pixels = F.interpolate(\n            pixels,\n            size=(MODEL_SIZE, MODEL_SIZE),\n            mode="bicubic",\n            align_corners=False,\n        )\n        if self.channels_last:\n            pixels = pixels.contiguous(memory_format=torch.channels_last)\n        pixels = (pixels - self.mean) / self.std\n        return self.backbone(pixels).reshape(batch, count, FEATURE_DIM)\n\n\ndef head_predictions(\n    head: TargetAttentionHead,\n    features: np.ndarray,\n    mask: np.ndarray,\n    device: torch.device,\n) -> np.ndarray:\n    predictions: list[np.ndarray] = []\n    head.eval()\n    with torch.inference_mode():\n        for start in range(0, len(features), 64):\n            feature = torch.from_numpy(\n                np.asarray(features[start:start + 64]).astype(np.float32)\n            ).to(device)\n            valid = torch.from_numpy(\n                np.asarray(mask[start:start + 64]).astype(bool)\n            ).to(device)\n            predictions.append(torch.sigmoid(head(feature, valid)).cpu().numpy())\n    return np.concatenate(predictions)\n\n\ndef gpu_snapshot() -> list[list[str]] | None:\n    if not torch.cuda.is_available():\n        return None\n    result = subprocess.run(\n        [\n            "nvidia-smi",\n            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",\n            "--format=csv,noheader,nounits",\n        ],\n        check=True,\n        capture_output=True,\n        text=True,\n    )\n    return [line.split(", ") for line in result.stdout.splitlines()]\n\n\ndef main() -> int:\n    started = time.time()\n    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n    p2_root = find_p2_root()\n    p2_manifest_path = p2_root / "p2_manifest.json"\n    assert sha256_file(p2_manifest_path) == EXPECTED_P2_MANIFEST_SHA256\n    p2_manifest = json.loads(p2_manifest_path.read_text())\n    assert p2_manifest["run_id"] == "P2-COATNET-v1"\n    assert p2_manifest["best_epoch"] == 8\n    assert p2_manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE\n    required_model_artifacts = [\n        "p2_coatnet_pseudo_best.pt",\n        *[f"p2_fold_{fold}_head.pt" for fold in range(5)],\n    ]\n    for name in required_model_artifacts:\n        path = p2_root / name\n        assert path.is_file() and sha256_file(path) == p2_manifest[\n            "artifacts_sha256"\n        ][name]\n\n    if ON_KAGGLE:\n        data_root = find_competition_root()\n        dicom_root = data_root / "test_series"\n        test = pd.read_csv(data_root / "test.csv", dtype={ID: str})\n        series = pd.read_csv(data_root / "test_series.csv", dtype={ID: str})\n        sample = pd.read_csv(data_root / "sample_submission.csv", dtype={ID: str})\n        workers = 2\n    else:\n        data_root = Path("data/raw").resolve()\n        dicom_root = Path("data/sample/files/train_series").resolve()\n        metadata = pd.read_csv(data_root / "train.csv", dtype={ID: str})\n        series_all = pd.read_csv(data_root / "train_series.csv", dtype={ID: str})\n        available = {path.name for path in dicom_root.iterdir() if path.is_dir()}\n        local_ids = [uid for uid in metadata[ID] if uid in available]\n        assert local_ids\n        richest = max(\n            local_ids,\n            key=lambda uid: sum(1 for _ in (dicom_root / uid).glob("*/*.dcm")),\n        )\n        test = pd.DataFrame({ID: [richest]})\n        series = series_all[series_all[ID].isin(test[ID])].copy()\n        sample = pd.DataFrame({ID: test[ID], **{target: 0.5 for target in TARGETS}})\n        workers = 0\n\n    assert dicom_root.is_dir()\n    assert test[ID].is_unique and len(test) > 0\n    assert sample.columns.tolist() == [ID, *TARGETS]\n    assert sample[ID].tolist() == test[ID].tolist()\n    assert set(series[ID].unique()) == set(test[ID])\n    study_ids = test[ID].astype(str).tolist()\n    series_by_study = {\n        str(uid): group.to_dict("records")\n        for uid, group in series.groupby(ID, sort=False)\n    }\n\n    print(\n        {\n            "P2_SUBMISSION_START": True,\n            "on_kaggle": ON_KAGGLE,\n            "studies": len(study_ids),\n            "series": len(series),\n            "p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,\n            "python": platform.python_version(),\n            "torch": torch.__version__,\n            "timm": timm.__version__,\n            "visible_gpus": torch.cuda.device_count(),\n        },\n        flush=True,\n    )\n\n    dataset = TestStudyDataset(study_ids, series_by_study, dicom_root)\n    loader_options: dict[str, object] = {\n        "batch_size": 1,\n        "shuffle": False,\n        "num_workers": workers,\n        "pin_memory": torch.cuda.is_available(),\n    }\n    if workers:\n        loader_options.update(\n            {\n                "prefetch_factor": 2,\n                "persistent_workers": True,\n                "multiprocessing_context": "spawn",\n            }\n        )\n    loader = DataLoader(dataset, **loader_options)\n\n    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n    channels_last = device.type == "cuda"\n    if channels_last:\n        torch.backends.cudnn.benchmark = True\n        torch.backends.cudnn.deterministic = True\n    model = CoAtNetMIL(channels_last=channels_last)\n    state = torch.load(\n        p2_root / "p2_coatnet_pseudo_best.pt",\n        map_location="cpu",\n        weights_only=True,\n    )\n    model.load_state_dict(state, strict=True)\n    del state\n    if channels_last:\n        model = model.to(device=device, memory_format=torch.channels_last)\n    else:\n        model = model.to(device)\n    model.eval()\n\n    features = np.zeros((len(test), K_EVAL, FEATURE_DIM), dtype=np.float16)\n    feature_mask = np.zeros((len(test), K_EVAL), dtype=np.uint8)\n    positions = np.full((len(test), K_EVAL), -1, dtype=np.int16)\n    selection_rows: list[dict[str, object]] = []\n    inference_started = time.time()\n    with torch.inference_mode():\n        for processed, batch in enumerate(loader, start=1):\n            indices, windows, window_mask, centers, selections_json = batch\n            index = int(indices.item())\n            windows = windows.to(device, non_blocking=True)\n            window_mask = window_mask.to(device, non_blocking=True).bool()\n            with torch.autocast(\n                device_type=device.type,\n                dtype=torch.float16,\n                enabled=device.type == "cuda",\n            ):\n                encoded = model.encode(windows)\n            features[index] = encoded.float().cpu().numpy()[0].astype(np.float16)\n            selected_mask = window_mask.cpu().numpy()[0]\n            feature_mask[index] = selected_mask.astype(np.uint8)\n            selected_centers = centers.numpy()[0]\n            positions[index, selected_mask] = selected_centers[selected_mask]\n            for selection in json.loads(selections_json[0]):\n                selection_rows.append({ID: study_ids[index], **selection})\n            if processed % 50 == 0 or processed == len(test):\n                print(\n                    {\n                        "P2_SUBMISSION_PROGRESS": f"{processed}/{len(test)}",\n                        "elapsed_minutes": (time.time() - inference_started) / 60.0,\n                        "gpu": gpu_snapshot(),\n                    },\n                    flush=True,\n                )\n\n    assert np.isfinite(features).all()\n    assert feature_mask.any(axis=1).all()\n    assert (positions[feature_mask.astype(bool)] >= 1).all()\n    del model\n    if device.type == "cuda":\n        torch.cuda.empty_cache()\n\n    fold_predictions = np.zeros((5, len(test), N_TARGETS), dtype=np.float32)\n    for fold in range(5):\n        head = TargetAttentionHead().to(device)\n        fold_state = torch.load(\n            p2_root / f"p2_fold_{fold}_head.pt",\n            map_location="cpu",\n            weights_only=True,\n        )\n        head.load_state_dict(fold_state, strict=True)\n        fold_predictions[fold] = head_predictions(\n            head, features, feature_mask.astype(bool), device\n        )\n        del head, fold_state\n\n    direct_state = torch.load(\n        p2_root / "p2_coatnet_pseudo_best.pt",\n        map_location="cpu",\n        weights_only=True,\n    )\n    direct_head = TargetAttentionHead().to(device)\n    direct_head.load_state_dict(\n        {\n            key.removeprefix("head."): value\n            for key, value in direct_state.items()\n            if key.startswith("head.")\n        },\n        strict=True,\n    )\n    direct_predictions = head_predictions(\n        direct_head, features, feature_mask.astype(bool), device\n    )\n    submission_predictions = np.clip(\n        fold_predictions.mean(axis=0), 1e-6, 1.0 - 1e-6\n    )\n    assert submission_predictions.shape == (len(test), N_TARGETS)\n    assert np.isfinite(submission_predictions).all()\n    assert ((submission_predictions > 0) & (submission_predictions < 1)).all()\n    if len(test) > 1:\n        assert (np.ptp(submission_predictions, axis=0) > 0).all()\n\n    submission = sample.copy()\n    submission.loc[:, TARGETS] = submission_predictions\n    assert submission.columns.tolist() == [ID, *TARGETS]\n    assert submission[ID].tolist() == study_ids\n    submission_path = OUTPUT_ROOT / "submission.csv"\n    submission.to_csv(submission_path, index=False)\n    np.save(OUTPUT_ROOT / "p2_test_predictions_by_fold.npy", fold_predictions)\n    np.save(OUTPUT_ROOT / "p2_test_feature_mask.npy", feature_mask)\n    np.save(OUTPUT_ROOT / "p2_test_feature_positions.npy", positions)\n    pd.DataFrame(selection_rows).to_csv(\n        OUTPUT_ROOT / "p2_test_selections.csv", index=False\n    )\n    pd.DataFrame({ID: study_ids, **{\n        target: direct_predictions[:, target_index]\n        for target_index, target in enumerate(TARGETS)\n    }}).to_csv(OUTPUT_ROOT / "p2_test_direct_predictions.csv", index=False)\n\n    artifacts = [\n        "submission.csv",\n        "p2_test_predictions_by_fold.npy",\n        "p2_test_feature_mask.npy",\n        "p2_test_feature_positions.npy",\n        "p2_test_selections.csv",\n        "p2_test_direct_predictions.csv",\n    ]\n    manifest = {\n        "schema_version": 1,\n        "run_id": "P2-SUBMISSION-v1",\n        "studies": len(test),\n        "series": len(series),\n        "targets": TARGETS,\n        "source_p2_run_id": p2_manifest["run_id"],\n        "source_p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,\n        "source_p2_best_epoch": p2_manifest["best_epoch"],\n        "source_p2_oof_macro_auc": p2_manifest["fold_oof"]["macro_auc"],\n        "geometry": {\n            "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,\n            "slot_specs": SLOT_SPECS,\n            "slot_bounds": SLOT_BOUNDS,\n            "relative_slice_range": [0.02, 0.98],\n            "physical_crop_mm": CROP_MM,\n            "cache_size": CACHE_SIZE,\n            "model_size": MODEL_SIZE,\n            "k_eval": K_EVAL,\n        },\n        "ensemble": "mean probabilities from five report-purged fold heads",\n        "submission_file": "submission.csv",\n        "test_metadata_sha256": {\n            "test.csv": sha256_file(data_root / (\n                "test.csv" if ON_KAGGLE else "train.csv"\n            )),\n            "test_series.csv": sha256_file(data_root / (\n                "test_series.csv" if ON_KAGGLE else "train_series.csv"\n            )),\n        },\n        "artifacts_sha256": {\n            name: sha256_file(OUTPUT_ROOT / name) for name in artifacts\n        },\n        "elapsed": {\n            "dicom_and_feature_minutes": (time.time() - inference_started) / 60.0,\n            "total_minutes": (time.time() - started) / 60.0,\n        },\n        "environment": {\n            "python": platform.python_version(),\n            "torch": torch.__version__,\n            "timm": timm.__version__,\n            "pydicom": pydicom.__version__,\n            "device": str(device),\n            "visible_gpus": torch.cuda.device_count(),\n            "loader_workers": workers,\n        },\n    }\n    manifest_path = OUTPUT_ROOT / "p2_submission_manifest.json"\n    manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n")\n    print(\n        {\n            "P2_SUBMISSION_COMPLETE": True,\n            "studies": len(submission),\n            "submission_sha256": sha256_file(submission_path),\n            "manifest_sha256": sha256_file(manifest_path),\n            "prediction_ranges": {\n                target: [\n                    float(submission[target].min()),\n                    float(submission[target].max()),\n                ]\n                for target in TARGETS\n            },\n            "elapsed_minutes": manifest["elapsed"]["total_minutes"],\n        },\n        flush=True,\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n', prep.__file__, 'exec'), prep.__dict__)
p5 = _bundle_types.ModuleType('p5_embedded_common')
p5.__file__ = '<embedded-p5-common>'
exec(compile('#!/usr/bin/env python3\n"""Benchmark the frozen P5 OrthoFoundation feature path on two Kaggle T4s."""\n\n# ruff: noqa: E402 - runtime paths and offline flags precede third-party imports.\n\nfrom __future__ import annotations\n\nimport csv\nimport hashlib\nimport json\nimport math\nimport os\nimport platform\nimport random\nimport subprocess\nimport sys\nimport threading\nimport time\nfrom pathlib import Path\n\nos.environ.setdefault("HF_HUB_OFFLINE", "1")\nos.environ.setdefault("TRANSFORMERS_OFFLINE", "1")\nos.environ.setdefault("TORCH_HOME", "/kaggle/working/torch-cache")\n\nimport numpy as np\nimport pandas as pd\nimport torch\nimport torch.distributed as dist\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader, Dataset\n\n\nID = "StudyInstanceUID"\nTARGETS = [\n    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",\n    "Medial OA", "Lateral OA", "PF OA", "Effusion",\n    "Synovitis", "Baker\'s", "Contusion", "Fracture",\n]\nRUN_ID = "P5A-ORTHOFOUNDATION-FROZEN-v1"\nEXECUTION_REVISION = "p5a_frozen_t4x2_smoke_v2"\nCACHE_ID = "P2-CACHE-v1"\nCACHE_SIGNATURE = "0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3"\nMODEL_SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"\nMODEL_BYTES = 1_213_056_638\nORTHO_COMMIT = "4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470"\nDINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"\nSEED = 20260904\nCACHE_SIZE = 336\nMODEL_SIZE = 224\nBACKBONE_DIM = 1024\nN_REGIONS = 2\nHEAD_DIM = 256\nK_SLICES = 32\nSLOT_QUOTAS = (9, 7, 6, 4, 6)\nSMOKE_STUDIES = 128\nSTUDY_BATCH_PER_RANK = 2\nWORKERS_PER_RANK = 2\nHEAD_SMOKE_STEPS = 5\nMODEL_MEAN = (0.485, 0.456, 0.406)\nMODEL_STD = (0.229, 0.224, 0.225)\nOUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(\n    "outputs/p5_orthofoundation_speed_smoke"\n)\n\n\ndef launch() -> int:\n    visible = torch.cuda.device_count()\n    if visible != 2:\n        raise RuntimeError(f"P5 speed smoke requires exactly two GPUs; observed {visible}")\n    command = [\n        sys.executable,\n        "-m",\n        "torch.distributed.run",\n        "--standalone",\n        "--nproc_per_node=2",\n        str(Path(__file__).resolve()),\n        "--worker",\n    ]\n    print({"P5_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)\n    subprocess.run(command, check=True)\n    print({"P5_DDP_PROCESS_COMPLETE": True}, flush=True)\n    return 0\n\n\ndef sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:\n    digest = hashlib.sha256()\n    with path.open("rb") as handle:\n        while chunk := handle.read(chunk_size):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef candidate_roots() -> list[Path]:\n    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]\n    if Path("/kaggle/input").is_dir():\n        roots.append(Path("/kaggle/input"))\n    return roots\n\n\ndef find_named_file(filename: str, minimum_bytes: int = 1) -> Path:\n    direct_map = {\n        "train.csv": Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"),\n        "training_targets.csv": Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-report-weak-supervision/training_targets.csv"\n        ),\n        "p2_cache_manifest_part_a.json": Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-p2-wide-cache-a/p2_cache_manifest_part_a.json"\n        ),\n        "p2_cache_manifest_part_b.json": Path(\n            "/kaggle/input/notebooks/vallepuvamsikrishna/"\n            "rsna-knee-p2-wide-cache-b/p2_cache_manifest_part_b.json"\n        ),\n    }\n    candidates = [direct_map[filename]] if filename in direct_map else []\n    for root in candidate_roots():\n        candidates.extend(\n            [\n                root / filename,\n                root / "data/raw" / filename,\n                root / "outputs/kaggle/rsna-report-weak-supervision" / filename,\n                root / "kaggle/rsna-p2-wide-cache-a/outputs/p2_cache_smoke_a" / filename,\n            ]\n        )\n    for path in candidates:\n        if path.is_file() and path.stat().st_size >= minimum_bytes:\n            return path.resolve()\n    kaggle_input = Path("/kaggle/input")\n    if kaggle_input.is_dir():\n        for path in kaggle_input.glob(f"**/{filename}"):\n            if path.is_file() and path.stat().st_size >= minimum_bytes:\n                return path.resolve()\n    raise FileNotFoundError(filename)\n\n\ndef find_runtime() -> tuple[Path, Path]:\n    for root in candidate_roots():\n        for checkpoint in root.glob("**/OrthoFoudation-L.pth"):\n            runtime = checkpoint.parent\n            if (runtime / "dinov3/hubconf.py").is_file():\n                return runtime.resolve(), checkpoint.resolve()\n    raise FileNotFoundError("OrthoFoundation-L checkpoint plus bundled DINOv3 runtime")\n\n\ndef cache_part(part: str) -> dict[str, object]:\n    manifest_path = find_named_file(f"p2_cache_manifest_part_{part}.json")\n    manifest = json.loads(manifest_path.read_text())\n    assert manifest["cache_id"] == CACHE_ID\n    assert manifest["cache_signature_sha256"] == CACHE_SIGNATURE\n    root = manifest_path.parent\n    result: dict[str, object] = {"manifest": manifest, "root": root}\n    for key, name in {\n        "pixels": f"p2_pixels_part_{part}.npy",\n        "mask": f"p2_valid_mask_part_{part}.npy",\n        "studies": f"p2_studies_part_{part}.csv",\n    }.items():\n        path = root / name\n        assert path.is_file(), path\n        result[key] = path\n    return result\n\n\nclass GpuTelemetry:\n    def __init__(self, output_path: Path, interval_seconds: int = 1) -> None:\n        self.output_path = output_path\n        self.interval_seconds = interval_seconds\n        self.stop = threading.Event()\n        self.thread: threading.Thread | None = None\n\n    def start(self) -> None:\n        self.output_path.write_text(\n            "unix_time,timestamp,index,name,utilization_gpu_percent,"\n            "utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,"\n            "temperature_c\\n"\n        )\n        self.thread = threading.Thread(target=self._sample, daemon=True)\n        self.thread.start()\n\n    def _sample(self) -> None:\n        query = (\n            "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,"\n            "memory.total,power.draw,temperature.gpu"\n        )\n        while not self.stop.is_set():\n            try:\n                result = subprocess.run(\n                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],\n                    check=True,\n                    capture_output=True,\n                    text=True,\n                )\n                rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))\n                now = time.time()\n                with self.output_path.open("a", newline="") as handle:\n                    writer = csv.writer(handle)\n                    for row in rows:\n                        writer.writerow([now, *row])\n                print({"P5_GPU": rows}, flush=True)\n            except Exception as error:\n                print({"P5_GPU_TELEMETRY_ERROR": repr(error)}, flush=True)\n            self.stop.wait(self.interval_seconds)\n\n    def close(self) -> None:\n        self.stop.set()\n        if self.thread is not None:\n            self.thread.join(timeout=10)\n\n\nclass HierarchicalDiagnosisHead(nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n        self.input_norm = nn.LayerNorm(BACKBONE_DIM)\n        self.projection = nn.Sequential(\n            nn.Linear(BACKBONE_DIM, HEAD_DIM),\n            nn.GELU(),\n            nn.Dropout(0.10),\n        )\n        self.region_embedding = nn.Parameter(torch.randn(N_REGIONS, HEAD_DIM) * 0.02)\n        self.slot_embedding = nn.Parameter(torch.randn(len(SLOT_QUOTAS), HEAD_DIM) * 0.02)\n        self.position_mlp = nn.Sequential(\n            nn.Linear(1, HEAD_DIM), nn.GELU(), nn.Linear(HEAD_DIM, HEAD_DIM)\n        )\n        self.region_query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)\n        self.slice_query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)\n        self.target_weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)\n        self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))\n\n    def forward(\n        self,\n        features: torch.Tensor,\n        mask: torch.Tensor,\n        slots: torch.Tensor,\n        positions: torch.Tensor,\n    ) -> torch.Tensor:\n        hidden = self.projection(self.input_norm(features.float()))\n        hidden = hidden + self.region_embedding[None, None]\n        region_scores = torch.einsum("bkrh,th->bkrt", hidden, self.region_query)\n        region_attention = (region_scores / math.sqrt(HEAD_DIM)).softmax(dim=2)\n        local = torch.einsum("bkrt,bkrh->bkth", region_attention, hidden)\n        context = self.slot_embedding[slots.clamp(0, len(SLOT_QUOTAS) - 1)]\n        context = context + self.position_mlp(positions[..., None])\n        local = local + context[:, :, None]\n        slice_scores = torch.einsum("bkth,th->bkt", torch.tanh(local), self.slice_query)\n        slice_scores = (slice_scores / math.sqrt(HEAD_DIM)).masked_fill(\n            ~mask[:, :, None], -1e4\n        )\n        slice_attention = slice_scores.softmax(dim=1)\n        pooled = torch.einsum("bkt,bkth->bth", slice_attention, local)\n        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias\n\n\ndef worker() -> int:\n    rank = int(os.environ["RANK"])\n    local_rank = int(os.environ["LOCAL_RANK"])\n    world_size = int(os.environ["WORLD_SIZE"])\n    assert world_size == 2\n    torch.cuda.set_device(local_rank)\n    device = torch.device("cuda", local_rank)\n    dist.init_process_group("nccl", device_id=device)\n    is_primary = rank == 0\n    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n    telemetry = GpuTelemetry(OUTPUT_ROOT / "p5_speed_gpu_telemetry.csv") if is_primary else None\n    if telemetry is not None:\n        telemetry.start()\n\n    random.seed(SEED + rank)\n    np.random.seed(SEED + rank)\n    torch.manual_seed(SEED + rank)\n    torch.cuda.manual_seed_all(SEED + rank)\n    torch.backends.cudnn.benchmark = True\n    torch.set_float32_matmul_precision("high")\n\n    runtime, checkpoint = find_runtime()\n    part_a, part_b = cache_part("a"), cache_part("b")\n    if is_primary:\n        assert checkpoint.stat().st_size == MODEL_BYTES\n        assert sha256_file(checkpoint) == MODEL_SHA256\n        provenance = (runtime / "PROVENANCE.md").read_text()\n        assert ORTHO_COMMIT in provenance and DINOV3_COMMIT in provenance\n        assert (runtime / "ORTHOFOUNDATION_LICENSE").is_file()\n        assert (runtime / "dinov3/LICENSE.md").is_file()\n        print(\n            {\n                "P5_SPEED_INPUTS_OK": True,\n                "checkpoint": str(checkpoint),\n                "checkpoint_sha256": MODEL_SHA256,\n                "runtime": str(runtime),\n            },\n            flush=True,\n        )\n    dist.barrier()\n\n    train_path = find_named_file("train.csv", 100_000)\n    target_path = find_named_file("training_targets.csv", 100_000)\n    train = pd.read_csv(train_path, dtype={ID: str})\n    targets = pd.read_csv(target_path, dtype={ID: str}).set_index(ID).loc[train[ID]].reset_index()\n    assert len(train) == len(targets) == 4_407\n    pixels_a = np.load(part_a["pixels"], mmap_mode="r")\n    pixels_b = np.load(part_b["pixels"], mmap_mode="r")\n    masks_a = np.load(part_a["mask"], mmap_mode="r")\n    masks_b = np.load(part_b["mask"], mmap_mode="r")\n    assert len(pixels_a) == 2_204 and len(pixels_b) == 2_203\n    slot_bounds = np.load(Path(part_a["root"]) / "p2_slot_bounds.npy").astype(int)\n    assert slot_bounds.tolist() == [[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]]\n\n    def read_study(index: int) -> tuple[np.ndarray, np.ndarray]:\n        if index < len(pixels_a):\n            return np.asarray(pixels_a[index]), np.asarray(masks_a[index]).astype(bool)\n        local = index - len(pixels_a)\n        return np.asarray(pixels_b[local]), np.asarray(masks_b[local]).astype(bool)\n\n    def select_slices(index: int):\n        volume, valid_mask = read_study(index)\n        slices = np.zeros((K_SLICES, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)\n        selected_mask = np.zeros(K_SLICES, dtype=bool)\n        slots = np.zeros(K_SLICES, dtype=np.int64)\n        positions = np.zeros(K_SLICES, dtype=np.float32)\n        cursor = 0\n        for slot, ((start, stop), quota) in enumerate(zip(slot_bounds, SLOT_QUOTAS)):\n            available = np.where(valid_mask[start:stop])[0] + start\n            count = min(len(available), quota)\n            if count:\n                if len(available) <= quota:\n                    chosen = available\n                else:\n                    chosen = available[np.round(np.linspace(0, len(available) - 1, quota)).astype(int)]\n                chosen = np.asarray(chosen[:count], dtype=int)\n                out = slice(cursor, cursor + count)\n                slices[out] = volume[chosen]\n                selected_mask[out] = True\n                slots[out] = slot\n                positions[out] = 2.0 * ((chosen - start) / max(stop - start - 1, 1)) - 1.0\n            slots[cursor:cursor + quota] = slot\n            cursor += quota\n        assert cursor == K_SLICES and selected_mask.any()\n        labels = targets.loc[index, TARGETS].to_numpy(np.float32)\n        weights = targets.loc[index, [f"weight__{target}" for target in TARGETS]].to_numpy(\n            np.float32\n        )\n        return index, slices, selected_mask, slots, positions, labels, weights\n\n    smoke_indices = np.round(np.linspace(0, len(train) - 1, SMOKE_STUDIES)).astype(int)\n    assert len(np.unique(smoke_indices)) == SMOKE_STUDIES\n    local_indices = smoke_indices[rank::world_size]\n\n    class SmokeDataset(Dataset):\n        def __len__(self) -> int:\n            return len(local_indices)\n\n        def __getitem__(self, position: int):\n            return select_slices(int(local_indices[position]))\n\n    loader = DataLoader(\n        SmokeDataset(),\n        batch_size=STUDY_BATCH_PER_RANK,\n        shuffle=False,\n        num_workers=WORKERS_PER_RANK,\n        pin_memory=True,\n        prefetch_factor=2,\n    )\n\n    load_started = time.time()\n    model = torch.hub.load(\n        str(runtime / "dinov3"),\n        "dinov3_vitl16",\n        source="local",\n        pretrained=False,\n    )\n    state = torch.load(checkpoint, map_location="cpu", weights_only=True)\n    assert isinstance(state, dict) and state\n    original_keys = list(state)\n    if all(key.startswith("backbone.") for key in original_keys):\n        state = {key.removeprefix("backbone."): value for key, value in state.items()}\n        checkpoint_namespace = "backbone."\n    elif all(key.startswith("module.backbone.") for key in original_keys):\n        state = {key.removeprefix("module.backbone."): value for key, value in state.items()}\n        checkpoint_namespace = "module.backbone."\n    else:\n        checkpoint_namespace = "unprefixed"\n    assert set(state) == set(model.state_dict()), {\n        "missing": sorted(set(model.state_dict()) - set(state))[:10],\n        "unexpected": sorted(set(state) - set(model.state_dict()))[:10],\n    }\n    model.load_state_dict(state, strict=True)\n    del state\n    model.requires_grad_(False).eval().to(device)\n    total_parameters = sum(parameter.numel() for parameter in model.parameters())\n    assert 300_000_000 < total_parameters < 310_000_000\n    model_load_seconds = time.time() - load_started\n    dist.barrier()\n    if is_primary:\n        print(\n            {\n                "P5_MODEL_READY": True,\n                "parameters": total_parameters,\n                "load_seconds": model_load_seconds,\n                "world_size": world_size,\n                "checkpoint_namespace": checkpoint_namespace,\n            },\n            flush=True,\n        )\n\n    mean = torch.tensor(MODEL_MEAN, device=device).view(1, 3, 1, 1)\n    std = torch.tensor(MODEL_STD, device=device).view(1, 3, 1, 1)\n    feature_batches: list[torch.Tensor] = []\n    mask_batches: list[torch.Tensor] = []\n    slot_batches: list[torch.Tensor] = []\n    position_batches: list[torch.Tensor] = []\n    label_batches: list[torch.Tensor] = []\n    weight_batches: list[torch.Tensor] = []\n    torch.cuda.reset_peak_memory_stats(device)\n    extract_started = time.time()\n    with torch.inference_mode():\n        for step, batch in enumerate(loader, start=1):\n            _, slices, slice_mask, slots, positions, labels, weights = batch\n            slices = slices.to(device, non_blocking=True)\n            batch_size, count = slices.shape[:2]\n            pixels = slices.reshape(-1, 1, CACHE_SIZE, CACHE_SIZE).float().div_(255.0)\n            pixels = F.interpolate(\n                pixels, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False\n            )\n            pixels = pixels.expand(-1, 3, -1, -1)\n            pixels = ((pixels - mean) / std).contiguous(memory_format=torch.channels_last)\n            with torch.autocast("cuda", dtype=torch.float16):\n                encoded = model.forward_features(pixels)\n                cls = encoded["x_norm_clstoken"]\n                patch_mean = encoded["x_norm_patchtokens"].mean(dim=1)\n                features = torch.stack((cls, patch_mean), dim=1)\n            features = features.reshape(batch_size, count, N_REGIONS, BACKBONE_DIM)\n            assert torch.isfinite(features).all()\n            feature_batches.append(features.half())\n            mask_batches.append(slice_mask.to(device, non_blocking=True).bool())\n            slot_batches.append(slots.to(device, non_blocking=True).long())\n            position_batches.append(positions.to(device, non_blocking=True).float())\n            label_batches.append(labels.to(device, non_blocking=True).float())\n            weight_batches.append(weights.to(device, non_blocking=True).float())\n            if step == 1 or step == len(loader):\n                print(\n                    {\n                        "P5_FEATURE_PROGRESS": f"rank={rank} step={step}/{len(loader)}",\n                        "feature_shape": list(features.shape),\n                    },\n                    flush=True,\n                )\n    torch.cuda.synchronize(device)\n    extraction_seconds = time.time() - extract_started\n\n    features_all = torch.cat(feature_batches)\n    masks_all = torch.cat(mask_batches)\n    slots_all = torch.cat(slot_batches)\n    positions_all = torch.cat(position_batches)\n    labels_all = torch.cat(label_batches)\n    weights_all = torch.cat(weight_batches)\n    head = HierarchicalDiagnosisHead().to(device)\n    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3, fused=True)\n    head_started = time.time()\n    final_loss = float("nan")\n    for step in range(HEAD_SMOKE_STEPS):\n        generator = torch.Generator(device=device).manual_seed(SEED + rank + step)\n        chosen = torch.randperm(len(features_all), generator=generator, device=device)[:64]\n        optimizer.zero_grad(set_to_none=True)\n        with torch.autocast("cuda", dtype=torch.float16):\n            logits = head(\n                features_all[chosen],\n                masks_all[chosen],\n                slots_all[chosen],\n                positions_all[chosen],\n            )\n            raw = F.binary_cross_entropy_with_logits(logits, labels_all[chosen], reduction="none")\n            loss = (raw * weights_all[chosen]).sum() / weights_all[chosen].sum().clamp_min(1.0)\n        assert torch.isfinite(loss)\n        loss.backward()\n        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)\n        optimizer.step()\n        final_loss = float(loss.detach())\n    torch.cuda.synchronize(device)\n    head_seconds = time.time() - head_started\n    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)\n\n    local_stats = torch.tensor(\n        [\n            model_load_seconds,\n            extraction_seconds,\n            head_seconds,\n            peak_memory_mib,\n            float(len(local_indices)),\n            final_loss,\n        ],\n        dtype=torch.float64,\n        device=device,\n    )\n    gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]\n    dist.all_gather(gathered, local_stats)\n    dist.barrier()\n    if is_primary:\n        rows = [item.cpu().tolist() for item in gathered]\n        max_load = max(row[0] for row in rows)\n        max_extract = max(row[1] for row in rows)\n        global_images = sum(row[4] for row in rows) * K_SLICES\n        images_per_second = global_images / max_extract\n        projected_full_feature_minutes = (4_407 * K_SLICES / images_per_second) / 60.0\n        projected_hidden_feature_minutes = (1_300 * K_SLICES / images_per_second) / 60.0\n        result = {\n            "run_id": RUN_ID,\n            "execution_revision": EXECUTION_REVISION,\n            "python": platform.python_version(),\n            "torch": torch.__version__,\n            "world_size": world_size,\n            "smoke_studies": SMOKE_STUDIES,\n            "slices_per_study": K_SLICES,\n            "slot_quotas": list(SLOT_QUOTAS),\n            "study_batch_per_rank": STUDY_BATCH_PER_RANK,\n            "image_batch_per_rank": STUDY_BATCH_PER_RANK * K_SLICES,\n            "feature_shape_per_study": [K_SLICES, N_REGIONS, BACKBONE_DIM],\n            "checkpoint_sha256": MODEL_SHA256,\n            "checkpoint_bytes": MODEL_BYTES,\n            "checkpoint_namespace": checkpoint_namespace,\n            "orthofoundation_commit": ORTHO_COMMIT,\n            "dinov3_commit": DINOV3_COMMIT,\n            "cache_id": CACHE_ID,\n            "cache_signature_sha256": CACHE_SIGNATURE,\n            "model_parameters": total_parameters,\n            "model_load_seconds_by_rank": [row[0] for row in rows],\n            "extraction_seconds_by_rank": [row[1] for row in rows],\n            "images_per_second_two_gpu": images_per_second,\n            "projected_full_feature_minutes": projected_full_feature_minutes,\n            "projected_hidden_feature_minutes": projected_hidden_feature_minutes,\n            "head_smoke_steps": HEAD_SMOKE_STEPS,\n            "head_seconds_by_rank": [row[2] for row in rows],\n            "head_final_loss_by_rank": [row[5] for row in rows],\n            "peak_memory_mib_by_rank": [row[3] for row in rows],\n            "licenses": ["MIT", "DINOv3 License"],\n            "finite": True,\n            "estimated_full_fixed_model_load_minutes": max_load / 60.0,\n        }\n        (OUTPUT_ROOT / "p5_speed_smoke.json").write_text(json.dumps(result, indent=2) + "\\n")\n        print({"P5_SPEED_SMOKE_COMPLETE": result}, flush=True)\n        if telemetry is not None:\n            telemetry.close()\n    dist.barrier()\n    dist.destroy_process_group()\n    return 0\n\n\ndef main() -> int:\n    if "--worker" not in sys.argv:\n        return launch()\n    return worker()\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n', p5.__file__, 'exec'), p5.__dict__)


RUN_ID = "P5-DIAGNOSTIC-SUBMISSION-v1"
EXPECTED_P5_MANIFEST_SHA256 = (
    "3b739d18f85347bbc457885bba72b1a20712d2ab56dda67ac68fc35840b3163d"
)
EXPECTED_P5_OOF_MACRO_AUC = 0.7688217022757898
OUTPUT_ROOT = Path("/kaggle/working")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_p5_root() -> Path:
    candidates = [
        Path(
            "/kaggle/input/notebooks/vallepuvamsikrishna/"
            "rsna-knee-p5-orthofoundation-frozen"
        ),
        Path("outputs/kaggle/rsna-p5-orthofoundation-frozen-v3"),
    ]
    for candidate in candidates:
        manifest = candidate / "p5_manifest.json"
        if manifest.is_file() and sha256_file(manifest) == EXPECTED_P5_MANIFEST_SHA256:
            return candidate.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = [
            path.parent
            for path in kaggle_input.glob("**/p5_manifest.json")
            if sha256_file(path) == EXPECTED_P5_MANIFEST_SHA256
        ]
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError("validated P5 output")


class P5TestStudyDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        study_ids: list[str],
        series_by_study: dict[str, list[dict[str, object]]],
        dicom_root: Path,
    ) -> None:
        self.indices = indices
        self.study_ids = study_ids
        self.series_by_study = series_by_study
        self.dicom_root = dicom_root

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        study_uid = self.study_ids[index]
        volume = np.zeros((64, p5.CACHE_SIZE, p5.CACHE_SIZE), dtype=np.uint8)
        valid_mask = np.zeros(64, dtype=np.uint8)
        selections: list[dict[str, object]] = []
        used: set[str] = set()
        cursor = 0
        for slot in prep.SLOT_SPECS:
            path, series_uid = prep.choose_series(
                self.dicom_root,
                study_uid,
                self.series_by_study.get(study_uid, []),
                slot,
                used,
            )
            stop = cursor + int(slot["count"])
            if path is not None:
                volume[cursor:stop], valid_mask[cursor:stop] = prep.build_slot(
                    path, int(slot["count"])
                )
            selections.append(
                {
                    "slot": slot["name"],
                    "series_uid": series_uid,
                    "valid_slices": int(valid_mask[cursor:stop].sum()),
                }
            )
            cursor = stop

        slices = np.zeros(
            (p5.K_SLICES, p5.CACHE_SIZE, p5.CACHE_SIZE), dtype=np.uint8
        )
        selected_mask = np.zeros(p5.K_SLICES, dtype=np.uint8)
        slots = np.zeros(p5.K_SLICES, dtype=np.int64)
        positions = np.zeros(p5.K_SLICES, dtype=np.float32)
        cursor = 0
        for slot_index, ((start, stop), quota) in enumerate(
            zip(prep.SLOT_BOUNDS, p5.SLOT_QUOTAS)
        ):
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
                output = slice(cursor, cursor + count)
                slices[output] = volume[chosen]
                selected_mask[output] = 1
                slots[output] = slot_index
                positions[output] = (
                    2.0 * ((chosen - start) / max(stop - start - 1, 1)) - 1.0
                )
            slots[cursor : cursor + quota] = slot_index
            cursor += quota
        if not selected_mask.any():
            raise RuntimeError(f"{study_uid} has no valid P5 slices")
        return (
            index,
            slices,
            selected_mask,
            slots,
            positions,
            json.dumps(selections, separators=(",", ":")),
        )


def load_backbone(device: torch.device):
    runtime, checkpoint = p5.find_runtime()
    assert checkpoint.stat().st_size == p5.MODEL_BYTES
    assert sha256_file(checkpoint) == p5.MODEL_SHA256
    provenance = (runtime / "PROVENANCE.md").read_text()
    assert p5.ORTHO_COMMIT in provenance and p5.DINOV3_COMMIT in provenance
    model = torch.hub.load(
        str(runtime / "dinov3"),
        "dinov3_vitl16",
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and state
    if all(key.startswith("backbone.") for key in state):
        state = {key.removeprefix("backbone."): value for key, value in state.items()}
    elif all(key.startswith("module.backbone.") for key in state):
        state = {
            key.removeprefix("module.backbone."): value for key, value in state.items()
        }
    assert set(state) == set(model.state_dict())
    model.load_state_dict(state, strict=True)
    del state
    return model.requires_grad_(False).eval().to(device)


def load_heads(
    root: Path, manifest: dict[str, object], device: torch.device
) -> list[p5.HierarchicalDiagnosisHead]:
    heads: list[p5.HierarchicalDiagnosisHead] = []
    artifact_hashes = manifest["artifacts_sha256"]
    for fold in range(5):
        name = f"p5_fold_{fold}_head.pt"
        path = root / name
        assert path.is_file() and sha256_file(path) == artifact_hashes[name]
        head = p5.HierarchicalDiagnosisHead().to(device)
        head.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        head.eval()
        heads.append(head)
    return heads


def launch() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P5 submission requires exactly two GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    print({"P5_SUBMISSION_DDP_LAUNCH": command, "visible_gpus": visible}, flush=True)
    subprocess.run(command, check=True)
    print({"P5_SUBMISSION_DDP_PROCESS_COMPLETE": True}, flush=True)
    return 0


def worker() -> int:
    started = time.time()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    p5_root = find_p5_root()
    manifest_path = p5_root / "p5_manifest.json"
    assert sha256_file(manifest_path) == EXPECTED_P5_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "P5A-ORTHOFOUNDATION-FROZEN-v1"
    assert manifest["fold_oof_macro_auc"] == EXPECTED_P5_OOF_MACRO_AUC
    assert manifest["representation_gold_rows"] == 0
    assert manifest["fold_report_hash_overlap"] == 0

    data_root = prep.find_competition_root()
    dicom_root = data_root / "test_series"
    test = pd.read_csv(data_root / "test.csv", dtype={p5.ID: str})
    series = pd.read_csv(data_root / "test_series.csv", dtype={p5.ID: str})
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={p5.ID: str})
    assert test[p5.ID].is_unique and len(test) > 0
    assert sample.columns.tolist() == [p5.ID, *p5.TARGETS]
    assert sample[p5.ID].tolist() == test[p5.ID].tolist()
    study_ids = test[p5.ID].astype(str).tolist()
    series_by_study = {
        str(uid): group.to_dict("records")
        for uid, group in series.groupby(p5.ID, sort=False)
    }
    local_indices = np.arange(len(test), dtype=np.int64)[rank::world_size]
    dataset = P5TestStudyDataset(
        local_indices, study_ids, series_by_study, dicom_root
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=len(dataset) > 2,
        multiprocessing_context="spawn",
    )

    model = load_backbone(device)
    heads = load_heads(p5_root, manifest, device)
    if rank == 0:
        print(
            {
                "P5_SUBMISSION_START": True,
                "studies": len(test),
                "series": len(series),
                "world_size": world_size,
                "source_manifest_sha256": EXPECTED_P5_MANIFEST_SHA256,
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            flush=True,
        )

    mean = torch.tensor(p5.MODEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(p5.MODEL_STD, device=device).view(1, 3, 1, 1)
    local_predictions = np.zeros(
        (5, len(local_indices), len(p5.TARGETS)), dtype=np.float32
    )
    local_masks = np.zeros((len(local_indices), p5.K_SLICES), dtype=np.uint8)
    local_slots = np.zeros((len(local_indices), p5.K_SLICES), dtype=np.int64)
    local_positions = np.zeros((len(local_indices), p5.K_SLICES), dtype=np.float32)
    selection_rows: list[dict[str, object]] = []
    offset = 0
    inference_started = time.time()
    with torch.inference_mode():
        for batch in loader:
            indices, slices, mask, slots, positions, selections_json = batch
            batch_size, count = slices.shape[:2]
            pixels = slices.to(device, non_blocking=True).reshape(
                -1, 1, p5.CACHE_SIZE, p5.CACHE_SIZE
            ).float().div_(255.0)
            pixels = F.interpolate(
                pixels,
                size=(p5.MODEL_SIZE, p5.MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            pixels = pixels.expand(-1, 3, -1, -1)
            pixels = ((pixels - mean) / std).contiguous(memory_format=torch.channels_last)
            mask_device = mask.to(device, non_blocking=True).bool()
            slots_device = slots.to(device, non_blocking=True).long()
            positions_device = positions.to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.float16):
                encoded = model.forward_features(pixels)
                features = torch.stack(
                    (
                        encoded["x_norm_clstoken"],
                        encoded["x_norm_patchtokens"].mean(dim=1),
                    ),
                    dim=1,
                ).reshape(batch_size, count, p5.N_REGIONS, p5.BACKBONE_DIM)
                probabilities = torch.stack(
                    [
                        torch.sigmoid(
                            head(features, mask_device, slots_device, positions_device)
                        )
                        for head in heads
                    ]
                )
            end = offset + batch_size
            local_predictions[:, offset:end] = probabilities.float().cpu().numpy()
            local_masks[offset:end] = mask.numpy().astype(np.uint8)
            local_slots[offset:end] = slots.numpy().astype(np.int64)
            local_positions[offset:end] = positions.numpy().astype(np.float32)
            for batch_offset, index in enumerate(indices.numpy().astype(int)):
                for selection in json.loads(selections_json[batch_offset]):
                    selection_rows.append({p5.ID: study_ids[index], **selection})
            offset = end
            if offset % 50 == 0 or offset == len(local_indices):
                print(
                    {
                        "P5_SUBMISSION_PROGRESS": f"rank={rank} {offset}/{len(local_indices)}",
                        "elapsed_minutes": (time.time() - inference_started) / 60.0,
                        "gpu": prep.gpu_snapshot(),
                    },
                    flush=True,
                )
    assert offset == len(local_indices)
    assert np.isfinite(local_predictions).all()
    assert local_masks.any(axis=1).all()
    np.savez(
        OUTPUT_ROOT / f"p5_submission_rank_{rank}.npz",
        indices=local_indices,
        predictions=local_predictions,
        masks=local_masks,
        slots=local_slots,
        positions=local_positions,
    )
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_ROOT / f"p5_submission_rank_{rank}_selections.csv", index=False
    )
    local_inference_seconds = torch.tensor(
        [time.time() - inference_started], dtype=torch.float64, device=device
    )
    inference_seconds_by_rank = [
        torch.zeros_like(local_inference_seconds) for _ in range(world_size)
    ]
    dist.all_gather(inference_seconds_by_rank, local_inference_seconds)
    dist.barrier()

    if rank == 0:
        fold_predictions = np.zeros(
            (5, len(test), len(p5.TARGETS)), dtype=np.float32
        )
        masks = np.zeros((len(test), p5.K_SLICES), dtype=np.uint8)
        slots_all = np.zeros((len(test), p5.K_SLICES), dtype=np.int64)
        positions_all = np.zeros((len(test), p5.K_SLICES), dtype=np.float32)
        selection_frames = []
        for shard_rank in range(world_size):
            shard = np.load(OUTPUT_ROOT / f"p5_submission_rank_{shard_rank}.npz")
            indices = shard["indices"].astype(int)
            fold_predictions[:, indices] = shard["predictions"]
            masks[indices] = shard["masks"]
            slots_all[indices] = shard["slots"]
            positions_all[indices] = shard["positions"]
            selection_frames.append(
                pd.read_csv(
                    OUTPUT_ROOT / f"p5_submission_rank_{shard_rank}_selections.csv",
                    dtype={p5.ID: str},
                )
            )
        predictions = np.clip(fold_predictions.mean(axis=0), 1e-6, 1.0 - 1e-6)
        assert np.isfinite(predictions).all()
        if len(test) > 1:
            assert (np.ptp(predictions, axis=0) > 0).all()
        submission = sample.copy()
        submission.loc[:, p5.TARGETS] = predictions
        submission_path = OUTPUT_ROOT / "submission.csv"
        submission.to_csv(submission_path, index=False)
        np.save(OUTPUT_ROOT / "p5_test_predictions_by_fold.npy", fold_predictions)
        np.save(OUTPUT_ROOT / "p5_test_mask.npy", masks)
        np.save(OUTPUT_ROOT / "p5_test_slots.npy", slots_all)
        np.save(OUTPUT_ROOT / "p5_test_positions.npy", positions_all)
        pd.concat(selection_frames, ignore_index=True).to_csv(
            OUTPUT_ROOT / "p5_test_selections.csv", index=False
        )
        artifacts = [
            "submission.csv",
            "p5_test_predictions_by_fold.npy",
            "p5_test_mask.npy",
            "p5_test_slots.npy",
            "p5_test_positions.npy",
            "p5_test_selections.csv",
        ]
        output_manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "studies": len(test),
            "series": len(series),
            "targets": p5.TARGETS,
            "source_p5_manifest_sha256": EXPECTED_P5_MANIFEST_SHA256,
            "source_p5_oof_macro_auc": EXPECTED_P5_OOF_MACRO_AUC,
            "source_checkpoint_sha256": p5.MODEL_SHA256,
            "source_fold_head_sha256": {
                str(fold): manifest["artifacts_sha256"][f"p5_fold_{fold}_head.pt"]
                for fold in range(5)
            },
            "geometry": {
                "slot_specs": prep.SLOT_SPECS,
                "slot_bounds": prep.SLOT_BOUNDS,
                "slot_quotas": list(p5.SLOT_QUOTAS),
                "relative_slice_range": [0.02, 0.98],
                "physical_crop_mm": prep.CROP_MM,
                "cache_size": p5.CACHE_SIZE,
                "model_size": p5.MODEL_SIZE,
                "slices_per_study": p5.K_SLICES,
            },
            "ensemble": "mean probabilities from five report-purged P5 fold heads",
            "test_metadata_sha256": {
                "test.csv": sha256_file(data_root / "test.csv"),
                "test_series.csv": sha256_file(data_root / "test_series.csv"),
            },
            "artifacts_sha256": {
                name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
            },
            "elapsed": {
                "inference_minutes_by_rank": [
                    float(value.item() / 60.0) for value in inference_seconds_by_rank
                ],
                "inference_minutes_slowest_rank": max(
                    float(value.item() / 60.0) for value in inference_seconds_by_rank
                ),
                "total_minutes_rank0": (time.time() - started) / 60.0,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "world_size": world_size,
                "batch_size_per_rank": 2,
                "loader_workers_per_rank": 2,
            },
        }
        manifest_output_path = OUTPUT_ROOT / "p5_submission_manifest.json"
        manifest_output_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
        print(
            {
                "P5_SUBMISSION_COMPLETE": True,
                "studies": len(test),
                "submission_sha256": sha256_file(submission_path),
                "manifest_sha256": sha256_file(manifest_output_path),
                "prediction_ranges": {
                    target: [
                        float(submission[target].min()),
                        float(submission[target].max()),
                    ]
                    for target in p5.TARGETS
                },
                "elapsed_minutes": output_manifest["elapsed"]["total_minutes_rank0"],
            },
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main() -> int:
    if "--worker" not in sys.argv:
        return launch()
    return worker()


if __name__ == "__main__":
    raise SystemExit(main())
