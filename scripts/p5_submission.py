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

import p2_submission as prep
import p5_orthofoundation_speed_smoke as p5


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
