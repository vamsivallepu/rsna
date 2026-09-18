"""Fold-safe P10 teachers and P12/P14/P15 students; offline DDP training."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .contracts import ID, TARGETS, VERSION, exclusion_groups, read_splits, sha256, write_json
from .data import PixelCache, StudyDataset
from .models import AnatomicalHead

INIT_HASHES = {"imagenet": "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e",
               "fastmri": "5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195",
               "dinov2": "1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050"}
ORIGINAL_TARGETS_SHA256 = "ca63cfff863c87286e07a8059fee88f86ff8b3f4b640a98078d62113f6b93f2a"


class ImageModel(nn.Module):
    def __init__(self, init, weight_path, head="baseline", checkpointing=False):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        if init == "dinov2":
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(str(Path(weight_path).parent), local_files_only=True)
            if (self.backbone.config.hidden_size, self.backbone.config.num_hidden_layers,
                    self.backbone.config.patch_size) != (384, 12, 14):
                raise ValueError("Not official DINOv2-small")
            if checkpointing:
                self.backbone.gradient_checkpointing_enable()
            dimension = 384
            self.size, mean, std = 336, (.485, .456, .406), (.229, .224, .225)
        else:
            self.backbone = timm.create_model("coatnet_rmlp_2_rw_384", pretrained=False)
            state = load_file(str(weight_path), device="cpu")
            self.size, mean, std = 384, (.5, .5, .5), (.5, .5, .5)
            self.backbone.load_state_dict(state, strict=True)
            self.backbone.reset_classifier(0)
            if checkpointing:
                self.backbone.set_grad_checkpointing(True)
            dimension = self.backbone.num_features
        self.is_dino = init == "dinov2"
        self.head = AnatomicalHead(dimension, head)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, windows, mask, slots, positions, augment=False):
        b, k = windows.shape[:2]
        x = windows.flatten(0, 1).float() / 255
        x = F.interpolate(x, (self.size, self.size), mode="bicubic", align_corners=False)
        if augment:
            gain = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(.9, 1.1)
            offset = torch.empty_like(gain).uniform_(-.05, .05)
            x = (x * gain + offset).clamp(0, 1)
        normalized = (x - self.mean) / self.std
        if self.is_dino:
            sequence = self.backbone(pixel_values=normalized).last_hidden_state
            features = (.5 * sequence[:, 0] + .5 * sequence[:, 1:].mean(dim=1)).reshape(b, k, -1)
        else:
            features = self.backbone(normalized).reshape(b, k, -1)
        return self.head(features, mask, slots, positions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True, help="Unchanged original report-consensus targets")
    p.add_argument("--student-targets", type=Path)
    p.add_argument("--cache-manifests", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["teacher", "student", "refit"], default="student")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if "RANK" not in os.environ and torch.cuda.device_count() > 1:
        return subprocess.call([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                f"--nproc_per_node={torch.cuda.device_count()}", "-m",
                                "scripts.rsna_next.train", *sys.argv[1:]])
    cfg = json.loads(args.config.read_text())
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    local = int(os.getenv("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Image training requires a GPU; use P11 for CPU experiments")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(2)
    seed = int(cfg.get("seed", 20260917)) + (args.fold if args.mode == "student" else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output.exists() and (args.output / "manifest.json").exists():
        raise FileExistsError("Completed output already exists")
    args.output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.train)
    splits = read_splits(args.splits, meta)
    if sha256(args.targets) != ORIGINAL_TARGETS_SHA256:
        raise ValueError("Original report-consensus validation targets changed")
    original = pd.read_csv(args.targets).set_index(ID).loc[meta[ID]]
    labels = original[TARGETS].to_numpy(np.float32)
    weights = original[["weight__" + t for t in TARGETS]].to_numpy(np.float32)
    original_y, original_w = labels.copy(), weights.copy()
    target_manifest = None
    if cfg.get("labels", "report") != "report":
        if args.student_targets is None or args.mode == "teacher":
            raise ValueError("Distilled student labels required; recursive teachers forbidden")
        target_manifest_path = args.student_targets.with_suffix(".manifest.json")
        target_manifest = json.loads(target_manifest_path.read_text())
        if target_manifest["split_sha256"] != sha256(args.splits) or target_manifest["targets_sha256"] != sha256(args.student_targets):
            raise ValueError("Student target provenance mismatch")
        if target_manifest["original_targets_sha256"] != sha256(args.targets) or target_manifest["arm"] != cfg["labels"]:
            raise ValueError("Student targets do not match the declared label arm")
        updated = pd.read_csv(args.student_targets).set_index(ID).loc[meta[ID]]
        labels = updated[TARGETS].to_numpy(np.float32)
        weights = updated[["weight__" + t for t in TARGETS]].to_numpy(np.float32)
    if not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid targets or weights")
    gold = splits.is_gold.to_numpy()
    if not np.array_equal(labels[gold], meta.loc[gold, TARGETS].to_numpy()):
        raise ValueError("Expert labels changed")
    if args.mode == "teacher":
        train_ix = np.flatnonzero(splits[f"teacher_train_{args.fold}"])
        predict_ix = np.flatnonzero(((splits.role == "pseudo_train") & (splits.teacher_fold == args.fold)) |
                                   (splits.role == "pseudo_valid") | splits.is_gold)
    elif args.mode == "student":
        train_ix = np.flatnonzero(splits[f"student_train_{args.fold}"])
        predict_ix = np.flatnonzero(gold & (splits.expert_fold == args.fold))
    else:
        train_ix = np.arange(len(splits))
        predict_ix = np.array([], dtype=int)
    valid_ix = np.flatnonzero(splits.role == "pseudo_valid")
    groups = exclusion_groups(splits)
    if args.mode != "refit" and set(groups.iloc[train_ix]) & set(groups.iloc[valid_ix]):
        raise ValueError("Validation leakage")
    oversample = int(cfg.get("expert_repeat", 1))
    if oversample < 1:
        raise ValueError("Invalid expert repetition")
    if args.mode != "teacher":
        train_ix = np.r_[train_ix, np.tile(train_ix[gold[train_ix]], oversample - 1)]
    init = cfg["initialization"]
    if sha256(args.weights) != INIT_HASHES[init]:
        raise ValueError("Unapproved or corrupt initialization weights")
    cache = PixelCache(args.cache_manifests, meta[ID], verify=(rank == 0))
    if world > 1:
        dist.barrier()
    if args.smoke:
        train_ix, valid_ix = train_ix[:64], valid_ix[:16]
        predict_ix = predict_ix[:12]
    epochs = 1 if args.smoke else int(cfg.get("epochs", 8))
    if args.mode == "refit" and "refit_epochs" not in cfg:
        raise ValueError("Refits require a frozen refit_epochs chosen before training")
    if args.mode == "refit" and not args.smoke:
        epochs = int(cfg["refit_epochs"])
    model = ImageModel(init, args.weights, cfg.get("head", "baseline"),
                       cfg.get("checkpointing", False)).to(device)
    wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 3e-5},
                            {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=.02)
    accum = int(cfg.get("accumulation", 4))
    microbatches = len(train_ix) // world
    steps = math.ceil(microbatches / accum)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=steps * epochs, pct_start=.15)
    scaler = torch.amp.GradScaler("cuda")
    started, best, best_epoch = time.time(), float("inf"), -1
    history = []

    def loader(indices, train=False, epoch=0, original_loss=False):
        ds = StudyDataset(cache, indices, original_y if original_loss else labels,
                          original_w if original_loss else weights,
                          count=int(cfg.get("train_windows", 8)) if train else 32,
                          seed=seed + epoch * 100003 if train else None,
                          balanced=cfg.get("sampling", "uniform") == "balanced")
        return DataLoader(ds, batch_size=1, num_workers=0 if args.smoke else 2, pin_memory=True)

    def move(batch):
        return [v.to(device, non_blocking=True) for v in batch]

    def evaluate(indices, original_loss=True):
        model.eval()
        sums = torch.zeros(2, device=device, dtype=torch.float64)
        for batch in loader(indices[rank::world], original_loss=original_loss):
            _, x, mask, slots, position, yy, ww = move(batch)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, mask, slots, position)
                raw = F.binary_cross_entropy_with_logits(logits.float(), yy, reduction="none")
            sums += torch.stack([(raw * ww).sum(), ww.sum()]).double()
        if world > 1:
            dist.all_reduce(sums)
        return float((sums[0] / sums[1].clamp_min(1)).item())

    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_ix)[:microbatches * world]
        opt.zero_grad(set_to_none=True)
        train_loss = torch.zeros(2, device=device)
        for step, batch in enumerate(loader(order[rank::world], True, epoch)):
            _, x, mask, slots, position, yy, ww = move(batch)
            end = (step + 1) % accum == 0 or step + 1 == microbatches
            denominator = ww.sum().detach()
            if world > 1:
                dist.all_reduce(denominator)
            group_steps = min(accum, microbatches - (step // accum) * accum)
            ctx = wrapped.no_sync() if world > 1 and not end else contextlib.nullcontext()
            with ctx, torch.autocast("cuda", dtype=torch.float16):
                logits = wrapped(x, mask, slots, position, augment=True)
                numerator = (F.binary_cross_entropy_with_logits(logits.float(), yy, reduction="none") * ww).sum()
                loss = numerator * world / denominator.clamp_min(1) / group_steps
                scaler.scale(loss).backward()
            train_loss += torch.stack([numerator.detach(), ww.sum().detach()])
            if end:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                scaler.step(opt)
                scaler.update()
                schedule.step()
                opt.zero_grad(set_to_none=True)
        if world > 1:
            dist.all_reduce(train_loss)
            for buffer in model.buffers():
                dist.broadcast(buffer, 0)
        validation = evaluate(valid_ix) if args.mode != "refit" else None
        distilled = evaluate(valid_ix, False) if target_manifest and args.mode != "refit" else None
        selected = validation is None or validation < best
        if selected:
            best, best_epoch = validation if validation is not None else 0., epoch + 1
            if rank == 0:
                torch.save(model.state_dict(), args.output / "best.pt")
        row = {"epoch": epoch + 1, "train_bce": float((train_loss[0] / train_loss[1]).item()),
               "original_report_valid_bce": validation, "distilled_valid_bce_diagnostic": distilled,
               "elapsed_minutes": (time.time() - started) / 60}
        history.append(row)
        if rank == 0:
            pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
            print(json.dumps(row), flush=True)
    if world > 1:
        dist.barrier()
    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True, map_location=device))
    model.eval()
    rows = []
    for batch in loader(predict_ix[rank::world]):
        index, x, mask, slots, position, _, _ = move(batch)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            probabilities = model(x, mask, slots, position).sigmoid().float().cpu().numpy()
        for ix, values in zip(index.cpu().tolist(), probabilities):
            rows.append({ID: meta.iloc[ix][ID], **dict(zip(TARGETS, values.tolist()))})
    pd.DataFrame(rows, columns=[ID, *TARGETS]).to_csv(args.output / f"predictions-rank-{rank}.csv", index=False)
    if world > 1:
        dist.barrier()
    peak_memory = torch.tensor(torch.cuda.max_memory_allocated() / 2**20, device=device)
    if world > 1:
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    if rank == 0:
        predictions = pd.concat([pd.read_csv(args.output / f"predictions-rank-{r}.csv") for r in range(world)], ignore_index=True)
        if len(predictions) != len(predict_ix) or predictions[ID].duplicated().any():
            raise ValueError("Incomplete predictions")
        predictions.to_csv(args.output / "predictions.csv", index=False)
        manifest = {"schema_version": VERSION, "status": "smoke_complete" if args.smoke else "trained_pending_independent_validation",
                    "mode": args.mode, "fold": args.fold, "seed": seed, "config": cfg, "world_size": world,
                    "split_sha256": sha256(args.splits), "original_targets_sha256": sha256(args.targets),
                    "initialization_sha256": sha256(args.weights), "cache_signature": cache.signature,
                    "cache_manifests": cache.manifest_hashes, "target_manifest": target_manifest,
                    "train_ids": meta.iloc[np.unique(train_ix)][ID].tolist(),
                    "validation_ids": meta.iloc[valid_ix][ID].tolist() if args.mode != "refit" else [],
                    "prediction_ids": meta.iloc[predict_ix][ID].tolist(), "best_epoch": best_epoch,
                    "best_original_report_bce": None if args.mode == "refit" else best,
                    "checkpoint_sha256": sha256(args.output / "best.pt"),
                    "predictions_sha256": sha256(args.output / "predictions.csv"),
                    "elapsed_minutes": (time.time() - started) / 60,
                    "peak_cuda_memory_mib": float(peak_memory.item()),
                    "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                    "numpy": np.__version__, "cuda": torch.version.cuda,
                                    "gpu": torch.cuda.get_device_name(local)},
                    "promotion_eligible": False, "refit_is_not_oof": args.mode == "refit"}
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({k: manifest[k] for k in ("status", "best_epoch", "elapsed_minutes")}), flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
