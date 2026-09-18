"""Offline DICOM inference for versioned next-generation model groups.

This writes predictions and runtime evidence; it never submits to Kaggle.
Group predictions are averaged, then rank blended using predeclared weights.
"""
# ruff: noqa: E402 -- time imports as part of offline inference startup.
import argparse
import hashlib
import json
import time
from pathlib import Path

IMPORT_STARTED = time.perf_counter()

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from .contracts import ID, TARGETS, VERSION, sha256, write_json
from .data import P2_SIGNATURE, sample_windows
from .geometry import build_study
from .screen import positions_to_metadata
from .train import INIT_HASHES, ImageModel


def blend_predictions(groups, weights, rank=True):
    weights = np.asarray(weights, float)
    values = np.asarray(groups, float)
    if values.ndim != 3 or values.shape[2] != len(TARGETS) or not len(values[0]):
        raise ValueError("Expected nonempty model-group probabilities")
    if weights.shape != (len(values),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Predeclared nonnegative weights must sum to one")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid group probabilities")
    if rank:
        values = np.stack([rankdata(v, axis=0, method="average") / len(v) for v in values])
    return np.einsum("g,gnt->nt", weights, values)


def release_fingerprint(spec):
    # File locations may differ between local and Kaggle mounts.
    contract = {"schema_version": spec["schema_version"], "groups": []}
    for group in spec["groups"]:
        contract["groups"].append({"weight": group["weight"], "preprocessing": group["preprocessing"],
            "models": [{k: m[k] for k in ("manifest_sha256", "checkpoint_sha256", "initialization_sha256")}
                       for m in group["models"]]})
    contract["rank_blend"] = spec.get("rank_blend", True)
    contract["code_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in
                               ("infer.py", "train.py", "models.py", "geometry.py", "data.py", "screen.py")}
    from scripts import p2_submission
    contract["legacy_preprocessing_sha256"] = sha256(p2_submission.__file__)
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def dicom_windows(uid, records, dicom_root, preprocessing):
    kind = preprocessing["kind"]
    if kind == "p2":
        from scripts.p2_submission import TestStudyDataset
        _, x, mask, centers, _ = TestStudyDataset([uid], {uid: records}, dicom_root)[0]
        slots, positions = positions_to_metadata(centers[None], mask[None].astype(bool))
        return x, mask.astype(bool), slots[0], positions[0]
    if kind != "p14":
        raise ValueError("Unknown preprocessing version")
    x, mask, slots, positions, _, _ = build_study(uid, records, dicom_root,
        preprocessing["size"], preprocessing["axial_policy"])
    slots[~mask] = -1
    pick, valid = sample_windows(slots, 32)
    if not valid.any():
        raise ValueError(f"No decoded windows for {uid}")
    return x[pick], valid, slots[pick], positions[pick]


def main(started=None):
    started = IMPORT_STARTED if started is None else started
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--studies", type=Path, required=True, help="test.csv or label-blind benchmark ID list")
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--dicom-root", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--hidden-studies", type=int, default=1300)
    args = p.parse_args()
    spec = json.loads(args.release.read_text())
    if spec["schema_version"] != VERSION or not spec["groups"]:
        raise ValueError("Incompatible release")
    fingerprint = release_fingerprint(spec)
    ids = pd.read_csv(args.studies, usecols=[ID], dtype={ID: str})[ID].tolist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Study IDs must be nonempty and unique")
    if args.benchmark and len(ids) < 200:
        raise ValueError("Runtime benchmark requires at least 200 label-blind studies")
    series = pd.read_csv(args.series, dtype={ID: str})
    records = {uid: rows.to_dict("records") for uid, rows in series.groupby(ID)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    groups = []
    for gi, group in enumerate(spec["groups"]):
        if not group["models"]:
            raise ValueError("Empty model group")
        predictions = []
        for mi, entry in enumerate(group["models"]):
            root = (args.release.parent / entry["run"]).resolve()
            weights = (args.release.parent / entry["weights"]).resolve()
            manifest = json.loads((root / "manifest.json").read_text())
            cfg = manifest["config"]
            for path, key in ((root / "manifest.json", "manifest_sha256"), (root / "best.pt", "checkpoint_sha256"), (weights, "initialization_sha256")):
                if sha256(path) != entry[key]:
                    raise ValueError(f"Release artifact mismatch: {path.name}")
            if manifest["schema_version"] != VERSION or manifest["status"] == "smoke_complete" or manifest["mode"] not in {"student", "refit"}:
                raise ValueError("Only full student/refit artifacts may enter a release")
            preprocessing = group["preprocessing"]
            expected = P2_SIGNATURE if preprocessing["kind"] == "p2" else preprocessing["cache_signature"]
            if expected != manifest["cache_signature"] or entry["initialization_sha256"] != INIT_HASHES[cfg["initialization"]]:
                raise ValueError("Training/deployment preprocessing or initialization mismatch")
            if preprocessing["kind"] == "p14":
                cache_contract = preprocessing["contract"]
                if (cache_contract["code_sha256"] != sha256(Path(__file__).with_name("geometry.py")) or
                    cache_contract["size"] != preprocessing["size"] or cache_contract["axial_policy"] != preprocessing["axial_policy"] or
                    hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest() != expected):
                    raise ValueError("Native cache contract changed")
            model = ImageModel(cfg["initialization"], weights, cfg.get("head", "baseline")).to(device)
            model.load_state_dict(torch.load(root / "best.pt", weights_only=True, map_location=device), strict=True)
            model.eval()
            values = []
            for uid in ids:
                tensors = [torch.as_tensor(a[None], device=device) for a in dicom_windows(uid, records.get(uid, []), args.dicom_root, preprocessing)]
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    values.append(model(*tensors).sigmoid().float().cpu().numpy()[0])
            values = np.asarray(values)
            pd.DataFrame({ID: ids, **dict(zip(TARGETS, values.T))}).to_csv(args.output / f"group-{gi}-model-{mi}.csv", index=False)
            predictions.append(values)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        groups.append(np.mean(predictions, axis=0))
    blend = blend_predictions(groups, [g["weight"] for g in spec["groups"]], spec.get("rank_blend", True))
    result = pd.DataFrame(blend, columns=TARGETS)
    result.insert(0, ID, ids)
    if args.sample_submission:
        sample = pd.read_csv(args.sample_submission, dtype={ID: str})
        if sample.columns.tolist() != [ID, *TARGETS] or sample[ID].duplicated().any() or set(sample[ID]) != set(ids):
            raise ValueError("Submission schema or row coverage mismatch")
        result = result.set_index(ID).loc[sample[ID]].reset_index()
    output = args.output / ("benchmark_predictions.csv" if args.benchmark else "submission.csv")
    result.to_csv(output, index=False)
    elapsed = time.perf_counter() - started
    # Conservatively scale all startup/model loads/decoding, plus 25% headroom.
    projected = 1.25 * elapsed * args.hidden_studies / len(ids) / 3600 if args.benchmark else None
    write_json(args.output / "inference.json", {"release_fingerprint": fingerprint,
        "elapsed_seconds": elapsed, "studies": len(ids), "device": str(device),
        "predictions_sha256": sha256(output), "benchmark": args.benchmark,
        "projected_hidden_hours": projected, "runtime_gate_passed": projected is not None and projected <= 7.5,
        "submission_eligible": False, "note": "Requires independent replay, promotion, and exact offline release validation"})


if __name__ == "__main__":
    main()
