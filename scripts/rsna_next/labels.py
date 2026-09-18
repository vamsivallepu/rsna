"""Compile provenance-checked OOF image teachers and P13 report-label arms."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ID, TARGETS, VERSION, read_splits, sha256, write_json
from .validate import validate_run

STATES = {"present", "absent", "uncertain", "unmentioned"}


def validate_extraction(record, report):
    findings = record.get("findings", {})
    if set(findings) != set(TARGETS):
        raise ValueError("Extractor must return exactly all 12 findings")
    for target, entry in findings.items():
        state, probability = entry["state"], float(entry["probability"])
        evidence = entry["evidence"]
        if state not in STATES or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid extraction for {target}")
        if state == "unmentioned":
            if evidence or probability != .5:
                raise ValueError("Unmentioned requires no evidence and probability 0.5")
        elif not evidence or evidence not in report:
            raise ValueError("Evidence must be an exact report substring")
        for field in ("compartment", "severity", "negation", "uncertainty"):
            if field not in entry:
                raise ValueError(f"Missing extraction field {field}")
    return findings


def teacher_predictions(roots, splits, split_path):
    n = len(splits)
    sums = np.zeros((n, 12), np.float64)
    counts = np.zeros(n, int)
    by_id = dict(zip(splits[ID], range(n)))
    source = []
    seen = set()
    configs = []
    for root in map(Path, roots):
        manifest_path = root / "manifest.json"
        m, _ = validate_run(root, splits, split_path)
        if m.get("schema_version") != VERSION or m["config"].get("labels") != "report":
            raise ValueError("Teacher must satisfy the current original-report contract")
        fold = m["fold"]
        if m["mode"] != "teacher" or m["status"] == "smoke_complete" or fold in seen:
            raise ValueError("Need five distinct full-run teacher folds")
        seen.add(fold)
        if m["split_sha256"] != sha256(split_path):
            raise ValueError("Teacher uses a different split")
        expected_train = set(splits.loc[splits[f"teacher_train_{fold}"], ID])
        if set(m["train_ids"]) != expected_train:
            raise ValueError("Teacher training membership is not the full allowed fold")
        if set(m["validation_ids"]) != set(splits.loc[splits.role == "pseudo_valid", ID]):
            raise ValueError("Teacher validation membership mismatch")
        for file, key in [("best.pt", "checkpoint_sha256"), ("predictions.csv", "predictions_sha256")]:
            if sha256(root / file) != m[key]:
                raise ValueError("Teacher artifact hash mismatch")
        pred = pd.read_csv(root / "predictions.csv")
        expected_predict = ((splits.role == "pseudo_train") & (splits.teacher_fold == fold)) | (splits.role == "pseudo_valid") | splits.is_gold
        if set(pred[ID]) != set(splits.loc[expected_predict, ID]) or pred[ID].duplicated().any():
            raise ValueError("Missing or in-sample teacher predictions")
        values = pred[TARGETS].to_numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid teacher probabilities")
        indices = np.array([by_id[x] for x in pred[ID]])
        sums[indices] += values
        counts[indices] += 1
        source.append({"manifest_sha256": sha256(manifest_path), "fold": fold,
                       "checkpoint_sha256": m["checkpoint_sha256"]})
        configs.append({"config": m["config"], "original_targets_sha256": m["original_targets_sha256"],
                        "cache_signature": m["cache_signature"], "initialization_sha256": m["initialization_sha256"]})
    if seen != set(range(5)) or any(c != configs[0] for c in configs[1:]):
        raise ValueError("Teacher bank must be five matched configurations")
    if not (counts[splits.role == "pseudo_train"] == 1).all():
        raise ValueError("Pseudo training rows must have exactly one OOF teacher")
    if not (counts[(splits.role == "pseudo_valid") | splits.is_gold] == 5).all():
        raise ValueError("Shared held-out rows require all five teachers")
    return sums / np.maximum(counts[:, None], 1), source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--teacher-roots", type=Path, nargs=5, required=True)
    p.add_argument("--extractions", type=Path)
    p.add_argument("--arm", choices=["distilled", "fourth_source", "mention_mask"], default="distilled")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    metadata = pd.read_csv(args.train)
    splits = read_splits(args.splits, metadata)
    targets = pd.read_csv(args.targets).set_index(ID).loc[metadata[ID]].reset_index()
    image, provenance = teacher_predictions(args.teacher_roots, splits, args.splits)
    report = targets[TARGETS].to_numpy().copy()
    weight_cols = ["weight__" + t for t in TARGETS]
    weights = targets[weight_cols].to_numpy().copy()
    report_weights, image_weights = weights.copy(), weights.copy()
    records = {}
    extractor_manifest = None
    if args.arm != "distilled":
        if args.extractions is None:
            raise ValueError("Report extraction required")
        extractor_manifest = json.loads(args.extractions.with_suffix(".manifest.json").read_text())
        if not extractor_manifest.get("complete"):
            raise ValueError("Benchmark extractions are not complete training labels")
        if extractor_manifest["extractions_sha256"] != sha256(args.extractions) or extractor_manifest["train_sha256"] != sha256(args.train):
            raise ValueError("Extractor provenance mismatch")
        for line in args.extractions.read_text().splitlines():
            r = json.loads(line)
            if r[ID] in records:
                raise ValueError("Duplicate extraction")
            records[r[ID]] = r
        if set(records) != set(metadata[ID]):
            raise ValueError("Incomplete extraction coverage")
    provenance_rows = []
    for i, row in metadata.iterrows():
        extracted = validate_extraction(records[row[ID]], row.Report) if records else None
        for j, target in enumerate(TARGETS):
            e = extracted[target] if extracted else None
            if e and args.arm == "fourth_source":
                report[i, j] = .75 * report[i, j] + .25 * e["probability"]
            if e and args.arm == "mention_mask" and e["state"] == "unmentioned":
                report_weights[i, j] = 0
            provenance_rows.append({ID: row[ID], "target": target,
                "mention_state": e["state"] if e else "not_extracted", "evidence": e["evidence"] if e else "",
                "report_probability": float(report[i, j]), "image_probability": float(image[i, j]),
                "report_weight": float(report_weights[i, j]), "image_weight": float(image_weights[i, j]),
                "teacher_fold": int(splits.iloc[i].teacher_fold), "extractor_version":
                    extractor_manifest["model_revision"] if extractor_manifest else "original-consensus"})
    # BCE is affine in y: this exactly represents separate half-weight report and
    # image losses, including the mention-mask arm (not a changed mixture search).
    combined_w = .5 * (report_weights + image_weights)
    combined_y = (.5 * report_weights * report + .5 * image_weights * image) / np.maximum(combined_w, 1e-8)
    eligible = splits.role.isin(["pseudo_train", "pseudo_valid"]).to_numpy()
    targets.loc[eligible, TARGETS] = combined_y[eligible]
    targets.loc[eligible, weight_cols] = combined_w[eligible]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output, index=False)
    prov_path = args.output.with_suffix(".provenance.csv")
    pd.DataFrame(provenance_rows).to_csv(prov_path, index=False)
    write_json(args.output.with_suffix(".manifest.json"), {
        "schema_version": VERSION, "arm": args.arm, "split_sha256": sha256(args.splits),
        "original_targets_sha256": sha256(args.targets), "targets_sha256": sha256(args.output),
        "provenance_sha256": sha256(prov_path), "teacher_sources": provenance,
        "extractor": extractor_manifest, "expert_targets_unchanged": True,
        "formula": "0.5 report loss + 0.5 cross-fitted image loss"})


if __name__ == "__main__":
    main()
