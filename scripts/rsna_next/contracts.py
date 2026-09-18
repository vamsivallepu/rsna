"""Shared split, provenance, evaluation and resource contracts."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
VERSION = "rsna-next-v1"
BOUNDS = np.array([[0, 18], [18, 32], [32, 44], [44, 52], [52, 64]])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def report_hash(value):
    text = "".join(c for c in unicodedata.normalize("NFKC", str(value)).casefold() if c.isalnum())
    if not text:
        raise ValueError("Empty report")
    return hashlib.sha256(text.encode()).hexdigest()


def group_validation(groups, seed=20260902, fraction=.1):
    unique, counts = np.unique(groups, return_counts=True)
    sizes = dict(zip(unique, counts))
    ordered = sorted(unique, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest())
    wanted, count, selected = round(len(groups) * fraction), 0, set()
    for g in ordered:
        if abs(count + sizes[g] - wanted) <= abs(count - wanted):
            count += sizes[g]
            selected.add(g)
        if count == wanted:
            break
    return np.isin(groups, list(selected))


def exclusion_groups(s):
    return s["exclusion_group"] if "exclusion_group" in s else s["report_group"]


def make_splits(metadata, image_pairs=None):
    if metadata[ID].duplicated().any() or metadata.Report.isna().any():
        raise ValueError("Invalid study IDs or reports")
    labelled = metadata[TARGETS].notna()
    if (labelled.any(axis=1) != labelled.all(axis=1)).any():
        raise ValueError("Partial gold labels require a new contract")
    gold = labelled.all(axis=1).to_numpy()
    out = metadata[[ID]].copy()
    out["report_group"] = metadata.Report.map(report_hash)
    out["is_gold"] = gold
    out["expert_fold"] = -1
    gi = np.flatnonzero(gold)
    for fold, (_, valid) in enumerate(KFold(5, shuffle=True, random_state=20260831).split(gi)):
        out.loc[gi[valid], "expert_fold"] = fold
    groups = out.report_group.to_numpy()
    twins = ~gold & np.isin(groups, groups[gold])
    # Preserve the historical stable split, then purge all expert report groups.
    pv = np.zeros(len(out), bool)
    pv[~gold] = group_validation(groups[~gold])
    out["role"] = np.where(gold, "expert", np.where(twins, "excluded_expert_twin",
                           np.where(pv, "pseudo_valid", "pseudo_train")))
    out["teacher_fold"] = -1
    sizes = out.loc[out.role == "pseudo_train"].groupby("report_group").size()
    loads = [0] * 5
    for g in sorted(sizes.index, key=lambda x: hashlib.sha256(f"20260917:{x}".encode()).digest()):
        fold = int(np.argmin(loads))
        out.loc[(out.report_group == g) & (out.role == "pseudo_train"), "teacher_fold"] = fold
        loads[fold] += int(sizes[g])
    for fold in range(5):
        out[f"teacher_train_{fold}"] = (out.role == "pseudo_train") & (out.teacher_fold != fold)
        out[f"student_train_{fold}"] = ((out.role == "pseudo_train") |
                                             (gold & (out.expert_fold != fold)))
    if image_pairs is not None:
        # Union report groups connected by verified (or conservatively quarantined)
        # image pairs, while preserving expert folds and the original valid set.
        parent = {g: g for g in out.report_group}
        by_id = out.set_index(ID).report_group.to_dict()

        def root(g):
            while parent[g] != g:
                parent[g] = parent[parent[g]]
                g = parent[g]
            return g

        for a, b in image_pairs:
            if a not in by_id or b not in by_id or a == b:
                raise ValueError("Invalid image duplicate pair")
            aa, bb = sorted((root(by_id[a]), root(by_id[b])))
            parent[bb] = aa
        out["exclusion_group"] = out.report_group.map(root)
        eg = out.exclusion_group
        expert_groups = set(eg[gold])
        valid_groups = set(eg[out.role == "pseudo_valid"])
        out.loc[~gold & eg.isin(expert_groups), "role"] = "excluded_expert_twin"
        out.loc[(out.role == "pseudo_train") & eg.isin(valid_groups), "role"] = "excluded_validation_twin"
        out.loc[out.role != "pseudo_train", "teacher_fold"] = -1
        # Preserve existing teacher assignments except where image groups merge.
        train_mask = out.role == "pseudo_train"
        folds = out.loc[train_mask].groupby("exclusion_group").teacher_fold.min()
        out.loc[train_mask, "teacher_fold"] = eg[train_mask].map(folds)
        for fold in range(5):
            out[f"teacher_train_{fold}"] = train_mask & (out.teacher_fold != fold)
            prohibited = set(eg[(out.role == "pseudo_valid") | (gold & (out.expert_fold == fold))])
            out[f"student_train_{fold}"] = (train_mask | (gold & (out.expert_fold != fold))) & ~eg.isin(prohibited)
    validate_splits(out)
    return out


def validate_splits(s):
    if s[ID].duplicated().any() or s.isna().any().any():
        raise ValueError("Split contains missing values or duplicate studies")
    groups = exclusion_groups(s)
    for fold in range(5):
        teacher = s[f"teacher_train_{fold}"].astype(bool)
        student = s[f"student_train_{fold}"].astype(bool)
        prohibited = (s.role != "pseudo_train")
        if set(groups[teacher]) & set(groups[prohibited]):
            raise ValueError("Teacher validation/expert group leakage")
        held = (s.role == "pseudo_train") & (s.teacher_fold == fold)
        if set(groups[teacher]) & set(groups[held]):
            raise ValueError("Teacher predicts its training group")
        valid = (s.role == "pseudo_valid") | (s.is_gold & (s.expert_fold == fold))
        if set(groups[student]) & set(groups[valid]):
            raise ValueError("Student validation group leakage")


def read_splits(path, metadata=None):
    s = pd.read_csv(path)
    validate_splits(s)
    if metadata is not None:
        if s[ID].tolist() != metadata[ID].tolist():
            raise ValueError("Split/metadata study order mismatch")
        pairs = None
        if "exclusion_group" in s:
            sidecar = json.loads(Path(path).with_suffix(".groups.json").read_text())
            if sidecar["split_sha256"] != sha256(path):
                raise ValueError("Image-group split provenance mismatch")
            pairs = sidecar["image_pairs"]
        pd.testing.assert_frame_equal(s, make_splits(metadata, pairs), check_dtype=False)
    return s


def metrics(y, p):
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid prediction array")
    auc = np.array([roc_auc_score(y[:, j], p[:, j]) for j in range(len(TARGETS))])
    return dict(zip(TARGETS + ["MACRO"], [*map(float, auc), float(auc.mean())]))


def paired_bootstrap(y, baseline, candidate, groups, repeats=5000, seed=20260917):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    # Pairwise AUC matrices avoid thousands of expensive sklearn calls.
    matrices = []
    for j in range(y.shape[1]):
        delta = ((candidate[:, j, None] > candidate[None, :, j]).astype(float) +
                 .5 * (candidate[:, j, None] == candidate[None, :, j]))
        delta -= ((baseline[:, j, None] > baseline[None, :, j]).astype(float) +
                  .5 * (baseline[:, j, None] == baseline[None, :, j]))
        matrices.append(delta * (y[:, j, None] == 1) * (y[None, :, j] == 0))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        ix = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        counts = np.bincount(ix, minlength=len(y))
        pos = counts @ y
        neg = counts.sum() - pos
        if (pos == 0).any() or (neg == 0).any():
            continue
        deltas.append(float(np.mean([(counts @ m @ counts) / (pos[j] * neg[j])
                                     for j, m in enumerate(matrices)])))
    if len(deltas) < repeats * .8:
        raise ValueError("Too few valid group bootstrap replicates")
    return {"valid_repeats": len(deltas), "ci95": np.quantile(deltas, [.025, .975]).tolist(),
            "probability_positive": float(np.mean(np.array(deltas) > 0))}


def save_predictions(path, ids, y, p, fold=None):
    rows = {ID: list(ids)}
    if fold is not None:
        rows["fold"] = fold
    for j, t in enumerate(TARGETS):
        rows[f"label__{t}"] = y[:, j]
        rows[f"pred__{t}"] = p[:, j]
    pd.DataFrame(rows).to_csv(path, index=False)


def assert_budget(entries, proposed_usd=0, cap=500):
    if any(not np.isfinite(float(e.get(k, 0))) or float(e.get(k, 0)) < 0
           for e in entries for k in ("actual_usd", "reserved_usd")):
        raise ValueError("Invalid budget ledger entry")
    committed = sum(float(e.get("actual_usd", 0)) + float(e.get("reserved_usd", 0)) for e in entries)
    if not np.isfinite(committed + proposed_usd) or proposed_usd < 0 or committed + proposed_usd > cap:
        raise ValueError("Paid-compute budget exceeded or invalid")
    return cap - committed - proposed_usd
