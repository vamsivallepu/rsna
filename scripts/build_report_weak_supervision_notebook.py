#!/usr/bin/env python3
"""Build the reproducible RSNA report weak-supervision notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/jupyter-notebook/rsna-report-weak-supervision.ipynb"),
    )
    args = parser.parse_args()

    cells = [
        markdown(
            """
# RSNA Knee: multilingual report weak supervision

**Question.** Can three independently generated, publicly accessible LLM label sets
turn 4,349 report-only studies into reliable soft targets without fitting blend weights
or thresholds on the tiny 58-study expert set?

**Primary method declared before inspecting the audit:** an equal-weight arithmetic
mean of the Pilkwang, Steven Lee, and Lixin label probabilities. The expert labels are
used only for reporting target-wise and macro ROC-AUC. They never determine source
weights, thresholds, or inclusion rules.

The output `training_targets.csv` is the contract for the image notebook: gold targets
are retained for the 58 expert studies; every other row receives the fixed three-LLM
mean plus an agreement-derived sample weight.
"""
        ),
        code(
            """
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20260831
RNG = np.random.default_rng(SEED)
ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PRIMARY_METHOD = "mean"
pd.set_option("display.max_columns", 30)
print({"python": sys.version.split()[0], "seed": SEED, "primary_method": PRIMARY_METHOD})
"""
        ),
        markdown(
            """
## Data interfaces

The discovery functions support both this repository and Kaggle's mounted competition
and dataset sources. No report text or DICOM pixel data is copied into notebook output.
"""
        ),
        code(
            """
COMPETITION_SLUG = "rsna-knee-abnormality-detection"
PUBLIC_FILES = {
    "pilkwang": "report_labels_v2.csv",
    "steven": "llm_labels_v4_blend.csv",
    "lixin": "report_labels_gpt56sol.csv",
    "nli": "nli_entailment_scores.csv",
}


def find_train_csv() -> Path:
    candidates = [
        Path.cwd() / "data/raw/train.csv",
        Path.cwd().parent / "data/raw/train.csv",
        Path.cwd().parent.parent / "data/raw/train.csv",
        Path("/kaggle/input") / COMPETITION_SLUG / "train.csv",
    ]
    root = Path("/kaggle/input")
    if root.is_dir():
        candidates.extend(root.glob("**/train.csv"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 100_000:
            frame = pd.read_csv(candidate, nrows=2)
            if {ID, "Report", *TARGETS}.issubset(frame.columns):
                return candidate.resolve()
    raise FileNotFoundError("Could not locate the competition train.csv")


def find_public_file(filename: str) -> Path:
    local_root = Path.cwd() / "data/sample/public_weak_labels"
    candidates = [local_root / filename]
    candidates.extend(local_root.glob(f"*{filename}"))
    for parent in (Path.cwd().parent, Path.cwd().parent.parent):
        candidates.extend((parent / "data/sample/public_weak_labels").glob(f"*{filename}"))
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        candidates.extend(kaggle_root.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate public weak-label file {filename}")


TRAIN_PATH = find_train_csv()
SOURCE_PATHS = {name: find_public_file(filename) for name, filename in PUBLIC_FILES.items()}
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path("outputs/report_weak_supervision")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_CONTEXT = {
    "on_kaggle": ON_KAGGLE,
    "train_path": str(TRAIN_PATH),
    "source_paths": {name: str(path) for name, path in SOURCE_PATHS.items()},
    "output_root": str(OUTPUT_ROOT.resolve()),
}
RUN_CONTEXT
"""
        ),
        markdown("## Load and validate all four public sources"),
        code(
            """
train = pd.read_csv(TRAIN_PATH)
assert train.columns.tolist() == [ID, "Report", *TARGETS]
assert len(train) == 4_407 and train[ID].is_unique
gold_mask = train[TARGETS].notna().all(axis=1)
assert int(gold_mask.sum()) == 58


def load_source(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[ID, *TARGETS])
    assert frame[ID].is_unique, f"{name}: duplicate study IDs"
    assert frame[TARGETS].notna().all().all(), f"{name}: missing target values inside source"
    return frame.rename(columns={target: f"{name}__{target}" for target in TARGETS})


merged = train[[ID, *TARGETS]].copy()
coverage_rows = []
for name, path in SOURCE_PATHS.items():
    source = load_source(path, name)
    merged = merged.merge(source, on=ID, how="left", validate="one_to_one")
    coverage_rows.append({"source": name, "rows": len(source), "path": str(path)})

coverage = pd.DataFrame(coverage_rows).set_index("source")
for name in SOURCE_PATHS:
    columns = [f"{name}__{target}" for target in TARGETS]
    coverage.loc[name, "matched_studies"] = int(merged[columns].notna().all(axis=1).sum())

assert int(coverage.loc["steven", "matched_studies"]) == len(train)
assert int(coverage.loc["lixin", "matched_studies"]) == len(train)
assert int(coverage.loc["nli", "matched_studies"]) == len(train)
assert int(coverage.loc["pilkwang", "matched_studies"]) >= len(train) - 1
coverage
"""
        ),
        markdown(
            """
## Fixed blends and agreement

`mean`, `median`, and rank averaging are calculated for transparency, but `mean` is the
predeclared training target. NLI is an external cross-lingual entailment ablation and is
not part of the primary three-LLM target. Missing Pilkwang coverage for at most one study
is handled by the mean of the other two LLM sources.
"""
        ),
        code(
            """
LLM_SOURCES = ["pilkwang", "steven", "lixin"]
weak = train[[ID]].copy()


def rank01(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


for target in TARGETS:
    llm_columns = [f"{name}__{target}" for name in LLM_SOURCES]
    llm_values = merged[llm_columns]
    weak[f"mean__{target}"] = llm_values.mean(axis=1)
    weak[f"median__{target}"] = llm_values.median(axis=1)
    weak[f"rank__{target}"] = pd.concat(
        [rank01(merged[column]) for column in llm_columns], axis=1
    ).mean(axis=1)
    weak[f"rank4__{target}"] = pd.concat(
        [rank01(merged[column]) for column in [*llm_columns, f"nli__{target}"]], axis=1
    ).mean(axis=1)
    votes = (llm_values > 0.5).astype(float).where(llm_values.notna())
    weak[f"vote_fraction__{target}"] = votes.mean(axis=1)
    weak[f"agreement__{target}"] = (votes.mean(axis=1) - 0.5).abs() * 2.0
    weak[f"spread__{target}"] = llm_values.max(axis=1) - llm_values.min(axis=1)

assert weak[[f"mean__{target}" for target in TARGETS]].notna().all().all()
assert weak.filter(like="mean__").apply(lambda col: col.between(0, 1).all()).all()

agreement_summary = pd.DataFrame(
    {
        "unanimous_fraction": {
            target: (weak[f"agreement__{target}"] == 1.0).mean() for target in TARGETS
        },
        "mean_source_spread": {
            target: weak[f"spread__{target}"].mean() for target in TARGETS
        },
    }
).sort_values("unanimous_fraction")
agreement_summary.style.format("{:.3f}")
"""
        ),
        markdown(
            """
## Expert-set audit, not model selection

ROC-AUC needs no threshold and matches the competition's per-target ranking objective.
Because 58 studies are far too few for stable target-specific tuning, the table and
bootstrap interval are diagnostic only. They do not alter `PRIMARY_METHOD`.
"""
        ),
        code(
            """
gold = merged.loc[gold_mask].reset_index(drop=True)
gold_weak = weak.loc[gold_mask].reset_index(drop=True)
METHODS = [*SOURCE_PATHS, "median", "mean", "rank", "rank4"]


def prediction_for(method: str, target: str) -> pd.Series:
    if method in SOURCE_PATHS:
        return gold[f"{method}__{target}"]
    return gold_weak[f"{method}__{target}"]


score_rows = []
for method in METHODS:
    target_aucs = []
    for target in TARGETS:
        prediction = prediction_for(method, target)
        valid = prediction.notna()
        y_true = gold.loc[valid, target].astype(int)
        auc = roc_auc_score(y_true, prediction[valid]) if y_true.nunique() == 2 else np.nan
        score_rows.append({"method": method, "target": target, "auc": auc})
        target_aucs.append(auc)
    score_rows.append({"method": method, "target": "MACRO", "auc": np.nanmean(target_aucs)})

scores = pd.DataFrame(score_rows)
score_table = scores.pivot(index="method", columns="target", values="auc")
score_table = score_table[[*TARGETS, "MACRO"]].sort_values("MACRO", ascending=False)
score_table.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None, vmin=0.5, vmax=1.0)
"""
        ),
        code(
            """
def bootstrap_macro(method: str, iterations: int = 2_000) -> np.ndarray:
    values = []
    for _ in range(iterations):
        index = RNG.integers(0, len(gold), len(gold))
        aucs = []
        for target in TARGETS:
            prediction = prediction_for(method, target).iloc[index].reset_index(drop=True)
            y_true = gold[target].iloc[index].reset_index(drop=True).astype(int)
            valid = prediction.notna()
            if y_true[valid].nunique() == 2:
                aucs.append(roc_auc_score(y_true[valid], prediction[valid]))
        if len(aucs) == len(TARGETS):
            values.append(float(np.mean(aucs)))
    return np.asarray(values)


bootstrap = bootstrap_macro(PRIMARY_METHOD)
primary_macro = float(score_table.loc[PRIMARY_METHOD, "MACRO"])
primary_ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
pd.Series(
    {
        "primary_method": PRIMARY_METHOD,
        "expert_rows": len(gold),
        "macro_auc": primary_macro,
        "bootstrap_95pct_low": primary_ci[0],
        "bootstrap_95pct_high": primary_ci[1],
        "valid_bootstrap_replicates": len(bootstrap),
    }
).to_frame("value")
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
score_table.loc[[PRIMARY_METHOD, "rank", "pilkwang", "steven", "lixin"], TARGETS].T.plot.bar(
    ax=axes[0], width=0.82
)
axes[0].set(title="Expert-set AUC by target", ylabel="ROC-AUC", xlabel="")
axes[0].set_ylim(0.45, 1.02)
axes[0].legend(fontsize=8, ncol=2)
agreement_summary["unanimous_fraction"].plot.bar(ax=axes[1], color="#4C78A8")
axes[1].set(title="Three-LLM unanimous vote fraction", ylabel="Fraction", xlabel="")
axes[1].set_ylim(0, 1.02)
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Export the image-training contract"),
        code(
            """
training_targets = train[[ID]].copy()
training_targets["is_gold"] = gold_mask.astype(np.uint8)
training_targets["label_source"] = np.where(gold_mask, "expert", "three_llm_mean")

for target in TARGETS:
    pseudo = weak[f"{PRIMARY_METHOD}__{target}"].astype(float)
    training_targets[target] = train[target].where(gold_mask, pseudo).astype(float)
    agreement = weak[f"agreement__{target}"].astype(float)
    training_targets[f"weight__{target}"] = np.where(
        gold_mask,
        2.0,
        0.50 + 0.50 * agreement,
    )

assert len(training_targets) == 4_407 and training_targets[ID].is_unique
assert training_targets[TARGETS].notna().all().all()
assert training_targets[TARGETS].apply(lambda col: col.between(0, 1).all()).all()
assert training_targets.loc[gold_mask, TARGETS].equals(train.loc[gold_mask, TARGETS].astype(float))

weak.to_csv(OUTPUT_ROOT / "consensus_weak_labels.csv", index=False)
training_targets.to_csv(OUTPUT_ROOT / "training_targets.csv", index=False)
scores.to_csv(OUTPUT_ROOT / "weak_label_auc.csv", index=False)
agreement_summary.to_csv(OUTPUT_ROOT / "source_agreement.csv")

manifest = {
    "schema_version": 1,
    "seed": SEED,
    "primary_method": PRIMARY_METHOD,
    "llm_sources": LLM_SOURCES,
    "nli_role": "audit_only",
    "studies": len(training_targets),
    "expert_studies": int(gold_mask.sum()),
    "pseudo_labeled_studies": int((~gold_mask).sum()),
    "targets": TARGETS,
    "expert_audit_macro_auc": primary_macro,
    "expert_audit_bootstrap_95pct": primary_ci,
    "run_context": RUN_CONTEXT,
}
(OUTPUT_ROOT / "weak_label_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

sorted(path.name for path in OUTPUT_ROOT.glob("*"))
"""
        ),
        markdown(
            """
## Result and next experiment

The fixed consensus is now a reproducible report teacher. The image model will train
against soft targets, use per-target agreement weights, and preserve a separate gold-only
evaluation path. The first imaging experiment should be a cheap single-fold 2.5D baseline;
only geometry, sampling, and label-loss changes that improve repeated gold-fold results
should graduate to five folds.
"""
        ),
        code(
            """
required = {
    "consensus_weak_labels.csv",
    "training_targets.csv",
    "weak_label_auc.csv",
    "source_agreement.csv",
    "weak_label_manifest.json",
}
written = {path.name for path in OUTPUT_ROOT.glob("*")}
assert required.issubset(written)
assert PRIMARY_METHOD == "mean"
assert len(bootstrap) >= 1_000

FINAL_CHECK = {
    "all_cells_executed": True,
    "study_count": len(training_targets),
    "expert_count": int(training_targets["is_gold"].sum()),
    "missing_targets": int(training_targets[TARGETS].isna().sum().sum()),
    "primary_macro_auc": round(primary_macro, 6),
    "artifacts": sorted(required),
}
FINAL_CHECK
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
