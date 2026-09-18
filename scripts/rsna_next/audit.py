"""Create the P10 split ledger and reproduce historical teacher exposure."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .contracts import ID, VERSION, make_splits, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--series", type=Path, default=Path("data/raw/train_series.csv"))
    p.add_argument("--output", type=Path, default=Path("outputs/experiments/P10-contract"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta, series = pd.read_csv(args.train), pd.read_csv(args.series)
    splits = make_splits(meta)
    splits.to_csv(args.output / "splits.csv", index=False)
    pseudo = np.flatnonzero(~splits.is_gold.to_numpy())
    old_train, _ = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=20260902).split(
        pseudo, groups=splits.report_group.to_numpy()[pseudo]))
    old_train = pseudo[old_train]
    old_valid = splits.index[splits.role == "pseudo_valid"].to_numpy()
    counts = series.groupby(ID).agg(series_count=("SeriesInstanceUID", "size"),
                                  plane_count=("Anatomical_Plane", "nunique"),
                                  fluid_count=("Fluid_Sensitive", "sum"))
    # Label-blind systematic sampling across metadata strata; raw DICOM size is
    # added by the remote benchmark instead of pretending series count is size.
    selection = splits[[ID]].join(counts, on=ID).sort_values(
        ["plane_count", "fluid_count", "series_count", ID])
    ix = np.round(np.linspace(0, len(selection) - 1, min(240, len(selection)))).astype(int)
    selection.iloc[ix].to_csv(args.output / "benchmark_studies.csv", index=False)
    summary = {"schema_version": VERSION, "status": "validated_local_split_contract",
               "studies": len(splits), "roles": splits.role.value_counts().to_dict(),
               "teacher_validation_exposure_historical": len(np.intersect1d(old_train, old_valid)),
               "pseudo_validation_studies": len(old_valid),
               "expert_report_twins_in_historical_representation": int(
                   (splits.loc[old_train, "role"] == "excluded_expert_twin").sum()),
               "teacher_fold_counts": splits.loc[splits.teacher_fold >= 0, "teacher_fold"].value_counts().to_dict(),
               "source_train_sha256": sha256(args.train), "source_series_sha256": sha256(args.series),
               "split_sha256": sha256(args.output / "splits.csv"),
               "benchmark_studies_sha256": sha256(args.output / "benchmark_studies.csv"),
               "remaining": ["full-mount pixel/near-duplicate audit", "200+ study exact P8 runtime benchmark",
                             "new split-safe model training"],
               "historical_report_matches_are_not_proof_of_patient_duplication": True}
    write_json(args.output / "manifest.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
