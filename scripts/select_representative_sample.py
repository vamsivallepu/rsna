#!/usr/bin/env python3
"""Select a deterministic, compact sample from RSNA competition metadata.

The selection covers both expert label states for every target, all series
descriptor combinations present in training metadata, and labeled/unlabeled
report scenarios. It does not inspect image pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


ID = "StudyInstanceUID"
REPORT = "Report"
SERIES_ID = "SeriesInstanceUID"
PLANE = "Anatomical_Plane"
FLUID = "Fluid_Sensitive"
FAT_SUPPRESSION = "Fat_Suppression"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def non_ascii_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(char) > 127 for char in letters) / len(letters)


def script_families(text: str) -> list[str]:
    families: set[str] = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for family in ("LATIN", "CYRILLIC", "ARABIC", "THAI", "CJK", "HANGUL"):
            if family in name:
                families.add(family)
    return sorted(families)


def series_scenario(row: dict[str, str]) -> str:
    return f"{row[PLANE]}|fluid={row[FLUID]}|fat_suppression={row[FAT_SUPPRESSION]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--series", type=Path, default=Path("data/raw/train_series.csv"))
    args = parser.parse_args()

    studies = read_csv(args.train)
    series = read_csv(args.series)
    targets = [column for column in studies[0] if column not in {ID, REPORT}]

    by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in series:
        by_study[row[ID]].append(row)

    labeled = [row for row in studies if all(row[target] in {"0", "1"} for target in targets)]
    unlabeled = [row for row in studies if all(row[target] == "" for target in targets)]

    label_universe = {f"label:{target}={value}" for target in targets for value in ("0", "1")}
    series_universe = {f"series:{series_scenario(row)}" for row in series}
    universe = label_universe | series_universe

    coverage: dict[str, set[str]] = {}
    row_by_id = {row[ID]: row for row in studies}
    for row in labeled:
        study_id = row[ID]
        covered = {f"label:{target}={row[target]}" for target in targets}
        covered |= {f"series:{series_scenario(item)}" for item in by_study[study_id]}
        coverage[study_id] = covered

    uncovered = set(universe)
    chosen_labeled: list[str] = []
    while uncovered:
        candidates = [
            (len(coverage[study_id] & uncovered), study_id)
            for study_id in coverage
            if study_id not in chosen_labeled
        ]
        gain, winner = max(candidates, key=lambda item: (item[0], item[1]))
        if gain == 0:
            break
        chosen_labeled.append(winner)
        uncovered -= coverage[winner]

    english_candidates = [
        row
        for row in unlabeled
        if non_ascii_ratio(row[REPORT]) < 0.01
        and re.search(r"\b(the|knee|findings|impression|no)\b", row[REPORT], re.I)
    ]
    english = min(english_candidates or unlabeled, key=lambda row: row[ID])
    multilingual = max(unlabeled, key=lambda row: (non_ascii_ratio(row[REPORT]), row[ID]))

    chosen_studies = list(dict.fromkeys(chosen_labeled + [english[ID], multilingual[ID]]))

    volume_series: list[dict[str, str]] = []
    for scenario in sorted(series_universe):
        match = next(
            (
                item
                for study_id in chosen_studies
                for item in by_study[study_id]
                if f"series:{series_scenario(item)}" == scenario
            ),
            None,
        )
        if match is None:
            match = next(item for item in series if f"series:{series_scenario(item)}" == scenario)
            if match[ID] not in chosen_studies:
                chosen_studies.append(match[ID])
        volume_series.append(match)

    represented = {item[ID] for item in volume_series}
    single_slice_series = [by_study[study_id][0] for study_id in chosen_studies if study_id not in represented]

    payload = {
        "selection_policy": {
            "label_coverage": "both 0 and 1 for all 12 expert targets",
            "series_coverage": sorted(value.removeprefix("series:") for value in series_universe),
            "report_coverage": ["expert-labeled", "unlabeled ASCII/English-like", "unlabeled non-ASCII"],
            "volume_policy": "one complete series per descriptor scenario; one middle slice for remaining studies",
        },
        "targets": targets,
        "uncovered_requirements": sorted(uncovered),
        "studies": [
            {
                "StudyInstanceUID": study_id,
                "label_status": "expert" if study_id in chosen_labeled else "unlabeled",
                "report_scripts": script_families(row_by_id[study_id][REPORT]),
                "report_non_ascii_ratio": round(non_ascii_ratio(row_by_id[study_id][REPORT]), 4),
                "labels": {target: row_by_id[study_id][target] or None for target in targets},
            }
            for study_id in chosen_studies
        ],
        "volume_series": [
            {
                ID: row[ID],
                SERIES_ID: row[SERIES_ID],
                "scenario": series_scenario(row),
            }
            for row in volume_series
        ],
        "single_slice_series": [
            {
                ID: row[ID],
                SERIES_ID: row[SERIES_ID],
                "scenario": series_scenario(row),
            }
            for row in single_slice_series
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
