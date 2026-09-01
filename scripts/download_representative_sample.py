#!/usr/bin/env python3
"""Download a bounded representative DICOM sample through Kaggle CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiListDataTreeFilesRequest


def list_series_files(api: KaggleApi, competition: str, path: str) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    token: str | None = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListDataTreeFilesRequest()
            request.competition_name = competition
            request.path = path
            request.page_size = 200
            request.page_token = token
            response = client.competitions.competition_api_client.list_data_tree_files(request)
            files.extend(
                {
                    "name": item.name,
                    "relative_url": item.relative_url,
                    "bytes": item.total_bytes,
                }
                for item in response.files
            )
            token = response.next_page_token or None
            if not token:
                break
    return sorted(files, key=lambda item: str(item["name"]))


def download_one(competition: str, root: Path, item: dict[str, object]) -> Path:
    relative = Path(str(item["relative_url"]))
    destination = root / relative.parent
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / relative.name
    if output.is_file() and output.stat().st_size == int(item["bytes"]):
        return output
    command = [
        "rtk",
        "kaggle",
        "competitions",
        "download",
        competition,
        "-f",
        str(relative),
        "-p",
        str(destination),
        "-q",
    ]
    last_error = ""
    for attempt in range(1, 5):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            break
        last_error = (result.stderr or result.stdout).strip()
        if attempt < 4:
            time.sleep(2 ** (attempt - 1))
    else:
        raise RuntimeError(f"Kaggle CLI failed after 4 attempts for {relative}: {last_error}")
    if not output.is_file():
        raise FileNotFoundError(output)
    if output.stat().st_size != int(item["bytes"]):
        raise RuntimeError(
            f"Size mismatch for {output}: expected {item['bytes']}, got {output.stat().st_size}"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("data/sample/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("data/sample/files"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    competition = selection["competition"]
    api = KaggleApi()
    api.authenticate()

    planned: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for mode, records in (
        ("complete_volume", selection["volume_series"]),
        ("representative_slice", selection["single_slice_series"]),
    ):
        for record in records:
            series_path = (
                f"train_series/{record['StudyInstanceUID']}/{record['SeriesInstanceUID']}"
            )
            available = list_series_files(api, competition, series_path)
            if not available:
                raise RuntimeError(f"Kaggle returned no files for {series_path}")
            chosen = available if mode == "complete_volume" else [available[len(available) // 2]]
            for item in chosen:
                item.update(
                    {
                        "mode": mode,
                        "scenario": record["scenario"],
                        "StudyInstanceUID": record["StudyInstanceUID"],
                        "SeriesInstanceUID": record["SeriesInstanceUID"],
                    }
                )
            planned.extend(chosen)
            summaries.append(
                {
                    **record,
                    "mode": mode,
                    "available_slices": len(available),
                    "selected_slices": len(chosen),
                    "selected_bytes": sum(int(item["bytes"]) for item in chosen),
                }
            )

    total_bytes = sum(int(item["bytes"]) for item in planned)
    print(
        json.dumps(
            {
                "series": len(summaries),
                "files": len(planned),
                "bytes": total_bytes,
                "gib": round(total_bytes / 1024**3, 3),
                "details": summaries,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, competition, args.output, item): item for item in planned}
        for index, future in enumerate(as_completed(futures), start=1):
            item = dict(futures[future])
            path = future.result()
            item["local_path"] = str(path)
            item["downloaded_bytes"] = path.stat().st_size
            completed.append(item)
            if index % 20 == 0 or index == len(planned):
                print(f"downloaded {index}/{len(planned)}", flush=True)

    manifest = {
        "competition": competition,
        "selection": str(args.selection),
        "file_count": len(completed),
        "downloaded_bytes": sum(int(item["downloaded_bytes"]) for item in completed),
        "files": sorted(completed, key=lambda item: str(item["relative_url"])),
    }
    manifest_path = args.output.parent / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
