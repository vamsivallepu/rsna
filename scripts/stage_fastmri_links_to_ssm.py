#!/usr/bin/env python3
"""Extract official fastMRI DICOM URLs from an email PDF into SSM SecureString.

The signed URLs are kept in memory only. Console output contains package names,
parameter names, and expiry timestamps, never URL values or query credentials.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import boto3


EXPECTED_HOST = "fastmri-dataset.s3.amazonaws.com"
PACKAGES = {
    "knee_singlecoil_train.tar.xz": "/rsna/p6/fastmri/knee-singlecoil-train-url",
    "knee_singlecoil_val.tar.xz": "/rsna/p6/fastmri/knee-singlecoil-val-url",
    "knee_multicoil_train_batch_0.tar.xz": (
        "/rsna/p6/fastmri/knee-multicoil-train-batch-0-url"
    ),
    "knee_multicoil_train_batch_1.tar.xz": (
        "/rsna/p6/fastmri/knee-multicoil-train-batch-1-url"
    ),
    "knee_multicoil_train_batch_2.tar.xz": (
        "/rsna/p6/fastmri/knee-multicoil-train-batch-2-url"
    ),
    "knee_multicoil_train_batch_3.tar.xz": (
        "/rsna/p6/fastmri/knee-multicoil-train-batch-3-url"
    ),
    "knee_multicoil_train_batch_4.tar.xz": (
        "/rsna/p6/fastmri/knee-multicoil-train-batch-4-url"
    ),
    "knee_multicoil_val.tar.xz": "/rsna/p6/fastmri/knee-multicoil-val-url",
    "knee_DICOMs_batch1.tar.xz": "/rsna/p6/fastmri/knee-dicoms-batch1-url",
    "knee_DICOMs_batch2.tar.xz": "/rsna/p6/fastmri/knee-dicoms-batch2-url",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--profile", default="personal")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_urls(pdf: Path) -> dict[str, tuple[str, int]]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = re.findall(
        r'"(https://fastmri-dataset\.s3\.amazonaws\.com/[^\"]+)"',
        result.stdout,
        flags=re.DOTALL,
    )
    selected: dict[str, tuple[str, int]] = {}
    for candidate in candidates:
        url = re.sub(r"\s+", "", candidate)
        parsed = urlsplit(url)
        package = Path(parsed.path).name
        if package not in PACKAGES:
            continue
        if parsed.scheme != "https" or parsed.hostname != EXPECTED_HOST:
            raise ValueError(f"Unexpected host or scheme for {package}")
        query = parse_qs(parsed.query, keep_blank_values=True)
        if not {"AWSAccessKeyId", "Signature", "Expires"}.issubset(query):
            raise ValueError(f"Required signed-query fields are absent for {package}")
        try:
            expires = int(query["Expires"][0])
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(f"Invalid expiry for {package}") from error
        if package in selected and selected[package][0] != url:
            raise ValueError(f"Multiple different signed URLs found for {package}")
        selected[package] = (url, expires)
    missing = set(PACKAGES) - set(selected)
    if missing:
        raise ValueError(f"Missing expected package commands: {sorted(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    selected = extract_urls(args.pdf)
    now = int(dt.datetime.now(tz=dt.UTC).timestamp())
    summaries = []

    ssm = None
    if not args.dry_run:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        ssm = session.client("ssm")

    for package, parameter_name in PACKAGES.items():
        url, expires = selected[package]
        if expires <= now:
            raise ValueError(f"Signed URL is expired for {package}")
        if ssm is not None:
            ssm.put_parameter(
                Name=parameter_name,
                Description=f"Temporary official NYU fastMRI URL for {package}",
                Value=url,
                Type="SecureString",
                Overwrite=True,
                Tier="Standard",
                DataType="text",
            )
            ssm.add_tags_to_resource(
                ResourceType="Parameter",
                ResourceId=parameter_name,
                Tags=[
                    {"Key": "Project", "Value": "rsna-p6"},
                    {"Key": "ManagedBy", "Value": "codex"},
                ],
            )
        summaries.append(
            {
                "package": package,
                "parameter": parameter_name,
                "expires_utc": dt.datetime.fromtimestamp(expires, tz=dt.UTC).isoformat(),
                "official_host": True,
                "stored": ssm is not None,
                "value_logged": False,
            }
        )
        del url

    print(json.dumps({"dry_run": args.dry_run, "packages": summaries}, indent=2))


if __name__ == "__main__":
    main()
