#!/usr/bin/env python3
"""Build the private Kaggle entry notebook used to attach the P6 fastMRI secret."""

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
        default=Path(
            "kaggle/rsna-p6-fastmri-staging/rsna-p6-fastmri-staging.ipynb"
        ),
    )
    args = parser.parse_args()

    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        markdown(
            """
# RSNA Knee P6 fastMRI Staging

**Purpose.** This private CPU notebook is the secure entry point for the official
NYU fastMRI image URL required by P6. It does not download the archive in this
version and never prints, hashes, or writes the secret value.

In the notebook editor, open **Add-ons -> Secrets**, create or enable
`FASTMRI_KNEE_DICOM_BATCH2_URL`, and grant this notebook access. Paste only the
signed URL for `knee_DICOMs_batch2`, not the surrounding `curl` command.

After the secret check below returns `secret_available: True`, the next committed
version will stream the official archive member-by-member into a compact cache.
The URL will remain outside source code, notebook output, Git, and cache manifests.
"""
        ),
        markdown("## Secret availability check"),
        code(
            """
from __future__ import annotations

import json
import platform
from pathlib import Path
from urllib.parse import urlsplit

SECRET_NAME = "FASTMRI_KNEE_DICOM_BATCH2_URL"
ON_KAGGLE = Path("/kaggle/working").is_dir()

secret_available = False
secret_official_host = False
secret_expected_package = False
secret_error_type = None
if ON_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient

        secret_value = UserSecretsClient().get_secret(SECRET_NAME)
        parsed = urlsplit(secret_value)
        secret_official_host = parsed.hostname == "fastmri-dataset.s3.amazonaws.com"
        secret_expected_package = parsed.path.endswith("/knee_DICOMs_batch2.tar.xz")
        secret_available = (
            parsed.scheme == "https" and secret_official_host and secret_expected_package
        )
        del secret_value
    except Exception as error:
        secret_error_type = type(error).__name__
else:
    secret_error_type = "NotOnKaggle"

result = {
    "experiment": "P6-FASTMRI-STAGING",
    "on_kaggle": ON_KAGGLE,
    "python": platform.python_version(),
    "secret_name": SECRET_NAME,
    "secret_available": secret_available,
    "secret_official_host": secret_official_host,
    "secret_expected_package": secret_expected_package,
    "secret_error_type": secret_error_type,
    "secret_value_logged": False,
    "download_started": False,
}
if ON_KAGGLE:
    output_path = Path("/kaggle/working/p6_secret_check.json")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
result
"""
        ),
        markdown(
            """
## Expected result and next step

- Before the secret is enabled: `secret_available` is `False`; this is expected.
- After it is enabled: rerun the cell and confirm `secret_available` is `True`.
- Do not display the secret or paste it into another cell.
- The notebook writes only the boolean result above to `p6_secret_check.json` so it
  can be verified through the Kaggle CLI; the URL is never included.
- Tell Codex once the secret is enabled. The subsequent staging version will add
  streaming download, annotation/hash validation, DICOM overlay checks, and the
  128-volume CPU cache smoke.
"""
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
