#!/usr/bin/env python3
"""Extract immutable BGE-M3 embeddings for every RSNA training report."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ID = "StudyInstanceUID"
RUN_ID = "P7-REPORT-EMBEDDINGS-v1"
MODEL_SOURCE = "yethukmutt/bge-m3/Transformers/m3/1"
UPSTREAM_MODEL = "BAAI/bge-m3"
MODEL_LICENSE = "MIT"
EXPECTED_STUDIES = 4_407
EXPECTED_DIMENSION = 1_024
MAX_TOKENS = 4_096
BATCH_SIZE_PER_RANK = 4
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(
    "outputs/p7_report_embeddings"
)


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_report(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split())


def find_train_csv() -> Path:
    candidates = [
        Path(
            "/kaggle/input/competitions/rsna-knee-abnormality-detection/"
            "train.csv"
        ),
        Path("data/raw/train.csv"),
    ]
    selected = next(
        (path.resolve() for path in candidates if path.is_file()), None
    )
    if selected is None:
        raise FileNotFoundError("train.csv")
    return selected


def find_model_root() -> Path:
    candidates = [
        Path(
            "/kaggle/input/models/yethukmutt/bge-m3/Transformers/m3/1"
        ),
        Path(
            "/kaggle/input/models/yethukmutt/bge-m3/transformers/m3/1"
        ),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(path.parent for path in input_root.glob("**/config.json"))
    for root in candidates:
        config_path = root / "config.json"
        if not config_path.is_file() or not (root / "tokenizer.json").is_file():
            continue
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        weights = [
            *root.glob("*.safetensors"),
            *root.glob("pytorch_model*.bin"),
        ]
        if (
            config.get("model_type") == "xlm-roberta"
            and sum(path.stat().st_size for path in weights) > 1_000_000_000
        ):
            return root.resolve()
    raise FileNotFoundError("pinned BGE-M3 model root")


def model_source_files(root: Path) -> list[Path]:
    names = [
        "README.md",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "pytorch_model.bin",
        "model.safetensors",
        "__huggingface_repos__.json",
    ]
    paths = [root / name for name in names if (root / name).is_file()]
    paths.extend(sorted(root.glob("model-*.safetensors")))
    paths.extend(sorted(root.glob("pytorch_model-*.bin")))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    assert any(path.stat().st_size > 1_000_000_000 for path in unique)
    return unique


def gpu_snapshot() -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def launch_ddp() -> int:
    visible = torch.cuda.device_count()
    if visible != 2:
        raise RuntimeError(f"P7-A requires exactly two GPUs; observed {visible}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    print({"P7_REPORT_EMBEDDINGS_DDP_LAUNCH": command}, flush=True)
    subprocess.run(command, check=True)
    print({"P7_REPORT_EMBEDDINGS_DDP_COMPLETE": True}, flush=True)
    return 0


class ReportDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        reports: list[str],
        tokenizer: object,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.reports = reports
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[int, str]:
        index = int(self.indices[position])
        return index, self.reports[index]

    def collate(self, batch: list[tuple[int, str]]) -> tuple[np.ndarray, dict]:
        indices, reports = zip(*batch, strict=True)
        tokens = self.tokenizer(
            list(reports),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        assert tokens["input_ids"].shape[1] <= MAX_TOKENS
        return np.asarray(indices, dtype=np.int64), dict(tokens)


def worker_main() -> int:
    if not ON_KAGGLE:
        raise RuntimeError("P7 report extraction is intentionally Kaggle-only")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    is_primary = rank == 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_path = find_train_csv()
    model_root = find_model_root()
    train = pd.read_csv(train_path, dtype={ID: str})
    assert len(train) == EXPECTED_STUDIES
    assert train[ID].is_unique and train["Report"].notna().all()
    reports = [normalize_report(value) for value in train["Report"]]
    assert all(reports)

    source_hashes: dict[str, str] = {}
    if is_primary:
        source_files = model_source_files(model_root)
        with ThreadPoolExecutor(max_workers=3) as pool:
            hashes = list(pool.map(sha256_file, source_files))
        source_hashes = {
            str(path.relative_to(model_root)): digest
            for path, digest in zip(source_files, hashes, strict=True)
        }
        readme = model_root / "README.md"
        if readme.is_file():
            assert "license: mit" in readme.read_text(errors="replace").lower()
        print(
            {
                "P7_REPORT_SOURCE_VALIDATED": True,
                "model_source": MODEL_SOURCE,
                "model_root": str(model_root),
                "source_files": len(source_hashes),
                "gpu": gpu_snapshot(),
            },
            flush=True,
        )
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    local_indices = np.arange(rank, len(train), world_size, dtype=np.int64)
    local_reports = [reports[int(index)] for index in local_indices]
    unpadded = tokenizer(
        local_reports,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    local_lengths = np.asarray([len(tokens) for tokens in unpadded], dtype=np.int32)
    del unpadded, local_reports
    assert len(local_lengths) == len(local_indices)
    assert int(local_lengths.max()) <= MAX_TOKENS
    ordered = local_indices[np.argsort(local_lengths, kind="stable")]
    dataset = ReportDataset(ordered, reports, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_RANK,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate,
    )

    model = AutoModel.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    assert hidden_size == EXPECTED_DIMENSION
    embeddings = np.empty(
        (len(ordered), EXPECTED_DIMENSION), dtype=np.float16
    )
    encoded_indices = np.empty(len(ordered), dtype=np.int64)

    encoded = 0
    encode_started = time.time()
    with torch.inference_mode():
        for indices, tokens in loader:
            tokens = {
                key: value.to(device, non_blocking=True)
                for key, value in tokens.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**tokens, return_dict=True)
            dense = F.normalize(
                output.last_hidden_state[:, 0].float(), dim=-1
            )
            stop = encoded + len(indices)
            encoded_indices[encoded:stop] = indices
            embeddings[encoded:stop] = dense.cpu().numpy().astype(np.float16)
            encoded = stop
            if encoded % 200 == 0 or encoded == len(ordered):
                print(
                    {
                        "P7_REPORT_EMBEDDINGS_PROGRESS": (
                            f"rank={rank} {encoded}/{len(ordered)}"
                        ),
                        "elapsed_minutes": (time.time() - encode_started) / 60.0,
                        "gpu": gpu_snapshot() if is_primary else None,
                    },
                    flush=True,
                )
    assert encoded == len(ordered) and np.isfinite(embeddings).all()
    np.save(OUTPUT_ROOT / f"p7_report_indices_rank_{rank}.npy", encoded_indices)
    np.save(OUTPUT_ROOT / f"p7_report_embeddings_rank_{rank}.npy", embeddings)
    np.save(OUTPUT_ROOT / f"p7_report_lengths_rank_{rank}.npy", local_lengths)
    np.save(OUTPUT_ROOT / f"p7_report_length_indices_rank_{rank}.npy", local_indices)
    elapsed = torch.tensor(
        [time.time() - encode_started], dtype=torch.float64, device=device
    )
    memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / (1024**2)],
        dtype=torch.float64,
        device=device,
    )
    elapsed_all = [torch.zeros_like(elapsed) for _ in range(world_size)]
    memory_all = [torch.zeros_like(memory) for _ in range(world_size)]
    dist.all_gather(elapsed_all, elapsed)
    dist.all_gather(memory_all, memory)
    dist.barrier()

    if is_primary:
        merged = np.empty(
            (len(train), EXPECTED_DIMENSION), dtype=np.float16
        )
        token_lengths = np.empty(len(train), dtype=np.int32)
        seen = np.zeros(len(train), dtype=bool)
        for shard_rank in range(world_size):
            indices = np.load(
                OUTPUT_ROOT / f"p7_report_indices_rank_{shard_rank}.npy"
            )
            values = np.load(
                OUTPUT_ROOT / f"p7_report_embeddings_rank_{shard_rank}.npy"
            )
            length_indices = np.load(
                OUTPUT_ROOT / f"p7_report_length_indices_rank_{shard_rank}.npy"
            )
            lengths = np.load(
                OUTPUT_ROOT / f"p7_report_lengths_rank_{shard_rank}.npy"
            )
            assert len(indices) == len(values)
            assert len(length_indices) == len(lengths)
            merged[indices] = values
            token_lengths[length_indices] = lengths
            seen[indices] = True
        assert seen.all() and np.isfinite(merged).all()
        norms = np.linalg.norm(merged.astype(np.float32), axis=1)
        assert np.max(np.abs(norms - 1.0)) < 0.002
        assert int(token_lengths.max()) <= MAX_TOKENS

        embedding_name = "p7_bge_m3_report_embeddings.npy"
        studies_name = "p7_bge_m3_report_studies.csv"
        lengths_name = "p7_bge_m3_token_lengths.npy"
        np.save(OUTPUT_ROOT / embedding_name, merged)
        np.save(OUTPUT_ROOT / lengths_name, token_lengths)
        pd.DataFrame({ID: train[ID]}).to_csv(
            OUTPUT_ROOT / studies_name, index=False
        )
        for shard_rank in range(world_size):
            for kind in (
                "indices",
                "embeddings",
                "lengths",
                "length_indices",
            ):
                (OUTPUT_ROOT / f"p7_report_{kind}_rank_{shard_rank}.npy").unlink()

        artifacts = [embedding_name, studies_name, lengths_name]
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "studies": len(train),
            "unique_studies": int(train[ID].nunique()),
            "nonempty_reports": int(pd.Series(reports).str.len().gt(0).sum()),
            "train_csv_sha256": sha256_file(train_path),
            "study_order_sha256": sha256_lines(train[ID].astype(str).tolist()),
            "normalized_report_order_sha256": sha256_lines(reports),
            "model": {
                "source": MODEL_SOURCE,
                "upstream": UPSTREAM_MODEL,
                "license": MODEL_LICENSE,
                "root": str(model_root),
                "source_files_sha256": source_hashes,
                "pooling": "first token followed by L2 normalization",
                "embedding_dimension": EXPECTED_DIMENSION,
                "supported_context_tokens": int(
                    getattr(model.config, "max_position_embeddings", 8194) - 2
                ),
                "extraction_max_tokens": MAX_TOKENS,
            },
            "tokens": {
                "minimum": int(token_lengths.min()),
                "median": float(np.median(token_lengths)),
                "p95": float(np.quantile(token_lengths, 0.95)),
                "maximum": int(token_lengths.max()),
                "truncated_reports": int((token_lengths > MAX_TOKENS).sum()),
            },
            "embedding_norm": {
                "minimum": float(norms.min()),
                "maximum": float(norms.max()),
                "max_abs_error_from_one": float(np.max(np.abs(norms - 1.0))),
            },
            "execution": {
                "world_size": world_size,
                "visible_gpus": torch.cuda.device_count(),
                "batch_size_per_rank": BATCH_SIZE_PER_RANK,
                "dtype": "float16",
                "elapsed_minutes_by_rank": [
                    float(value.item() / 60.0) for value in elapsed_all
                ],
                "peak_allocated_mib_by_rank": [
                    float(value.item()) for value in memory_all
                ],
                "total_minutes": (time.time() - started) / 60.0,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            "artifacts_sha256": {
                name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
            },
        }
        manifest_path = OUTPUT_ROOT / "p7_report_embeddings_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            {
                "P7_REPORT_EMBEDDINGS_COMPLETE": True,
                "studies": len(train),
                "max_tokens": int(token_lengths.max()),
                "manifest_sha256": sha256_file(manifest_path),
                "elapsed_minutes": manifest["execution"]["total_minutes"],
            },
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main() -> int:
    if ON_KAGGLE and "--ddp-worker" not in sys.argv:
        return launch_ddp()
    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
