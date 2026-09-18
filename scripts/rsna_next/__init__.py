"""Versioned P10-P15 experiment infrastructure; historical pipelines stay immutable."""
import os
import sys
from pathlib import Path

for root in (Path.cwd() / ".deps", Path(__file__).resolve().parents[2] / ".deps"):
    if root.is_dir() and str(root) not in sys.path:
        sys.path.append(str(root))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
