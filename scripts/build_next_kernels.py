#!/usr/bin/env python3
"""Build private offline P10-P15 jobs without launching or submitting them."""
import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path

from rsna_next.contracts import sha256, write_json


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Source drift at {old[:80]!r}")
    return source.replace(old, new, 1)


def build_runtime(output, benchmark_studies=None):
    original = Path("kaggle/rsna-p8-blend-canary/rsna-p8-blend-canary.py")
    if sha256(original) != "69f1d0c828d218b445f3571ec879c4ab689b93ca5c7e1e00926dbad35777a9fe":
        raise ValueError("P8 champion source changed")
    source = original.read_text()
    old = '''        dicom_root = data_root / "test_series"
        test = pd.read_csv(data_root / "test.csv", dtype={ID: str})
        series = pd.read_csv(data_root / "test_series.csv", dtype={ID: str})
        sample = pd.read_csv(
            data_root / "sample_submission.csv", dtype={ID: str}
        )'''
    new = '''        dicom_root = data_root / "train_series"
        # Only acquisition metadata and IDs are read for benchmark selection.
        all_ids = pd.read_csv(data_root / "train.csv", usecols=[ID], dtype={ID: str})
        all_series = pd.read_csv(data_root / "train_series.csv", dtype={ID: str})
        counts = all_series.groupby(ID).agg(series_count=("SeriesInstanceUID", "size"),
            plane_count=("Anatomical_Plane", "nunique"), fluid_count=("Fluid_Sensitive", "sum"))
        ordered = all_ids.join(counts, on=ID).sort_values(["plane_count", "fluid_count", "series_count", ID])
        chosen = np.round(np.linspace(0, len(ordered) - 1, 240)).astype(int)
        test = ordered.iloc[chosen][[ID]].reset_index(drop=True)
        series = all_series[all_series[ID].isin(test[ID])].copy()
        sample = pd.DataFrame({ID: test[ID], **{t: 0.5 for t in TARGETS}})
        test.to_csv(OUTPUT_ROOT / "p10_benchmark_studies.csv", index=False)'''
    selection_hash = None
    if benchmark_studies:
        import pandas as pd
        selected = pd.read_csv(benchmark_studies)
        ids = selected["StudyInstanceUID"].astype(str).tolist()
        known = set(pd.read_csv("data/raw/train.csv", usecols=["StudyInstanceUID"])["StudyInstanceUID"])
        if len(ids) < 200 or len(ids) != len(set(ids)) or not set(ids) <= known:
            raise ValueError("Benchmark must contain at least 200 unique training study IDs")
        selection_hash = sha256(benchmark_studies)
        new = new.replace('test = ordered.iloc[chosen][[ID]].reset_index(drop=True)',
                          f'test = pd.DataFrame({{ID: {ids!r}}})')
    source = replace_once(source, old, new)
    tail = '''if __name__ == "__main__":
    raise SystemExit(main())'''
    replacement = '''if __name__ == "__main__":
    main()
    manifest = json.loads((OUTPUT_ROOT / "p8_blend_submission_manifest.json").read_text())
    elapsed = manifest["elapsed"]
    n = manifest["studies"]
    assert n >= 200
    startup = elapsed["total_minutes"] - elapsed["dicom_and_model_inference_minutes"]
    projected = (max(0, startup) + 1.25 * elapsed["dicom_and_model_inference_minutes"] * 1300 / n) / 60
    result = {"experiment_id": "P10-RUNTIME", "studies": n, "hidden_studies": 1300,
        "safety_multiplier": 1.25, "projected_hidden_hours": projected,
        "runtime_gate_passed": projected <= 7.5, "pipeline": "exact P8 model/preprocessing/scheduler",
        "benchmark_selection": "label-blind acquisition-stratified quantiles",
        "is_submission": False, "original_source_sha256": "69f1d0c828d218b445f3571ec879c4ab689b93ca5c7e1e00926dbad35777a9fe"}
    (OUTPUT_ROOT / "p10_runtime.json").write_text(json.dumps(result, indent=2))
    (OUTPUT_ROOT / "submission.csv").rename(OUTPUT_ROOT / "p10_benchmark_predictions.csv")
    print(result, flush=True)'''
    source = replace_once(source, tail, replacement)
    if selection_hash:
        source = source.replace('"benchmark_selection": "label-blind acquisition-stratified quantiles",',
                                f'"benchmark_selection": "frozen label-blind header-audit selection", "selection_sha256": "{selection_hash}",')
    meta = json.loads(Path("kaggle/rsna-p8-blend-canary/kernel-metadata.json").read_text())
    meta.update(id="vallepuvamsikrishna/rsna-knee-p10-representative-runtime", title="RSNA Knee P10 Representative Runtime", code_file="run.py")
    if selection_hash:
        meta.update(id="vallepuvamsikrishna/rsna-knee-p10-scanner-size-runtime", title="RSNA Knee P10 Scanner Size Runtime")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.py").write_text(source)
    write_json(output / "kernel-metadata.json", meta)
    write_json(output / "build.json", {"kind": "runtime", "code_sha256": sha256(output / "run.py"), "selection_sha256": selection_hash})


def build_training(args):
    plan = json.loads(Path("experiments/next_plan.json").read_text())
    cfg = {**plan["defaults"], **plan["recipes"][args.recipe]}
    cfg["seed"] = args.seed
    if args.folds and (args.mode != "teacher" or args.smoke or len(set(args.folds)) != len(args.folds) or len(args.folds) > 3):
        raise ValueError("Teacher banks may contain one to three distinct full folds")
    if args.head:
        cfg["head"] = args.head
    if args.mode == "refit":
        if args.smoke or not args.winner_config or not args.refit_epochs or args.refit_epochs < 1:
            raise ValueError("Refits require a frozen --winner-config and --refit-epochs")
        winner = json.loads(args.winner_config.read_text())
        for key in ("initialization", "head", "train_windows", "sampling", "labels", "expert_repeat", "checkpointing"):
            cfg[key] = winner[key]
        cfg["refit_epochs"] = args.refit_epochs
    if cfg.get("requires_winning_recipe_lock"):
        if not args.winner_config:
            raise ValueError("Diversity run requires --winner-config frozen from confirmation")
        winner = json.loads(args.winner_config.read_text())
        cfg.update({k: winner[k] for k in ("epochs", "train_windows", "sampling", "head", "labels")})
    if not args.smoke:
        if not args.smoke_evidence:
            raise ValueError("Full runs require --smoke-evidence from a validated two-GPU smoke")
        evidence = json.loads(args.smoke_evidence.read_text())
        if evidence.get("status") != "validated_smoke" or not all(evidence["gates"].values()):
            raise ValueError("Smoke validation has not passed")
        if not args.replay_evidence or json.loads(args.replay_evidence.read_text()).get("status") != "replay_validated":
            raise ValueError("Full runs require --replay-evidence from full smoke prediction replay")
        replay = json.loads(args.replay_evidence.read_text())
        if replay["run_manifest_sha256"] != evidence["manifest_sha256"]:
            raise ValueError("Smoke and replay evidence refer to different runs")
        for key in ("initialization", "head", "train_windows", "checkpointing"):
            if cfg.get(key) != evidence["config"].get(key):
                raise ValueError(f"Resource-changing configuration needs its own smoke: {key}")
    split_groups = args.splits.with_suffix(".groups.json")
    if not args.splits.exists() or not split_groups.exists():
        raise ValueError("New training requires the versioned report/image split and its group manifest")
    split_contract = json.loads(split_groups.read_text())
    if split_contract["split_sha256"] != sha256(args.splits):
        raise ValueError("Split artifact changed")
    if not args.smoke and not split_contract.get("full_training_allowed"):
        raise ValueError("Raw duplicate verification is required before full training")
    if args.mode == "teacher" and cfg["labels"] != "report":
        raise ValueError("Teacher must use original report labels")
    if cfg.get("requires_p14_cache") and not args.cache_sources:
        raise ValueError("Geometry arm needs the four validated P14 cache sources")
    if cfg["labels"] != "report" and not args.label_source:
        raise ValueError("Student label artifact source required")
    payload = {str(p): p.read_text() for p in Path("scripts/rsna_next").glob("*.py")}
    payload["scripts/__init__.py"] = ""
    payload["recipe.json"] = json.dumps(cfg)
    payload["splits.csv.b64z"] = base64.b64encode(zlib.compress(args.splits.read_bytes())).decode()
    payload["splits.groups.json"] = split_groups.read_text()
    program = '''import json, os, sys, base64, zlib
from pathlib import Path
ROOT = Path('/kaggle/working/next_runtime')
ROOT.mkdir(parents=True, exist_ok=True)
PAYLOAD = __PAYLOAD__
for name, text in PAYLOAD.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
(ROOT / 'splits.csv').write_bytes(zlib.decompress(base64.b64decode((ROOT / 'splits.csv.b64z').read_text())))
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
def find(name):
    candidates = list(Path('/kaggle/input').glob('**/' + name))
    if len(candidates) != 1:
        raise ValueError(f'Expected one {name}, found {len(candidates)}')
    return candidates[0]
from scripts.rsna_next.contracts import read_splits
import pandas as pd
train = find('train.csv')
splits = ROOT / 'splits.csv'
read_splits(splits, pd.read_csv(train))
args = __ARGS__
args += ['--train', str(train), '--splits', str(splits), '--targets', str(find('training_targets.csv')),
         '--weights', str(find(__WEIGHTS__)), '--config', str(ROOT / 'recipe.json'),
         '--output', '/kaggle/working/next_run']
__CACHE__
__LABELS__
sys.argv = ['train', *args]
from scripts.rsna_next.train import main
for fold in __FOLDS__:
    run_args = args.copy()
    run_args[run_args.index('--fold') + 1] = str(fold)
    if __BANK__:
        run_args[run_args.index('--output') + 1] = f'/kaggle/working/next_run/fold-{fold}'
    sys.argv = ['train', *run_args]
    result = main()
    if result:
        raise SystemExit(result)
    if '--smoke' not in args:
        from scripts.rsna_next.validate import validate_run
        from scripts.rsna_next.contracts import read_splits, write_json
        run_root = Path(run_args[run_args.index('--output') + 1])
        if __MODE__ != 'refit':
            manifest, pred = validate_run(run_root, read_splits(splits, pd.read_csv(train)), splits)
            write_json(run_root / 'membership_validation.json', {'status':'passed', 'rows':len(pred), 'fold':fold})
if '--smoke' in args:
    import subprocess
    common = ['--train', str(train), '--splits', str(splits)]
    subprocess.run([sys.executable, '-m', 'scripts.rsna_next.validate', '--roots', '/kaggle/working/next_run',
                    '--smoke', '--output', '/kaggle/working/next_run/validation', *common], check=True)
    cache_index = args.index('--cache-manifests')
    cache_paths = []
    for item in args[cache_index + 1:]:
        if item.startswith('--'):
            break
        cache_paths.append(item)
    subprocess.run([sys.executable, '-m', 'scripts.rsna_next.replay', '--run', '/kaggle/working/next_run',
                    '--cache-manifests', *cache_paths, '--weights', str(find(__WEIGHTS__)),
                    '--output', '/kaggle/working/next_run/replay.json', *common], check=True)
'''
    train_args = ["--mode", args.mode, "--fold", str(args.fold)] + (["--smoke"] if args.smoke else [])
    filename = {"fastmri": "p6_localization_backbone_best.safetensors", "imagenet": "model.safetensors", "dinov2": "pytorch_model.bin"}[cfg["initialization"]]
    cache = "args += ['--cache-manifests', str(find('p2_cache_manifest_part_a.json')), str(find('p2_cache_manifest_part_b.json'))]"
    if args.cache_sources:
        cache = "manifests = sorted(p for p in Path('/kaggle/input').glob('**/manifest.json') if json.loads(p.read_text()).get('cache_id', '').startswith('P14'))\nargs += ['--cache-manifests', *map(str, sorted(manifests, key=lambda p: json.loads(p.read_text())['part']))]"
    labels = "args += ['--student-targets', str(find('student_targets.csv'))]" if args.label_source else ""
    program = program.replace("__PAYLOAD__", repr(payload)).replace("__ARGS__", repr(train_args)).replace("__WEIGHTS__", repr(filename)).replace("__CACHE__", cache).replace("__LABELS__", labels)
    program = program.replace("__FOLDS__", repr(args.folds or [args.fold])).replace("__BANK__", repr(bool(args.folds))).replace("__MODE__", repr(args.mode))
    slug = f"rsna-knee-next-{args.mode}-{args.recipe}-{args.fold}" + (f"-{args.head}" if args.head else "")
    slug += "-smoke" if args.smoke else f"-s{args.seed}"
    if args.folds:
        slug = f"rsna-knee-teacher-bank-{''.join(map(str, args.folds))}-s{args.seed}"
    # Kaggle creates new notebook slugs from titles and rejects titles >50 chars.
    if len(slug) > 50:
        slug = slug.replace("rsna-knee-next-", "rsna-", 1)
    if len(slug) > 50:
        raise ValueError("Notebook title exceeds Kaggle's 50-character limit")
    sources = ["vallepuvamsikrishna/rsna-knee-report-weak-supervision"]
    sources += args.cache_sources or ["vallepuvamsikrishna/rsna-knee-p2-wide-cache-a", "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b"]
    if args.label_source:
        sources.append(args.label_source)
    dataset = {"fastmri": ["vallepuvamsikrishna/rsna-p6-fastmri-localization-init"],
               "imagenet": ["vallepuvamsikrishna/rsna-coatnet-rmlp2-imagenet-init"], "dinov2": []}[cfg["initialization"]]
    metadata = {"id": f"vallepuvamsikrishna/{slug}", "title": slug.replace("-", " ").title(),
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": dataset,
        "kernel_sources": sources, "model_sources": ["metaresearch/dinov2/pytorch/small/1"] if cfg["initialization"] == "dinov2" else []}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.py").write_text(program)
    write_json(args.output / "kernel-metadata.json", metadata)
    write_json(args.output / "build.json", {"recipe": cfg, "mode": args.mode, "fold": args.fold,
        "smoke": args.smoke, "code_sha256": sha256(args.output / "run.py"),
        "split_sha256": sha256(args.splits), "split_revision": split_contract["split_revision"],
        "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in payload.items()}})
    if args.folds:
        build = json.loads((args.output / "build.json").read_text())
        build["folds"] = args.folds
        write_json(args.output / "build.json", build)


def build_data(args):
    payload = {str(p): p.read_text() for p in Path("scripts/rsna_next").glob("*.py")}
    payload["scripts/__init__.py"] = ""
    if args.kind == "geometry":
        module = "geometry"
        extra = ["--part", str(args.part), "--parts", "4", "--size", str(args.size), "--axial-policy", args.axial_policy]
        setup = "args += ['--dicom-root', str(train.parent / 'train_series')]"
        slug = f"rsna-knee-p14-native-{args.size}-{args.axial_policy}-{args.part}"
        sources = []
    elif args.kind == "header-audit":
        module, extra = "header_audit", []
        setup = "args += ['--dicom-root', str(train.parent / 'train_series')]"
        slug, sources = "rsna-knee-p10-header-audit", []
    else:
        module, extra = "pixel_audit", []
        setup = "args += ['--cache-manifests', str(find('p2_cache_manifest_part_a.json')), str(find('p2_cache_manifest_part_b.json'))]"
        slug = "rsna-knee-p10-pixel-audit"
        sources = ["vallepuvamsikrishna/rsna-knee-p2-wide-cache-a", "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b"]
    program = '''import sys
from pathlib import Path
ROOT = Path('/kaggle/working/next_runtime')
PAYLOAD = __PAYLOAD__
for name, text in PAYLOAD.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
sys.path.insert(0, str(ROOT))
def find(name):
    paths = list(Path('/kaggle/input').glob('**/' + name))
    if len(paths) != 1:
        raise ValueError(f'Expected one {name}: found {len(paths)}')
    return paths[0]
train = find('train.csv')
args = ['--train', str(train), '--series', str(find('train_series.csv')), '--output', '/kaggle/working/result', *__EXTRA__]
__SETUP__
sys.argv = ['data', *args]
from scripts.rsna_next.__MODULE__ import main
main()
'''
    program = program.replace("__PAYLOAD__", repr(payload)).replace("__EXTRA__", repr(extra)).replace("__SETUP__", setup).replace("__MODULE__", module)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.py").write_text(program)
    write_json(args.output / "kernel-metadata.json", {"id": f"vallepuvamsikrishna/{slug}",
        "title": slug.replace("-", " ").title(), "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [], "kernel_sources": sources})
    write_json(args.output / "build.json", {"kind": args.kind, "code_sha256": sha256(args.output / "run.py"),
        "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in payload.items()}})


def build_inference(args):
    if not args.release:
        raise ValueError("Inference requires --release with frozen model groups and Kaggle input paths")
    release = json.loads(args.release.read_text())
    if release.get("schema_version") != "rsna-next-v1" or not release.get("groups"):
        raise ValueError("Invalid release spec")
    payload = {str(p): p.read_text() for p in Path("scripts/rsna_next").glob("*.py")}
    payload["scripts/__init__.py"] = ""
    payload["scripts/p2_submission.py"] = Path("scripts/p2_submission.py").read_text()
    payload["release.json"] = json.dumps(release)
    program = '''import time
RUN_STARTED = time.perf_counter()
import sys
from pathlib import Path
ROOT = Path('/kaggle/working/next_runtime')
PAYLOAD = __PAYLOAD__
for name, text in PAYLOAD.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.rsna_next.contracts import ID
def find(name):
    paths = list(Path('/kaggle/input').glob('**/' + name))
    if len(paths) != 1:
        raise ValueError(f'Expected one {name}, found {len(paths)}')
    return paths[0]
args = ['--release', str(ROOT / 'release.json'), '--output', '/kaggle/working/inference']
if __BENCHMARK__:
    train = find('train.csv')
    all_ids = pd.read_csv(train, usecols=[ID])
    series = pd.read_csv(find('train_series.csv'))
    counts = series.groupby(ID).agg(series_count=('SeriesInstanceUID', 'size'), plane_count=('Anatomical_Plane', 'nunique'), fluid_count=('Fluid_Sensitive', 'sum'))
    ordered = all_ids.join(counts, on=ID).sort_values(['plane_count', 'fluid_count', 'series_count', ID])
    import numpy as np
    ordered.iloc[np.round(np.linspace(0, len(ordered)-1, 240)).astype(int)][[ID]].to_csv(ROOT / 'benchmark.csv', index=False)
    args += ['--benchmark', '--studies', str(ROOT / 'benchmark.csv'), '--series', str(find('train_series.csv')), '--dicom-root', str(train.parent / 'train_series')]
else:
    test = find('test.csv')
    args += ['--studies', str(test), '--series', str(find('test_series.csv')), '--dicom-root', str(test.parent / 'test_series'), '--sample-submission', str(find('sample_submission.csv'))]
sys.argv = ['infer', *args]
from scripts.rsna_next.infer import main
main(RUN_STARTED)
if not __BENCHMARK__:
    (Path('/kaggle/working/inference') / 'submission.csv').replace('/kaggle/working/submission.csv')
'''
    program = program.replace("__PAYLOAD__", repr(payload)).replace("__BENCHMARK__", repr(args.benchmark))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.py").write_text(program)
    slug = "rsna-knee-next-release" + ("-benchmark" if args.benchmark else "-canary")
    write_json(args.output / "kernel-metadata.json", {"id": f"vallepuvamsikrishna/{slug}",
        "title": slug.replace("-", " ").title(), "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4", "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": release.get("dataset_sources", []), "kernel_sources": release.get("kernel_sources", []),
        "model_sources": release.get("model_sources", [])})
    write_json(args.output / "build.json", {"kind": "inference", "benchmark": args.benchmark,
        "release_sha256": sha256(args.release), "code_sha256": sha256(args.output / "run.py"),
        "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in payload.items()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["runtime", "training", "geometry", "pixel-audit", "header-audit", "inference"])
    p.add_argument("--recipe", default="fastmri-report")
    p.add_argument("--mode", choices=["teacher", "student", "refit"], default="teacher")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--folds", type=int, nargs="+", choices=range(5))
    p.add_argument("--seed", type=int, choices=[20260917, 20260918], default=20260917)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--splits", type=Path, default=Path("outputs/experiments/P10-contract-v2/splits.csv"))
    p.add_argument("--smoke-evidence", type=Path)
    p.add_argument("--replay-evidence", type=Path)
    p.add_argument("--winner-config", type=Path)
    p.add_argument("--refit-epochs", type=int)
    p.add_argument("--head", choices=["baseline", "position", "hierarchical"])
    p.add_argument("--cache-sources", nargs="+")
    p.add_argument("--label-source")
    p.add_argument("--part", type=int, choices=range(4), default=0)
    p.add_argument("--size", type=int, choices=[336, 384], default=336)
    p.add_argument("--axial-policy", choices=["longest", "fluid"], default="longest")
    p.add_argument("--release", type=Path)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-studies", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.kind == "runtime":
        build_runtime(args.output, args.benchmark_studies)
    elif args.kind == "training":
        build_training(args)
    elif args.kind == "inference":
        build_inference(args)
    else:
        build_data(args)
    compile((args.output / "run.py").read_text(), str(args.output / "run.py"), "exec")
    print(args.output)


if __name__ == "__main__":
    main()
