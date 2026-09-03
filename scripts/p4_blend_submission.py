#!/usr/bin/env python3
"""Create the promoted equal-rank P2+P4 submission on Kaggle.

The deployment builder prepends the proven P2 preprocessing/model definitions to
this file. Local execution imports those definitions directly from p2_submission.
"""

# ruff: noqa: E402 - the deployment builder supplies common definitions in place.

from concurrent.futures import ThreadPoolExecutor

# BEGIN COMMON IMPORTS
from p2_submission import (
    CACHE_SIZE,
    DataLoader,
    EXPECTED_CACHE_SIGNATURE,
    ID,
    K_EVAL,
    MODEL_SIZE,
    SLOT_BOUNDS,
    SLOT_SPECS,
    TARGETS,
    CoAtNetMIL,
    TargetAttentionHead,
    TestStudyDataset,
    find_competition_root,
    find_p2_root,
    gpu_snapshot,
    json,
    np,
    pd,
    platform,
    pydicom,
    sha256_file,
    time,
    timm,
    torch,
    Path,
)
# END COMMON IMPORTS


RUN_ID = "P4-BLEND-SUBMISSION-v1"
EXPECTED_P2_MANIFEST_SHA256 = (
    "ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508"
)
EXPECTED_P4_MANIFEST_SHA256 = {
    0: "6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932",
    1: "9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5",
    2: "93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac",
    3: "9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c",
    4: "e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228",
}
EXPECTED_P4_CHECKPOINT_SHA256 = {
    0: "340576ef981922173e056ced97c98f34529bcaaeec4831ea41c6c308638782ee",
    1: "bd11c89940f93cf819f3244387ffe16b49216180d5bc529431b13daa078bd0d9",
    2: "49340b64c4fd1dee9253e5712ae776b14d3805ddec73b4195b0b55caca38fae5",
    3: "16bac1a05b81fba82e2fa1e2f49b4a660de0e616662894f0a63707bdd1b89bd7",
    4: "596b267f51f52bfe489034dadaa7759342e271825834def1f4aab7a05ed069b5",
}
P2_OOF_MACRO_AUC = 0.8911253543643616
P4_OOF_MACRO_AUC = 0.89116089660126
BLEND_OOF_MACRO_AUC = 0.8958210633119559
ON_KAGGLE = Path("/kaggle/working").is_dir()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(
    "outputs/p4_blend_submission_smoke"
)


def find_p4_roots() -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for fold in range(5):
        direct = [
            Path(
                "/kaggle/input/notebooks/vallepuvamsikrishna/"
                f"rsna-knee-p4-coatnet-e2e-fold-{fold}"
            ),
            Path("outputs/kaggle/rsna-p4-coatnet-e2e-v1") / f"fold-{fold}",
        ]
        selected = next(
            (
                candidate.resolve()
                for candidate in direct
                if (candidate / f"p4_fold_{fold}_manifest.json").is_file()
            ),
            None,
        )
        if selected is None and Path("/kaggle/input").is_dir():
            matches = list(
                Path("/kaggle/input").glob(
                    f"**/p4_fold_{fold}_manifest.json"
                )
            )
            if len(matches) == 1:
                selected = matches[0].parent.resolve()
        if selected is None:
            raise FileNotFoundError(f"validated P4 fold {fold} output")
        roots[fold] = selected
    return roots


def rank_columns(predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            pd.Series(predictions[:, index]).rank(
                method="average", pct=True
            ).to_numpy(np.float64)
            for index in range(predictions.shape[1])
        ]
    )


def load_full_model(checkpoint: Path, device: torch.device) -> CoAtNetMIL:
    channels_last = device.type == "cuda"
    model = CoAtNetMIL(channels_last=channels_last)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model.eval()
    return model


def load_p2_heads(
    p2_root: Path, device: torch.device
) -> list[TargetAttentionHead]:
    heads: list[TargetAttentionHead] = []
    for fold in range(5):
        head = TargetAttentionHead().to(device)
        state = torch.load(
            p2_root / f"p2_fold_{fold}_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        head.load_state_dict(state, strict=True)
        head.eval()
        heads.append(head)
        del state
    return heads


def infer_device_group(
    device: torch.device,
    windows_cpu: torch.Tensor,
    mask_cpu: torch.Tensor,
    p2_model: CoAtNetMIL | None,
    p2_heads: list[TargetAttentionHead],
    p4_models: list[tuple[int, CoAtNetMIL]],
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    windows = windows_cpu.to(device, non_blocking=device.type == "cuda")
    mask = mask_cpu.to(device, non_blocking=device.type == "cuda").bool()
    p2_result: np.ndarray | None = None
    p4_result: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        if p2_model is not None:
            features = p2_model.encode(windows)
            probabilities = torch.stack(
                [torch.sigmoid(head(features, mask)) for head in p2_heads]
            ).mean(dim=0)
            p2_result = probabilities.float().cpu().numpy()
            del features, probabilities
        for fold, model in p4_models:
            features = model.encode(windows)
            probabilities = torch.sigmoid(model.head(features, mask))
            p4_result[fold] = probabilities.float().cpu().numpy()
            del features, probabilities
    return p2_result, p4_result


def prediction_frame(study_ids: list[str], values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID: study_ids,
            **{
                target: values[:, target_index]
                for target_index, target in enumerate(TARGETS)
            },
        }
    )


def main() -> int:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not ON_KAGGLE:
        raise RuntimeError(
            "Full six-checkpoint inference is intentionally Kaggle-only; "
            "the local host does not have a safe model-loading memory margin."
        )

    p2_root = find_p2_root()
    p2_manifest_path = p2_root / "p2_manifest.json"
    assert sha256_file(p2_manifest_path) == EXPECTED_P2_MANIFEST_SHA256
    p2_manifest = json.loads(p2_manifest_path.read_text())
    assert p2_manifest["run_id"] == "P2-COATNET-v1"
    assert p2_manifest["best_epoch"] == 8
    assert p2_manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
    p2_names = [
        "p2_coatnet_pseudo_best.pt",
        *[f"p2_fold_{fold}_head.pt" for fold in range(5)],
    ]
    for name in p2_names:
        path = p2_root / name
        assert path.is_file()
        assert sha256_file(path) == p2_manifest["artifacts_sha256"][name]

    p4_roots = find_p4_roots()
    p4_manifests: dict[int, dict[str, object]] = {}
    for fold, root in p4_roots.items():
        manifest_path = root / f"p4_fold_{fold}_manifest.json"
        assert sha256_file(manifest_path) == EXPECTED_P4_MANIFEST_SHA256[fold]
        manifest = json.loads(manifest_path.read_text())
        checkpoint_name = f"p4_fold_{fold}_best.pt"
        checkpoint = root / checkpoint_name
        assert manifest["run_id"] == f"P4-COATNET-E2E-FOLD-{fold}-v1"
        assert manifest["fold"] == fold and manifest["best_epoch"] == 8
        assert manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE
        assert manifest["expert_valid_gradient_rows"] == 0
        assert manifest["expert_valid_report_overlap"] == 0
        assert manifest["selection_report_overlap"] == 0
        assert checkpoint.is_file()
        assert sha256_file(checkpoint) == EXPECTED_P4_CHECKPOINT_SHA256[fold]
        assert (
            manifest["artifacts_sha256"][checkpoint_name]
            == EXPECTED_P4_CHECKPOINT_SHA256[fold]
        )
        p4_manifests[fold] = manifest

    if ON_KAGGLE:
        data_root = find_competition_root()
        dicom_root = data_root / "test_series"
        test = pd.read_csv(data_root / "test.csv", dtype={ID: str})
        series = pd.read_csv(data_root / "test_series.csv", dtype={ID: str})
        sample = pd.read_csv(
            data_root / "sample_submission.csv", dtype={ID: str}
        )
        workers = 2
    else:
        data_root = Path("data/raw").resolve()
        dicom_root = Path("data/sample/files/train_series").resolve()
        metadata = pd.read_csv(data_root / "train.csv", dtype={ID: str})
        series_all = pd.read_csv(
            data_root / "train_series.csv", dtype={ID: str}
        )
        available = {
            path.name for path in dicom_root.iterdir() if path.is_dir()
        }
        local_ids = [uid for uid in metadata[ID] if uid in available]
        assert local_ids
        richest = max(
            local_ids,
            key=lambda uid: sum(
                1 for _ in (dicom_root / uid).glob("*/*.dcm")
            ),
        )
        test = pd.DataFrame({ID: [richest]})
        series = series_all[series_all[ID].isin(test[ID])].copy()
        sample = pd.DataFrame(
            {ID: test[ID], **{target: 0.5 for target in TARGETS}}
        )
        workers = 0

    assert dicom_root.is_dir()
    assert test[ID].is_unique and len(test) > 0
    assert sample.columns.tolist() == [ID, *TARGETS]
    assert sample[ID].tolist() == test[ID].tolist()
    assert set(series[ID].unique()) == set(test[ID])
    study_ids = test[ID].astype(str).tolist()
    series_by_study = {
        str(uid): group.to_dict("records")
        for uid, group in series.groupby(ID, sort=False)
    }

    visible_gpus = torch.cuda.device_count()
    devices = (
        [torch.device("cuda:0"), torch.device("cuda:1")]
        if visible_gpus >= 2
        else [torch.device("cuda:0" if visible_gpus else "cpu")]
    )
    for device in devices:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    print(
        {
            "P4_BLEND_SUBMISSION_START": True,
            "on_kaggle": ON_KAGGLE,
            "studies": len(study_ids),
            "series": len(series),
            "p2_manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
            "p4_manifest_sha256": EXPECTED_P4_MANIFEST_SHA256,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "visible_gpus": visible_gpus,
        },
        flush=True,
    )

    model_load_started = time.time()
    p2_device = devices[0]
    p2_model = load_full_model(
        p2_root / "p2_coatnet_pseudo_best.pt", p2_device
    )
    p2_heads = load_p2_heads(p2_root, p2_device)
    if len(devices) == 2:
        assignments = {0: [0, 2], 1: [1, 3, 4]}
    else:
        assignments = {0: list(range(5))}
    p4_by_device: dict[int, list[tuple[int, CoAtNetMIL]]] = {}
    for device_index, folds in assignments.items():
        device = devices[device_index]
        p4_by_device[device_index] = [
            (
                fold,
                load_full_model(
                    p4_roots[fold] / f"p4_fold_{fold}_best.pt", device
                ),
            )
            for fold in folds
        ]
    print(
        {
            "P4_BLEND_MODELS_READY": True,
            "execution_mode": "resident dual-device inference",
            "device_assignments": {
                str(devices[index]): [
                    *(["P2"] if index == 0 else []),
                    *[f"P4-fold-{fold}" for fold in assignments[index]],
                ]
                for index in assignments
            },
            "elapsed_minutes": (time.time() - model_load_started) / 60.0,
            "gpu": gpu_snapshot(),
        },
        flush=True,
    )

    batch_size = 2 if visible_gpus else 1
    dataset = TestStudyDataset(study_ids, series_by_study, dicom_root)
    loader_options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(visible_gpus),
    }
    if workers:
        loader_options.update(
            {
                "prefetch_factor": 2,
                "persistent_workers": True,
                "multiprocessing_context": "spawn",
            }
        )
    loader = DataLoader(dataset, **loader_options)

    p2_predictions = np.zeros((len(test), len(TARGETS)), dtype=np.float32)
    p4_fold_predictions = np.zeros(
        (5, len(test), len(TARGETS)), dtype=np.float32
    )
    feature_mask = np.zeros((len(test), K_EVAL), dtype=np.uint8)
    positions = np.full((len(test), K_EVAL), -1, dtype=np.int16)
    selection_rows: list[dict[str, object]] = []
    inference_started = time.time()
    executor = ThreadPoolExecutor(max_workers=len(devices))
    try:
        processed = 0
        for batch in loader:
            indices, windows, window_mask, centers, selections_json = batch
            batch_indices = indices.numpy().astype(np.int64)
            futures = []
            for device_index, device in enumerate(devices):
                futures.append(
                    executor.submit(
                        infer_device_group,
                        device,
                        windows,
                        window_mask,
                        p2_model if device_index == 0 else None,
                        p2_heads if device_index == 0 else [],
                        p4_by_device[device_index],
                    )
                )
            batch_results = [future.result() for future in futures]
            for p2_batch, p4_batches in batch_results:
                if p2_batch is not None:
                    p2_predictions[batch_indices] = p2_batch
                for fold, values in p4_batches.items():
                    p4_fold_predictions[fold, batch_indices] = values

            masks = window_mask.numpy().astype(bool)
            center_values = centers.numpy()
            feature_mask[batch_indices] = masks.astype(np.uint8)
            for batch_offset, index in enumerate(batch_indices):
                positions[index, masks[batch_offset]] = center_values[
                    batch_offset, masks[batch_offset]
                ]
                for selection in json.loads(selections_json[batch_offset]):
                    selection_rows.append(
                        {ID: study_ids[index], **selection}
                    )
            processed += len(batch_indices)
            if processed % 50 == 0 or processed == len(test):
                print(
                    {
                        "P4_BLEND_SUBMISSION_PROGRESS": (
                            f"{processed}/{len(test)}"
                        ),
                        "elapsed_minutes": (
                            time.time() - inference_started
                        ) / 60.0,
                        "gpu": gpu_snapshot(),
                    },
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True)

    assert np.isfinite(p2_predictions).all()
    assert np.isfinite(p4_fold_predictions).all()
    assert feature_mask.any(axis=1).all()
    assert (positions[feature_mask.astype(bool)] >= 1).all()
    p4_predictions = p4_fold_predictions.mean(axis=0)
    blend_predictions = np.clip(
        (rank_columns(p2_predictions) + rank_columns(p4_predictions)) / 2.0,
        1e-6,
        1.0 - 1e-6,
    )
    assert blend_predictions.shape == (len(test), len(TARGETS))
    assert np.isfinite(blend_predictions).all()
    assert ((blend_predictions > 0) & (blend_predictions < 1)).all()
    if len(test) > 1:
        assert (np.ptp(p2_predictions, axis=0) > 0).all()
        assert (np.ptp(p4_predictions, axis=0) > 0).all()

    submission = sample.copy()
    submission.loc[:, TARGETS] = blend_predictions
    assert submission.columns.tolist() == [ID, *TARGETS]
    assert submission[ID].tolist() == study_ids
    submission_path = OUTPUT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    prediction_frame(study_ids, p2_predictions).to_csv(
        OUTPUT_ROOT / "p4_blend_p2_predictions.csv", index=False
    )
    prediction_frame(study_ids, p4_predictions).to_csv(
        OUTPUT_ROOT / "p4_blend_p4_predictions.csv", index=False
    )
    np.save(
        OUTPUT_ROOT / "p4_blend_p4_predictions_by_fold.npy",
        p4_fold_predictions,
    )
    np.save(OUTPUT_ROOT / "p4_blend_feature_mask.npy", feature_mask)
    np.save(OUTPUT_ROOT / "p4_blend_feature_positions.npy", positions)
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_ROOT / "p4_blend_test_selections.csv", index=False
    )

    artifacts = [
        "submission.csv",
        "p4_blend_p2_predictions.csv",
        "p4_blend_p4_predictions.csv",
        "p4_blend_p4_predictions_by_fold.npy",
        "p4_blend_feature_mask.npy",
        "p4_blend_feature_positions.npy",
        "p4_blend_test_selections.csv",
    ]
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "studies": len(test),
        "series": len(series),
        "targets": TARGETS,
        "source_p2": {
            "run_id": p2_manifest["run_id"],
            "manifest_sha256": EXPECTED_P2_MANIFEST_SHA256,
            "checkpoint_sha256": p2_manifest["artifacts_sha256"][
                "p2_coatnet_pseudo_best.pt"
            ],
            "oof_macro_auc": P2_OOF_MACRO_AUC,
        },
        "source_p4": {
            str(fold): {
                "run_id": p4_manifests[fold]["run_id"],
                "manifest_sha256": EXPECTED_P4_MANIFEST_SHA256[fold],
                "checkpoint_sha256": EXPECTED_P4_CHECKPOINT_SHA256[fold],
            }
            for fold in range(5)
        },
        "oof": {
            "p4_macro_auc": P4_OOF_MACRO_AUC,
            "equal_rank_p2_p4_macro_auc": BLEND_OOF_MACRO_AUC,
        },
        "geometry": {
            "cache_signature_sha256": EXPECTED_CACHE_SIGNATURE,
            "slot_specs": SLOT_SPECS,
            "slot_bounds": SLOT_BOUNDS,
            "relative_slice_range": [0.02, 0.98],
            "physical_crop_mm": 140.0,
            "cache_size": CACHE_SIZE,
            "model_size": MODEL_SIZE,
            "k_eval": K_EVAL,
        },
        "ensemble": {
            "p2": "mean probabilities from five report-purged P2 heads",
            "p4": "mean probabilities from five end-to-end P4 folds",
            "final": (
                "equal-rank mean of P2 and P4 by target, clipped to "
                "[1e-6, 1-1e-6]"
            ),
            "weights": {"p2": 0.5, "p4": 0.5},
        },
        "submission_file": "submission.csv",
        "test_metadata_sha256": {
            "test.csv": sha256_file(
                data_root / ("test.csv" if ON_KAGGLE else "train.csv")
            ),
            "test_series.csv": sha256_file(
                data_root
                / ("test_series.csv" if ON_KAGGLE else "train_series.csv")
            ),
        },
        "artifacts_sha256": {
            name: sha256_file(OUTPUT_ROOT / name) for name in artifacts
        },
        "elapsed": {
            "model_load_minutes": (
                inference_started - model_load_started
            ) / 60.0,
            "dicom_and_six_model_inference_minutes": (
                time.time() - inference_started
            ) / 60.0,
            "total_minutes": (time.time() - started) / 60.0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "pydicom": pydicom.__version__,
            "visible_gpus": visible_gpus,
            "devices": [str(device) for device in devices],
            "batch_size": batch_size,
            "loader_workers": workers,
            "device_assignments": {
                str(devices[index]): [
                    *(["P2"] if index == 0 else []),
                    *[f"P4-fold-{fold}" for fold in assignments[index]],
                ]
                for index in assignments
            },
        },
    }
    manifest_path = OUTPUT_ROOT / "p4_blend_submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        {
            "P4_BLEND_SUBMISSION_COMPLETE": True,
            "studies": len(submission),
            "submission_sha256": sha256_file(submission_path),
            "manifest_sha256": sha256_file(manifest_path),
            "prediction_ranges": {
                target: [
                    float(submission[target].min()),
                    float(submission[target].max()),
                ]
                for target in TARGETS
            },
            "elapsed_minutes": manifest["elapsed"]["total_minutes"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
