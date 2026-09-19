import copy

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.rsna_next.contracts import (ID, TARGETS, assert_budget, make_splits,
                                        paired_bootstrap, validate_splits)
from scripts.rsna_next.data import sample_windows
from scripts.rsna_next.geometry import canonical_basis, native_indices, orient_crop
from scripts.rsna_next.labels import validate_extraction
from scripts.rsna_next.models import AnatomicalHead
from scripts.rsna_next.screen import positions_to_metadata


def metadata():
    rows = []
    for i in range(100):
        rows.append({ID: f"study-{i}", "Report": f"unique report {i}",
                     **{t: i % 2 if i < 20 else np.nan for t in TARGETS}})
    rows[-1]["Report"] = rows[0]["Report"]
    return pd.DataFrame(rows)


def test_split_rejects_leakage_at_every_stage():
    s = make_splits(metadata())
    assert s.iloc[-1].role == "excluded_expert_twin"
    assert len(s.loc[s.is_gold]) == 20
    assert s.loc[s.role == "pseudo_train", [f"teacher_train_{i}" for i in range(5)]].sum(axis=1).eq(4).all()
    contaminated = s.copy()
    contaminated.loc[s.role == "pseudo_valid", "teacher_train_0"] = True
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(contaminated)
    contaminated = s.copy()
    contaminated.loc[s.expert_fold == 0, "student_train_0"] = True
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(contaminated)
    pd.testing.assert_frame_equal(s, make_splits(metadata()))


def test_report_groups_stay_together():
    m = metadata()
    m.loc[40:44, "Report"] = "same template"
    s = make_splits(m)
    assert s.loc[40:44, "role"].nunique() == 1
    assert s.loc[40:44, "teacher_fold"].nunique() == 1


@pytest.mark.parametrize("variant", ["baseline", "position", "hierarchical"])
def test_heads_mask_invariance_and_empty_study(variant):
    torch.manual_seed(7)
    model = AnatomicalHead(16, variant).eval()
    x = torch.randn(2, 7, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    slots = torch.tensor([[0, 1, 4, -1, -1, -1, -1]] * 2)
    pos = torch.zeros(2, 7)
    a = model(x, mask, slots, pos)
    changed = x.clone()
    changed[~mask] = 999
    b = model(changed, mask, slots, pos)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[1], model.target_bias)
    a.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_position_head_starts_as_exact_baseline():
    baseline = AnatomicalHead(16).eval()
    model = AnatomicalHead(16, "position").eval()
    model.load_adapted(baseline.state_dict())
    x, mask = torch.randn(2, 6, 16), torch.ones(2, 6, dtype=torch.bool)
    slots = torch.tensor([[0, 1, 2, 3, 4, 4]] * 2)
    pos = torch.rand(2, 6)
    torch.testing.assert_close(baseline(x, mask), model(x, mask, slots, pos), atol=0, rtol=0)


def test_slot_balancing_and_boundary_rejection():
    slots = np.array([0] * 20 + [1, 2, 3, 4])
    pick, valid = sample_windows(slots, 8, np.random.default_rng(0), True)
    assert set(slots[pick]) == set(range(5)) and valid.all()
    p, mask = sample_windows([], 8)
    assert not mask.any()
    with pytest.raises(ValueError):
        positions_to_metadata(np.array([[17]]), np.array([[True]]))
    sid, pos = positions_to_metadata(np.array([[1, 19, 33, 45, 53]]), np.ones((1, 5), bool))
    assert sid.tolist() == [[0, 1, 2, 3, 4]]
    assert (abs(pos) <= 1).all()


def test_native_triplets_are_native_neighbors():
    triple = native_indices(43, 12)
    assert triple.shape == (12, 3)
    assert (np.diff(triple, axis=1) == 1).all()
    assert triple.min() >= 0 and triple.max() < 43
    assert native_indices(2, 6).size == 0


def test_orientation_respects_patient_axes_and_rejects_wrong_plane():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    a = orient_crop(image, (1, 1), (1, 0, 0, 0, 1, 0), "Axial", 8, 0, 63, crop_mm=8)
    b = orient_crop(image[:, ::-1], (1, 1), (-1, 0, 0, 0, 1, 0), "Axial", 8, 0, 63, crop_mm=8)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="plane"):
        canonical_basis((1, 0, 0, 0, 1, 0), "Sagittal")


def test_evidence_validation_never_invents_negative():
    r = {"findings": {t: {"state": "unmentioned", "probability": .5, "evidence": "",
          "compartment": "", "severity": "", "negation": "", "uncertainty": ""} for t in TARGETS}}
    validate_extraction(r, "No effusion.")
    wrong = copy.deepcopy(r)
    wrong["findings"]["ACL"]["probability"] = 0
    with pytest.raises(ValueError):
        validate_extraction(wrong, "No effusion.")
    wrong = copy.deepcopy(r)
    wrong["findings"]["ACL"].update(state="present", evidence="ACL rupture")
    with pytest.raises(ValueError):
        validate_extraction(wrong, "No effusion.")


def test_bootstrap_identity_and_direction():
    y = np.tile(np.arange(40)[:, None] % 2, (1, 12))
    low = np.full(y.shape, .5)
    good = .1 + .8 * y
    same = paired_bootstrap(y, low, low, np.arange(40), repeats=200)
    assert same["ci95"] == [0., 0.]
    improved = paired_bootstrap(y, low, good, np.arange(40), repeats=200)
    assert improved["probability_positive"] == 1
    assert improved["ci95"] == [.5, .5]


def test_budget_accounts_for_reservations():
    assert assert_budget([{"actual_usd": 20, "reserved_usd": 30}], 100) == 350
    with pytest.raises(ValueError):
        assert_budget([{"actual_usd": 490}], 20)
    with pytest.raises(ValueError):
        assert_budget([], float("nan"))


@pytest.mark.parametrize("variant", ["baseline", "position", "hierarchical"])
def test_half_precision_all_missing_is_finite(variant):
    model = AnatomicalHead(16, variant).half().eval()
    output = model(torch.randn(1, 6, 16).half(), torch.zeros(1, 6, dtype=torch.bool),
                   torch.full((1, 6), -1), torch.zeros(1, 6).half())
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0], model.target_bias)


def test_promotion_requires_both_seeds_and_runtime():
    from scripts.rsna_next.validate import promotion
    y = np.tile(np.arange(40)[:, None] % 2, (1, 12))
    b, c = np.full(y.shape, .5), .1 + .8 * y
    good = promotion(y, [b, b], [c, c], np.arange(40), 7., repeats=200)
    assert good["passed"]
    one = promotion(y, [b, b], [c, b], np.arange(40), 7., repeats=200)
    assert not one["passed"] and not one["gates"]["both_seed_wins"]
    slow = promotion(y, [b, b], [c, c], np.arange(40), 7.51, repeats=200)
    assert not slow["passed"] and not slow["gates"]["runtime"]


def test_rank_blend_independent_arithmetic_and_ties():
    from scipy.stats import rankdata
    from scripts.rsna_next.infer import blend_predictions
    rng = np.random.default_rng(72)
    a, b = rng.integers(0, 4, (2, 58, 12)) / 4
    expected = .8 * rankdata(a, axis=0) / 58 + .2 * rankdata(b, axis=0) / 58
    np.testing.assert_allclose(blend_predictions([a, b], [.8, .2]), expected)
    with pytest.raises(ValueError):
        blend_predictions([a, b], [.8, .3])


def test_teacher_bank_rejects_membership_and_hash_corruption(tmp_path):
    import json
    from scripts.rsna_next.contracts import VERSION, sha256, write_json
    from scripts.rsna_next.labels import teacher_predictions
    splits = make_splits(metadata())
    path = tmp_path / "splits.csv"
    splits.to_csv(path, index=False)
    roots = []
    for fold in range(5):
        root = tmp_path / str(fold)
        root.mkdir()
        (root / "best.pt").write_bytes(b"synthetic hash fixture")
        mask = ((splits.role == "pseudo_train") & (splits.teacher_fold == fold)) | (splits.role == "pseudo_valid") | splits.is_gold
        pd.DataFrame({ID: splits.loc[mask, ID], **{t: .25 for t in TARGETS}}).to_csv(root / "predictions.csv", index=False)
        pd.DataFrame({"epoch": [1, 2], "train_bce": [.6, .5], "original_report_valid_bce": [.6, .5]}).to_csv(root / "history.csv", index=False)
        write_json(root / "manifest.json", {"schema_version": VERSION, "mode": "teacher", "status": "trained_pending_independent_validation",
            "fold": fold, "config": {"labels": "report"}, "split_sha256": sha256(path),
            "train_ids": splits.loc[splits[f"teacher_train_{fold}"], ID].tolist(),
            "validation_ids": splits.loc[splits.role == "pseudo_valid", ID].tolist(), "best_epoch": 2,
            "checkpoint_sha256": sha256(root / "best.pt"), "predictions_sha256": sha256(root / "predictions.csv"),
            "original_targets_sha256": "fixed", "cache_signature": "fixed", "initialization_sha256": "fixed"})
        roots.append(root)
    values, provenance = teacher_predictions(roots, splits, path)
    np.testing.assert_allclose(values[splits.role == "pseudo_train"], .25)
    assert len(provenance) == 5
    manifest = json.loads((roots[0] / "manifest.json").read_text())
    manifest["train_ids"].append(splits.loc[splits.role == "pseudo_valid", ID].iloc[0])
    write_json(roots[0] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="membership"):
        teacher_predictions(roots, splits, path)
    manifest["train_ids"].pop()
    write_json(roots[0] / "manifest.json", manifest)
    (roots[0] / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        teacher_predictions(roots, splits, path)


def test_compressed_native_dicom_and_deployment_cache_agreement(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, RLELossless, generate_uid
    from scripts.rsna_next.geometry import build_study
    from scripts.rsna_next.infer import dicom_windows
    uid, sid = generate_uid(), generate_uid()
    folder = tmp_path / uid / sid
    folder.mkdir(parents=True)
    for i in range(5):
        fm = FileMetaDataset()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.MediaStorageSOPClassUID = MRImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(folder / f"{i}.dcm"), {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, fm.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = uid, sid
        ds.Rows, ds.Columns = 32, 32
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelSpacing = [4., 4.]
        ds.ImageOrientationPatient = [1., 0., 0., 0., -1., 0.]
        ds.ImagePositionPatient = [0., 0., float(i * 3)]
        ds.PixelData = (np.arange(1024, dtype=np.uint16).reshape(32, 32) + 10 * i).tobytes()
        ds.compress(RLELossless)
        ds.save_as(folder / f"{i}.dcm", enforce_file_format=True)
    records = [{"SeriesInstanceUID": sid, "Anatomical_Plane": "Axial", "Fluid_Sensitive": 1, "Fat_Suppression": 1}]
    x, mask, slots, pos, provenance, errors = build_study(uid, records, tmp_path, 32)
    assert not errors and mask.sum() == 6
    assert all(np.diff(r["slice_positions_mm"]).min() > 0 for r in provenance)
    pick, valid = sample_windows(slots, 32)
    deployed = dicom_windows(uid, records, tmp_path, {"kind": "p14", "size": 32, "axial_policy": "longest"})
    for actual, expected in zip(deployed, (x[pick], valid, slots[pick], pos[pick])):
        np.testing.assert_array_equal(actual, expected)


def test_kernel_builder_is_offline_and_full_jobs_require_evidence(tmp_path):
    import json
    import subprocess
    import sys
    from scripts.rsna_next.contracts import sha256, write_json
    split = tmp_path / "splits.csv"
    make_splits(metadata(), []).to_csv(split, index=False)
    write_json(split.with_suffix(".groups.json"), {"split_sha256": sha256(split), "image_pairs": [],
        "split_revision": "test", "full_training_allowed": False})
    command = [sys.executable, "scripts/build_next_kernels.py", "training", "--output", str(tmp_path), "--splits", str(split)]
    blocked = subprocess.run(command, text=True, capture_output=True)
    assert blocked.returncode != 0 and "Full runs require --smoke-evidence" in blocked.stderr
    built = subprocess.run([*command, "--smoke"], text=True, capture_output=True)
    assert built.returncode == 0, built.stderr
    kernel_metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert kernel_metadata["is_private"] and not kernel_metadata["enable_internet"]
    compile((tmp_path / "run.py").read_text(), "run.py", "exec")
    # Regression: the full distilled title previously caused HTTP 400 on Kaggle.
    write_json(split.with_suffix('.groups.json'), {'split_sha256': sha256(split),
        'image_pairs': [], 'split_revision': 'test', 'full_training_allowed': True})
    smoke, replay = tmp_path / 'smoke.json', tmp_path / 'replay.json'
    write_json(smoke, {'status': 'validated_smoke', 'gates': {'test': True},
        'manifest_sha256': 'fixture', 'config': {'initialization': 'fastmri',
        'head': 'baseline', 'train_windows': 8, 'checkpointing': False}})
    write_json(replay, {'status': 'replay_validated', 'run_manifest_sha256': 'fixture'})
    distilled = subprocess.run([*command, '--recipe', 'fastmri-distilled', '--mode', 'student',
        '--seed', '20260917', '--label-source', 'fixture/labels', '--smoke-evidence', str(smoke),
        '--replay-evidence', str(replay), '--replay'], text=True, capture_output=True)
    assert distilled.returncode == 0, distilled.stderr
    meta = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert len(meta['title']) <= 50
    assert meta['id'].split('/')[1] == meta['title'].lower().replace(' ', '-')
    assert meta['id'].endswith('fastmri-distilled-0-s20260917')
    source = (tmp_path / 'run.py').read_text()
    assert "if '--smoke' in args or True:" in source
    compile(source, 'run.py', 'exec')


def test_confirmation_read_retry(monkeypatch):
    from scripts.rsna_next import confirm_duration as queue
    calls = []
    def flaky(args, timeout):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('Temporary failure in name resolution')
        return 'COMPLETE'
    monkeypatch.setattr(queue, 'command', flaky)
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    assert queue.read_command(['status'], 120) == 'COMPLETE'
    assert len(calls) == 2
    monkeypatch.setattr(queue, 'command', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Permission denied')))
    with pytest.raises(RuntimeError, match='Permission denied'):
        queue.read_command(['status'], 120)


def test_confirmation_rejects_partial_replay(tmp_path):
    from scripts.rsna_next.confirm_duration import verify_replay
    from scripts.rsna_next.contracts import sha256, write_json
    manifest = {'checkpoint_sha256': 'checkpoint', 'predictions_sha256': 'predictions'}
    write_json(tmp_path / 'manifest.json', manifest)
    replay = {'status': 'replay_validated', 'rows': 12, 'maximum_absolute_error': 0,
              'run_manifest_sha256': sha256(tmp_path / 'manifest.json'), **manifest}
    write_json(tmp_path / 'replay.json', replay)
    verify_replay(tmp_path, manifest, 12)
    replay['rows'] = 11
    write_json(tmp_path / 'replay.json', replay)
    with pytest.raises(ValueError, match='Full replay failed'):
        verify_replay(tmp_path, manifest, 12)


def test_image_duplicate_closure_preserves_validation_and_blocks_leakage(tmp_path):
    from scripts.rsna_next.contracts import ID, exclusion_groups, read_splits, sha256, write_json
    m = metadata()
    original = make_splits(m)
    valid_id = original.loc[original.role == "pseudo_valid", ID].iloc[0]
    train_id = original.loc[original.role == "pseudo_train", ID].iloc[0]
    pairs = [[valid_id, train_id]]
    repaired = make_splits(m, pairs)
    pd.testing.assert_series_equal(repaired.expert_fold, original.expert_fold)
    assert repaired.loc[repaired[ID] == train_id, "role"].item() == "excluded_validation_twin"
    assert set(repaired.loc[repaired.role == "pseudo_valid", ID]) == set(original.loc[original.role == "pseudo_valid", ID])
    assert exclusion_groups(repaired)[repaired[ID].isin([valid_id, train_id])].nunique() == 1
    contaminated = repaired.copy()
    contaminated.loc[contaminated[ID] == train_id, "teacher_train_0"] = True
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(contaminated)
    path = tmp_path / "splits.csv"
    repaired.to_csv(path, index=False)
    write_json(path.with_suffix(".groups.json"), {"image_pairs": pairs, "split_sha256": sha256(path)})
    pd.testing.assert_frame_equal(read_splits(path, m), repaired, check_dtype=False)
