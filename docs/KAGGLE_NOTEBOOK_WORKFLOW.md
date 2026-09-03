# Kaggle notebook workflow

## Current remote notebooks

| Purpose | Kaggle notebook | Accelerator | State |
|---|---|---|---|
| Full metadata/DICOM audit | [rsna-knee-abnormality-detection-data-audit](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-abnormality-detection-data-audit) | CPU | Version 2 verified complete |
| Report weak supervision | [rsna-knee-report-weak-supervision](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-report-weak-supervision) | CPU | Version 1 verified complete; outputs downloaded and byte-checked |
| 2.5D image baseline | [rsna-knee-2-5d-radimagenet-baseline](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-2-5d-radimagenet-baseline) | Nvidia Tesla T4 | Version 3 verified complete; 4,407/4,407 usable, macro OOF AUC 0.6943 |
| DINOv2 pseudo adaptation | [rsna-knee-dinov2-pseudo-adaptation](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-dinov2-pseudo-adaptation) | Nvidia Tesla T4 | Version 1 verified complete; 4,407/4,407 usable, direct expert AUC 0.7852, fold-head OOF AUC 0.7976 |
| P2 wide cache A | [rsna-knee-p2-wide-cache-a](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p2-wide-cache-a) | CPU | Version 2 complete; 2,204/2,204, failures=0, 95.85 min total |
| P2 wide cache B | [rsna-knee-p2-wide-cache-b](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p2-wide-cache-b) | CPU | Version 2 complete; 2,203/2,203, failures=0, 77.04 min total |
| P2 CoAtNet-RMLP2 | [rsna-knee-p2-coatnet-rmlp2](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p2-coatnet-rmlp2) | GPU T4 x2 | Version 2 verified complete; OOF 0.891125, direct 0.878427, epoch 8 selected, both GPUs validated |
| P2 DDP speed smoke | [rsna-knee-p2-ddp-speed-smoke](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p2-ddp-speed-smoke) | GPU T4 x2 | V3 passed: both GPUs 100%, 8,084.9 MiB peak/rank, 1.2021 s/global step including warm-up; V4 infrastructure-stalled and was bounded/cancelled |
| P2 submission | [rsna-knee-p2-submission](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p2-submission) | GPU T4 | Canary v1 validated; canonical submission `55965846` complete with public score 0.908; duplicate `55965923` matched exactly |
| P3 speed smoke | [rsna-knee-p3-dinov2-ddp-speed-smoke](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p3-dinov2-ddp-speed-smoke) | GPU T4 x2 | V2 independently passed: 93%/99% GPU maxima, 481.2 MiB per-rank peak, 36.73 min full-path projection |
| P3 DINOv2-HMIL | [rsna-knee-p3-dinov2-hierarchical-mil](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p3-dinov2-hierarchical-mil) | GPU T4 x2 | Version 1 independently verified complete in 30.89 min; OOF 0.805988, fixed P2+P3 rank blend 0.863908; rejected versus P2 |
| P4 fold-0 speed smoke | [rsna-knee-p4-coatnet-e2e-speed-smoke](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-speed-smoke) | GPU T4 x2 | Version 1 independently passed; exact mixed-label fold path, 12 held-out predictions, both GPUs 100% peak |
| P4 CoAtNet E2E fold 0 | [rsna-knee-p4-coatnet-e2e-fold-0](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-0) | GPU T4 x2 | Version 1 complete and independently validated; epoch 8, BCE 0.463703, 130.59 min representation |
| P4 CoAtNet E2E fold 1 | [rsna-knee-p4-coatnet-e2e-fold-1](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-1) | GPU T4 x2 | Version 1 complete and independently validated; epoch 8, BCE 0.462888, 140.58 min representation |
| P4 CoAtNet E2E fold 2 | [rsna-knee-p4-coatnet-e2e-fold-2](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-2) | GPU T4 x2 | Version 1 complete and independently validated; epoch 8, BCE 0.466997, 129.61 min representation |
| P4 CoAtNet E2E fold 3 | [rsna-knee-p4-coatnet-e2e-fold-3](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-3) | GPU T4 x2 | Version 1 complete and independently validated; epoch 8, BCE 0.461624, 140.98 min representation |
| P4 CoAtNet E2E fold 4 | [rsna-knee-p4-coatnet-e2e-fold-4](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p4-coatnet-e2e-fold-4) | GPU T4 x2 | Version 1 complete and independently validated; epoch 8, BCE 0.464586, 126.86 min representation |

All notebooks are private with internet disabled. The verified competition mount is
`/kaggle/input/competitions/rsna-knee-abnormality-detection`.

The successful remote run loaded all 4,407 training studies and 24,371 series,
confirmed 58 fully labeled and 4,349 unlabeled studies, and decoded all six
selected complete MRI series. The downloaded remote audit is under
`outputs/kaggle/rsna-data-audit/`.

The verified report-teacher run generated fixed-consensus soft targets for all
4,407 studies. Its remote `training_targets.csv` exactly matches the locally
verified artifact (SHA-256
`ca63cfff863c87286e07a8059fee88f86ff8b3f4b640a98078d62113f6b93f2a`).

The verified DINOv2 run cached all 4,407 studies in 128.1 minutes with zero
failures and 18 valid windows per study, then completed all three pseudo-only
backbone-adaptation epochs and five fold-safe head fits. The downloaded model,
feature cache, masks, predictions, metrics, and teacher targets passed the local
integrity checker. The untouched-expert direct macro AUC is **0.7852** and the
five-fold head OOF macro AUC is **0.7976**.

The 116.2 MiB adapted token cache is retained in the downloaded outputs and can be
reused for head-only experiments without decoding DICOM again. The 11.12 GiB pixel
cache was temporary and is not in the notebook outputs, so changes to image geometry
or backbone adaptation currently require rebuilding it. P2 therefore creates
`P2-CACHE-v1` as two immutable, version-pinned private Kaggle kernel outputs. This
keeps the roughly 32 GB cache entirely on Kaggle and avoids downloading and re-uploading
it locally. Every consumer validates the two manifests, row ranges, study-order hash,
cache-signature hash, and array hashes before training.

## Local and deployment artifacts

- Executed local notebook: `output/jupyter-notebook/rsna-data-exploration.ipynb`
- Clean Kaggle source: `kaggle/rsna-data-audit/rsna-data-audit.ipynb`
- Kaggle metadata: `kaggle/rsna-data-audit/kernel-metadata.json`
- Reproducible builder: `scripts/build_kaggle_notebook.py`

- Executed local teacher notebook: `output/jupyter-notebook/rsna-report-weak-supervision.ipynb`
- Clean teacher source and metadata: `kaggle/rsna-report-weak-supervision/`
- Teacher builder: `scripts/build_report_weak_supervision_notebook.py`
- Local public-label audit helper: `scripts/audit_public_weak_labels.py`

- Executed local image smoke notebook: `output/jupyter-notebook/rsna-2p5d-gpu-baseline.ipynb`
- Clean GPU source and metadata: `kaggle/rsna-2p5d-gpu-baseline/`
- GPU notebook builder: `scripts/build_gpu_baseline_notebook.py`

- Executed local DINO smoke notebook: `output/jupyter-notebook/rsna-dinov2-pseudo-adaptation.ipynb`
- Clean DINO source and metadata: `kaggle/rsna-dinov2-pseudo-adaptation/`
- DINO notebook builder: `scripts/build_dino_adaptation_notebook.py`
- Downloaded verified DINO outputs: `outputs/kaggle/rsna-dinov2-pseudo-adaptation/`
- Full-output integrity checker: `scripts/validate_gpu_baseline_outputs.py`
- DINO full-output integrity checker: `scripts/validate_dino_adaptation_outputs.py`
- Reconnect-safe Kaggle log monitor: `scripts/follow_kaggle_milestones.py`

- P2 cache builders and sources: `scripts/build_p2_cache_notebook.py`,
  `kaggle/rsna-p2-wide-cache-a/`, and `kaggle/rsna-p2-wide-cache-b/`
- P2 training builder and source: `scripts/build_p2_coatnet_notebook.py` and
  `kaggle/rsna-p2-coatnet-rmlp2/`
- P2 canonical two-process runner: `scripts/p2_coatnet_ddp.py`; the historical
  builder name now materializes a Kaggle script rather than an `.ipynb`
- P2 speed-smoke builder and source: `scripts/build_p2_speed_smoke.py` and
  `kaggle/rsna-p2-speed-smoke/`
- P2 submission source, builder, and validator: `scripts/p2_submission.py`,
  `scripts/build_p2_submission.py`, `scripts/validate_p2_submission.py`, and
  `kaggle/rsna-p2-submission/`
- P2 offline official-weight provenance: `kaggle/rsna-coatnet-rmlp2-init/`
- Frozen P2 protocols: `docs/P2_PROTOCOL.md` and `docs/P2_SUBMISSION_PROTOCOL.md`
- P3 source, builders, and validators: `scripts/p3_dinov2_hmil_ddp.py`,
  `scripts/build_p3_dinov2_notebook.py`, `scripts/build_p3_speed_smoke.py`,
  `scripts/validate_p3_speed_smoke.py`, and `scripts/validate_p3_outputs.py`
- P3 Kaggle bundles: `kaggle/rsna-p3-dinov2-hmil/` and
  `kaggle/rsna-p3-speed-smoke/`; frozen protocol: `docs/P3_PROTOCOL.md`
- Downloaded P3 full outputs: `outputs/kaggle/rsna-p3-dinov2-hmil-v1/`
  (ignored by Git); compact validated result: `experiments/results/P3.json`
- P4 canonical mode, builders, and prelaunch/speed validators:
  `scripts/p2_coatnet_ddp.py`, `scripts/build_p4_coatnet_folds.py`,
  `scripts/build_p4_speed_smoke.py`, `scripts/validate_p4_prelaunch.py`, and
  `scripts/validate_p4_speed_smoke.py`
- P4 per-fold and aggregate validation: `scripts/validate_p4_fold.py`,
  `scripts/aggregate_p4_outputs.py`, and `scripts/validate_p4_outputs.py`
- P4 Kaggle bundles: `kaggle/rsna-p4-coatnet-folds/fold-{0..4}/` and
  `kaggle/rsna-p4-speed-smoke/`; frozen protocol: `docs/P4_PROTOCOL.md`
- Downloaded P4 fold outputs: `outputs/kaggle/rsna-p4-coatnet-e2e-v1/fold-{0..4}/`
  (ignored by Git); aggregate validation outputs:
  `outputs/experiments/P4-coatnet-e2e-oof/` (ignored by Git); compact validated
  result: `experiments/results/P4.json`

All five P4 folds completed and independently passed their artifact, split, report
purge, OOF coverage, and dual-GPU checks. P4 alone scored **0.891161** OOF, effectively
tying P2, while the fixed equal-rank P2+P4 blend scored **0.895821** and passed its
predeclared promotion gate. Only the blend is promoted; its five-checkpoint hidden-test
inference path still requires a representative-data smoke and Kaggle canary before a
submission is spent.

The local notebook is executed against the representative DICOM sample. The
clean deployment notebook uses the same UIDs and loader against Kaggle's full
mounted competition source.

## Update and run

Rebuild both notebook forms after changing the builder:

```bash
python3 scripts/build_kaggle_notebook.py
python3 scripts/build_kaggle_notebook.py \
  --output kaggle/rsna-data-audit/rsna-data-audit.ipynb
```

Run locally before uploading:

```bash
jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=600 \
  output/jupyter-notebook/rsna-data-exploration.ipynb
```

Upload a new private version; Kaggle automatically starts execution:

```bash
kaggle kernels push -p kaggle/rsna-data-audit -t 1200
```

Inspect the remote run:

```bash
kaggle kernels status \
  vallepuvamsikrishna/rsna-knee-abnormality-detection-data-audit
kaggle kernels logs \
  vallepuvamsikrishna/rsna-knee-abnormality-detection-data-audit
kaggle kernels output \
  vallepuvamsikrishna/rsna-knee-abnormality-detection-data-audit \
  -p outputs/kaggle/rsna-data-audit
```

## GPU policy

The data-audit and report-teacher notebooks are CPU-only. Both image-model notebooks
use GPU. Their exact DICOM preprocessing, checkpoint load, feature extraction,
attention-head forward pass, backward pass, and optimizer update must pass locally
before each new remote GPU version is launched.

The image notebook prefers the successful output of the report-teacher notebook when
attached. It also has a deterministic fallback that rebuilds the same fixed three-LLM
mean contract directly from the three public source datasets, so GPU execution does not
depend on Kaggle CPU-queue timing.

The DINO notebook attaches Kaggle's official Meta DINOv2-small model source and never
uses third-party competition-trained weights. Backbone adaptation uses only the 4,349
pseudo-labelled rows; the 58 expert studies remain untouched until direct evaluation or
fold-safe head fitting.
