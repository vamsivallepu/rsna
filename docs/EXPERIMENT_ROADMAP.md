# Experiment roadmap

Snapshot date: 2026-09-01

This is the canonical execution plan for the competition. Conversation history is
not a source of truth. Change the ordering here before starting an unlisted run,
and record every completed, failed, or abandoned run in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Objective and selection contract

The primary objective is mean AUC over the 12 competition targets. We optimize for
a robust hidden-test ranking, not the highest accidental score on the 58 labelled
training studies or one public-leaderboard split.

Every candidate must use the following validation hierarchy:

1. report-group-purged pseudo validation for representation and training choices;
2. pooled five-fold OOF predictions covering each of the 58 expert studies once;
3. target-wise stability, paired bootstrap uncertainty, and repeated seeds for a
   promising low-cost change;
4. the public leaderboard as secondary evidence only;
5. runtime, reproducibility, licensing, and hidden-rerun safety before promotion.

Because 58 rows are extremely noisy, a macro-AUC change smaller than roughly 0.02
is inconclusive unless it is supported by paired uncertainty, multiple seeds, and
consistent target-level behaviour. No per-target threshold, blend weight, prompt,
or hyperparameter may be selected on those 58 labels.

## Cache policy

- Head, loss, weighting, pooling, and calibration experiments reuse the existing
  `(4407, 6, 3, 768)` DINOv2 token cache. They do **not** decode DICOM again.
- Geometry, slice coverage, crop, resolution, augmentation, or backbone changes
  require a new pixel/feature cache because the image inputs or encoder changed.
- The next full-image notebook must write a versioned pixel cache to a private
  Kaggle Dataset. Subsequent training notebooks attach that dataset instead of
  repeating the multi-hour DICOM pass.
- A cache is identified by data version, study ordering hash, series-selection
  policy, crop in millimetres, slice positions, image size, preprocessing, encoder
  checkpoint, and encoder weights hash. A changed signature means a new cache.

## Ordered experiment queue

The IDs and order are stable. Sub-experiments may be added beneath an item, but a
later expensive stage advances only after its prerequisites pass.

| Order | ID | Experiment | Compute / cache | Promotion gate |
|---:|---|---|---|---|
| 0 | **P0** | Validation hardening: paired replay of the current DINO head and report-group-purged OOF training | Local CPU; existing token cache | Exact audit, all 58 rows OOF, no report twin in a held-out fold's training set, independently reproduced metrics |
| 1 | **P1** | Cached-token head and objective ablations | Local CPU; existing token cache | Improve purged macro AUC or show stable target gains without a material regression |
| 2 | **P2** | Strong compact convolutional baseline: CoAtNet-RMLP2-style model at about 384 px with fuller physical slice coverage | Kaggle GPU; new pixel cache | One honest fixed configuration beats the purged DINO baseline and fits inference budget |
| 3 | **P3** | DINOv2-small at 336 px with fuller coverage, LoRA/last-block adaptation, and hierarchical diagnosis-query MIL | Kaggle GPU; reuse P2 pixel cache when compatible | Clear validation gain over P0/P2 with stable targets and acceptable runtime |
| 4 | **P4** | Five end-to-end folds for only the winning P2/P3 configuration | Kaggle GPU; winning cache | Fold-safe representation training and repeatable pooled OOF improvement |
| 5 | **P5** | OrthoFoundation frozen probe, then limited LoRA if the probe is useful | Kaggle GPU; compatible cache if preprocessing permits | License verified; frozen probe adds signal; LoRA only after that gate |
| 6 | **P6** | fastMRI+ knee ROI/localization pretraining or auxiliary supervision | Kaggle GPU | Dataset agreement/license verified and ROI pretraining improves held-out pseudo groups |
| 7 | **P7** | Multilingual report-image contrastive auxiliary loss during training | Kaggle GPU | Image-only inference preserved; no report identity leakage; robust improvement |
| 8 | **P8** | Noisy-student relabelling and a rank ensemble of genuinely diverse finalists | Kaggle GPU | Each member independently clears validation; ensemble diversity is measured |
| 9 | **P9** | OrthoDiffusion or compact 3D specialist for complementary targets | Kaggle GPU; high cost | Pursue only if 2.5D finalists plateau and licensing/runtime are acceptable |

### P0 — report-group-purged DINO OOF

The original 0.7976 DINO OOF run used every pseudo-labelled row in every head fold.
Normalized report auditing found repeated reports, including a group containing one
expert row and four pseudo rows. P0 keeps the original expert KFold assignment but,
for each fold, removes every pseudo row whose normalized report hash occurs in that
fold's expert validation rows.

P0 has two paired arms initialized from the same adapted head and trained with the
same folds, optimizer, epochs, and per-fold seeds:

- **P0-A replay:** all pseudo rows, matching the original training contract;
- **P0-B purged:** exclude validation-report twins from that fold's training set.

The paired replay separates purge effects from changes in local random-number state.
This is a head-only experiment; the DICOM and DINO encoders are not rerun.

### P1 — cached-token ablation matrix

Run in this order and change one dimension at a time:

1. **P1-A label weights:** current agreement weights versus raw uncertainty weights
   from the fixed report teacher. The Steven v4 synovitis repair is already present;
   do not apply it again.
2. **P1-B head:** current target-query attention versus shared attention plus
   target-specific output, and mean/max/query pooling controls.
3. **P1-C objective:** weighted BCE versus asymmetric focal loss; add pairwise ranking
   or consistency only if the simpler objective is stable.
4. **P1-D regularization:** global dropout/weight decay and two repeated seeds for
   the best candidate. Do not tune target-specific settings.

Only a P1 winner is allowed to influence the expensive P2/P3 notebook.

### P2–P4 — high-value image models

P2 tests the strongest compact convolutional family under a fixed, documented
physical-geometry policy: roughly 130 mm crops, broader volume coverage, and enough
windows to expose small lesions. P3 holds geometry as constant as practical while
testing higher-resolution DINOv2-small and hierarchical MIL. Avoid DINOv2-base at
224 px: the present bottleneck is coverage/resolution and label noise, not raw model
size. P4 pays for five end-to-end folds only after one configuration wins cheaply.

### P5–P9 — complementary signal

Foundation or external-data stages require a documented license audit before data or
weights enter a final submission. P8 ensembles ranks, not raw logits, and admits only
models with meaningfully different error patterns. P9 is deliberately last because a
full 3D system is expensive, harder to validate on 58 rows, and unlikely to be the
best first use of compute.

## Stop, go, and reprioritization rules

- **Go:** advance a configuration when macro AUC, target stability, and runtime all
  improve or when a diverse model adds validated ensemble value.
- **Repeat:** rerun a cheap candidate when its apparent gain is below 0.02 or driven
  by one target; use matched seeds and paired study bootstrap intervals.
- **Stop:** abandon a configuration after two well-formed runs show no useful signal,
  a licence is incompatible, the cache cannot be reproduced, or inference cannot fit
  the competition's nine-hour limit with margin.
- **Reorder:** update this file first and add a dated rationale to the experiment log.
  Public leaderboard movement alone is not sufficient reason to reorder the plan.

## Required artifacts for every run

Each run record must contain:

- stable experiment/run ID, status, date, hypothesis, and parent baseline;
- exact code commit, seed, software/device, data and cache signatures;
- train/validation identities and leakage audit;
- 12 target AUCs, macro AUC, deltas, and uncertainty where applicable;
- extraction/training/inference time, failures, and resource usage;
- output paths and hashes for small manifests/predictions;
- conclusion: promote, repeat, reject, or blocked, with the next action.

Large caches and checkpoints remain outside Git. Small manifests, metrics, and the
decision record belong in the repository so the experiment history survives notebook
cleanup, Kaggle session expiry, and conversation compaction.

## Explicitly deprioritized choices

- DINOv2-base at 224 px before fixing coverage and resolution;
- a full 3D model trained from scratch;
- aggressive left/right flips, colour transforms, MixUp, or CutMix without an
  anatomy-specific validation case;
- per-target tuning on the 58 expert labels;
- ensembles of many correlated public checkpoints;
- DINOv3-derived core weights before their final-submission licence is verified.
