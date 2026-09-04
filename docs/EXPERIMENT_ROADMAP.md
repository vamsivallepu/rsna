# Experiment roadmap

Snapshot date: 2026-09-04

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
- The next full-image notebook must write a versioned private Kaggle pixel cache.
  A private Dataset or immutable version-pinned private kernel output is acceptable;
  subsequent training notebooks attach it instead of repeating the DICOM pass.
- A cache is identified by data version, study ordering hash, series-selection
  policy, crop in millimetres, slice positions, image size, preprocessing, encoder
  checkpoint, and encoder weights hash. A changed signature means a new cache.

## Ordered experiment queue

The IDs and order are stable. Sub-experiments may be added beneath an item, but a
later expensive stage advances only after its prerequisites pass.

| Order | ID | State | Experiment | Compute / cache | Promotion gate |
|---:|---|---|---|---|---|
| 0 | **P0** | validated | Validation hardening: paired replay of the current DINO head and report-group-purged OOF training | Local CPU; existing token cache | Passed: all 58 rows OOF, zero purged overlap, saved-head and hash validation |
| 1 | **P1** | validated | Cached-token head and objective ablations | Local CPU; existing token cache | Closed after P1-D; no change promoted over P0-B |
| 2 | **P2** | **validated / promoted / scored** | Strong compact convolutional baseline: CoAtNet-RMLP2 at 384 px with fuller physical slice coverage | Kaggle CPU cache + GPU training; `P2-CACHE-v1` | Passed at 0.891125 OOF with +0.094337 over P0-B; first public leaderboard score 0.908 |
| 3 | **P3** | **rejected** | DINOv2-small at 336 px with fuller coverage, final-two-block adaptation, and hierarchical diagnosis-query MIL | Kaggle GPU; reuse `P2-CACHE-v1` | Failed: 0.805988 OOF and 0.863908 fixed rank blend both underperformed P2 |
| 4 | **P4** | **validated / blend promoted / scored** | Five end-to-end folds of the winning P2 configuration | Five Kaggle T4 x2 jobs; `P2-CACHE-v1` | Fixed P2+P4 blend passed at 0.895821 OOF and scored 0.916 public, +0.008 over P2 |
| 5 | **P5** | **validated / rejected** | OrthoFoundation-L frozen 2D-slice probe; limited LoRA only if the frozen probe is useful | Kaggle T4 x2; reuse `P2-CACHE-v1` | Failed: 0.768822 OOF and 0.886740 fixed blend; LoRA advance gate failed |
| 6 | **P6** | **active / full-source staging** | fastMRI+ knee ROI/localization pretraining or auxiliary supervision | Private on-demand AWS `g6.2xlarge`; all smokes passed; official 78.02 GB training archive downloading | GPU smoke passed at 99.44% utilization, 31.68 images/s, 6.89 GiB peak, projected 43.66 min; proceed to source training |
| 7 | **P7** | queued | Multilingual report-image contrastive auxiliary loss during training | Kaggle GPU | Image-only inference preserved; no report identity leakage; robust improvement |
| 8 | **P8** | queued | Noisy-student relabelling and a rank ensemble of genuinely diverse finalists | Kaggle GPU | Each member independently clears validation; ensemble diversity is measured |
| 9 | **P9** | queued | OrthoDiffusion or compact 3D specialist for complementary targets | Kaggle GPU; high cost | Pursue only if 2.5D finalists plateau and licensing/runtime are acceptable |

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

**Validated result:** P0-A scored 0.797796 and P0-B scored 0.796788. The
`-0.001008` paired delta had a 5,000-sample study-bootstrap 95% interval of
`[-0.003610, 0.001324]`. The leakage path is now closed, but its measured effect is
negligible and statistically inconclusive. P0-B is the comparison baseline for P1.

### P1 — cached-token ablation matrix

Run in this order and change one dimension at a time:

1. **P1-A label weights:** current agreement weights versus Steven v1's documented
   raw confidence `2*abs(p-0.5)`, where raw `0.5` means the report did not address
   that finding and receives zero loss weight. Keep the fixed consensus targets
   unchanged. The Steven v4 synovitis repair is already present; do not apply it again.
2. **P1-B head:** current target-query attention versus three predeclared controls:
   one shared attention query plus target-specific output, masked mean pooling, and
   masked max pooling. Every control inherits the adapted projection, slot/position
   embeddings, target weights, and bias; the shared query starts as the mean of the
   12 adapted target queries. All other training settings remain fixed to P0-B.
3. **P1-C objective:** weighted BCE versus the official multi-label asymmetric loss
   defaults (`gamma_neg=4`, `gamma_pos=1`, probability clip `0.05`). Extend its
   positive and negative terms continuously to the fixed soft targets, retain gradients
   through the focal factor, and apply the current agreement weights afterward. Add
   pairwise ranking or consistency only if this simpler objective is stable and promising.
   Primary references: [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)
   and [official implementation](https://github.com/Alibaba-MIIL/ASL).
4. **P1-D regularization:** close the cached-token phase with a factor-at-a-time
   primary-seed screen around P0-B: dropout `0.10` and `0.30` at weight decay
   `1e-3`, plus weight decay `3e-4` and `3e-3` at dropout `0.20`. The default is
   dropout `0.20`, weight decay `1e-3`. Select the highest primary-seed macro arm
   using the declared order for ties, then pair it against the default at training
   seeds `20260901` and `20260902`; retain the original expert folds. Do not tune
   target-specific settings.

   Promotion requires every frozen condition: positive mean macro delta across the
   three seeds, positive delta in at least two seeds, a positive three-seed ensemble
   delta, at least eight nonnegative target deltas, worst target delta above `-0.02`,
   and paired-bootstrap probability of a positive ensemble delta at least `0.80`.
   Whether it promotes or not, P1-D closes P1 and hands a frozen configuration to P2.

Only a promoted P1 change may influence the expensive P2/P3 notebook. Because no
change passed, P0-B supplies the frozen head/loss reference for their comparisons.

**P1-A result:** raw not-addressed weighting scored 0.800197 versus the 0.796788
P0-B baseline (`+0.003409`; paired 95% interval `[-0.010395, 0.015791]`). The
change is inconclusive and driven by Synovitis (`+0.0669`) while seven targets
declined. It was not promoted; P0-B remained the baseline for P1-B.

**P1-B result:** shared attention scored 0.794658 (`-0.002130`, interval
`[-0.007557, 0.002770]`), masked mean 0.794122 (`-0.002666`, interval
`[-0.009680, 0.003682]`), and masked max 0.780986 (`-0.015802`, interval
`[-0.036439, 0.006588]`). No control is promoted. Target-specific query attention
remains the P1-C baseline.

**P1-C result:** asymmetric loss scored 0.800046 versus P0-B's 0.796788
(`+0.003257`; paired 95% interval `[-0.008962, 0.015640]`; probability of a
positive delta `0.6932`). Eight targets improved, but Fracture declined by 0.03611.
The candidate is stable but inconclusive and fails the predeclared target-stability
gate, so it is not promoted and ranking/consistency losses are not attempted. P1-D
uses P0-B as the primary configuration. The now-frozen P1-D scope does not combine
or repeat P1-A/P1-C; their OOF predictions remain available only for a later diversity
audit, not as target-specific tuning branches.

**P1-D result:** the primary-seed screen scored 0.795970 for dropout `0.10`,
0.796559 for dropout `0.30`, and exactly 0.796788 for both weight decay `3e-4`
and `3e-3`, versus P0-B's 0.796788. The declared tie order selected `3e-4` for
confirmation. At seeds `20260831`, `20260901`, and `20260902`, candidate-minus-default
macro deltas were exactly `0.0`, and every target AUC also tied in the three-seed
ensemble (`0.798224` for both). No arm passes the frozen promotion gate. Retain
P0-B's dropout `0.20` and weight decay `1e-3`, close P1, and proceed to P2.

### P2–P4 — high-value image models

P2 tests the strongest compact convolutional family under the frozen
[`P2_PROTOCOL.md`](P2_PROTOCOL.md): 140 mm crops, 64 slices over five acquisition
slots, 2–98% volume coverage, official ImageNet-only CoAtNet-RMLP2 initialization,
and target-specific attention MIL. The representation checkpoint is selected only
by report-grouped pseudo validation; expert labels are reserved for the direct gate
and fold-safe cached-feature OOF heads. P3 holds geometry as constant as practical
while testing higher-resolution DINOv2-small and hierarchical MIL. Avoid DINOv2-base
at 224 px: the present bottleneck is coverage/resolution and label noise, not raw
model size. P3 scored `0.805988` OOF versus P2's `0.891125`; its fixed equal-rank
blend with P2 also declined to `0.863908`. P3 is rejected and P2 remains the best
standalone model. P4 then trained five fold-safe, end-to-end P2 models. P4 alone tied
P2 at `0.891161`, but the predeclared equal-rank P2+P4 blend reached `0.895821`
(`+0.004696`) and passed the uncertainty and target-stability gate. Promote only that
blend, retain P2 as a required member, validate its hidden-test inference path, and
advance model research to P5. The validated blend submission subsequently scored
`0.916` publicly versus P2's `0.908`; this supports the OOF promotion without changing
the frozen blend or using leaderboard feedback for tuning.

### P5–P9 — complementary signal

Foundation or external-data stages require a documented license audit before data or
weights enter a final submission. P8 ensembles ranks, not raw logits, and admits only
models with meaningfully different error patterns. P9 is deliberately last because a
full 3D system is expensive, harder to validate on 58 rows, and unlikely to be the
best first use of compute.

P5-A is frozen in [`P5_PROTOCOL.md`](P5_PROTOCOL.md). It reuses the P2 pixel cache
but encodes 32 individual MRI slices per study with the released OrthoFoundation-L
ViT-L/16 checkpoint. The backbone remains frozen; a report-group-safe hierarchical
MIL head determines whether this knee-MRI-specific representation adds signal to the
current P2+P4 blend. A two-T4 speed/memory smoke precedes the full feature extraction.
The corrected version-2 smoke passed: both T4s reached 100% utilization during
extraction, peak allocation was 1.85 GiB per rank, and the measured two-GPU rate of
159.45 images/s projected the 4,407-study feature pass to 14.74 minutes. The full
version-3 run completed and independently validated at 0.768822 OOF; adding the
predeclared 25% P5 rank component reduced P2+P4 from 0.895821 to 0.886740. Both the
promotion and LoRA advance gates failed. P5 is closed without LoRA; proceed to P6.

P6-A is frozen in [`P6_PROTOCOL.md`](P6_PROTOCOL.md). The current Kaggle annotation
package is byte-identical to the hash-pinned Microsoft fastMRI+ release, and the audit
validates 16,154 boxes over 974 annotated volumes plus a 1,172-volume reviewed list.
The annotations are MIT-licensed, but the underlying MRI images require separate
official fastMRI agreement and access. The user's official NYU download email confirms
that access; its temporary URLs are not tracked. P6 still rejects unofficial Kaggle
image mirrors and direct CSV-only coordinate priors. Kaggle secret inheritance made
repeatable automated runs unnecessarily fragile, so P6 moved to the private AWS host
recorded in [`AWS_P6_RUNBOOK.md`](AWS_P6_RUNBOOK.md). After the official images are
checksum-verified on persistent EBS, pretrain the P2 CoAtNet backbone on
class-agnostic lesion localization, then compare it with the original ImageNet
initialization under the unchanged report-grouped pseudo and expert-OOF contract.

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
