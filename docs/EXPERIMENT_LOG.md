# Experiment log

Last updated: 2026-09-01

This is the durable run and decision record. The ordered plan and acceptance rules
live in [`EXPERIMENT_ROADMAP.md`](EXPERIMENT_ROADMAP.md). Status values are
`queued`, `running`, `validated`, `repeat`, `rejected`, or `blocked`.

## Baseline registry

| ID | Status | Experiment | Validation result | Decision |
|---|---|---|---:|---|
| B00 | validated | Fixed three-report-model arithmetic-mean teacher | 0.894811 macro AUC on 58 expert rows | Use as fixed pseudo-label source; not an image inference model |
| B01 | validated | Frozen RadImageNet ResNet-50, six-slot attention head | 0.6943 five-fold expert OOF | Superseded; established extraction and geometry baseline |
| B02 | validated | L2-normalized mean-pooled ridge on B01 cache | 0.7053 five-fold expert OOF | Rejected as next direction; frozen global representation is limiting |
| B03 | validated | Pseudo-only adapted DINOv2-small, direct untouched expert gate | 0.7851916 macro AUC | Representation promoted to fold-head evaluation |
| B04 | validated | Adapted DINO tokens plus five fold-specific query heads | 0.7976172 expert OOF | Current image baseline; report-group purge not yet applied |
| P0 | validated | Paired replay and report-group-purged DINO fold heads | 0.7967883 purged expert OOF | Honest cached-token baseline for P1; purge effect was negligible |
| P1-A | repeat | Raw Steven `2*abs(p-0.5)` pseudo-cell confidence weights | 0.8001969 purged expert OOF | Do not promote: +0.0034 is inconclusive and driven by Synovitis |

### B04 target-wise OOF AUC

| Target | AUC | Target | AUC |
|---|---:|---|---:|
| ACL | 0.768382 | MCL | 0.714286 |
| Medial Meniscus | 0.786058 | Lateral Meniscus | 0.751553 |
| Medial OA | 0.919380 | Lateral OA | 0.814313 |
| PF OA | 0.779923 | Effusion | 0.875776 |
| Synovitis | 0.685783 | Baker's | 0.869565 |
| Contusion | 0.828610 | Fracture | 0.777778 |

B04 used seed `20260831`, 224 px images, 130 mm crops, six acquisition slots,
three positions per slot, three pseudo-only adaptation epochs, and 30 head epochs.
The reusable token cache has shape `(4407, 6, 3, 768)`. Its independent validator is
[`scripts/validate_dino_adaptation_outputs.py`](../scripts/validate_dino_adaptation_outputs.py).

## Latest validated run

### P0 — paired report-group purge audit

- **Status:** validated
- **Hypothesis:** duplicate normalized reports allow pseudo-labelled twins of a held
  expert study into its fold-head training set; purging them gives a more honest OOF
  estimate and may change generalization.
- **Parent:** B04.
- **Data audit:** Unicode-alphanumeric normalized report hashes produce 52 duplicate
  groups containing 223 studies. One group contains one expert study and four pseudo
  studies. No expert-expert duplicate group was found in the current metadata.
- **Arms:** P0-A paired baseline replay and P0-B fold-wise report-twin purge.
- **Constants:** original five expert folds, adapted-head initialization, 30 epochs,
  weighted BCE, AdamW `lr=2e-4`, `weight_decay=1e-3`, batch size 256, matched per-fold
  seeds, and the existing DINO token cache.
- **Success checks:** every expert row predicted exactly once; zero training/validation
  report-hash overlap in P0-B; finite predictions; metrics independently recomputed.
- **Raw artifact location:** `outputs/experiments/P0-report-group-purged-dino-oof/`
  (local and intentionally ignored by Git).
- **Result:** P0-A replay `0.7977963`; P0-B purged `0.7967883`; paired delta
  `-0.0010081`. The 5,000-sample paired study-bootstrap 95% interval was
  `[-0.0036101, 0.0013239]`, with probability of a positive delta `0.1946`.
- **Runtime:** 455.1 seconds on four local CPU threads.
- **Independent validation:** all 58 OOF rows per arm, saved-head prediction replay,
  zero P0-B hash overlap, and every source/output hash passed.
- **Tracked result:** [`experiments/results/P0.json`](../experiments/results/P0.json).
- **Decision:** the possible leakage is closed, but removing four twins has no
  meaningful measured effect. Use P0-B as the honest baseline and proceed to P1-A;
  no DICOM or DINO feature recaching is required.

### P1-A — raw not-addressed confidence weights

- **Status:** validated; repeat before any promotion.
- **Change:** pseudo-cell loss weight `2*abs(p-0.5)` from Steven's raw v1 labels;
  raw `0.5` means “not addressed.” Expert weight remains 2.0 and the fixed three-model
  consensus targets do not change.
- **Data effect:** 25.3947% of pseudo target cells receive zero weight; Synovitis is
  the outlier at 83.8814% not addressed.
- **Result:** P1-A `0.8001969` versus P0-B `0.7967883`, delta `+0.0034086`.
  The independently recomputed 5,000-sample paired 95% interval is
  `[-0.0103949, 0.0157913]`; probability of positive delta is `0.7094`.
- **Target pattern:** Synovitis `+0.0669`, MCL `+0.0249`, ACL `+0.0123`, and
  Effusion `+0.0112`; seven targets declined, including both menisci and all three
  OA targets.
- **Runtime and validation:** 371.0 seconds on four CPU threads; all heads,
  predictions, bootstraps, fold purges, and source/output hashes independently passed.
- **Tracked result:** [`experiments/results/P1-A.json`](../experiments/results/P1-A.json).
- **Decision:** do not promote or use target-specific weighting. Keep P0-B as the
  comparison baseline, run P1-B next, and revisit P1-A only during repeated-seed
  confirmation if later candidates support the same global change.

## Queued next runs

| ID | Status | Run | Dependency |
|---|---|---|---|
| P1-A | repeat | Agreement weight versus documented raw `2*abs(p-0.5)` not-addressed confidence; targets unchanged | +0.0034, wide interval, one-target dominated; no promotion |
| P1-B | **next** | Query/head pooling controls | P0-B remains the baseline |
| P1-C | queued | BCE versus asymmetric/ranking/consistency objectives | P1-B winner |
| P1-D | queued | Repeated-seed regularization confirmation | Best P1 candidate |
| P2 | queued | CoAtNet-RMLP2-style 384 px/full-coverage image model | P1 decision and versioned pixel-cache notebook |
| P3 | queued | DINOv2-small 336 px LoRA hierarchical MIL | P2 cache/geometry contract |
| P4 | queued | Five end-to-end folds of the P2/P3 winner | A clear single-run winner |
| P5–P9 | queued | Foundation, auxiliary, relabelling, ensemble, and 3D stages | Gates in the roadmap |

## Decision history

- **2026-09-01:** Made the roadmap and this log canonical so ordering, assumptions,
  failures, and results do not depend on conversation retention.
- **2026-09-01:** Inserted P0 before architecture work after finding normalized report
  duplicates crossing the expert/pseudo boundary. P0 reuses tokens and does not repeat
  DICOM caching.
- **2026-09-01:** P0 independently validated at 0.796788 purged OOF. The paired
  `-0.001008` delta was inconclusive, so the purge becomes a permanent validation
  guardrail and made P1-A the next run at that point.
- **2026-09-01:** P1-A independently validated at 0.800197 (`+0.003409`), but the
  interval crosses zero and the gain is driven by Synovitis. It is not promoted;
  P0-B remains baseline and P1-B becomes next.
