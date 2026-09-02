# Experiment log

Last updated: 2026-09-02

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
| P1-B | validated | Shared attention, masked mean, and masked max controls | Best control 0.7946578 purged expert OOF | No control promoted; retain P0-B target queries |
| P1-C | validated | Official-default asymmetric loss with fixed soft labels | 0.8000455 purged expert OOF | Do not promote; positive point estimate is inconclusive and Fracture declined |
| P1-D | validated | Global regularization screen and matched repeated seeds | Three-seed candidate/default ensemble both 0.7982239 | No regularization promoted; P1 closed with P0-B configuration |

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

## Earlier validated runs

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
  comparison baseline. P1-D has since frozen a global-regularization-only scope, so
  P1-A is retained for a later diversity audit rather than repeated during P1-D.

## Queued next runs

| ID | Status | Run | Dependency |
|---|---|---|---|
| P1-A | rejected | Agreement weight versus documented raw `2*abs(p-0.5)` not-addressed confidence; targets unchanged | +0.0034, wide interval, one-target dominated; retained only for diversity audit |
| P1-B | validated | Query/head pooling controls | All three controls underperformed P0-B; no promotion |
| P1-C | validated | BCE versus asymmetric objective | Gate failed; no ranking/consistency follow-up and no promotion |
| P1-D | validated | Factor-at-a-time global regularization and matched repeated-seed confirmation | No promotion; P0-B retained and P1 closed |
| P2 | **queued** | CoAtNet-RMLP2 384 px/64-slice full-coverage image model | Cache complete; `ddp_t4_v3` must pass its short speed/memory gate before relaunch |
| P3 | queued | DINOv2-small 336 px LoRA hierarchical MIL | P2 cache/geometry contract |
| P4 | queued | Five end-to-end folds of the P2/P3 winner | A clear single-run winner |
| P5–P9 | queued | Foundation, auxiliary, relabelling, ensemble, and 3D stages | Gates in the roadmap |

## Active run

### P2 — wide-coverage CoAtNet-RMLP2

- **Status:** cache complete; the first training version was cancelled after one full
  epoch plus 250 steps of epoch 2 because it used only GPU 0 from a T4 x2 allocation.
  Its partial checkpoint is not used. The replacement `ddp_t4_v3` execution path is
  locally smoke-tested and must pass a short remote throughput/memory gate before the
  full eight-epoch launch.
- **Hypothesis:** denser physical volume coverage and a strong 384 px compact
  convolution/attention backbone will recover spatially sparse abnormalities missed
  by the current 224 px/18-window DINO representation.
- **Parent:** P0-B purged OOF macro AUC `0.7967883` and its unchanged weighted-BCE,
  target-specific-attention, and report-purge guardrails.
- **Cache:** `P2-CACHE-v1`; five slots, 64 total slices, 2–98% coverage, 140 mm crop,
  336 px uint8 storage, and boundary-safe adjacent-slice windows. Two versioned private
  Kaggle kernel outputs keep each part below the practical output-size limit.
- **Model:** official Apache-2.0 timm
  `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` initialization; no public
  competition-trained weights.
- **Selection:** train the representation on pseudo-labelled rows only and select its
  checkpoint by grouped pseudo-validation weighted BCE. Expert labels cannot select
  the epoch, checkpoint, or retry.
- **Evaluation:** direct untouched-expert gate followed by five report-group-purged
  feature-head folds and paired comparison with P0-B.
- **Frozen contract:** [`P2_PROTOCOL.md`](P2_PROTOCOL.md).
- **Launch evidence:** cache A reached 100/2,204 studies in 4.3 minutes and cache B
  reached 100/2,203 in 4.1 minutes, both with zero failures. The two workers run
  concurrently and write disjoint row ranges.
- **Version 1 incident:** both otherwise healthy jobs were mistakenly launched with
  CLI `--timeout 1200` and received `CANCEL_ACKNOWLEDGED` near 20 minutes. A stopped
  at 400/2,204 (17.3 min) and B at 400/2,203 (15.9 min), both with zero failures and
  no traceback. Those partial outputs are not reused. Version 2 omits `--timeout`.
- **Version 2 cache result:** all 4,407 studies cached with zero failures. Part A
  completed 2,204 studies in 95.85 minutes with pixel SHA-256
  `a79c740be3da78f0e9a396a31f66214148a9897cacf96ad4c8a6ac2b0ffc1335`; part B
  completed 2,203 studies in 77.04 minutes with pixel SHA-256
  `520b08734af7e86a05e8562efbedab261747aeadf78038bc26080f2ba354e5bf`.
  Both manifests report the frozen cache signature
  `0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3`.
- **Training version 1 evidence:** input verification finished at 1,240 seconds.
  Epoch 1 completed at 4,192 seconds with 49.06 minutes of representation time and
  pseudo-validation weighted BCE `0.5926257`; epoch 2 reached step 250/1,963 before
  cancellation. The code used `nn.DataParallel` only after the subsequent local
  revision; the actual version 1 model was plain `cuda` and left the second T4 idle.
- **Execution revision `ddp_t4_v3`:** true two-process NCCL DDP, `no_sync()` during
  gradient accumulation, channels-last fp16, fused AdamW, fixed-shape cuDNN autotuning,
  no per-step scalar synchronization, two loader workers per rank, parallel full-file
  cache hashing, two-study-per-rank inference, and two-GPU feature sharding. The global
  training study batch, accumulation,
  sampling, labels, loss, optimizer schedule, validation, and expert gates remain fixed.
- **DDP speed-gate incidents:** remote smoke v1 initialized both NCCL ranks but failed
  on the first accumulated update because PyTorch 2.10's static reducer is incompatible
  with this `no_sync()` pattern. Smoke v2 disabled the static reducer but exceeded the
  ten-minute diagnostic ceiling and was cancelled; Kaggle discarded its buffered logs
  and partial output. Neither smoke produced a reusable model artifact.
- **Next action:** run the 64-study/32-global-step DDP speed/memory smoke with streamed
  four-step timing and symmetric NCCL result collection. Launch the full run without
  a CLI timeout only if both T4s are active, peak memory is safely below 15 GB, and the
  projected completion time is comfortably within Kaggle's limit.

## Previous completed run

### P1-C — asymmetric focal objective

- **Status:** independently validated; no objective change promoted.
- **Baseline:** P0-B target-query head with weighted BCE at 0.7967883.
- **Candidate:** the official multi-label ASL defaults: `gamma_neg=4`,
  `gamma_pos=1`, and probability clip `0.05`, with gradients retained through the
  focal factor.
- **Soft-label contract:** interpolate the positive and negative ASL terms using the
  fixed consensus `y` and `1-y`, then multiply by the unchanged agreement weight.
  A built-in identity test must prove `gamma_neg=gamma_pos=clip=0` equals weighted BCE.
- **Constants:** target-query architecture, adapted initialization, purged folds,
  30 epochs, AdamW `lr=2e-4`, `weight_decay=1e-3`, batch size 256, and matched seeds.
- **Gate:** ranking or consistency loss is attempted only if ASL is finite, target-stable,
  and promising versus P0-B. Otherwise P1-C stops after ASL.
- **Raw artifact location:** `outputs/experiments/P1-C-asymmetric-focal-loss/`.
- **Result:** P1-C `0.8000455` versus P0-B `0.7967883`, delta `+0.0032573`.
  The independently recomputed 5,000-sample paired 95% interval is
  `[-0.0089623, 0.0156402]`; probability of positive delta is `0.6932`.
- **Target pattern:** eight of 12 targets improved, led by MCL `+0.02494`,
  Contusion `+0.02024`, and Effusion `+0.01615`; four declined, with Fracture
  falling `-0.03611` and Lateral Meniscus `-0.01366`.
- **Runtime and validation:** 384.7 seconds on four CPU threads; the float64 identity
  contract differed from weighted BCE by only `2.22e-16`; all five saved heads,
  predictions, purges, bootstrap values, and source/output hashes independently passed.
- **Tracked result:** [`experiments/results/P1-C.json`](../experiments/results/P1-C.json).
- **Decision:** do not promote ASL. It is stable but not sufficiently target-stable or
  statistically persuasive; the predeclared ranking/consistency gate therefore fails.
  Keep ASL OOF predictions only for a later diversity audit and advance P1-D with
  P0-B as the primary configuration.

## Latest completed run

### P1-D — global regularization and repeated-seed confirmation

- **Status:** independently validated; no regularization arm promoted; P1 closed.
- **Primary-seed screen:** dropout `0.10`/`0.30` at the default weight decay and
  weight decay `3e-4`/`3e-3` at the default dropout; one factor changes per arm.
- **Default:** P0-B weighted BCE, target-query head, dropout `0.20`, weight decay
  `1e-3`, and seed `20260831`.
- **Confirmation:** the highest primary-seed macro candidate is paired with the
  default at training seeds `20260901` and `20260902`; expert fold assignment stays
  fixed to seed `20260831`.
- **Constants:** current agreement weights, report-group purge, adapted initialization,
  30 epochs, AdamW `lr=2e-4`, batch size 256, and the existing token cache.
- **Promotion gate:** positive mean three-seed delta; at least two seed wins; positive
  seed-ensemble delta; at least eight nonnegative target deltas; worst target delta
  above `-0.02`; bootstrap probability of positive delta at least `0.80`. All required.
- **Scope:** no P1-A/P1-C combination and no target-specific settings. P1-D closes
  cached-token tuning and hands either the promoted arm or unchanged P0-B to P2.
- **Raw artifact location:**
  `outputs/experiments/P1-D-regularization-seed-confirmation/`.
- **Screen result:** dropout `0.10` scored `0.7959703` (`-0.0008180`) and dropout
  `0.30` scored `0.7965593` (`-0.0002290`). Weight decay `3e-4` and `3e-3`
  both tied P0-B exactly at `0.7967883`; declared ordering selected `3e-4`.
- **Seed confirmation:** candidate-minus-default macro deltas were exactly zero at
  all three seeds. Default/candidate seed-ensemble macro AUC was `0.7982239` for
  both, every target AUC tied, and the paired interval was `[0.0, 0.0]`.
- **Runtime and validation:** 2,145.7 seconds on four CPU threads; all 40 heads,
  eight trained-run OOF files, four purges, metrics, ensemble, bootstrap, and every
  source/output hash independently reproduced.
- **Tracked result:** [`experiments/results/P1-D.json`](../experiments/results/P1-D.json).
- **Decision:** promote nothing. Retain dropout `0.20`, weight decay `1e-3`, weighted
  BCE, and the P0-B target-query head. P1 is closed and P2 is next.

## Previous completed run

### P1-B — query and pooling controls

- **Status:** independently validated; no control promoted.
- **Baseline:** P0-B target-specific query attention at 0.7967883.
- **P1-B1:** one shared attention query, target-specific output.
- **P1-B2:** masked mean pooling, target-specific output.
- **P1-B3:** masked max pooling, target-specific output.
- **Initialization:** load the adapted projection, slot/position embeddings, target
  weights, and target bias for all controls; initialize the shared query as the mean
  of the 12 adapted target queries.
- **Constants:** current agreement label weights, report-group-purged folds, 30 epochs,
  weighted BCE, AdamW `lr=2e-4`, `weight_decay=1e-3`, batch size 256, and matched seeds.
- **Promotion rule:** observed macro alone does not select a head. Require target
  stability and paired uncertainty; a small winner advances only to repeated-seed P1-D.
- **Raw artifact location:** `outputs/experiments/P1-B-head-pooling-controls/`.
- **Results:** shared attention `0.7946578` (`-0.0021305`, paired 95% interval
  `[-0.0075572, 0.0027704]`); masked mean `0.7941219` (`-0.0026663`, interval
  `[-0.0096801, 0.0036816]`); masked max `0.7809864` (`-0.0158019`, interval
  `[-0.0364390, 0.0065878]`).
- **Target pattern:** shared and mean improved Effusion by `+0.0224` and `+0.0248`
  respectively, but lost enough ordering on OA, meniscus, Baker's, and/or Fracture
  to reduce macro AUC. Max pooling produced large, unstable target swings.
- **Runtime and validation:** 1,118.0 seconds on four CPU threads; all 15 heads,
  predictions, three 5,000-sample bootstraps, fold purges, and hashes independently passed.
- **Tracked result:** [`experiments/results/P1-B.json`](../experiments/results/P1-B.json).
- **Decision:** target-specific query attention remains the baseline. Advance no
  P1-B control; P1-C objective ablations are next.

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
- **2026-09-01:** All P1-B controls underperformed P0-B. Shared attention and mean
  pooling were slightly worse; max pooling was much less stable. Target-specific
  query attention remains baseline and P1-C becomes next.
- **2026-09-01:** P1-C ASL independently validated at 0.800046 (`+0.003257`),
  but its interval crosses zero and Fracture declined by 0.03611. ASL is not promoted,
  ranking/consistency is skipped under the frozen gate, and P1-D becomes next.
- **2026-09-01:** P1-D screened four global regularization arms and confirmed the
  selected lower-weight-decay arm at two additional seeds. It tied the default on
  every target and seed; no arm is promoted, P1 closes with P0-B, and P2 becomes next.
- **2026-09-02:** Froze P2 before remote execution. P2 uses a new 64-slice,
  five-slot, 140 mm/336 px cache, official ImageNet-only CoAtNet-RMLP2 initialization,
  pseudo-group checkpoint selection, untouched direct expert evaluation, and
  report-group-purged cached-feature OOF heads.
- **2026-09-02:** Published the hashed official CoAtNet initialization as private
  Kaggle Dataset `vallepuvamsikrishna/rsna-coatnet-rmlp2-imagenet-init`, pushed the
  P2 implementation at `d0ce2c7`, and launched both `P2-CACHE-v1` version 1 workers.
- **2026-09-02:** Diagnosed both version 1 cache cancellations as the accidental
  1,200-second CLI runtime limit, not a cache failure. Relaunched identical version 2
  jobs without `--timeout`; both reached authoritative `RUNNING` state.
- **2026-09-02:** Both version 2 cache jobs reached `COMPLETE`: 4,407/4,407 studies,
  zero failures, matching contract signatures, and finalized per-part pixel hashes.
- **2026-09-02:** Launched private P2 CoAtNet GPU notebook version 1 without a CLI
  timeout after both cache sources completed; authoritative state reached `RUNNING`.
