# Experiment log

Last updated: 2026-09-04

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
| P6-A | validated | fastMRI+ provenance, schema, access, and transfer audit | Official annotation hashes, release invariants, and user access validated | P6-B advances to secure private Kaggle image staging; unofficial mirrors rejected |
| P6-B-SMOKE | validated | Official single-coil HDF5 data, visual, and localization GPU smoke | 128/128 decoded; 16/16 overlays passed; 99.44% GPU; 31.68 images/s; 6.89 GiB peak | All gates passed; stage official training archive for source pretraining |
| P6-B | validated | Full fastMRI+ class-agnostic localization pretraining | Epoch 2: validation loss 1.056847, lesion recall 0.464186; strict P2 backbone compatibility passed | Advance selected source checkpoint to a matched P2 transfer comparison; no promotion yet |
| P6-C | validated | Matched ImageNet-control versus P6-initialized P2 reruns | Control 0.899966 OOF; transfer 0.909878; fixed 25% P6 blend 0.905421 | Standalone fails Effusion stability floor; promote the fixed 75% P2+P4 plus 25% P6 rank blend |

## Active experiment

### P6 — fastMRI+ localization transfer

- **Status:** P6 source pretraining and both matched downstream Kaggle version-1 arms
  completed and independently validated. The fixed 75% P2+P4 plus 25% P6 rank blend
  passed every frozen gate and is promoted for hidden-test inference validation.
- **Hypothesis:** lesion-localization supervision on sagittal knee MRI can improve the
  P2 CoAtNet backbone's sensitivity to small abnormalities.
- **Parent:** promoted P2+P4 equal-rank blend at `0.8958211` OOF and `0.916` public;
  the controlled initialization comparison reruns the standalone P2 architecture.
- **Provenance:** Microsoft `microsoft/fastmri-plus` commit
  `67ed9a68ee33d3197f93893e8125688470c4a577`. The small Kaggle annotation package
  exactly matches the official `knee.csv` and `knee_file_list.csv` SHA-256 hashes.
- **Release audit:** 16,167 rows comprise 16,154 boxes and 13 study-level artifact
  rows; 22 labels including artifact; and 974 annotated volumes. The reviewed-file
  CSV has 1,173 rows, but the final `Knee data` row is a footer rather than a file ID.
  Its 1,172 valid IDs match the paper, and 198 have no annotation row. The audit also
  found six negative-coordinate boxes, one zero-height box, and one label with trailing
  whitespace; P6-B will sanitize and log them explicitly.
- **Transfer audit:** ACL, MCL, and Effusion have grouped direct relationships. The
  meniscus, OA, Baker's, Contusion, and Fracture mappings are ambiguous or proxy-only;
  Synovitis has no direct label. The annotation CSV also omits image geometry and the
  boxes follow a documented up/down image flip. Direct annotation-only spatial priors
  are rejected.
- **Access decision:** annotation files are MIT-licensed, but underlying images require
  the official fastMRI data-use agreement and access path. The user's official NYU
  download email dated 2026-09-04 satisfies the access confirmation gate. Temporary
  signed URLs are not stored in Git, artifacts, notebook source, or logs. Do not use
  or republish an unofficial Kaggle image mirror; stage an official package through a
  private Kaggle Secret and retain filename/hash/overlay checks.
- **Frozen downstream contract:** class-agnostic lesion-center and box-size
  pretraining of the P2 CoAtNet backbone, followed by a matched
  ImageNet-versus-localization initialization comparison under the existing
  report-grouped pseudo and five-fold expert OOF contract. Full details:
  [`P6_PROTOCOL.md`](P6_PROTOCOL.md).
- **Secure staging:** private CPU notebook
  `vallepuvamsikrishna/rsna-knee-p6-fastmri-staging` version 1 was pushed on
  2026-09-04. It checks only whether `FASTMRI_KNEE_DICOM_BATCH2_URL` is enabled and
  whether it targets the official host and expected package; it never prints the
  secret or downloads data. CLI-pushed validation versions 3 and 4 both returned a
  Kaggle `ConnectionError` because API-launched runs do not inherit editor-attached
  secrets. The same cell then ran successfully in the authenticated editor session:
  `secret_available`, `secret_official_host`, and `secret_expected_package` were all
  `True`, while `secret_value_logged` and `download_started` remained `False`. Future
  Kaggle secret-dependent runs would need that editor session rather than
  `kaggle kernels push`.
- **AWS execution decision (2026-09-04):** move P6 execution to the private on-demand
  `g6.2xlarge` in `ap-south-1` documented in
  [`AWS_P6_RUNBOOK.md`](AWS_P6_RUNBOOK.md). It has one 24 GB NVIDIA L4, 32 GiB host
  memory, SSM-only access, encrypted persistent EBS, a least-privilege instance role,
  and no inbound security-group rules. This removes Kaggle runtime and secret
  inheritance friction without changing the frozen source task, matched P2 control,
  or promotion gate.
- **AWS data correction:** `knee_DICOMs_batch2.tar.xz` completed at 30,570,247,404
  bytes and passed official SHA-256
  `91efa0626c1ce7af68915bbd0589fc02f1aab5101569abdfbe7815fa265d5e89`.
  Archive and header inspection then showed anonymized clinical-series paths, JPEG
  2000 DICOMs, no `file<digits>` value in the probed header, and a PatientID that was
  not a fastMRI file ID. Because no verified mapping to the annotation identities was
  present, that package is excluded from P6-B localization rather than guessed.
- **Correct P6-B image source:** the official fastMRI+ example reads `file*.h5`
  `reconstruction_rss` and applies `pixels[:, ::-1, :]`. P6 therefore uses the
  official `knee_singlecoil_val.tar.xz` for the 128-volume smoke: 16,050,975,296 bytes,
  official SHA-256
  `cd65a27f25d11369ad73fd329ff0c68ea8d2510f98e9f46aaeff7fcaf0dba43b`.
  Its signed URL is an encrypted SSM SecureString and is never logged. Full source
  pretraining will add `knee_singlecoil_train.tar.xz` only after the smoke passes.
- **HDF5 smoke result:** the archive passed its official hash and extracted to 199
  reviewed HDF5 volumes. Commit `2cc14e7` decoded a deterministic 128-volume cohort
  (96 annotated, 32 reviewed negatives), produced a 103/25 volume-isolated split,
  retained all 1,605 selected boxes with no missing slice, and generated 16 overlays.
  All 16 manifest-verified overlays passed visual alignment with no systematic flip,
  mirror, slice offset, or coordinate shift. GPU utilization, peak memory, throughput,
  and full-runtime projection are still unmeasured.
- **GPU smoke version 1:** commit `826da99` completed without OOM and processed the
  30-step timed window at 31.64 images/s, with 6.88 GiB peak allocated memory and a
  projected 0.73 hours for eight source epochs. Its two remaining gates were invalid:
  the utilization monitor included roughly 70 seconds of lazy worker startup around
  only 7.58 seconds of timed training, and the ten-step fixed-batch comparison used
  training-mode running statistics. Mean utilization was therefore reported as
  17.52% and fixed-batch loss rose 11.33%. Record this as a harness repeat, not model
  evidence. Version 2 starts workers before CUDA/model construction, monitors only
  timed optimizer steps, compares fixed-batch loss in evaluation mode over 30 steps,
  and uses a safer `3e-4` localization-head learning rate.
- **GPU smoke version 2:** commit `2888ee4` passed every gate. Batch 8 sustained
  31.68 images/s with 99.44% mean GPU utilization (99% p10, 100% max), 6.89 GiB
  peak allocated memory, and finite losses. The evaluation-mode fixed-batch loss fell
  from 3.4640 to 0.7201 (`79.21%`), and the unchanged throughput projects 82,984
  images over eight source epochs to 43.66 minutes. This estimate excludes the
  one-time training-archive download and extraction. SSM command
  `71de23d3-270d-4c02-99c1-fbf9890366ec` produced the validated manifest.
- **Full-source staging:** after version 2 passed, SSM command
  `1d6ef4e8-a253-4192-a8e7-a4b7c81c23fd` started the resumable official
  `knee_singlecoil_train.tar.xz` download to encrypted persistent EBS. Expected size
  is 78,016,052,604 bytes and SHA-256 is
  `1e8ab7f602bb0fea649736020aa9317ebf3ba9c417c211b286b85c81a28bce5f`.
- **Frozen full run:** eight epochs, batch 8, official 973/199 archive split, every
  unique positive slice, four fixed slices per reviewed negative volume, five
  artifact-only volumes excluded, intensity-only augmentation, FP16/channels-last,
  fused AdamW plus OneCycleLR, `3e-5` backbone LR, and `3e-4` FPN LR. Minimum
  held-out localization loss selects the backbone; top-K lesion-center recall at
  radius two breaks exact ties. The separate validator must replay selection and
  prove strict P2 backbone compatibility before downstream transfer.
- **Full-source result:** corrected commit `e2be22d` completed in 0.687 hours. Epoch 2
  minimized held-out loss at `1.0568469` (baseline `4.1179232`) with lesion-center
  recall `0.4641857` (baseline `0.0008003`). Mean L4 utilization was `97.66%`, peak
  observed NVIDIA memory was 7,628 MiB, and all hashes, selection replay, data gates,
  and strict compatibility of 480 backbone tensors passed independent validation.
  The selected checkpoint SHA-256 is
  `5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195`.
- **Downstream staging:** the selected checkpoint and two small manifests, but no
  medical images, are in the private ready Kaggle Dataset
  `vallepuvamsikrishna/rsna-p6-fastmri-localization-init`. The matched kernels
  `rsna-knee-p6-p2-matched-control` and `rsna-knee-p6-p2-transfer` use the same source,
  P2 cache, seeds, optimizer, batches, epochs, checkpoint-selection boundary, and
  five-fold expert OOF procedure; initialization is their only intended difference.
  Both version-1 kernels completed successfully and their outputs were downloaded on
  2026-09-05.
- **Matched downstream result:** control pseudo BCE was `0.4736813` and OOF macro AUC
  was `0.8999658`; transfer improved them to `0.4625987` and `0.9098782`. Standalone
  transfer gained `+0.0099125` with bootstrap probability `0.8844`, but is not promoted
  alone because Effusion declined `-0.0708075`, beyond the `-0.05` stability floor.
  The fixed 75% P2+P4 plus 25% P6 rank blend reached `0.9054206`, a `+0.0095995`
  improvement over P2+P4. Its paired 95% interval `[0.0029767, 0.0172338]`, `0.9976`
  improvement probability, nine nonnegative targets, and `-0.0071685` worst decline
  pass every frozen promotion gate. Promote only this fixed blend.
- **Downstream validation:** both manifests and every hashed artifact passed; all 58
  expert rows were OOF exactly once; saved direct and five fold heads replayed; report
  overlap remained zero; and both T4s were active. Control GPU means were 97.79% and
  97.14%; transfer means were 96.77% and 96.74%.
- **AWS cost control:** after the selected checkpoint was verified locally and in the
  private Kaggle Dataset, instance `i-0952ebd05fab9e563` was stopped and then
  terminated with explicit approval on 2026-09-04. Its 100 GiB root and 500 GiB data
  EBS volumes, eight versioned staging bundles, and private staging bucket were
  deleted. A post-cleanup audit found no RSNA EBS volumes, snapshots, Elastic IPs, or
  staging bucket; only non-chargeable IAM, security-group, and Standard SSM
  configuration remains.
- **Tracked result:** [`experiments/results/P6-AUDIT.json`](../experiments/results/P6-AUDIT.json).
  Partial P6-B gate result: [`experiments/results/P6-B-SMOKE.json`](../experiments/results/P6-B-SMOKE.json).
  First GPU-smoke attempt: [`experiments/results/P6-B-GPU-SMOKE-v1.json`](../experiments/results/P6-B-GPU-SMOKE-v1.json).
  Validated GPU smoke: [`experiments/results/P6-B-GPU-SMOKE-v2.json`](../experiments/results/P6-B-GPU-SMOKE-v2.json).
  Validated full source run: [`experiments/results/P6-B.json`](../experiments/results/P6-B.json).
  Validated matched comparison: [`experiments/results/P6.json`](../experiments/results/P6.json).

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
| P2 | **validated / promoted** | CoAtNet-RMLP2 384 px/64-slice full-coverage image model | OOF 0.891125 versus P0-B 0.796788; integrity, uncertainty, and runtime gates passed |
| P2-SUB | **validated / scored** | Five-head P2 hidden-test inference and `submission.csv` | Public leaderboard 0.908; both identical submissions completed successfully |
| P3 | **rejected** | DINOv2-small 336 px final-two-block adaptation and hierarchical diagnosis-query MIL | Valid run, but OOF 0.805988 and fixed P2+P3 rank blend 0.863908 both underperformed P2 |
| P4 | **validated / blend promoted** | Five end-to-end folds of the P2 winner | P4 alone tied P2 at 0.891161; fixed equal-rank P2+P4 reached 0.895821 and passed its gate |
| P4-SUB | **validated / scored** | Optimized six-model P2+P4 equal-rank submission | Public leaderboard 0.916, improving canonical P2 by 0.008 |
| P5 | **validated / rejected** | OrthoFoundation-L frozen single-slice feature probe | OOF 0.768822; fixed 25% blend fell to 0.886740; promotion and LoRA gates failed |
| P6 | **validated / blend promoted / scored** | fastMRI+ localization transfer into matched P2 reruns | Fixed blend submission `56026228` completed at public 0.923, +0.007 over P4 |
| P7 | **validated / rejected** | BGE-M3 report-image auxiliary training | Pseudo BCE improved, but 0.904283 OOF and 0.904354 fixed blend both missed P6; retain P6 and proceed to P8 |
| P8 | **validated / OOF gate passed / canary required** | Fold-safe P2/P4/P6 image-teacher relabelling and five student folds | P8 OOF 0.902912; frozen candidate blend 0.909798 versus current 0.905421; runtime canary next |
| P9 | queued | Compact 3D specialist | Gates in the roadmap |

## Current P6 submission

- **Frozen candidate:** 75% of the P2+P4 equal-rank baseline plus 25% P6-transfer
  target rank; OOF `0.905421` versus `0.895821`, delta `+0.009600`.
- **Canary v1:** all hashes and predictions validated, but `1.137270` inference
  minutes projected to `8.2136` hidden hours and only `0.7864` hours of headroom.
  This missed the predeclared `0.9`-hour safety gate and was not submitted.
- **Canary v2:** converted the shared normalized tensor to FP16 once per device,
  avoiding a repeated identical input cast for every resident backbone. It completed
  the inference segment in `1.053519` minutes, projecting to `7.6087` hours and
  `1.3913` hours of headroom. The final `submission.csv` remained byte-identical to
  v1 (`f31fa1d...`) and independent replay passed every source, blend, schema, and
  runtime check.
- **Leaderboard submission:** exact private Kaggle kernel version 2 from commit
  `ac8aa92` completed on 2026-09-05 as submission `56026228` with public score
  `0.923`. This is `+0.007` over P4 and `+0.015` over P2; the live leader was `0.954`
  at verification, leaving a `0.031` gap. Tracked evidence:
  [`experiments/results/P6-SUBMISSION.json`](../experiments/results/P6-SUBMISSION.json).

## Current P7 execution

- **P7-A:** Kaggle version 1 completed and independently validated on 2026-09-05.
  All 4,407 reports have finite unit-normalized 1,024-dimensional BGE-M3 embeddings;
  maximum input length was 1,238 tokens, so none reached the frozen 4,096-token
  ceiling. Peak allocation was 1,260/1,231 MiB on T4 x2. The manifest is pinned at
  SHA-256 `f82ddf5f5b95224a21cb83b2f1e8b3ed1bf3bad2e8495396d5c398956a9556a8`.
- **P7-B:** the matched report-image trainer and 64-study speed-smoke bundles are
  generated from one canonical source. BGE-M3 is absent from both; they consume only
  the 8.6 MiB embedding matrix. The report projection is training-only and excluded
  from the deployable image checkpoint. Smoke version 1 completed and independently
  validated: the full fixed bank contains 3,926 pseudo-training rows/3,761 report
  groups, report InfoNCE was finite at `1.0142`, both ranks peaked at 8,148 MiB, and
  no traceback or CUDA OOM occurred. The post-warmup representation projection is
  1.89 hours; the all-in conservative projection is 5.62 hours.
- **Full P7-B:** version 1 completed on 2026-09-06 at
  `vallepuvamsikrishna/rsna-knee-p7-report-image-auxiliary` using the exact trainer
  frozen in commit `6dbbc05`. All artifact hashes, the full report-bank split, saved
  direct and five fold heads, deployable checkpoint schema, and dual-GPU telemetry
  independently passed. Representation took 147.21 minutes and feature extraction
  plus head evaluation took 23.26 minutes; GPU utilization averaged 96.91% and
  96.17%.
- **P7 result:** pseudo-validation BCE improved from P6's `0.462599` to `0.459584`,
  but standalone expert OOF declined from `0.909878` to `0.904283`. Replacing P6 in
  the frozen 25% slot reduced the blend from `0.905421` to `0.904354` (delta
  `-0.001067`); its paired-bootstrap probability of improvement was only `0.3724`.
  Three mandatory gates failed, so P7 is rejected without tuning and P6 remains the
  promoted model.
- **Tracked evidence:**
  [`experiments/results/P7-A.json`](../experiments/results/P7-A.json),
  [`experiments/results/P7-B-SMOKE.json`](../experiments/results/P7-B-SMOKE.json),
  [`experiments/results/P7.json`](../experiments/results/P7.json), and the frozen
  [`P7_PROTOCOL.md`](P7_PROTOCOL.md).

## Current P8 execution

- **Frozen teacher audit:** the exact `0.50` report plus `0.50` equal P2/P4/P6
  image-teacher formula reached `0.920920` expert macro AUC versus `0.894811` for
  the report teacher. Its paired-bootstrap probability of a positive delta was
  `0.9984`; all advance gates passed without target-specific tuning.
- **Teacher generation V1:** the private T4 x2 job consumed the correct immutable,
  hash-pinned P2/P4/P6 outputs, weak labels, and `P2-CACHE-v1`. All sources passed at
  8.07 minutes, both GPUs sustained 100% observed utilization with 8,845 MiB peak
  allocation, and ranks completed 2,204/2,204 and 2,203/2,203 studies in 95.58 and
  89.39 inference minutes. It then failed at 104.16 total minutes on a final
  `np.allclose(..., atol=1e-5)` check between batch-of-two FP16 replay predictions
  and the original batch-of-one P4 OOF values. This was not an OOM, model, cache, or
  data failure, but Kaggle retained no artifacts.
- **Teacher generation V2 fix:** commit `be3508b` moves the replay gate to all 58
  expert studies before full inference, records maximum/mean/p99 differences, and
  preserves the already-frozen canonical P2/P4/P6 expert values for the audit. Only
  pseudo-study predictions use the faster batch-of-two replay. Exact generated-source,
  lint, compile, source/hash, formula, fold-safety, and prelaunch checks passed locally.
  Failure evidence is tracked in
  [`P8-A-TEACHER-LABELS-V1-FAILURE.json`](../experiments/results/P8-A-TEACHER-LABELS-V1-FAILURE.json).
- **Teacher generation V2:** Kaggle version 2 passed every immutable source
  hash and the full 58-study batch-of-two replay preflight. The worst P4 absolute
  difference was `0.000566` against a `0.005` gate; per-fold mean differences were
  `0.000022` to `0.000051` against a `0.0005` gate. P2/P6 maxima were below
  `2.7e-7`. The completed artifacts were downloaded and independently replayed:
  all SHA-256 hashes, all five fold-specific target equations, expert-label
  preservation, source ordering, and fold safety passed. Runtime was `102.16`
  minutes; both T4s reached 100% utilization, averaged 91.61%/91.52%, and peaked at
  7,784.61 MiB allocated per rank. Manifest SHA-256 is
  `c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f`.
  Tracked evidence:
  [`P8-A-TEACHER-LABELS.json`](../experiments/results/P8-A-TEACHER-LABELS.json) and
  [`P8-A-TEACHER-LABELS-V2-PREFLIGHT.json`](../experiments/results/P8-A-TEACHER-LABELS-V2-PREFLIGHT.json).
- **Student speed smoke:** the 64-study smoke and all five fold bundles were generated
  from the canonical trainer with the exact teacher manifest hash embedded. Local
  compilation, lint, expert-label, non-target-column, initialization, metadata, and
  generated-source checks passed. Kaggle version 1 completed and its downloaded
  artifacts independently passed the exact teacher/target and P6 initialization
  hashes, complete role ledger, report isolation, checkpoint selection, held-out
  predictions, artifact hashes, and two-GPU gates. Its 64-row epoch took `0.93`
  minutes, held-out inference `0.07` minutes, both T4s reached 99% utilization and
  7,539 MiB used, and there was no traceback or OOM. Calibrating against the same-path
  P4 smoke and five full P4 folds projects approximately `124–138` total minutes per
  P8 fold. Tracked evidence:
  [`P8-B-SPEED-SMOKE.json`](../experiments/results/P8-B-SPEED-SMOKE.json).
- **Full student wave 1:** folds 0 and 1 ran concurrently as private Kaggle version 1
  jobs. They
  embed teacher manifest `c9fa6080...e5bc85f` and fold-target hashes
  `ec306bb8...e9790b` / `fdc11652...b55596`. Both downloaded outputs independently
  passed every artifact hash, exact split ledger, report-isolation, P6 initialization,
  teacher-target, checkpoint-selection, held-out prediction, and dual-GPU check.
  Fold 0 selected epoch 8 at pseudo BCE `0.467890` and took `143.32` representation
  minutes; fold 1 selected epoch 8 at `0.466821` and took `141.64` minutes. GPU means
  were 96.13–96.92%, and all devices reached 100%. Evidence:
  [`P8-B-FOLD-0.json`](../experiments/results/P8-B-FOLD-0.json) and
  [`P8-B-FOLD-1.json`](../experiments/results/P8-B-FOLD-1.json).
- **Full student wave 2:** folds 2 and 3 ran concurrently and their downloaded outputs
  independently passed the same complete validation contract. Fold 2 selected epoch 8
  at pseudo BCE `0.467777` in `129.06` representation minutes; fold 3 selected epoch 8
  at `0.467245` in `136.33` minutes. All GPUs averaged above 96% utilization and
  reached 100%. Evidence:
  [`P8-B-FOLD-2.json`](../experiments/results/P8-B-FOLD-2.json) and
  [`P8-B-FOLD-3.json`](../experiments/results/P8-B-FOLD-3.json).
- **Final student fold:** private Kaggle version 1 completed and its downloaded
  output independently passed the same artifact, teacher-target, initialization,
  split-ledger, report-isolation, held-out prediction, and dual-GPU contract. Fold 4
  selected epoch 8 at pseudo BCE `0.468403` in `149.62` representation minutes;
  both GPUs averaged above 97% utilization, reached 100%, and peaked at 7,541 MiB.
  Evidence: [`P8-B-FOLD-4.json`](../experiments/results/P8-B-FOLD-4.json).
- **Five-fold aggregation:** the five held-out files cover all 58 expert studies
  exactly once. P8 standalone reached `0.902912` versus P4's `0.891161`, a
  `+0.011751` delta with 9/12 nonnegative target deltas, worst target delta
  `-0.017544`, and paired-bootstrap probability of improvement `0.9560`.
- **Frozen ensemble gate:** replacing only P4 in
  `0.375 rank(P2) + 0.375 rank(P4/P8) + 0.25 rank(P6)` raised macro AUC from
  `0.905421` to `0.909798`, a `+0.004378` delta. Nine of 12 target deltas were
  nonnegative, the worst was `-0.009070`, and the 5,000-repeat paired-bootstrap
  probability of improvement was `0.8992` (95% interval `[-0.002435, 0.011119]`).
  All predeclared OOF gates passed. Promotion remains blocked only by the mandatory
  hidden-test runtime canary with at least 0.9 hours of projected headroom. Aggregate
  evidence: [`experiments/results/P8.json`](../experiments/results/P8.json).

## Completed P5 run

- **Run ID:** `P5A-ORTHOFOUNDATION-FROZEN-v1`.
- **Frozen protocol:** [`P5_PROTOCOL.md`](P5_PROTOCOL.md).
- **Speed smoke version 1:** failed before inference because the official released
  checkpoint stores backbone tensors under `backbone.*` while the raw DINOv3 model
  expects unprefixed keys. The exact checkpoint hash and provenance had already
  passed; no metric or speed result was produced.
- **Speed smoke version 2:** normalized that single namespace deterministically,
  asserted exact equality of checkpoint/model key sets, then loaded strictly. The
  remote run completed and the downloaded artifacts independently passed
  [`validate_p5_speed_smoke.py`](../scripts/validate_p5_speed_smoke.py).
- **Measured result:** 128 studies and 4,096 slices completed at 159.4534 images/s
  across two T4s. Full 4,407-study feature extraction projects to 14.7404 minutes;
  1,300-study hidden feature inference projects to 4.3482 minutes. Peak allocated
  memory was 1,847.16 MiB on each rank. Both GPUs reached 100% utilization during
  extraction, all features/losses were finite, and five head backward updates passed.
- **Decision:** pass the speed, memory, utilization, model-integrity, and numerical
  gates. Proceed to the full frozen feature and fold-safe OOF run. LoRA remains
  prohibited until the frozen representation passes its separate signal gate.
- **Full run version 1:** failed at 9.6 seconds, before GPU allocation or input
  access, with `ModuleNotFoundError: p5_orthofoundation_speed_smoke`. Kaggle's
  script upload retained only the declared `code_file` and omitted the builder's
  helper sidecar. This is a packaging failure with no experimental result. Version
  2 embeds the same canonical helper source into one compiled Kaggle script.
- **Full run version 2:** the self-contained bundle passed startup and all input
  integrity checks. Both T4s sustained 99–100% utilization; rank 1 completed
  2,203/2,203 studies and rank 0 reached 2,000/2,204. The launcher had mistakenly
  supplied CLI `--timeout 1200`, so Kaggle returned `CANCEL_ACKNOWLEDGED` at about
  20 minutes. This was an external runtime cap, not a code or data failure, and its
  incomplete artifacts are not used.
- **Full run version 3:** completed in 38.03 minutes with the unchanged
  self-contained code and no CLI timeout. All 4,407 studies were encoded; the
  slowest feature rank took 15.85 minutes and peak allocation was 1,839.09 MiB on
  each T4. The downloaded artifacts, hashes, five fold heads, 58-row OOF coverage,
  and reported metrics independently passed `validate_p5_outputs.py`.
- **Result:** direct expert macro AUC `0.765714`; fold-safe P5 OOF macro AUC
  `0.768822`. The fixed 25% P5 rank blend reduced the promoted P2+P4 comparator
  from `0.895821` to `0.886740` (`-0.009081`). Its paired-bootstrap probability of
  improvement was `0.028` with 95% interval `[-0.018692, 0.000147]`; only four of
  twelve targets were nonnegative.
- **Decision:** reject P5-A and do not run P5-B LoRA because the predeclared advance
  gate failed. Preserve the frozen features and positive target diagnostics for P8,
  but do not tune target-specific weights on the 58-row expert set. Advance to P6.
- **Diagnostic submission:** requested for leaderboard information only. The frozen
  five-fold probability mean is isolated under `P5-DIAGNOSTIC-SUBMISSION-v1`; its
  public score cannot promote P5 or authorize target/weight tuning. A two-T4 private
  canary completed in 9.64 wall minutes and independently passed source hashes,
  geometry, sample order, prediction, and five-fold replay checks. Its slower-rank
  inference segment was 0.304 minutes for the three visible studies, projecting the
  1,300-study hidden run to about 1.79 hours including startup. Exact Kaggle kernel
  version 1 was submitted as reference `56008122`; evaluation is pending. The
  submission description contains an incidental commit-hash typo (`a9447a1`); the
  immutable kernel version was built from actual code commit `a80f042` and is traced
  by manifest SHA-256 `ddec9141e892bda4db4b6c3b75590cb65c0fe0c16e34460f9396606195086fd2`.

## Completed P2 run

### P2 — wide-coverage CoAtNet-RMLP2

- **Status:** full Kaggle version 2 completed and independently validated. The first
  training version was cancelled after one full
  epoch plus 250 steps of epoch 2 because it used only GPU 0 from a T4 x2 allocation.
  Its partial checkpoint is not used. The replacement `ddp_t4_v3` execution path has
  passed local validation and the remote two-GPU throughput/memory gate; the full
  eight-epoch relaunch is ready.
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
  cache hashing, one-study-per-rank inference, and two-GPU feature sharding. The global
  training study batch, accumulation,
  sampling, labels, loss, optimizer schedule, validation, and expert gates remain fixed.
- **DDP speed-gate incidents:** remote smoke v1 initialized both NCCL ranks but failed
  on the first accumulated update because PyTorch 2.10's static reducer is incompatible
  with this `no_sync()` pattern. Smoke v2 disabled the static reducer but exceeded the
  ten-minute diagnostic ceiling and was cancelled; Kaggle discarded its buffered logs
  and partial output. Smoke v4, a speculative two-study-per-rank inference benchmark,
  never exposed script output during its ten-minute bound and was cancelled; the
  unproven larger inference batch was reverted. None produced a reusable model artifact.
- **DDP speed gate passed:** smoke v3 completed 32 real global training steps in
  38.47 seconds including one-time cuDNN warm-up (`1.2021 s/step` conservative average;
  approximately `0.41 s/step` after warm-up). Both T4s sampled at 100% utilization and
  each peaked at 8,084.9 MiB of 15,360 MiB. The downloaded JSON, telemetry, and complete
  log are retained under `outputs/kaggle/rsna-p2-ddp-speed-smoke-v3/`.
- **Full version 2 launch:** Kaggle run `346781024` was accepted from Git commit
  `9a3bd4d` with the proven one-study-per-rank inference setting. At the initial live
  check the worker was `RUNNING` on T4 x2, while Kaggle's log-stream API returned HTTP
  500 and the signed-in log viewer remained on `Loading`; DDP startup therefore still
  requires direct log evidence once Kaggle exposes it.
- **Full version 2 result:** all eight epochs completed; epoch 8 minimized grouped
  pseudo-validation BCE at `0.4762114`. Direct untouched-expert macro AUC was
  `0.8784271`; five-fold report-purged OOF macro AUC was `0.8911254` versus P0-B
  `0.7967883`, delta `+0.0943371`. The paired 5,000-sample interval was
  `[0.0635023, 0.1280301]` with probability positive `1.0`. Eleven targets improved;
  Lateral OA declined `-0.05416`, with its paired interval crossing zero. All hashes,
  feature geometry, report purges, saved-head replays, rows, and metrics passed the
  independent validator.
- **Runtime:** representation plus selection took `133.16` minutes and final feature,
  head, and evaluation work took `20.76` minutes. Median GPU utilization was 100% on
  both T4s, with mean utilization 97.27% and 96.65%.
- **Submission path:** `P2-SUBMISSION-v1` is frozen in
  [`P2_SUBMISSION_PROTOCOL.md`](P2_SUBMISSION_PROTOCOL.md). Its local representative
  DICOM smoke and independent five-head-mean validator passed. Private Kaggle canary
  version 1 then processed all three visible test studies in `0.51` minutes with
  non-constant predictions for every target. Its `submission.csv` SHA-256 is
  `6addb8b64dc7c5ccc09c709ee560700f9fd94bcb37b4e9a1e4bcd5f5b9e137d0`, and the
  downloaded artifact bundle passed the independent validator.
- **Leaderboard submission:** notebook version 1 from Git commit `8a4f410` was
  submitted to the code competition as canonical reference `55965846`. It completed
  successfully with public score `0.908`. The accidental duplicate reference
  `55965923` also completed with the identical `0.908` score, confirming deterministic
  notebook output. The live leading score at verification time was `0.952`, leaving a
  `0.044` public-leaderboard gap.
- **Next action:** retain `55965846` as the canonical P2 submission. P3 was rejected;
  P4 subsequently promoted only the fixed equal-rank P2+P4 blend. Do not spend
  another submission slot on unchanged P2.

## Completed P3 run

- **Run ID:** `P3-DINOV2-HMIL-v1`.
- **Frozen protocol:** [`P3_PROTOCOL.md`](P3_PROTOCOL.md).
- **Parent:** validated P2 OOF `0.8911254`, public score `0.908`.
- **Core change:** cache-native 336 px DINOv2-small, final-two-block adaptation,
  five retained region tokens per window, and two-stage diagnosis-query pooling.
- **Integrity:** reuse exact `P2-CACHE-v1`; pseudo-only representation selection;
  untouched direct expert gate; report-group-purged five-fold expert OOF heads.
- **Execution plan:** local one-study forward/backward, dual-T4 64-study speed smoke,
  then full P3 only if memory, utilization, numerical, and runtime gates pass.
- **Speed smoke v1:** the exact dual-T4 path completed with finite validation BCE
  `0.648608`, `481.2` MiB peak allocated memory per rank, and a projected full
  representation/validation/extraction time of `50.05` minutes. The independent
  gate withheld approval because the 4.38-second training segment was shorter than
  the five-second telemetry interval and therefore did not capture peak utilization.
  V2 changes only the telemetry sampling interval to one second.
- **Speed smoke v2:** passed the independent gate. It reproduced validation BCE
  `0.648608`, measured `3.05` seconds for 64 training studies and `0.72` seconds
  for eight 32-window validation studies, projected `36.73` minutes for the full
  representation/validation/extraction path, and used `481.2` MiB peak allocated
  memory per rank. Telemetry captured maxima of 93% and 99% utilization on the two
  T4s. The full P3 run is approved.
- **Full launch:** private Kaggle kernel
  [`rsna-knee-p3-dinov2-hierarchical-mil`](https://www.kaggle.com/code/vallepuvamsikrishna/rsna-knee-p3-dinov2-hierarchical-mil)
  version 1 was launched from the frozen implementation at Git commit `024db68`.
  Kaggle completed the worker successfully in `30.89` wall-clock minutes.
- **Representation result:** all eight epochs completed and grouped pseudo-validation
  BCE improved monotonically from `0.559016` to `0.507069`, selecting epoch 8.
  Representation plus selection took `15.46` minutes; feature extraction, expert-head
  fitting, and evaluation took `10.69` minutes after `3.96` minutes of cache hashing.
- **Expert result:** direct untouched-expert macro AUC was `0.8041672`; five-fold
  report-purged OOF macro AUC was `0.8059881`, which is `-0.0851373` below P2's
  `0.8911254`. The independently recomputed 5,000-sample paired interval was
  `[-0.1160286, -0.0577528]`, with probability positive `0.0`.
- **Diversity result:** the predeclared equal-rank P2+P3 blend scored `0.8639078`,
  `-0.0272176` below P2. Its paired interval was `[-0.0421689, -0.0133201]`, with
  probability positive `0.0002`; P3 therefore does not supply useful aggregate
  ensemble diversity.
- **Target pattern:** P3 improved only Lateral OA (`+0.05996`) and Effusion
  (`+0.01118`). Ten targets declined, led by Medial Meniscus (`-0.24639`), MCL
  (`-0.18821`), Baker's (`-0.13587`), and ACL (`-0.12868`). The rank blend improved
  only Lateral OA, Effusion, and Synovitis and still lost decisively in macro AUC.
- **Independent validation:** all artifact hashes and feature geometry passed; every
  direct and fold head replayed its saved predictions; all 58 expert rows appeared
  exactly once OOF; and every fold had zero validation-report overlap. Both T4s were
  active with utilization medians `97%` and `96%` and maxima `99%` and `100%`.
- **Tracked result:** [`experiments/results/P3.json`](../experiments/results/P3.json).
- **Decision:** reject P3 and the P2+P3 blend. This is a valid negative result rather
  than an infrastructure failure: pseudo BCE continued improving, but the learned
  DINOv2 hierarchy transferred poorly to the expert targets. Retain P2 as the P2/P3
  winner and advance the ordered roadmap to P4; do not spend a submission on P3.

## Completed P4 run

- **Run IDs:** `P4-COATNET-E2E-FOLD-{0..4}-v1`.
- **Frozen protocol:** [`P4_PROTOCOL.md`](P4_PROTOCOL.md).
- **Parent:** promoted P2 CoAtNet OOF `0.8911254`, public score `0.908`; rejected
  P3 is not used.
- **Core change:** train one full P2 backbone and target-attention head per expert
  fold using 46–47 in-fold expert rows plus group-pure pseudo training. Each fold's
  11–12 expert validation rows and all report twins contribute zero gradients.
- **Selection:** every checkpoint is selected only by grouped pseudo-validation BCE.
  Expert-validation AUC cannot choose an epoch or trigger a retry.
- **Execution:** one private T4 x2 Kaggle script per fold, preserving P2's proven
  DDP, image, optimizer, augmentation, and eight-epoch contracts. A short fold-0
  remote smoke exercises the new expert-mixing, held-out inference, shard merge,
  ledger, and manifest path before launching the five full jobs.
- **Prelaunch validation:** all five generated scripts compile and independently
  reconstruct complete 58-study OOF coverage. The version-independent grouped split
  contains 3,914 pseudo training and 435 pseudo-validation rows. Fold 2 removes the
  four known validation-report twins; folds 3 and 4 each deterministically drop one
  pseudo row to balance DDP shards. No expert row is dropped or duplicated, and all
  training/selection/held-out report intersections are empty.
- **Speed smoke result:** private Kaggle version 1 completed successfully. It trained
  62 pseudo and two expert rows through the exact fold-0 DDP branch, selected epoch 1
  at pseudo-validation BCE `0.5126671`, and wrote all 12 held-out predictions. The
  representation segment took `0.97` minutes and held-out inference `0.08` minutes
  after `2.95` minutes of full cache hashing. Both T4s reached 100% utilization. The
  independent validator passed the script/artifact hashes, complete 4,407-row role
  ledger, label/prediction rows, report isolation, and telemetry.
- **Full wave 1 result:** fold 0 and fold 1 version 1 completed and passed the
  independent per-fold validator. Fold 0 selected epoch 8 at pseudo-validation BCE
  `0.4637027` in `130.59` representation minutes; fold 1 selected epoch 8 at
  `0.4628877` in `140.58` minutes. Both produced the exact 12 held-out expert rows,
  had zero training/selection/validation report overlap, reproduced all split roles
  and artifact hashes, and sustained 100% median utilization on both T4s. Manifest
  SHA-256 values are `6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932`
  and `9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5`.
- **Full wave 2 result:** folds 2 and 3 completed and passed the independent validator.
  Fold 2 correctly excluded the four validation-report twins, selected epoch 8 at
  pseudo-validation BCE `0.4669970`, and trained for `129.61` representation minutes.
  Fold 3 selected epoch 8 at `0.4616242` and trained for `140.98` minutes. Both had
  zero forbidden report overlap, exact split and artifact hashes, correct held-out
  rows, and 100% median utilization on both T4s. Their manifest SHA-256 values are
  `93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac` and
  `9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c`.
- **Final fold result:** fold 4 completed and passed the independent validator. It
  selected epoch 8 at pseudo-validation BCE `0.4645861`, trained for `126.86`
  representation minutes, produced the exact 11 held-out expert predictions, and
  sustained 100% median utilization on both T4s. Its manifest SHA-256 is
  `e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228`.
- **Aggregate result:** all 58 expert studies appeared exactly once OOF. P4 scored
  `0.8911609` versus P2's `0.8911254`, a negligible `+0.0000355`. Its paired
  5,000-sample interval was `[-0.0149955, 0.0147569]` with probability positive
  `0.4968`; only 8/12 target deltas were nonnegative and MCL declined `-0.03855`.
  P4 alone therefore fails the declared uncertainty and worst-target gates.
- **Fixed blend result:** the predeclared equal-rank P2+P4 blend scored `0.8958211`,
  delta `+0.0046957` over P2. Its paired interval was
  `[-0.0040290, 0.0136381]`, probability positive `0.8622`, 9/12 target deltas were
  nonnegative, and the worst delta was MCL at `-0.02041`. It passes every frozen
  blend-promotion condition, although the interval crosses zero and the gain does
  not meet the strong-evidence threshold.
- **Independent validation:** all five source manifests, checkpoints, scripts, split
  ledgers, report-twin purges, hashes, fold assignments, and aggregate predictions
  passed. Every fold selected epoch 8 and recorded 100% median utilization on both
  GPUs. Aggregate manifest SHA-256 is
  `a05fbb69c839ee1e0eb05c450ebf0aa6c150bc5c25e091aa6ec13d8094d48d05`.
- **Tracked smoke result:**
  [`experiments/results/P4-SPEED-SMOKE.json`](../experiments/results/P4-SPEED-SMOKE.json).
- **Tracked full result:** [`experiments/results/P4.json`](../experiments/results/P4.json).
- **Decision:** reject P4 alone as a replacement, but promote the fixed equal-rank
  P2+P4 blend. Retain P2 as a required member, validate the five-checkpoint P4
  hidden-test inference and blend path before spending a submission, and advance the
  ordered model-research queue to P5.
- **Submission preparation:** `P4-BLEND-SUBMISSION-v1` is frozen in
  [`P4_SUBMISSION_PROTOCOL.md`](P4_SUBMISSION_PROTOCOL.md). It decodes each study
  once, runs P2 plus five P4 models over the same 32 windows, averages P4 probabilities,
  and applies the exact target-wise equal-rank P2/P4 formula. The generated script is
  deterministic and passes compile and lint checks. Two local full-model attempts
  were stopped when the 15 GiB workstation exhausted swap while materializing model
  state; neither reached inference or produced an artifact. Full execution validation
  is therefore assigned to the private three-study Kaggle T4 x2 canary, not retried
  locally.
- **Submission canary v1:** all six immutable model sources loaded, both T4s were
  assigned three models, all three visible studies completed, and the independent
  validator exactly reproduced the five-fold P4 mean and final rank blend. Total time
  was `1.77` minutes and the DICOM-plus-six-model segment took `1.1647` minutes.
  Naively scaling the measured segment to roughly 1,300 hidden studies gives `8.41`
  hours, only about 35 minutes below the nine-hour ceiling. This fails the runtime
  margin gate, so version 1 is not submitted. Version 2 uses three-study batches and
  shares resize/normalization across each GPU's three models before repeating the
  canary.
- **Submission canary v2:** output validation passed and its final `submission.csv`
  was byte-identical to version 1, but the three-study batch raised observed GPU memory
  to `12,621 MiB` and slowed the inference segment to `1.3089` minutes. Its linear
  hidden-test projection exceeds nine hours, so version 2 is also rejected. Version 3
  returns to batch size 2, retains shared preprocessing, and stores inference-only
  model weights in FP16 to eliminate repeated autocast weight conversion.
- **Submission canary v3:** version 3 completed all three studies in `1.4153` minutes;
  its DICOM-plus-six-model segment fell to `1.0076` minutes. The independent validator
  rechecked every source hash, exact five-fold P4 mean, rank blend, sample order,
  geometry, and artifact hash. Its `submission.csv` is byte-identical to versions 1
  and 2. Linear projection of the measured inference segment is `7.28` hours for
  1,300 hidden studies, leaving `1.72` hours or 19% below the nine-hour ceiling.
  Version 3 passes the release gate.
- **Leaderboard submission:** exact notebook version 3 from commit `d1e9f15` was
  accepted as competition submission `55990677` on 2026-09-03 and completed on
  2026-09-04 with public score `0.916`. This improves canonical P2's `0.908` by
  `0.008`. The verified live leader remains `0.952`, reducing our observed gap from
  `0.044` to `0.036`. Retain this as the current submitted leader and do not tune the
  fixed blend from public feedback. Tracked evidence:
  [`experiments/results/P4-SUBMISSION.json`](../experiments/results/P4-SUBMISSION.json).

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
