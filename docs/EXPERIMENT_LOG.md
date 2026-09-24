# Experiment log

Last updated: 2026-09-24

This is the durable run and decision record. The ordered plan and acceptance rules
live in [`EXPERIMENT_ROADMAP.md`](EXPERIMENT_ROADMAP.md). Status values are
`queued`, `running`, `validated`, `repeat`, `rejected`, or `blocked`.

### 2026-09-24 — P12 duration confirmation complete; do not promote

The six-run final AWS batch completed at 06:39 IST and stopped automatically.
All six checkpoints were downloaded and independently passed frozen-config,
artifact-hash, exclusion, epoch-count, held-out coverage and saved replay checks.
All four five-fold OOF sets contain exactly one prediction for each of 58 experts.
Independent pairwise ranking reproduced all 48 target AUCs; all ten matched
control/candidate pairs have identical recorded hardware/software environments.
The predeclared AWS fold2 seed2 control is used; the original Kaggle control remains.

| Seed | Eight epochs | Sixteen epochs | Difference |
|---|---:|---:|---:|
| 20260917 | 0.907588 | 0.905830 | -0.001757 |
| 20260918 | 0.903195 | 0.908377 | +0.005182 |
| Mean predictions across seeds | 0.908713 | 0.910885 | +0.002173 |

The grouped paired bootstrap gives probability of improvement 0.6233 (required
0.90), with 95% interval [-0.011340, 0.016727] from 4,999 valid replicates.
The both-seeds gate fails. Standalone mean-prediction target regressions include
synovitis -0.066906, fracture -0.048611, effusion -0.040994 and medial meniscus
-0.020433. This target diagnostic is not a deployed-blend assessment; no blend
weights were searched. Duration16 is not promoted and P8 remains the fallback.

Two reused seed2 fold0 models still lack pinned full-image replay evidence locally.
Their checkpoint hashes and memberships passed; the gap is not called a replay
success. An exact-release runtime benchmark is also absent. Statistical failure
already prevents promotion, so no additional paid replay or runtime job was launched.
Results and all selected source hashes: `experiments/results/P12-confirmation-oof.json`.
Private OOF rows remain outside Git in `outputs/experiments/P12-confirmation/`.

Both AWS hosts are stopped; retained disks/S3 still incur storage charges.
The temporary `rsna-p12-batch` role policy was removed and absence verified;
the original role remains. The temporary encrypted Kaggle parameter is absent.
The next planned training hypothesis is slot-balanced coverage at eight epochs;
this decision does not authorize an unrecorded recipe change or automatic launch.

### 2026-09-23 — AWS fold-3 batch independently validated

All four runs completed around 06:12 IST; the host stopped automatically.
Downloaded all checkpoints and predictions and independently verified their hashes,
exact frozen configurations, exclusion memberships, full 11-study held-out coverage,
original-report checkpoint selection and saved replay evidence (max error1.11e-16).
Eight versus sixteen epochs improved report-validation BCE in both seeds:
0.456939 to 0.448032 and 0.455669 to 0.447950. This is not expert OOF promotion evidence.
Seven new Kaggle runs plus four AWS runs are validated; five configurations remain.

Kaggle quota resets in 66 hours per the user. The continuation includes those five
configurations plus an AWS repeat of the fold2 seed20260918 eight-epoch control,
so its duration comparison uses the same execution backend. The original Kaggle
control is preserved. The fixed continuation permits no additional configurations,
uses separate `results/final/` storage, and stops after 36 hours or success/failure.
The original g6.2xlarge had no capacity across five supported zones; us-east-1e
returned Unsupported. A g6.4xlarge fallback hit the eight-vCPU account quota. The quota-compatible
g5.2xlarge (A10G) launched in us-east-1f after three further capacity failures.
Verified pricing is USD 1.212/hour, maximum 36-hour compute USD 43.632, within the
existing USD 80 reserve. Host i-0f1a75d1094f23f27 passed its fresh A10G smoke (33 seconds) and all
12 replay predictions (maximum error9.19e-17); the first full eight-epoch fold2
seed2 control started. The previous host remains stopped. Temporary Kaggle
credentials were removed from both the host and SSM after cache staging.
Verified automatic shutdown: September24 17:53:43 UTC (23:23 IST).

### 2026-09-22 13:43 UTC — Two completed Kaggle fold-2 runs validated

Downloaded duration16 seed20260917 and eight-epoch report seed20260918.
Both passed local checkpoint/prediction hashes, frozen recipe and split checks,
zero prohibited group overlap, complete 12-study held-out coverage, and original
report-target checkpoint selection. Their saved full image replays cover all
12 rows each with maximum absolute error 1.11e-16; no new local GPU replay ran.
Duration16 selected epoch15 with report BCE0.447216 (288.03 minutes total);
the eight-epoch control selected epoch8 with BCE0.455518 (150.29 minutes).
These are different seeds and must not be treated as a matched duration comparison.

Seven of the sixteen new confirmation jobs are now validated on Kaggle. Four
fold-3 jobs are assigned to the existing AWS batch, whose live status was not
refreshed here; five additional jobs remain unlaunched. No jobs were launched,
no dispatcher restarted, and no promotion or competition submission occurred.
Aggregate evidence and artifact hashes are recorded in
`experiments/results/P12-confirmation-fold2-retrieval.json`. Complete matched
five-fold OOF, promotion checks, and final deployment runtime validation remain.

### 2026-09-22 — Switch to manual Kaggle batches at user request

Stopped `rsna-p12-confirmation.service`; host verification confirms the service
inactive and no dispatcher process. Existing remote jobs remain live RUNNING:
duration16 fold2 seed20260917 and report fold2 seed20260918. Five of 16 new runs
are locally validated, two run remotely, and nine are unlaunched. The queue state
is retained for bookkeeping, with dispatch mode marked manual. No new job was
launched during this switch. Future batches are launched manually when the user
returns after completion; do not restart automatic dispatch or provision a cloud
dispatcher. This preference is recorded in `AGENTS.md`.

### 2026-09-22 — Five confirmation jobs complete; dispatcher moved to user service

Live Kaggle checks confirm duration16 fold1 seed20260918 and report fold2
seed20260917 COMPLETE. Five of 16 new jobs have now finished; three were already
locally validated and two await retrieval/validation. Eleven had not launched.
The detached dispatcher was absent in the host process context with no new saved
error. It did not reliably survive between checks.

Resumed the same queue under the native user service manager as
`rsna-p12-confirmation.service`, with its working directory set to this repository
and stdout/stderr appended to `outputs/experiments/P12-confirmation.log`.
Verified `ActiveState=active`, `SubState=running`, `ExecMainStatus=0` in a separate
host call. The existing exclusive lock, two-job concurrency limit, finite 72-hour
bound, and full replay gates remain unchanged. The service does not submit to the
competition or provision paid compute. It still requires this host to stay online;
it is not a boot-enabled or recurring job. Do not start a competing dispatcher.

Read status with `rtk systemctl --user status rsna-p12-confirmation.service` from
the host context; the sandbox process view cannot establish whether it is alive.

### 2026-09-21 — Confirmation dispatcher recovery

Live checks found both previously active jobs COMPLETE: duration16 fold1 seed
20260917 and report fold1 seed20260918. Together with the already validated first
control, three of 16 new runs had finished on Kaggle, while 13 had not launched.
The queue file had stopped advancing without a new recorded error. Resumed the
existing queue as a detached host process; verified it alive from the host process
context. No completed model was resubmitted. The duration fold then passed local
artifact/replay validation; the second control was downloading at the snapshot.
The dispatcher will refill slots after retrieval/validation. This interruption
invalidates the original estimate measured from initial dispatch. Current snapshot:
`experiments/results/P12-confirmation-status.json`; live queue remains under
`outputs/experiments/P12-confirmation/queue.json`. No paid compute or submission.

### 2026-09-19 — Full two-seed, five-fold duration confirmation launched

**Queue recovery:** a temporary DNS failure during a Kaggle status request stopped
the local dispatcher. Live checks showed the first eight-epoch fold COMPLETE and
the paired 16-epoch fold RUNNING. Added bounded retries for transient read-only
status/download errors, preserving stop-on-error behavior for non-transient errors
and every uncertain push. Resume preserves the original error and accepted jobs.
The dispatcher was restarted to retrieve/validate the completed fold and fill the
free slot. Validation: 23 tests pass, including transient retry and permission-error
behavior; lint clean. This interruption delays the original wall-clock estimate.

Frozen queue `experiments/P12-confirmation.json` contains 16 remaining runs:
folds 1–4, seeds 20260917/20260918, fastMRI/report at 8 and 16 epochs. The four
completed fold-0 models are reused with pinned manifest hashes. All recipes and
model/data source hashes match the corresponding fold-0 run. No expert outcomes
were used to modify these recipes.

`python3 -m scripts.rsna_next.confirm_duration` runs the finite queue with a maximum
of two active Kaggle jobs, a 72-hour limit and an exclusive local lock. Each new
run performs full prediction replay after training. Local membership/hash/replay
checks must pass before its queue status becomes validated. The queue stops on
failure or uncertain launch, saving errors in
`outputs/experiments/P12-confirmation/queue.json`; its log is
`outputs/experiments/P12-confirmation.log`. Reruns never blindly repeat a launch
left in `submitting` state. Local host availability is required for dispatch.

The first matched pair (fold 1, seed 20260917) was accepted as Kaggle v1. The other
14 jobs are pending and will launch as slots free. After all jobs validate, four
five-fold OOF sets are assembled. Paired bootstrap, deployed-blend target checks,
remaining fold-0 replays and exact final notebook/runtime validation remain before
promotion/submission. No automatic competition submission or paid provisioning.
Validation: 22 tests pass, lint clean, all 16 generated bundles compile and match
the fixed recipes. Generated split-bearing source remains excluded from Git.

### 2026-09-19 — Two-seed duration screen passes

Second-seed duration16 is live COMPLETE. All 16 epochs, saved checkpoint and
prediction hashes, 12 held-out expert rows, memberships, matched source/settings
and zero prohibited overlap passed independent checks. Best checkpoint is epoch
15: original report-validation BCE **0.447302**, versus the matched eight-epoch
control **0.457841** (delta **-0.010538**). Runtime was 317.58 minutes for
training/evaluation/prediction. Full second-seed image replay remains pending.

Both seeds support the duration change: seed 20260917 **0.456168 → 0.448599**;
seed 20260918 **0.457841 → 0.447302**. The frozen screening gate (both seed BCE
improvements plus the first duration checkpoint's full replay) passes. Advance to
five-fold OOF confirmation with both matched seeds; no expert AUC was used to
choose this screen and no recipe is promoted or submitted yet. Evidence:
`experiments/results/P12-duration16-f0-seed2.json` and
`experiments/results/P12-duration-two-seed-screen.json`.

### 2026-09-19 — Second-seed duration confirmation dispatched

**Subsequent status check:** duration replay COMPLETE and independently verified
against all pinned hashes: 12 rows, maximum error 1.11e-16. The launcher then
successfully pushed the second-seed 16-epoch candidate, now live RUNNING.
The second-seed eight-epoch control is COMPLETE and passes artifact, membership,
zero-overlap and prediction-coverage checks: best epoch 8, original report BCE
**0.457841**, 148.15 minutes training/evaluation/prediction. Its full checkpoint
image replay remains pending. No second-seed duration conclusion is possible
until the 16-epoch candidate finishes. Evidence:
`experiments/results/P12-report-f0-seed2.json` and
`experiments/results/P12-duration16-replay.json`.

The existing student replay builder now accepts `--duration16`; the pinned
duration checkpoint replay and seed 20260918 eight-epoch fastMRI/report control
are both live RUNNING as private offline Kaggle v1 jobs. The paired 16-epoch
seed-20260918 candidate is frozen with identical settings except duration and its
full learning-rate schedule. Kaggle rejected its initial launch with
`Maximum batch GPU session count of 2 reached.`

A bounded one-shot local launcher waits for the replay, validates all 12 rows
against the pinned manifest/checkpoint/prediction hashes, then pushes the candidate.
It records errors in `experiments/results/P12-seed2-launch.json`, with a full log
at `outputs/experiments/P12-seed2-launch.log`. The candidate is locally queued,
not yet running on Kaggle. No paid resource was provisioned.

The gate to five-fold confirmation is a passing duration replay and improved
original report-validation BCE in both matched seeds. Full OOF, paired promotion
checks and exact offline runtime validation still precede a new submission.
Design: `experiments/results/P12-duration-confirmation-design.json`.
All three bundles compile; configuration/source/manifest checks and builder lint
pass. Generated source with embedded splits remains outside Git.

### 2026-09-19 — Duration screen improves pseudo validation; student replays pass

Live Kaggle checks confirm duration16 fold 0 and both prior student replays COMPLETE.
All downloaded hashes, training/validation memberships, 12 held-out expert rows and
checkpoint-selection history passed independent validation, with zero prohibited
group overlap. The 16-epoch run selected epoch 16: unchanged report-validation BCE
**0.448599**, versus eight-epoch fastMRI/report **0.456168** (delta **-0.007569**).
Training/evaluation/prediction took 321.35 minutes. Configuration, source hashes,
labels, cache, seed and memberships match except the predeclared duration/schedule.
This supports a second matched-seed duration comparison; it is not OOF AUC or
promotion evidence. The 16-epoch checkpoint still requires independent image replay.

Full ImageNet and distilled checkpoint replays passed on all 12 saved predictions
each, maximum error 1.11e-16, with pinned manifests/checkpoint/prediction hashes.
Evidence: `experiments/results/P12-duration16-f0.json` and
`experiments/results/P12-student-replay.json`. No new jobs, paid compute, or
competition submissions were launched during this status check.

### P12 — Duration screen and remaining student replays launched

Private offline Kaggle version 1 jobs `rsna-p12-student-replay` and
`rsna-knee-next-student-duration16-0-s20260917` both reached live RUNNING.
The replay pins the completed ImageNet and distilled checkpoint manifests and
checks all 12 held-out predictions from each model. The duration experiment uses
fastMRI/original-report labels, fold 0, seed 20260917, baseline head and eight
globally sampled windows. Only epochs (8 to 16) and the corresponding full-duration
OneCycle schedule change. Source hashes and all other configuration fields match
the completed eight-epoch control. This is a screen, not multi-fold promotion.

Frozen comparison: `experiments/results/P12-duration16-design.json`.
Both bundles compile, satisfy private/offline/title constraints, and the replay
builder passes lint. No paid compute or competition submission was made.

### 2026-09-18 — Full replay and controlled P12 comparison launched

**Distilled completion:** live COMPLETE; downloaded checkpoint, all eight epochs
and 12 held-out expert predictions pass hash, split and zero-overlap checks. Labels
match the replay-verified teacher manifest. Configuration, initialization, cache,
training/validation membership and model/training source hashes match fastMRI/report
except the intended label change. Best unchanged report-validation BCE is
**0.455944**, versus fastMRI/report **0.456168** and ImageNet/report **0.466329**;
all select epoch 8. Distillation's 0.000224 reduction is a tiny single-fold/seed
difference, not grounds for promotion. Full distilled/ImageNet checkpoint replay
and further controlled screens remain. Evidence: `experiments/results/P12-distilled-f0.json`.

**Completion check:** replay/labels and ImageNet/report fold 0 both COMPLETE.
All six checkpoints reproduced all 6,386 saved study predictions with maximum
absolute error 1.11e-16. Independently recomputed labels agree to 1e-12. The
ImageNet run passes hashes, exclusions, epoch history and held-out coverage checks;
configuration, training/validation IDs, targets and cache match the fastMRI control.
Best original report-validation BCE is ImageNet 0.466329 versus fastMRI 0.456168
(both epoch 8). This favors fastMRI in the single-fold screen only; full ImageNet
replay and multi-fold/seed confirmation remain pending.

The distilled launch was rejected with HTTP 400: `The title cannot exceed 50
characters.` The builder now shortens long notebook names and checks the limit;
the regression case is tested. Recovered distilled version 1 is live RUNNING at
`vallepuvamsikrishna/rsna-student-fastmri-distilled-0-s20260917`.
The prior failure is retained in `experiments/results/P12-distilled-launch.json`.
Evidence: `experiments/results/P12-full-replay.json` and
`experiments/results/P12-initialization-f0.json`.

Private offline Kaggle version 1 jobs `rsna-knee-p12-replay-labels` and
`rsna-knee-next-student-imagenet-report-0-s20260917` both reached live RUNNING.
The first pins the five teacher manifests and completed fastMRI/report baseline
manifest, replays every saved prediction, then builds fixed 50/50 report/image
labels only after all replays pass. The second changes only initialization against
the completed fold-0 baseline (same seed, split, labels, head and eight epochs).

The one-shot `scripts/rsna_next/advance_p12.py` coordinator waits up to four hours,
downloads and verifies six complete replays and their original artifact hashes,
recomputes labels locally and checks probabilities/provenance before launching
the fastMRI/distilled fold-0 arm. State/errors are saved in
`experiments/results/P12-distilled-launch.json`; the process log is
`outputs/experiments/P12-distilled-launch.log`. The distilled arm is **not launched
yet**. No new student OOF claim, promotion, paid compute or submission is implied.
All 21 existing tests and lint on the two new scripts passed.

### 2026-09-18 — Teacher bank complete; baseline launch recovered

**Subsequent live check:** baseline student fold 0 is now COMPLETE. Independent
artifact checks passed: 3,955 training studies, 435 shared validation studies,
12 held-out expert predictions, zero prohibited group overlap, matching checkpoint
and prediction hashes. All eight epochs completed; original report-validation BCE
fell from 0.542743 to 0.456168 and selected epoch 8. Total logged time was 160.06
minutes (139.31 minutes training/evaluation/prediction). This single-fold result
is not full OOF evidence or promotion; full-image replay remains pending.
Evidence: `experiments/results/P12-fastmri-baseline-f0.json`.

Live Kaggle checks confirm both teacher banks (012 and 34) COMPLETE. Downloaded
all five checkpoints and predictions; independent validation passes artifact
hashes, full fold memberships, zero prohibited group overlap and exactly one
held-out teacher per 3,909 pseudo-training studies. Each shared validation/expert
study has all five teacher predictions. All folds selected epoch 8, with original
report-validation BCE 0.45888–0.46038. This is teacher label-source validation,
not student OOF or a promotion decision; full checkpoint image replay remains.
Evidence: `experiments/results/P10-teacher-bank-validation.json`.

The prior coordinator completed duplicate verification, smoke replay and both
teacher launches, but did not finish the baseline launch. Its exact failure reason
was not retained. The baseline slug returned 404; its existing bundle was pushed
successfully as version 1 and live status verified QUEUED on Sep 18. No paid compute
or competition submission was made. P8 remains the fallback.

### 2026-09-17 — P10 full header audit

All 24,371 series across 4,407 studies were read without errors. Independent checks
confirmed artifact hashes, unique IDs, volume aggregation and coverage of all 47
scanner-model domains in the new 240-study benchmark selection. The earlier P8
runtime sample covered 33 domains and omitted the native-size extremes. The
replacement benchmark bundle is prepared, not launched. There are 407 series
with plane-normal agreement below 0.85 (365 coronal, 28 axial, 14 sagittal), across
385 studies; P14 must inspect these flags rather than assume metadata errors.
Evidence: `experiments/results/P10-header-audit.json`.

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
| P8 | **validated / blend promoted / scored** | Fold-safe P2/P4/P6 image-teacher relabelling and five student folds | Submission `56077025` scored 0.930, +0.007 over P6; live gap 0.024 |
| P9 | **validated / rejected** | One shared S3D true-3D representation plus five cached-feature heads | OOF 0.753855; frozen 15% blend declined to 0.907284; retain P8 |

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
  All predeclared OOF gates passed, advancing the frozen candidate to the mandatory
  hidden-test runtime canary with at least 0.9 hours of projected headroom. Aggregate
  evidence: [`experiments/results/P8.json`](../experiments/results/P8.json).
- **Submission canary:** exact private Kaggle version 1 from commit `a669b51`
  completed successfully. Its three-study/15-series DICOM-plus-model segment took
  `1.108033` minutes, projecting to `8.002461` hours for 1,300 hidden studies and
  `0.997539` hours of headroom. This passes the frozen 0.9-hour gate by about 5.85
  minutes. Independent validation recomputed the five-fold P8 mean and final blend,
  verified every source/artifact hash, schema, row order, selection geometry, and
  dual-GPU P6 alternation, and found no traceback or OOM. P8 is promoted, and the
  canary made exact version 1 eligible for competition submission. Evidence:
  [`P8-SUBMISSION-CANARY.json`](../experiments/results/P8-SUBMISSION-CANARY.json) and
  [`P8_SUBMISSION_PROTOCOL.md`](P8_SUBMISSION_PROTOCOL.md).
- **Leaderboard submission:** after explicit approval, exact private Kaggle version 1
  was accepted as competition reference `56077025` with the correct commit `a669b51`
  in its description. It completed at public score `0.930`, improving P6 by `0.007`,
  P4 by `0.014`, and P2 by `0.022`. The live leader remained `0.954`, so the gap is
  now `0.024`. Retain P8 as the current submitted leader and advance to P9 without
  changing the frozen blend. Evidence:
  [`P8-SUBMISSION.json`](../experiments/results/P8-SUBMISSION.json).

## Current P9 execution

- **Architecture decision:** do not attempt OrthoDiffusion reproduction. Its public
  implementation describes three orientation-specific 3D diffusion backbones, but
  the pretrained knee checkpoints and 15,948-exam source corpus used by the paper
  are not publicly packaged with the code. P9 instead uses the official Torchvision
  S3D Kinetics-400 checkpoint: a compact 8.32-million-parameter separable 3D CNN.
- **Frozen protocol:** [`P9_PROTOCOL.md`](P9_PROTOCOL.md). One shared S3D backbone
  processes five uniformly resampled 16-frame clips per study at 160 px and produces
  five 1,024-dimensional slot features. Target-specific attention fuses the slots.
  The representation sees no expert labels.
- **Fold-safe pseudo supervision:** `0.50 report + 0.25 P2 direct + 0.25 P6 direct`.
  P4 is intentionally excluded from the shared representation because its five
  checkpoints carry fold-specific expert gradients. Expert rows remain unchanged.
- **Prelaunch state:** the official 33,524,997-byte checkpoint was strictly loaded
  into `torchvision.models.video.s3d` and pinned at SHA-256
  `d76dad2f4b2a6701ad7f64e85b294b2d1989f1ea0d3f8b240446ed2152850bbb`.
  The P8 teacher manifest and P2/P6 arrays passed hash, row-order, shape, and
  no-expert-gradient checks. Canonical source, generated smoke/full kernels, syntax,
  lint, metadata, pseudo formula, and expert preservation all pass locally.
- **Speed smoke result:** the private initialization Dataset's remote checkpoint is
  exactly 33,524,997 bytes. Kaggle kernel
  `vallepuvamsikrishna/rsna-knee-p9-s3d-speed-smoke` version 1, frozen at commit
  `98d10af`, completed successfully. Both DDP ranks executed the real five-clip
  forward/backward path; training and validation losses were finite; peak allocation
  was 912.63 MiB per rank; and the conservative six-epoch representation projection
  is 0.9474 hours. Independent validation found no traceback or CUDA OOM and passed
  every frozen advance gate. Evidence:
  [`P9-SPEED-SMOKE.json`](../experiments/results/P9-SPEED-SMOKE.json).
- **Full run result:** exact version 1 from commit `98d10af` completed successfully
  at `vallepuvamsikrishna/rsna-knee-p9-s3d-specialist`. Epoch 5 minimized grouped
  pseudo-validation BCE at `0.585310`. Representation training took 15.94 minutes;
  feature extraction and five heads took 1.50 minutes; total model time was 17.44
  minutes. Both GPUs reached 100% utilization, with full-run utilization means of
  57.19% and 63.21% including hashing and I/O, and maximum observed memory of 1,301
  MiB.
- **Independent validation:** all 17 artifact hashes, the generated source hash,
  checkpoint and teacher hashes, 4,407-row safe-target equation, unchanged expert
  targets, P4 exclusion, `(4407, 5, 1024)` feature cache, five report-purged splits,
  five saved-head prediction replays, direct-head replay, 58-row OOF coverage,
  telemetry, and completion log passed. There was no traceback or CUDA OOM. Source
  manifest SHA-256 is
  `c0e9b4b7f6a8ddcf44968451c116523994d8038a0367b047919635a20c6d77f5`.
- **OOF result:** the direct representation head scored `0.662476`; five-fold expert
  OOF reached `0.753855`. Adding the frozen 15% P9 rank component reduced the current
  P8 blend from `0.909798` to `0.907284` (`-0.002514`). Only 6/12 target deltas were
  nonnegative, MCL was worst at `-0.024943`, and the 5,000-sample paired-bootstrap
  probability of a positive macro delta was `0.2126`.
- **Decision:** all four mandatory promotion gates fail. Reject P9 without tuning its
  blend weight or selecting targets on the 58 expert rows. Do not build a hidden-test
  runtime canary or spend a competition submission. Retain P8 as the promoted system;
  the ordered P0–P9 ladder is complete. Evidence:
  [`P9.json`](../experiments/results/P9.json).

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

- **2026-09-17, image-group split correction:** Added transitive report/image
  exclusion groups and a hash-bound sidecar, preserving the original validation
  membership and expert folds. One pseudo-training twin is quarantined: 3,909
  pseudo train, 435 validation, 58 experts, four expert-report twins and one
  validation-image twin. Raw verification and full prediction replay launched in
  private Kaggle job `rsna-knee-p10-duplicate-replay`; full builds enforce the gate.
- **2026-09-17, P11 closed:** Replayed all 60 fold heads, each within 1e-6 of its
  saved predictions. Mean two-seed position gains are +0.000278 on P2 and +0.000234
  on P6; seed-ensemble bootstrap probabilities are 0.644 and 0.373, respectively.
  Hierarchical pooling fails both-seed/target-regression checks. Neither variant
  advances; preserve baseline head. Complete summaries: `P11-complete.json` and
  `P11-replay.json` under `experiments/results/`.
- **2026-09-17, P12 preparation:** ImageNet student fold-0 smoke submitted with the
  corrected split and automatic prediction replay. An initial HTTP 400 submission
  failed; compact split packaging was subsequently accepted as version 1. This is
  an infrastructure event, not model evidence. Prepared two sequential teacher
  banks (0–2 and 3–4) plus fastMRI/report student fold-0 baseline, contingent on gates.

- **2026-09-17, completed Kaggle output review:** All three P10 jobs reached
  COMPLETE. Teacher smoke independently passed checkpoint/prediction hashes,
  original-report checkpoint selection, group memberships and finite predictions:
  two GPUs, 64 training / 16 validation / 12 prediction rows; reported rank-zero
  peak 7,108.7 MiB. The 0.559-minute training/evaluation time excludes cache setup.
  Independent image replay is still pending; this is not model-quality evidence.
- **2026-09-17, pixel audit finding:** Verified 4,407 unique fingerprint rows and
  acquisition totals against source metadata. Found two studies in one identical
  selected-cache-pixel group; the sole near-duplicate candidate is the same pair.
  Their reports differ, and one is pseudo-validation while the other is pseudo-train
  (teacher holdout 1). This exposes four full teacher folds to image overlap.
  Neither row occurred in the smoke training/validation subset. Full training is
  held until source-image/cache verification and versioned split correction; selected
  pixel equality alone is not a claim of patient identity. Results are recorded in
  `experiments/results/P10-pixel-audit.json` and `P10-teacher-smoke.json`.

- **2026-09-17, P10–P15 implementation:** Added the [active runbook](P10_P15_PROTOCOL.md),
  machine-readable recipe matrix and $500 reservation ledger, grouped teacher/student
  splits, DDP image trainer, anatomical heads, report/image provenance compiler,
  pinned offline Qwen extractor, native DICOM cache builder, independent replay,
  rank-blend/promotion checks and hashed offline deployment runner. These are new
  experimental paths; the historical P8 submission is retained.
- **2026-09-17, P10 audit:** Materialized 3,910 pseudo train / 435 shared pseudo
  validation / 58 expert / four excluded report-twin studies. Five teacher holdouts
  each contain 782 pseudo-training studies. Reconstructed 400/435 historical P8
  pseudo-validation exposures and four representation-stage expert-report twins.
  [Tracked split audit](../experiments/results/P10-contract.json).
- **2026-09-17, initial execution:** Launched private free Kaggle teacher smoke,
  exact P8 240-study representative runtime benchmark, and selected-pixel duplicate
  audit. Authoritative RUNNING states observed; acceptance/running is not completion.
  No paid compute provisioned and no new competition submission made.
- **2026-09-17, P10 runtime completed:** Exact P8 inference processed 240 selected
  training studies in 8.217 minutes on two T4s. The 1,300-study projection including
  25% inference headroom is 0.876 hours. Verified all hashes, 240 unique prediction
  rows, all 32 windows per study, fold averaging, and final blend arithmetic to
  1.11e-16. [Tracked result](../experiments/results/P10-runtime.json). This is a
  substantially stronger benchmark than three canary studies, but native size and
  scanner-domain stratification remain outstanding; do not transfer its timing to
  a changed release without another benchmark.
- **2026-09-17, P11 partial:** Baseline P2-cache first seed scored 0.89182733;
  slot/position scored 0.89188653. Paired report-group bootstrap probability of
  positive gain was 0.52871, CI [-0.003085, 0.003105]. Inconclusive; no promotion.
  [Partial evidence](../experiments/results/P11-partial.json). Two-seed P2/P6 matrices
  preserve completed fold checkpoints. Local process interruptions are execution
  failures, not model results; resume uses archived exact source modules.
  The P2 hierarchical first seed subsequently scored 0.89236092, with bootstrap
  probability 0.57532 and CI [-0.006770, 0.008023]; it is also inconclusive.
- **2026-09-17, geometry checks:** One real local study produced 32 valid native
  triplets across all five slots with no series error. Added synthetic RLE-compressed
  DICOM tests, canonical channel-order checks, empty-slot fp16 checks, corrupt-teacher
  rejection, independent blend arithmetic and training/deployment cache agreement.
  Full-mount geometry QA, GPU prediction replay and final runtime gates remain open.

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

- **2026-09-22, P12 AWS fallback:** User authorized paid compute and selected
  `us-east-1` after Mumbai capacity failures. `g6.2xlarge` L4 instance
  `i-0d0ec06ac8023cffe` in `us-east-1c` costs USD 0.9776/hour. Four matched
  fold-3 jobs are reserved for AWS; the first eight-epoch control is running.
  GPU smoke and all 11 saved predictions passed independent integrity, membership,
  and replay checks (maximum error 9.71e-17); measured full-training GPU utilization
  was 98-100% across a one-minute sample. Host deadline remains September 23
  06:32 UTC. No full training job has completed and no candidate is promoted.
  Software setup and buffered-download failures were repaired before full training;
  they are infrastructure failures, not modelling results. Source and aggregate
  evidence are in the AWS runbook and `experiments/results/P12-aws-batch.json`.
  The unused Mumbai instance and disk were deleted with explicit approval; the
  temporary Kaggle token was removed from both the active host and SSM.

- **2026-09-22 10:36 UTC, P12 AWS status:** First fold-3 eight-epoch control
  (seed20260917) completed in 173.87 minutes. Local artifact/membership validation
  and all 11 replayed predictions passed (max error 1.11e-16). Best original-report
  BCE was 0.456939 at epoch 8; this is not an OOF promotion result. The matched
  16-epoch run completed epoch 1 and is training epoch 2 at 100% GPU utilization.
  Two second-seed runs remain queued within the fixed batch. Early measured
  throughput suggests another 14-15 hours for the batch; the existing shutdown
  deadline remains active.
