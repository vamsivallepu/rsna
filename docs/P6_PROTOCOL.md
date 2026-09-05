# P6 protocol — fastMRI+ localization transfer

Frozen audit stage: 2026-09-04

P6 tests whether supervised knee-pathology localization before RSNA training can
improve the promoted CoAtNet representation. P6-A is an annotation, provenance,
access, and transferability audit. P6-B source pretraining has completed and passed
independent validation on private AWS. Its selected backbone is now an input to the
matched downstream comparison, not a promoted RSNA model.

## Hypothesis and parent

- **Hypothesis:** lesion-localization supervision on sagittal knee MRI can make the
  P2 CoAtNet backbone attend to small structural abnormalities that study-level
  report labels do not localize.
- **Parent:** the promoted P2+P4 equal-rank blend, OOF macro AUC `0.8958211` and
  public score `0.916`. P2 remains the standalone architecture used for the controlled
  initialization comparison.
- **P6-A run ID:** `P6A-FASTMRI-PLUS-AUDIT-v1`.
- **P6-B run ID:** `P6B-FASTMRI-PLUS-LOCALIZATION-v1`.

## P6-A provenance and access gate

- Annotation source: Microsoft `microsoft/fastmri-plus`, commit
  `67ed9a68ee33d3197f93893e8125688470c4a577`, MIT licensed.
- Official `knee.csv` SHA-256:
  `c1f4a083646cec815b67f005f8d7764313d1b82a028b90c272a4d000330d065b`.
- Official `knee_file_list.csv` SHA-256:
  `4b09e5523709815d5314a4f652882c40f8c14af26b67a00ea769054ca7e1d6b7`.
- The small Kaggle annotation package was verified byte-for-byte against those two
  official files. It contains no images.
- The released annotation file contains 16,154 slice-level boxes, 13 study-level
  artifact rows, 22 labels including artifact, and 974 annotated volumes. The
  released reviewed-file CSV has 1,173 rows, but its final row is the literal footer
  `Knee data`, not a volume identifier. Filtering by the released `file<digits>` ID
  contract yields 1,172 unique reviewed volumes, matching the paper; 198 have no row
  in `knee.csv`. The audit asserts this footer explicitly rather than silently treating
  arbitrary malformed rows as data.
- The release has six boxes with `y=-1`, one box with zero height, and one raw label
  (`Joint Effusion `) with trailing whitespace. P6-B must strip label whitespace,
  clamp boxes to the actual reconstructed image bounds, discard boxes that remain
  zero-area, and record affected row identities and counts in its manifest.
- The underlying fastMRI images are separately controlled. They may be used only
  after the user has accepted the current fastMRI data-use agreement and obtained
  the data through the official access path. They must not be committed, republished,
  or copied into a public Kaggle Dataset. Unofficial Kaggle image mirrors are not an
  approved P6 source.

The reproducible audit is:

```bash
python3 scripts/p6_fastmri_plus_audit.py \
  --annotations /path/to/official/Annotations/knee.csv \
  --reviewed-files /path/to/official/Annotations/knee_file_list.csv \
  --output-dir outputs/experiments/P6-A-fastmri-plus-audit
```

## Transferability finding

The annotation CSV alone is not a sound spatial prior for the RSNA pixels:

- it has boxes but no image dimensions, pixel spacing, or volume depth;
- the official conversion flipped pixel arrays up/down before annotation;
- the released knee images and boxes are sagittal, while P2 uses five acquisition
  slots spanning sagittal, coronal, and axial series;
- meniscus side and cartilage compartment are absent; fracture/contusion/dislocation
  share one label; Synovitis has no defensible direct mapping; and Baker's has only a
  broader periarticular-cyst proxy;
- labels come from a single radiologist without other views and are explicitly not
  an exhaustive clinical ground truth.

Therefore P6-A does not inject coordinate priors or convert boxes into RSNA targets.
The tracked crosswalk exists to expose ambiguity, not hide it.

## Frozen P6-B design after authorized image access

P6-B will use only official knee HDF5 volumes corresponding to the hash-pinned
reviewed-file list. Read `reconstruction_rss` and reproduce the pinned fastMRI+
example's labeling frame exactly with `pixels[:, ::-1, :]`; verify random overlays
before training. Split by volume, never slice. The split manifest must be hash-pinned
and no volume may cross train/validation. Apply the declared whitespace and box
sanitation only after retaining the original row values for auditability.

The source task initializes the same CoAtNet-RMLP2 backbone used by P2. A lightweight
feature-pyramid head predicts a class-agnostic lesion-center heatmap plus box size on
annotated slices. This avoids pretending the ambiguous 22-to-12 label crosswalk is
exact. A secondary 21-label presence head may be logged diagnostically, but it cannot
select the checkpoint or alter RSNA targets. Background loss is downweighted because
the single-reader annotations are not exhaustive. Geometric transforms must update
boxes exactly; no left/right transform is allowed.

Select one source checkpoint by held-out-volume localization loss and lesion-center
recall. Then discard the localization head, initialize P2's backbone with the selected
weights, and rerun the otherwise unchanged P2 training and validation contract. The
matched control starts from the original ImageNet checkpoint with the same code,
seed, batches, and runtime environment. P2 report-grouped pseudo validation selects
the representation checkpoint; the 58 expert rows remain excluded from selection and
are evaluated only through the existing five-fold OOF procedure.

## Gates

The user supplied an official NYU fastMRI download email dated 2026-09-04, so the
agreement/access confirmation gate is satisfied. Its temporary signed URLs are not
stored in Git, tracked artifacts, notebook source, or logs. P6-B may launch only when
the remaining checks pass:

1. the selected official download URL is retrieved only from an encrypted AWS SSM
   SecureString by the least-privilege EC2 role rather than embedded in code;
2. source filenames match the official reviewed list and remain private;
3. annotation and reviewed-list hashes match P6-A;
4. the official HDF5 archive matches NYU's published SHA-256 and overlays pass after
   applying the documented labeling-time up/down flip exactly once;
5. a 128-volume smoke establishes decoding correctness, memory use, GPU utilization,
   throughput, and a viable projected full runtime.

On 2026-09-04, the official single-coil validation archive passed its published
SHA-256, extracted to 199 reviewed `file*.h5` volumes, and completed the deterministic
128-volume data smoke at code commit `2cc14e7`. All 128 volumes decoded, the 103/25
volume-level train/validation split had zero overlap, all 1,605 selected boxes mapped
to present slices, and no box was discarded after clamping. Sixteen private overlays
were generated, transferred through a loopback-only SSM tunnel after explicit user
authorization, and verified against their manifest hashes. All 16 passed visual
alignment with no systematic flip, mirror, slice offset, or coordinate shift. This
validates the data and visual paths but does not yet satisfy the GPU utilization,
peak-memory, throughput, or projected-runtime part of gate 5. The tracked gate result is
[`P6-B-SMOKE.json`](../experiments/results/P6-B-SMOKE.json).

GPU smoke version 1 established that batch 8 fits comfortably (6.88 GiB peak allocated)
and measured 31.64 images/s, projecting eight source epochs to about 0.73 hours. It is
recorded as a harness repeat because utilization sampling incorrectly included lazy
worker startup and the fixed-batch loss comparison used training-mode running
statistics. Version 2 corrects those measurements without changing the source task,
cohort, backbone, FPN, loss, or promotion contract.

GPU smoke version 2 at commit `2888ee4` passed every gate. It sustained 31.68
images/s at batch 8 with 99.44% mean GPU utilization (99% p10), 6.89 GiB peak
allocated memory, finite losses, and a 79.21% fixed-batch loss reduction. At that
measured rate, 82,984 images across eight projected source epochs take approximately
43.66 minutes, excluding one-time archive download and extraction. The official
single-coil training archive may therefore be staged for full source pretraining.
The tracked result is
[`P6-B-GPU-SMOKE-v2.json`](../experiments/results/P6-B-GPU-SMOKE-v2.json).

## Frozen full-source execution

Before the training archive completes, the full source run is frozen as follows:

- the official archive boundary is the split: 973 training volumes and 199 held-out
  validation volumes, together matching all 1,172 reviewed file IDs with zero overlap;
- use every unique annotated slice plus four fixed 20–80% depth samples from each
  reviewed volume with no annotation row; exclude the five volumes containing only
  study-level artifact rows rather than treating them as negatives;
- sanitize all 16,154 released boxes under the audited rules, retaining 16,153 and
  discarding the single zero-area box after clamping;
- train the exact 73,881,264-parameter P2 CoAtNet-RMLP2 backbone and the validated
  985,859-parameter four-level class-agnostic FPN for eight epochs at batch 8;
- use FP16, channels-last tensors, fused AdamW, OneCycleLR, backbone learning rate
  `3e-5`, head learning rate `3e-4`, weight decay `0.02`, background weight `0.10`,
  and box-size loss weight `0.10`;
- allow intensity gain/offset only. No geometric or left/right augmentation is used,
  so the audited box geometry cannot drift;
- select the checkpoint by minimum held-out-volume localization loss, with top-K
  lesion-center recall within radius two as the deterministic tie-breaker;
- save the full backbone and localization head separately as safetensors, hash every
  input/output manifest, and require the independent validator to prove archive
  coverage, zero volume leakage, box counts, selection replay, finite tensors, and
  strict state-key/shape compatibility with the unchanged P2 backbone.

The full run cannot change these settings in response to its validation curve. Any
change would be a separately declared repeat rather than an overwrite of P6-B v1.

## Validated full-source result

The corrected full run at commit `e2be22d` completed all eight epochs in 0.687 hours
on one NVIDIA L4. It used 973 training and 199 held-out volumes with zero overlap,
8,673 training samples and 1,700 validation samples, retained 16,153 boxes, discarded
the one audited zero-area box, and excluded exactly five artifact-only volumes. The
independent validator passed every input/output hash and strict-load checked all 480
backbone tensors against the unchanged P2 architecture.

Epoch 2 was selected by the frozen primary criterion: held-out localization loss
improved from 4.117923 to 1.056847 and lesion-center recall improved from 0.000800 to
0.464186. Mean GPU utilization was 97.66% and peak observed NVIDIA memory was
7,628 MiB. The selected 295,695,280-byte backbone has SHA-256
`5222ed8edc4802d7e189e5c8656fadc64292cb4c7ca563832ceebe5237c2a195`.
The compact result is [`P6-B.json`](../experiments/results/P6-B.json).

For downstream execution, only that checkpoint and its hash-pinned manifests were
copied—no fastMRI images—to the private, ready Kaggle Dataset
`vallepuvamsikrishna/rsna-p6-fastmri-localization-init`. Two self-contained T4 x2
kernels implement the matched comparison: `rsna-knee-p6-p2-matched-control` starts
from the original ImageNet checkpoint and `rsna-knee-p6-p2-transfer` starts from the
selected P6 checkpoint. They share the P2 cache, code, seeds, batches, optimizer,
epochs, pseudo-validation selection, and expert-OOF procedure.

## Validated matched-transfer result

Both Kaggle version-1 arms completed and passed independent artifact hashes, saved-head
prediction replay, report-group purge, exact 58-row OOF coverage, and dual-GPU checks.
The matched ImageNet control reached `0.8999658` OOF; P6 transfer reached `0.9098782`,
a `+0.0099125` delta with paired-bootstrap probability `0.8844`. Transfer also lowered
the pseudo-validation weighted BCE from `0.4736813` to `0.4625987`.

The standalone transfer is not promoted by itself because Effusion declined
`-0.0708075`, exceeding the frozen `-0.05` per-target floor. The predeclared fixed
blend—75% of the existing P2+P4 equal-rank model plus 25% P6 rank—reached `0.9054206`
OOF versus `0.8958211`, a `+0.0095995` gain. Its 5,000-sample paired-bootstrap 95%
interval is `[0.0029767, 0.0172338]` with probability of improvement `0.9976`; nine
targets are nonnegative and the worst decline is only `-0.0071685`. It passes every
frozen gate and is promoted for a hidden-test inference canary. The tracked result is
[`P6.json`](../experiments/results/P6.json).

The frozen hidden-inference and runtime-gating procedure is documented in
[`P6_SUBMISSION_PROTOCOL.md`](P6_SUBMISSION_PROTOCOL.md). It preserves the exact
75/25 blend and requires a hash-validated T4 x2 canary with at least 0.9 hours of
projected headroom before a competition submission is spent.

The private secret entry notebook is
`vallepuvamsikrishna/rsna-knee-p6-fastmri-staging`. Version 1 only tests secret
availability and validates the official host/package path; it does not display the
value or start a download. The required secret name is
`FASTMRI_KNEE_DICOM_BATCH2_URL`.

Kaggle API/CLI-pushed runs do not inherit editor-attached secrets. Versions 3 and 4
therefore returned `ConnectionError` from `UserSecretsClient` despite the enabled
editor toggle. An authenticated interactive execution on 2026-09-04 passed all three
checks: the secret was available, its host was the official fastMRI S3 host, and its
path selected `knee_DICOMs_batch2.tar.xz`; the value was neither printed nor written.
The cache smoke would have required the authenticated Kaggle editor rather than
`kaggle kernels push`. On 2026-09-04 execution moved to the private AWS environment
described in [`AWS_P6_RUNBOOK.md`](AWS_P6_RUNBOOK.md): an on-demand `g6.2xlarge`
with SSM-only access and persistent encrypted EBS. The signed NYU links are encrypted
SSM parameters, the official archive hashes are checked before extraction, and no
fastMRI image is copied into Git, public S3, or a Kaggle Dataset.

The initially selected `knee_DICOMs_batch2.tar.xz` downloaded and matched NYU's
published hash, but inspection found anonymized clinical-series paths rather than the
`file*.h5` identities used by fastMRI+; a privacy-preserving header probe found no
`file<digits>` mapping. It is therefore excluded from P6-B localization. The smoke
uses `knee_singlecoil_val.tar.xz`, whose `file*.h5` volumes contain the same
`reconstruction_rss` field used by the official fastMRI+ example. This execution-data
correction does not change the frozen source task, matched control, or promotion gate.

Promotion requires a positive matched change on report-grouped pseudo validation,
followed by at least `+0.003` macro AUC in expert OOF or a positive fixed 25% rank-blend
delta versus P2+P4, paired-bootstrap probability of improvement at least `0.80`, at
least seven nonnegative target deltas, and no target below `-0.05`. Public leaderboard
feedback does not select checkpoints, target mappings, or blend weights.

Primary sources: the [Microsoft fastMRI+ repository](https://github.com/microsoft/fastmri-plus),
the [fastMRI+ paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/), and the
[official fastMRI data-access page](https://fastmri.med.nyu.edu/).
