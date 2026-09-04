# P6 protocol — fastMRI+ localization transfer

Frozen audit stage: 2026-09-04

P6 tests whether supervised knee-pathology localization before RSNA training can
improve the promoted CoAtNet representation. P6-A is an annotation, provenance,
access, and transferability audit. P6-B is not authorized to launch until the image
access gate below is satisfied. A Kaggle mirror is not a substitute for that gate.

## Hypothesis and parent

- **Hypothesis:** lesion-localization supervision on sagittal knee MRI can make the
  P2 CoAtNet backbone attend to small structural abnormalities that study-level
  report labels do not localize.
- **Parent:** the promoted P2+P4 equal-rank blend, OOF macro AUC `0.8958211` and
  public score `0.916`. P2 remains the standalone architecture used for the controlled
  initialization comparison.
- **P6-A run ID:** `P6A-FASTMRI-PLUS-AUDIT-v1`.
- **P6-B run ID:** reserved as `P6B-FASTMRI-PLUS-LOCALIZATION-v1`; it does not yet
  denote an authorized or launched training run.

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

P6-B will use only the official validation knee volumes corresponding to the
hash-pinned reviewed-file list. Reconstruct the source image exactly as documented by
fastMRI+, including the labeling-time up/down flip, and verify random overlays before
training. Split by volume, never slice. The split manifest must be hash-pinned and no
volume may cross train/validation. Apply the declared whitespace and box sanitation
only after retaining the original row values for auditability.

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

1. the selected official download URL is supplied to the private Kaggle runtime as a
   Kaggle Secret rather than embedded in code;
2. source filenames match the official reviewed list and remain private;
3. annotation and reviewed-list hashes match P6-A;
4. overlays pass in the exact DICOM coordinate frame or after the documented HDF5
   up/down flip, depending on the selected official package;
5. a 128-volume smoke stays within Kaggle memory/runtime limits.

The private secret entry notebook is
`vallepuvamsikrishna/rsna-knee-p6-fastmri-staging`. Version 1 only tests secret
availability and validates the official host/package path; it does not display the
value or start a download. The required secret name is
`FASTMRI_KNEE_DICOM_BATCH2_URL`.

Promotion requires a positive matched change on report-grouped pseudo validation,
followed by at least `+0.003` macro AUC in expert OOF or a positive fixed 25% rank-blend
delta versus P2+P4, paired-bootstrap probability of improvement at least `0.80`, at
least seven nonnegative target deltas, and no target below `-0.05`. Public leaderboard
feedback does not select checkpoints, target mappings, or blend weights.

Primary sources: the [Microsoft fastMRI+ repository](https://github.com/microsoft/fastmri-plus),
the [fastMRI+ paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/), and the
[official fastMRI data-access page](https://fastmri.med.nyu.edu/).
