# RSNA Knee Abnormality Detection: competition brief

Snapshot date: 2026-09-01  
Source: live Kaggle CLI 2.2.4 responses and downloaded official CSV metadata  
Competition: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>

## Current account and contest state

- The Kaggle account is authenticated and has accepted/entered the competition.
- Competition category: Research.
- Current team count reported by the CLI: 2,808.
- Current submissions: 0 lifetime and 0 today; 5 submissions remain today.
- Total prize pool: USD 77,000.

## Objective

Build a multimodal model that predicts the per-study probability of 12 clinically
important abnormalities from knee MRI studies. The training set pairs every MRI
study with its original radiology report, often in one of several languages.

Targets:

1. ACL injury
2. MCL injury
3. Medial meniscus tear
4. Lateral meniscus tear
5. Medial tibiofemoral osteoarthritis
6. Lateral tibiofemoral osteoarthritis
7. Patellofemoral osteoarthritis
8. Joint effusion
9. Synovitis
10. Baker's cyst
11. Bone contusion
12. Fracture

## Evaluation and submission

The main score is macro-averaged ROC AUC:

`score = mean(AUC for each of the 12 targets)`

Predictions are one probability per study and target. The first column is
`StudyInstanceUID`; the remaining columns are the 12 target names in the sample
submission.

This is a notebook-only code competition:

- CPU runtime must be at most 9 hours.
- GPU runtime must be at most 9 hours.
- Internet must be disabled.
- Freely and publicly available external data and pretrained models are allowed,
  subject to the competition rules.
- The generated file must be named `submission.csv`.
- The hidden scoring test set has about 1,300 studies. Reports are not available
  at inference time.

## Timeline

All deadlines are 23:59 UTC unless Kaggle changes the timeline.

- Start: 2026-07-30
- Entry deadline: 2026-10-15
- Team merger deadline: 2026-10-15
- Final submission deadline: 2026-10-22
- Winners' requirements deadline: 2026-11-05

## Official training data observed locally

The downloaded metadata contains:

- 4,407 training studies, all with non-empty radiology reports.
- 24,371 training MRI series across those 4,407 studies.
- Exactly 58 fully expert-labeled studies; the other 4,349 have no provided
  binary target labels.
- No partially labeled studies.
- Series by plane: 9,864 sagittal, 8,609 coronal, and 5,898 axial.
- 14,010 fluid-sensitive/fat-suppressed series and 10,361 other series in the
  supplied descriptors.

Expert-label positives among the 58 labeled studies:

| Target | Positives | Prevalence |
|---|---:|---:|
| ACL | 24 | 41.4% |
| MCL | 9 | 15.5% |
| Medial Meniscus | 26 | 44.8% |
| Lateral Meniscus | 23 | 39.7% |
| Medial OA | 15 | 25.9% |
| Lateral OA | 11 | 19.0% |
| PF OA | 21 | 36.2% |
| Effusion | 35 | 60.3% |
| Synovitis | 27 | 46.6% |
| Baker's | 12 | 20.7% |
| Contusion | 19 | 32.8% |
| Fracture | 18 | 31.0% |

The crucial modeling implication is that report-derived weak supervision is not
optional for a competitive solution: only 58 of 4,407 studies have expert binary
labels, while all 4,407 have reports.

## Data layout

- `train.csv`: one row per training study, containing `StudyInstanceUID`, a
  multilingual `Report`, and the 12 binary label columns.
- `train_series.csv`: one row per MRI series, with study/series UIDs,
  `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`.
- `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`:
  one DICOM file per slice.
- `test.csv`, `test_series.csv`, and `test_series/`: three example public-test
  studies locally; Kaggle replaces them with the real hidden test data at scoring.
- `sample_submission.csv`: valid all-0.5 example submission.

Series usually contain 20-45 slices (median 30), with a long tail into the
hundreds. Image intensity, orientation, and resolution vary. Transfer syntaxes
include uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, and
Implicit VR Little Endian. DICOM metadata is restricted to an allowlist of 86
tags.

The complete competition archive is 247 GB. This machine had 118 GB free, so the
full archive was not downloaded. The five official CSV metadata files are in
`data/raw/` and are excluded from Git.

## Rules that directly affect the solution

- Maximum team size: 5.
- Maximum submissions per day: 5.
- Up to 2 final submissions may be selected for judging.
- Competition data may be used for commercial and academic research subject to
  the MIRA license and official rules.
- Competition data must not be transmitted, duplicated, published,
  redistributed, or made available to anyone who has not accepted the rules.
- External data and models are allowed only when publicly available and equally
  accessible at no cost, or when they satisfy Kaggle's reasonableness/minimal-cost
  standard.
- Hand-labeling or human prediction of validation/test records is prohibited.
- Winner submission/code license named by the rules: CC BY-NC 4.0.
- Winners must provide reproducible training and inference code, model weights,
  documentation, and the computational environment. The host additionally asks
  for a short solution video, public code and weights, and a publicly distributed
  final model.

Always re-read the complete live rules before selecting external datasets/models,
forming a team, publishing artifacts, or submitting a final solution.

## Prizes

Main leaderboard: USD 9,000; 7,000; 6,500; 6,000; 5,500; then USD 5,000 each for
places 6-10.

Efficiency track: USD 7,000; 6,000; and 5,000. An efficiency-track submission
must be eligible under Kaggle's selection rules and beat the all-0.5 benchmark on
the private leaderboard. Kaggle combines private AUC and runtime (normalized to
the 32,400-second notebook limit), with lower efficiency score ranked better.

## Immediate strategic implications

1. Build report-derived labels with multilingual medical terminology, negation,
   uncertainty, laterality, and severity handling. Use the 58 expert-labeled studies
   as a high-uncertainty audit/OOF set, not as enough data for target-specific blend,
   prompt, or threshold fitting.
2. Split by study and likely by source/site proxies to test generalization across
   the international scanner/protocol mix; never split at slice or series level.
3. Use plane- and sequence-aware MRI encoders, aggregate slices to series and
   series to studies, and train all 12 targets jointly.
4. Optimize validation for macro AUC and per-target robustness, not overall
   accuracy; rare targets can dominate failure modes.
5. Design inference for the 9-hour offline Kaggle notebook limit from the start,
   keeping an efficiency-track candidate alongside the strongest ensemble.
