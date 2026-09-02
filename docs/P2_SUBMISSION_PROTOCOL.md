# P2 submission protocol

## Frozen source

- Submission run ID: `P2-SUBMISSION-v1`.
- Source model: validated `P2-COATNET-v1`, best pseudo-validation checkpoint at
  epoch 8.
- Required source manifest SHA-256:
  `ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508`.
- The inference notebook must verify the source checkpoint and all five fold-head
  hashes from that manifest before reading test pixels.

## Hidden-test inference

- Reproduce `P2-CACHE-v1` geometry exactly: five acquisition slots, 64 sampled
  slices over 2–98% of each selected series, 140 mm central crop, per-selected-series
  2nd/98th percentile scaling, 336 px bilinear storage geometry, and boundary-safe
  adjacent-slice windows.
- Use 32 deterministic valid windows per test study and the unchanged 384 px
  CoAtNet normalization.
- Stream DICOM preprocessing through two spawned CPU workers into one GPU. Do not
  write a full hidden-test pixel cache or assume the three-study visible canary count.
- A study with no valid adjacent window is a hard failure; it must not silently receive
  a constant prediction.

## Ensemble and output

- Replay each of the five report-group-purged feature heads independently.
- The submitted probability for each study and target is the arithmetic mean of the
  five sigmoid probabilities. No target-specific blending, calibration, thresholding,
  public-leaderboard fitting, or direct-head mixing is permitted in this first P2
  submission.
- Preserve the exact `sample_submission.csv` study order and columns. All 12 outputs
  must be finite and strictly between zero and one.
- Write `submission.csv`, fold predictions, direct-head audit predictions, feature
  masks/positions, selection audit, and a cryptographic manifest.
- Validate that `submission.csv` exactly reproduces the five-fold mean before using
  the notebook version as a competition submission.

The visible three-study mount is only a canary. Kaggle's notebook-only submission
rerun evaluates the same code on the hidden test set of roughly 1,300 studies under
the nine-hour GPU limit.
