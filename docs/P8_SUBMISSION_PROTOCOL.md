# P8 submission protocol

Frozen: 2026-09-07 after every P8 OOF promotion gate passed.

## Frozen candidate

The P8 candidate changes exactly one component of the promoted P6 system:

1. P2 prediction: mean probability from the five report-purged P2 heads.
2. P8 prediction: mean probability from the five independently validated
   fold-safe noisy-student backbones.
3. P2+P8 baseline: equal target-wise percentile-rank mean of P2 and P8.
4. P6 prediction: mean probability from the five report-purged P6-transfer heads.
5. Final prediction: `0.75 * P2+P8 baseline + 0.25 * rank(P6)`, clipped to
   `[1e-6, 1 - 1e-6]`.

This is algebraically `0.375 rank(P2) + 0.375 rank(P8) + 0.25 rank(P6)`.
It keeps the same seven-backbone count, preprocessing, 32-window geometry,
two-study batching, FP16 resident weights, and dual-device scheduling as the
validated P6 submission. No target, fold, model, or blend weight may change after
the canary or leaderboard result is observed.

## OOF evidence

- P8 standalone: `0.9029118074637345` versus P4 `0.89116089660126`.
- Equal-rank P2+P8 baseline: `0.9022377777850243`.
- Frozen seven-model P8 blend: `0.9097982866758175` versus the current P6 blend
  `0.9054205862136073`.
- Nine of 12 blend target deltas were nonnegative, the worst was `-0.0090703`,
  and the paired-bootstrap probability of a positive delta was `0.8992`.

All predeclared OOF gates in `P8_PROTOCOL.md` passed.

## Runtime gate

The visible three-study Kaggle run is a runtime and integrity canary. Before a
competition submission is spent:

1. Run the exact private T4 x2 bundle against the visible test mount.
2. Download and independently validate every artifact and source hash.
3. Reconstruct the five-fold P8 mean and final blend from component predictions.
4. Project the measured DICOM-plus-model segment linearly to 1,300 hidden studies.
5. Require at least `0.9` hours of headroom below Kaggle's nine-hour limit.
6. Submit the exact validated Kaggle version, never a rebuilt copy.

## Reproducible artifacts

- Canonical inference source: `scripts/p4_blend_submission.py`.
- P8 derivation builder: `scripts/build_p8_blend_submission.py`.
- Prelaunch validator: `scripts/validate_p8_blend_prelaunch.py`.
- Output validator: `scripts/validate_p8_blend_submission.py`.
- Kaggle bundle: `kaggle/rsna-p8-blend-canary/`.

The builder replaces only the P4 fold family and frozen OOF constants, enables the
already-validated P6 path, and asserts that no P4 source remains in the generated
deployment script.
