# P6 submission protocol

## Frozen candidate

Submit only the predeclared blend promoted by the independently validated P6
matched comparison:

1. P2 prediction: mean probability from the five report-purged P2 heads.
2. P4 prediction: mean probability from the five end-to-end P4 folds.
3. Baseline prediction: equal target-wise percentile-rank mean of P2 and P4.
4. P6 prediction: mean probability from the five report-purged P6-transfer heads.
5. Final prediction: `0.75 * baseline + 0.25 * rank(P6)`, clipped to
   `[1e-6, 1 - 1e-6]`.

No weight, target, seed, checkpoint, or preprocessing choice may be changed after
observing the canary or leaderboard result.

## Immutable inputs

- P2 manifest SHA-256:
  `ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508`.
- P4 fold manifest and checkpoint hashes are the same hash-pinned artifacts used by
  the scored P4 submission.
- P6-transfer manifest SHA-256:
  `885b58c5a9af1bffaac1f52c9703db3538d25f8d9660b036b563b351b1067bce`.
- Geometry remains the validated P2/P4 geometry: five series slots, 32 windows,
  336-pixel cache geometry, and 384-pixel CoAtNet input.

The local validator must recompute all artifact hashes, the five-fold P4 mean, and
the final rank blend from raw component predictions.

## Runtime gate

The visible three-study Kaggle execution is a canary; competition submission reruns
the same code against approximately 1,300 hidden studies under the nine-hour GPU
limit. The canary uses two-study batches, resident FP16 models, one preprocessing
pass per device, and two independent single-thread device queues. P6 weights are
resident on both GPUs, while P6 batches alternate between devices. This balances
the seventh backbone over consecutive batches without a per-batch cross-device
barrier. The shared normalized image tensor is converted to FP16 once per device
group, matching the dtype autocast supplies to each first convolution while avoiding
the same large input conversion inside every resident backbone.

Before spending a competition submission:

1. Run the exact private T4 x2 bundle on the visible test mount.
2. Download and independently validate every canary artifact.
3. Linearly project the measured DICOM-plus-model segment to 1,300 studies.
4. Require at least `0.9` hours of headroom below the nine-hour limit. A valid
   prediction whose projection exceeds `8.1` hours is not submitted; optimize and
   repeat the canary instead.
5. Submit the exact validated Kaggle kernel version, never a rebuilt local copy.

## Reproducible artifacts

- Canonical source: `scripts/p4_blend_submission.py` with the immutable P6 switch
  enabled only by the P6 builder.
- Builder: `scripts/build_p6_blend_submission.py`.
- Independent validator: `scripts/validate_p6_blend_submission.py`.
- Kaggle bundle: `kaggle/rsna-p6-blend-canary/`.

## Execution record

Version 1 independently validated the complete prediction path but projected to
`8.2136` hours, leaving only `0.7864` hours below the limit, and therefore failed the
frozen margin gate. Version 2 cast the shared normalized input once to the same FP16
dtype already used by autocast. It produced a byte-identical `submission.csv`, passed
independent replay, and reduced the projection to `7.6087` hours with `1.3913` hours
of headroom. Exact version 2 from commit `ac8aa92` was accepted as competition
submission `56026228` on 2026-09-05 and is pending scoring.
