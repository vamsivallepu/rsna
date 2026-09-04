# P4 blend submission protocol

Frozen: 2026-09-03

## Frozen sources

- Submission run ID: `P4-BLEND-SUBMISSION-v1`.
- P2 source manifest SHA-256:
  `ce537b678242cecab9e63b3e98a3d62f33a4487107379d26c741750a12396508`.
- P4 fold manifest SHA-256 values, folds 0 through 4:
  `6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932`,
  `9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5`,
  `93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac`,
  `9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c`, and
  `e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228`.
- Each manifest and every P2/P4 checkpoint or head must match its recorded hash
  before test DICOM inference begins.

## Hidden-test inference

- Reproduce the frozen P2/P4 geometry exactly: five acquisition slots, 64 sampled
  slices over 2–98% coverage, 140 mm central crops, 336 px storage geometry, 384 px
  model input, and 32 deterministic valid windows per study.
- Decode and select each study once. Run P2 and P4 over the same in-memory windows;
  do not perform six independent DICOM passes or write a hidden-test pixel cache.
- Use two-study inference batches. Keep inference-only model weights resident in FP16
  to avoid recasting every checkpoint on every batch. On each GPU, convert, resize,
  and normalize the shared window tensor once, then feed that identical tensor through
  the three resident backbones assigned to the device. Autocast remains enabled for
  the inference operations.
- P2 is the mean probability from its five report-purged heads over the selected P2
  backbone. P4 is the mean probability from the five end-to-end fold checkpoints.
- On a T4 x2 allocation, keep P2 plus two P4 folds on GPU 0 and the remaining three
  P4 folds on GPU 1, and execute the two device groups concurrently. Fall back to
  one device without changing predictions.

## Frozen blend and output

- For each target independently, convert the complete hidden-test P2 and P4
  prediction vectors to average percentile ranks using pandas `rank(method="average",
  pct=True)`. The final value is the arithmetic mean of the two ranks.
- P2 and P4 weights are fixed at `0.5/0.5`. Do not tune weights, targets,
  calibration, or thresholds from public-leaderboard feedback.
- Clip only the final ranks to `[1e-6, 1-1e-6]`, preserve sample-submission order and
  columns, and save the P2, P4, per-fold P4, geometry, selection, and hash audits.
- This is the only candidate approved by P4 OOF validation: OOF macro AUC
  `0.8958211` versus P2's `0.8911254`.

## Release gates

1. The generated self-contained Kaggle script compiles and matches its canonical
   source builder.
2. Reuse the already passed local P2 representative-DICOM geometry smoke and verify
   statically that all five P4 checkpoint schemas exactly match the P2 model. Do not
   load six full models on the 15 GiB local host: two attempted loads exhausted swap
   before inference and were stopped without producing an artifact.
3. A private three-study Kaggle canary is the full execution gate. It verifies all
   six source hashes, both GPUs,
   finite/nonconstant P2 and P4 predictions, exact fold means, and exact rank blend.
4. Project the canary's model-inference rate to roughly 1,300 hidden studies. Submit
   only with a defensible margin under the competition's nine-hour limit.
5. Download and independently validate the canary outputs before submitting that
   exact notebook version to the competition.

## Recorded release

Versions 1 and 2 were functionally valid but failed the runtime-margin gate. Version
3 retained batch size 2, shared preprocessing per device, and resident FP16 inference
weights. It reproduced the exact same canary `submission.csv`, passed the independent
validator, and reduced the three-study inference segment to `1.0076` minutes. The
linear 1,300-study projection is `7.28` hours, leaving `1.72` hours below the limit.
Exact notebook version 3 from commit `d1e9f15` was submitted as reference `55990677`;
it completed with public score `0.916`, improving canonical P2 by `0.008`. The live
leader remained `0.952` at verification, leaving a `0.036` public gap.
