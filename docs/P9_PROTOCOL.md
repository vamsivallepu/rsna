# P9 protocol — compact true-3D S3D specialist

Frozen: 2026-09-07 before the P9 speed/memory smoke.

## Objective and parent

P9 tests whether explicit through-plane convolutions add complementary signal after
the promoted P8 2.5D ensemble plateaued at `0.909798` expert OOF and `0.930` public.
The parent remains the fixed rank blend
`0.375 * P2 + 0.375 * P8 + 0.25 * P6`.

OrthoDiffusion is not the executable P9 arm. Its public repository contains the
architecture and training workflow, but not the three knee-diffusion checkpoints or
the 15,948-exam pretraining corpus used by the paper. Reproducing that pretraining is
outside the competition data and Kaggle budget. P9 therefore uses the official
Torchvision S3D Kinetics-400 initialization: a separable 3D CNN with 8.32 million
parameters and a substantially smaller published compute footprint than R3D-18 or
R(2+1)D-18.

## Frozen P9-A representation

- Reuse immutable `P2-CACHE-v1`: 64 grayscale 336 px slices over five acquisition
  slots, 140 mm crop, and the existing orientation/coverage contract.
- Build one 16-frame clip per slot by deterministic uniform resampling of valid
  cached slices. Missing slots are masked. No left/right flip or temporal reversal.
- Replicate grayscale to RGB, resize to `160 x 160`, and apply the official S3D
  Kinetics normalization (`mean=[0.43216, 0.394666, 0.37645]`,
  `std=[0.22803, 0.22145, 0.216989]`).
- Load the hash-pinned official checkpoint `s3d-d76dad2f.pth`, SHA-256
  `d76dad2f4b2a6701ad7f64e85b294b2d1989f1ea0d3f8b240446ed2152850bbb`.
- Pool each slot's 3D feature map adaptively to 1,024 dimensions. A target-specific
  attention head, with learned slot embeddings, fuses the five slot features into
  the 12 logits.
- Train one shared representation for six epochs on pseudo studies only. Select its
  checkpoint solely by report-group-pure pseudo-validation weighted BCE. Expert
  labels do not enter representation gradients or checkpoint selection.

To avoid carrying fold-specific P4 expert gradients into a shared representation,
P9 pseudo targets intentionally exclude P4. For each pseudo row and target the fixed
target is:

`0.50 * original report mean + 0.25 * P2 direct + 0.25 * P6 direct`.

P2 and P6 direct heads contain no expert-label gradients. Original report-agreement
weights are retained. Expert targets and their weight `2.0` remain unchanged.

## Frozen P9-B expert OOF evaluation

After representation selection, extract and cache five 1,024-dimensional slot
features per study. Freeze the backbone and train five cheap attention heads from
the selected representation head initialization. Each fold trains on all allowed
pseudo rows plus four expert folds, removes pseudo rows sharing a normalized report
with the held-out expert fold, and predicts that expert fold exactly once. The fold
assignment, purge, optimizer, 30 head epochs, and seeds match the established P2
OOF contract.

Only one S3D backbone is required for hidden inference; the five fold heads share its
features. This is intentionally not five end-to-end 3D models.

## Speed-smoke gate

Before the full run, a private T4 x2 smoke executes 64 pseudo-training studies and
16 grouped pseudo-validation studies through the real DDP, mixed-precision,
five-clip, backward, optimizer, and validation path. Advance only if:

- both GPUs participate and no CUDA OOM/traceback occurs;
- peak allocated memory is below `14,500 MiB` on both ranks;
- losses are finite;
- the conservative six-epoch representation projection is below four hours.

The full run must remain below Kaggle's execution limit and emit the checkpoint,
features, OOF predictions, split ledger, telemetry, histories, hashes, and manifest.

## Frozen promotion gate

P9 is a diversity specialist, so the primary comparison is the predeclared rank
blend, not standalone P9 alone:

`candidate = 0.85 * current_P8_blend + 0.15 * rank(P9)`.

Equivalently its component weights are `0.31875 P2 + 0.31875 P8 + 0.21250 P6 +
0.15000 P9`. The `0.15` weight is frozen before observing P9 OOF and will not be
tuned per target or from the public leaderboard.

Promotion requires every condition:

- candidate OOF exceeds current P8 blend `0.909798`;
- at least 8 of 12 target deltas are nonnegative;
- worst target delta is greater than `-0.02`;
- paired 5,000-sample bootstrap probability of a positive macro delta is at least
  `0.80`;
- an exact hidden-test runtime canary retains at least `0.9` hours below Kaggle's
  nine-hour limit.

If P9 adds OOF value but cannot fit as an extra backbone, test replacement of P6
only under a separately frozen blend before any leaderboard submission. If the gate
fails, P9 is rejected without weight search on the 58 expert rows.
