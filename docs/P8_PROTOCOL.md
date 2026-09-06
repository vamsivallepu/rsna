# P8 protocol — fold-safe noisy-student relabelling

Frozen: 2026-09-06 before the P8 teacher audit.

## Objective and parent

P8 tests whether fold-safe image-teacher probabilities can reduce report-label noise
for a new five-fold CoAtNet student ensemble. The parent is the promoted P6
submission system: `0.375 * rank(P2) + 0.375 * rank(P4) + 0.25 * rank(P6)`, with
expert OOF macro AUC `0.9054206` and public score `0.923`.

P8 replaces the five P4 backbones rather than adding five more. A promoted P8 system
therefore remains `0.375 * rank(P2) + 0.375 * rank(P8) + 0.25 * rank(P6)` and retains
the current seven-backbone hidden-inference count.

## Frozen fold-safe teacher

For each expert fold `f`, create probabilities for every study from exactly three
image teachers:

1. the P2 representation checkpoint's direct pseudo-only attention head;
2. the P6 representation checkpoint's direct pseudo-only attention head;
3. P4 end-to-end fold `f`, whose representation saw only the other expert folds.

P2 and P6 direct heads contain no expert-label gradients. P4 fold `f` excludes expert
validation fold `f`, its labels, and pseudo rows sharing its normalized report hashes.
Thus no teacher used for P8 fold `f` has seen that fold's expert labels.

The image teacher is the unweighted arithmetic mean of the three probabilities. For
each of the 4,349 pseudo studies and target, the P8 soft target is fixed as:

`0.50 * original three-LLM report mean + 0.50 * image-teacher probability`.

Original target-specific report-agreement weights remain unchanged. Expert-training
rows retain their exact binary labels and weight `2.0`; expert-validation rows never
enter gradients or checkpoint selection. There is no thresholding, sharpening,
temperature scaling, target-specific weighting, or teacher-weight search.

## P8-A teacher audit and generation

Before GPU generation, audit the frozen formula on the 58 expert rows using P2 and P6
direct predictions plus P4 OOF predictions. P8-A advances only if every condition
holds against the original three-LLM report mean:

- relabel-teacher macro AUC is at least `0.90` and strictly higher;
- at least 8 of 12 target deltas are nonnegative;
- worst target delta is greater than `-0.05`;
- paired 5,000-sample bootstrap probability of a positive macro delta is at least
  `0.80`.

If the audit passes, one private Kaggle T4 x2 kernel consumes the immutable P2/P4/P6
outputs and `P2-CACHE-v1`, predicts all 4,407 rows, and writes five fold-specific
target tables plus component predictions, row order, split evidence, telemetry, and
SHA-256 manifest. It must independently verify all source manifests/checkpoints and
all fold exclusions before emitting labels.

The frozen audit passed on 2026-09-06. The original report teacher scored `0.894811`,
the equal three-image teacher scored `0.903487`, and their fixed 50/50 relabel teacher
scored `0.920920`. The relabel-minus-report bootstrap mean was `+0.026200`, its 95%
interval was `[+0.008877, +0.044085]`, probability positive was `0.9984`, 8/12 target
deltas were nonnegative, and the worst delta was `-0.024943`. P8-A advances to GPU
teacher generation without changing the frozen formula.

## P8-B student training

Train five independent fold students. Each fold:

- starts from the exact P6 fastMRI+ localization backbone, not a finished P2/P4/P6
  competition-trained checkpoint;
- reuses `P2-CACHE-v1`, the P4 fold assignment, grouped pseudo split, report-twin
  purge, two-process T4 DDP, CoAtNet-RMLP2 architecture, stochastic eight-window
  training, 32-window evaluation, weighted BCE, optimizer, schedule, augmentation,
  eight epochs, and checkpoint-selection rules;
- differs from P4 only in the P6 localization initialization and its frozen
  fold-specific pseudo targets;
- selects the checkpoint only by its fold-safe grouped pseudo-validation BCE;
- outputs its held-out expert predictions, complete split ledger, checkpoint,
  telemetry, histories, and hashes.

The five fold jobs may run concurrently within Kaggle quota, after a 64-study T4 x2
smoke validates the new target input and unchanged training path.

## Final promotion gate

Independent validation must reconstruct all teacher labels and student splits,
replay every saved held-out prediction, and cover every expert row exactly once.
Promotion requires every condition:

- P8 standalone OOF exceeds P4 OOF `0.8911609`;
- the fixed seven-model P8 blend exceeds the current P6 blend `0.9054206`;
- at least 8 of 12 P8-blend target deltas versus the current blend are nonnegative;
- worst P8-blend target delta is greater than `-0.02`;
- paired 5,000-sample bootstrap probability of a positive blend delta is at least
  `0.80`;
- a hidden-test canary retains at least `0.9` hours below Kaggle's nine-hour limit.

If any gate fails, do not tune the 50/50 relabel mix, component weights, target
weights, or folds against the expert rows or public leaderboard. Retain P6 and advance
to P9.
