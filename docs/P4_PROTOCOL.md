# P4 protocol — five end-to-end CoAtNet folds

Frozen: 2026-09-03

P4 is the end-to-end cross-validation stage for the promoted P2 CoAtNet system.
P3 and its blend failed their declared gates, so P4 changes neither architecture
nor image geometry. It pays for five independently trained representations so each
expert study is evaluated by a model whose backbone and head never saw that study,
its label, or a pseudo-labelled report twin.

## Hypothesis and parent

- **Hypothesis:** adding the 46–47 in-fold expert studies to each full P2 training
  run will align the image representation with expert labels, while five independent
  fold models provide a reproducible hidden-test ensemble.
- **Parent:** P2 CoAtNet-RMLP2, report-purged OOF macro AUC `0.8911254` and public
  leaderboard score `0.908`.
- **Run IDs:** `P4-COATNET-E2E-FOLD-{0..4}-v1`.
- **Seeds:** fixed expert folds `20260831`, fixed grouped pseudo split `20260902`,
  and representation seeds `20260904 + fold`.

## Frozen data contract

- Reuse `P2-CACHE-v1` exactly: five acquisition slots, 64 physical slices over
  2–98% volume coverage, 140 mm crops, 336 px uint8 storage, and cache signature
  `0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3`.
- Use the same 58 expert studies and shuffled five-fold assignment as P2. Each fold
  contains 11 or 12 expert validation studies and 46 or 47 expert training studies.
- Preserve P2's grouped 90/10 pseudo-split semantics and seed, but make membership
  version-independent by ranking normalized report hashes with SHA-256 of the seed
  and group hash, then selecting the closest group-pure 10% validation set. This
  avoids scikit-learn version drift discovered by the prelaunch audit. For each fold,
  remove from both pseudo partitions every row sharing a normalized report with the expert
  validation fold. Also remove from pseudo validation any row sharing a report with
  an expert training study, so checkpoint selection has zero training-report overlap.
- Train on the remaining pseudo-training rows plus only the fold's expert-training
  rows. The expert validation rows contribute zero gradients and cannot select a
  checkpoint. If equal DDP shards require it, deterministically drop at most one
  pseudo-training row and record its study ID; never drop or duplicate an expert row.

## Frozen model and optimization

- Use the exact P2 model: official Apache-2.0 timm
  `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` initialization with SHA-256
  `91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e`,
  384 px bicubic model input, 32 deterministic evaluation windows, and the promoted
  target-specific attention MIL head.
- Preserve P2 normalization, gain/offset augmentation, weighted BCE, eight epochs,
  eight stochastic windows per training study, two-process NCCL DDP, global study
  batch 2, four-step accumulation, fused AdamW, OneCycle schedule, gradient clipping,
  and all learning rates and weight decays.
- Select the fold checkpoint only by minimum grouped pseudo-validation weighted BCE.
  Do not use fold AUC, any expert-validation loss, or the leaderboard to choose an
  epoch or retry.
- Run one private Kaggle T4 x2 script per fold. This preserves the proven P2 DDP path,
  lets Kaggle schedule folds concurrently, and keeps each kernel below the runtime
  ceiling. The already validated P2 speed gate covers the unchanged compute path.

## Outputs and independent validation

Each fold saves its selected full model, epoch history, held-out expert predictions,
complete row-role ledger, GPU telemetry, exact hashes, and manifest. The aggregate
validator must independently reconstruct both splitters and every report purge,
verify all 58 expert studies appear exactly once, recompute all 12 AUCs and macro AUC,
and compare P4 with the exact P2 OOF predictions using a 5,000-sample paired study
bootstrap. It also evaluates one fixed equal-rank P2+P4 OOF blend; no blend weight is
tuned on the 58 studies.

The five selected full-model checkpoints form the only P4 hidden-test ensemble.
Submission inference must average their probabilities study by study and must pass a
separate representative-data and three-study Kaggle canary before submission.

## Decision gate

- **Promote P4:** macro OOF delta over P2 is positive, paired-bootstrap probability
  of a positive delta is at least `0.80`, at least 8 of 12 target deltas are
  nonnegative, the worst target delta is above `-0.03`, and every integrity gate
  passes. A delta at least `0.02` or a confidence interval wholly above zero is
  considered strong evidence.
- **Promote only the fixed blend:** P4 alone misses the primary gate, but the fixed
  equal-rank P2+P4 blend independently meets the same uncertainty and target-stability
  conditions.
- **Reject:** integrity failure, nonpositive P4 and blend deltas with no persuasive
  paired signal, multi-target instability, or unsafe inference/runtime.

No architecture, fold, epoch, per-target weight, or ensemble-weight sweep is part of
P4. Any such change receives a new experiment ID.

## Recorded outcome

All five folds completed on 2026-09-03 and passed independent artifact, split,
report-purge, and exact OOF-coverage validation. P4 alone scored `0.8911609` versus
P2's `0.8911254`; its bootstrap probability of improvement was only `0.4968` and its
worst target delta was `-0.03855`, so it failed the primary gate. The predeclared
equal-rank P2+P4 blend scored `0.8958211` (`+0.0046957`), with probability positive
`0.8622`, 9/12 nonnegative target deltas, and worst delta `-0.02041`. Promote only
the fixed blend. The complete evidence is recorded in
[`experiments/results/P4.json`](../experiments/results/P4.json).
