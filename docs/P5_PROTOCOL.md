# P5 protocol — OrthoFoundation-L frozen probe

Frozen: 2026-09-04

P5 tests whether an MRI-specific foundation representation adds error diversity to
the promoted P2+P4 system. P5-A is a frozen-backbone probe. P5-B (limited LoRA) is
not authorized by this protocol unless P5-A passes its explicit advance gate.
Material changes to the inputs, slice policy, representation, validation, or blend
create a new run ID rather than silently changing P5-A.

## Hypothesis and parent

- **Hypothesis:** OrthoFoundation-L's continued self-supervised pretraining on knee
  MRI slices provides pathology features that are complementary to the ImageNet
  CoAtNet models, even when the ViT-L backbone is completely frozen.
- **Parent comparator:** the fixed equal-rank P2+P4 OOF blend, macro AUC
  `0.8958211`; its corresponding public submission scored `0.916`.
- **Run ID:** `P5A-ORTHOFOUNDATION-FROZEN-v1`.
- **Seeds:** `20260904` for the report-grouped pseudo split and head training;
  `20260831` for the unchanged five expert folds.

## Frozen external model and licence audit

- Upstream project: `ytrsk/OrthoFoundation`, source commit
  `4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470`.
- Backbone: DINOv3 ViT-L/16, 24 blocks, 1,024-dimensional tokens, 224 px input,
  four storage tokens, initialized without network access and loaded strictly from
  `OrthoFoudation-L.pth`.
- Official Git-LFS checkpoint SHA-256:
  `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`;
  expected size `1,213,056,638` bytes.
- Offline Kaggle runtime source:
  `yanisbelami/orthofoundation-l-official-runtime`. The notebook must recompute the
  official checkpoint hash before loading it and use the pinned bundled DINOv3
  source commit `6876159a11b4df116f30f667f8c9888617df0751`.
- OrthoFoundation's code and released checkpoint are MIT-licensed. DINOv3 code and
  inherited materials retain the DINOv3 License. Both notices and model attribution
  must remain with any redistributed runtime or winning solution. The competition's
  external-model permission does not erase those upstream obligations.

## Frozen cache, sampling, and representation

- Reuse `P2-CACHE-v1` exactly: five acquisition slots, 64 physically sampled
  slices, 2–98% coverage, 140 mm crop, 336 px uint8 pixels, and cache signature
  `0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3`.
- P5 uses single 2D slices because OrthoFoundation was pretrained on individual
  MRI/X-ray images converted to RGB, not on neighbouring slices packed as channels.
- Select 32 deterministic valid slices per study, stratified over the five cache
  slots with quotas `(9, 7, 6, 4, 6)`. Within each slot, select evenly across its
  valid physical coverage. Missing quota positions are masked; there is no
  cross-slot borrowing or stochastic sampling.
- Resize 336 to 224 with bilinear interpolation, duplicate grayscale into three
  channels, scale to `[0,1]`, and apply ImageNet mean/std normalization. No crop,
  flip, colour jitter, or test-time augmentation is permitted in P5-A.
- Keep two 1,024-dimensional tokens per slice: the normalized CLS token and the
  mean of normalized patch tokens. Storage tokens are excluded. Persist float16
  features plus slice mask, slot ID, and continuous within-slot position.
- The backbone is frozen in evaluation mode for every study, including pseudo and
  expert rows. Feature extraction is label-independent and sharded over exactly two
  T4 processes.

## Frozen head training and validation

- A target-query hierarchical MIL head projects the two region tokens to 256
  dimensions, attends within each slice, then attends across slices after adding
  learned slot and continuous position embeddings.
- Pretrain the head on 90% of the 4,349 pseudo-labelled studies and select one
  checkpoint by minimum weighted BCE on the remaining report-grouped 10%. No expert
  label participates in this checkpoint selection.
- Evaluate that pseudo-only head directly on the untouched 58 expert rows. Then fit
  five fold heads initialized from the selected pseudo checkpoint. For each unchanged
  expert fold, include the other expert rows plus pseudo rows after removing every
  normalized-report twin of that fold's validation rows.
- Use weighted BCE, AdamW, fixed global batch 128, pretraining LR `3e-4`, fold LR
  `2e-4`, weight decay `1e-3`, dropout `0.10`, 40 pseudo epochs, 30 fold epochs,
  and gradient clipping at `2.0`. No target-specific tuning is allowed.
- Save the 58-row P5 OOF predictions, target metrics, five heads, pseudo head,
  features and geometry, telemetry, hashes, split audit, and a manifest. An
  independent validator must replay saved heads and reproduce all reported metrics.

## Predeclared comparison and gates

- **Current baseline:** `(rank(P2) + rank(P4)) / 2` target by target.
- **Primary P5 candidate:** `0.75 * current_baseline + 0.25 * rank(P5)`.
- **Diagnostic only:** equal three-way rank mean. It cannot replace the primary
  candidate based on these 58 labels.
- **Promote P5-A blend:** primary macro delta at least `+0.003`, paired-bootstrap
  probability of a positive delta at least `0.80`, at least seven nonnegative target
  deltas, and no target below `-0.05`.
- **Advance to P5-B LoRA:** P5 standalone macro at least `0.84`, primary blend delta
  above `+0.001`, paired-bootstrap probability positive at least `0.65`, and the
  speed smoke projects a LoRA run plus feature extraction inside eight hours. This
  is an advance gate, not automatic promotion of LoRA.
- **Reject:** integrity/leakage failure, no positive blend signal, unstable target
  behaviour, or projected hidden inference above eight hours.

Before the full P5-A run, a 128-study remote smoke must load the exact checkpoint,
exercise 32-slice feature extraction and head backward on both T4s, show finite
outputs, peak below 14.5 GiB per GPU, at least 80% observed peak utilization on both
devices, and project full feature extraction below six hours.

Primary sources are the
[OrthoFoundation repository](https://github.com/ytrsk/OrthoFoundation), the
[OrthoFoundation paper](https://arxiv.org/abs/2601.18250), and the
[DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md).
