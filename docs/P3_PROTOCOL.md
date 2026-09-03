# P3 protocol — DINOv2-small hierarchical diagnosis MIL

Frozen: 2026-09-03

P3 is the predeclared high-resolution transformer comparison against the promoted
P2 CoAtNet system. It reuses the immutable P2 pixel cache and changes the
representation and MIL hierarchy only. Any material change to the items below
creates a new run ID rather than silently changing P3.

## Hypothesis and parent

- **Hypothesis:** DINOv2-small at the cache-native 336 px resolution, retaining
  coarse spatial patch structure and pooling it with diagnosis-specific queries,
  will provide complementary signal beyond P2's one-vector-per-window CoAtNet.
- **Parent comparator:** P2, macro OOF AUC `0.8911254` and public leaderboard
  score `0.908`.
- **Run ID:** `P3-DINOV2-HMIL-v1`.
- **Seeds:** `20260903` for the report-grouped pseudo split and representation;
  `20260831` for the unchanged five expert folds.

## Frozen inputs and licensing

- Reuse `P2-CACHE-v1` exactly: five acquisition slots, 64 physically sampled
  slices, 2–98% coverage, 140 mm crop, 336 px uint8 pixels, and cache signature
  `0a9332278587c80c6994da1bb8bb2c27501a198cf8a93a1041f5ab1788a3c1e3`.
- Backbone: classic Meta DINOv2 ViT-S/14 without registers, from the official
  Kaggle model source `metaresearch/dinov2/PyTorch/small/1`. The exact
  `pytorch_model.bin` SHA-256 is
  `1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050`.
- The classic DINOv2 model card identifies the weights as Apache-2.0. This is not
  Cell-DINO or XRay-DINO, whose later research-only licenses do not apply here.
  The competition permits reasonably accessible external pretrained models, but
  every final dependency still retains its own license obligations.

## Frozen representation and hierarchy

- Each input is a three-adjacent-slice 2.5D window from one cache slot. Windows
  never cross slot boundaries. Training samples 8 valid windows per study per
  epoch; validation and feature extraction use 32 deterministic windows.
- DINO receives native `336 × 336` inputs, yielding a `24 × 24` patch grid. Each
  window retains five region tokens: the class token plus adaptive `2 × 2`
  averages of the patch grid.
- A shared projection maps 384-dimensional region tokens to 256 dimensions.
  Twelve learned diagnosis queries first attend over the five regions within
  each window. A second set of twelve queries attends across windows after adding
  learned five-slot embeddings and a continuous within-slot position embedding.
  Target-specific linear weights produce the 12 logits.
- Freeze the patch embedding and first ten transformer blocks. Fully train only
  the final two transformer blocks, final backbone layer norm, and hierarchical
  MIL head. This is the declared last-block adaptation arm; P3 does not combine a
  LoRA sweep with the primary comparison.
- Use the DINOv2 ImageNet normalization. The only augmentation is P2's light
  intensity gain/offset jitter; no flips or spatial transforms.

## Frozen training and validation

- Representation data: 4,349 pseudo-labelled studies only. All 58 expert rows
  are excluded from backbone gradients and checkpoint selection.
- Split pseudo studies 90/10 by normalized report hash with seed `20260903`; zero
  report hashes may cross the split. Select the single checkpoint with minimum
  grouped pseudo-validation weighted BCE.
- Train for 8 epochs with two-process NCCL DDP, global study batch 2, four-step
  accumulation, mixed float16, fused AdamW, backbone LR `1e-5`, head LR `5e-4`,
  weight decay `0.02`, OneCycle schedule (`pct_start=0.15`), and gradient clipping
  at `3.0`.
- Before the full run, a 64-study remote speed smoke must exercise the exact
  336 px forward/backward, 8-window training, 32-window validation, both T4s,
  and gradient-accumulation path. Launch the full run only if it is finite, both
  GPUs are active, peak memory is safely below 15 GB, and projected runtime fits
  the nine-hour competition limit.
- After checkpoint selection, extract five region tokens for 32 windows from all
  4,407 studies. Evaluate the pseudo-only head directly on the untouched 58 expert
  studies, then fit five report-group-purged feature-only heads using the unchanged
  P2 fold seed, 30 epochs, LR `2e-4`, weight decay `1e-3`, batch 256, and weighted
  BCE.
- Save OOF predictions, all heads, features/masks/slots/positions, telemetry,
  histories, hashes, and a manifest. An independent validator must replay every
  head and reproduce P3-versus-P2 target metrics and paired bootstrap uncertainty.

## Decision gate

- **Promote:** P3 exceeds P2's `0.8911254` OOF score with stable target behaviour,
  valid purges/hashes, and safe inference runtime; or a P2+P3 rank ensemble yields
  a clear paired improvement that demonstrates useful diversity.
- **Repeat:** an apparent gain below `0.02` with broad target support but an
  uncertainty interval crossing zero.
- **Reject:** integrity failure, multi-target instability, no useful P2 diversity,
  or unsafe hidden-test inference time.

Primary architecture references are the
[DINOv2 paper](https://arxiv.org/abs/2304.07193) and the
[official Meta model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).
