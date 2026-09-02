# P2 protocol — wide-coverage CoAtNet-RMLP2

Frozen: 2026-09-02

P2 is one predeclared image-model experiment. Its purpose is to test whether a
strong compact convolution/attention backbone and substantially denser physical
coverage improve on the honest P0-B DINO baseline without using the 58 expert
labels for representation selection. This document is the run contract; a change
to any frozen item creates a new run ID rather than silently changing P2.

## Hypothesis and parent

- **Hypothesis:** a 384 px CoAtNet-RMLP2 representation with 64 physically sampled
  slices and target-specific attention MIL will recover small and spatially sparse
  knee abnormalities missed by the current 18-window, 224 px DINO representation.
- **Parent comparator:** P0-B, macro OOF AUC `0.7967883`, with the current
  agreement-weighted soft BCE, target-specific attention, report-group purge,
  dropout `0.20`, and head weight decay `1e-3`.
- **Run IDs:** `P2-CACHE-v1` for pixels and `P2-COATNET-v1` for training/evaluation.
- **Seed:** `20260831` for fixed folds and head fitting; `20260902` for the
  pseudo train/validation group split and representation training.

## Frozen pixel-cache contract

The cache is built once from the mounted competition DICOM and is reused by P2 and
compatible later experiments. It is split into two immutable private Kaggle notebook
outputs so neither local disk nor one Kaggle output version must hold the roughly
32 GB full cache. The concatenated order must exactly equal `train.csv` order.

| Slot | Series policy | Slices | Relative centers |
|---|---|---:|---|
| Sagittal fluid-sensitive | highest deterministic fluid score | 18 | evenly spaced 2–98% |
| Sagittal structural | highest deterministic structural score | 14 | evenly spaced 2–98% |
| Coronal fluid-sensitive | highest deterministic fluid score | 12 | evenly spaced 2–98% |
| Coronal structural | highest deterministic structural score | 8 | evenly spaced 2–98% |
| Axial any | highest slice count after geometry checks | 12 | evenly spaced 2–98% |

- Correct slice order comes from ImageOrientationPatient/ImagePositionPatient, with
  deterministic instance-number and filename fallbacks.
- A series may fill at most one slot in a study. Ties use score, valid slice count,
  and SeriesInstanceUID in that order.
- Decode modality values using rescale slope/intercept, invert MONOCHROME1, then use
  a per-selected-series 2nd/98th-percentile intensity window.
- Crop a centered `140 mm × 140 mm` field of view from DICOM PixelSpacing and resize
  it to `336 × 336` uint8 pixels. Cache values are not ImageNet-normalized.
- A 2.5D input is three adjacent slices from the same slot. Windows crossing a slot
  boundary are forbidden. Slot bounds and validity masks are cache artifacts.
- Part A contains rows `[0, 2204)` and part B `[2204, 4407)`. Each part writes the
  pixel array, mask, ordered IDs, failures, and a JSON manifest. The manifest includes
  the competition/train ordering hashes, selection/crop/intensity/sampling policy,
  array shape/dtype/hash, row range, code commit, and cache-signature hash.
- Any change to data version/order, geometry, slot definitions, sampling, crop,
  intensity processing, resize, or cache code invalidates `P2-CACHE-v1`.

The two versioned private Kaggle kernel outputs are the immutable cache source. This
is equivalent to a private Kaggle Dataset for reproducibility while avoiding a local
download and re-upload of about 32 GB. Downstream metadata pins the exact successful
kernel versions and verifies both manifests before training.

## Frozen model and representation training

- Backbone: `timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, initialized only from
  the official Apache-2.0 timm checkpoint (ImageNet-12k pretraining followed by
  ImageNet-1k fine-tuning). The private Kaggle weight source records its original URL
  and SHA-256; competition-trained public weights are prohibited.
- Input: valid adjacent-slice RGB windows resized from cached 336 px to 384 px with
  the official checkpoint normalization (mean/std `0.5/0.5` per channel). No
  left/right flip or geometric transform is used. The only
  augmentation is a fixed light intensity gain/offset jitter of at most 10%.
- Pooling: each window becomes one backbone vector. A target-specific attention MLP
  (`LayerNorm`, `Linear(F,256)`, `Tanh`, dropout `0.20`, `Linear(256,12)`) softmaxes
  over valid windows, followed by target-specific linear weights and bias.
- Data: the 4,349 pseudo-labelled studies only. All 58 expert studies are excluded
  from backbone gradients, epoch selection, stopping, and configuration choice.
- Pseudo validation: normalized-report-hash grouped 90/10 split using seed
  `20260902`. Zero normalized report hashes may cross the split.
- Objective: the unchanged three-model consensus soft targets and agreement weights,
  optimized with weighted BCE. No ASL, ranking loss, per-target tuning, or confidence
  replacement is introduced in P2.
- Optimization: 8 complete epochs, AdamW, backbone LR `3e-5`, head LR `1e-3`, weight
  decay `0.02`, OneCycle schedule with `pct_start=0.15`, gradient clipping at `3.0`,
  mixed float16, study batch size `2`, and four-step gradient accumulation. Training
  samples 8 valid windows per study per epoch. Gradient checkpointing is disabled in
  execution revision `ddp_t4_v3`: the speed smoke must first prove that the unchanged
  microbatch fits safely in each 15 GB T4, otherwise the full run is not launched.
- Execution revision `ddp_t4_v3` uses one NCCL
  `DistributedDataParallel` process per T4. Each rank receives one study from every
  frozen global batch, and DDP gradient averaging is rescaled to reproduce the same
  global weighted-BCE denominator as the original two-study calculation. The first
  three accumulation backwards use `no_sync()`, so gradients communicate only at the
  optimizer boundary. Validation and feature extraction shard rows across both GPUs
  without changing their deterministic windows.
- Execution also uses channels-last fp16 tensors, deterministic cuDNN kernels with
  fixed-shape autotuning, fused AdamW, two loader workers per rank, pinned asynchronous
  transfers, and no per-step GPU-to-CPU scalar synchronization. The two independent
  cache-part SHA-256 streams are read concurrently while still validating every byte.
  Deterministic validation and feature extraction batch two studies on each rank
  (global inference batch four); the selected windows and their order are unchanged.
  Per-GPU utilization, memory, power, and temperature are logged every 60 seconds.
  These are execution-only changes; labels, study order within each global batch,
  windows, effective batch, optimizer updates, epoch selection, folds, and decision
  gates are unchanged.
- The only representation checkpoint is the epoch with minimum grouped pseudo-
  validation weighted BCE. Validation uses 32 deterministic valid windows per study.
  The expert labels cannot choose an epoch, checkpoint average, or retry.

## Frozen expert evaluation

1. Evaluate the selected pseudo-only checkpoint directly on all 58 untouched expert
   studies using 32 deterministic valid windows. This is the representation gate.
2. Cache 32 CoAtNet feature vectors per study for all 4,407 studies.
3. Starting from the selected pseudo-trained attention head, fit five feature-only
   heads. Each fold uses all pseudo studies except those sharing a normalized report
   hash with its held-out expert rows, plus the non-held-out expert rows. Use the
   unchanged P0-B head constants: 30 epochs, AdamW LR `2e-4`, weight decay `1e-3`,
   dropout `0.20`, weighted BCE, batch size `256`, and fixed seed/folds `20260831`.
4. Concatenate the five held-out expert predictions into one 58-row OOF file. Verify
   every row occurs once, no purged hash crosses a fold, saved heads replay, and all
   values and hashes are finite and stable.
5. Compare P2 OOF with P0-B using target AUCs and a paired 5,000-sample study
   bootstrap. Record extraction/training/evaluation time and measured or conservatively
   projected 1,322-study inference time.

## Decision gate

- **Promote to the P2/P3 comparison:** P2 macro OOF exceeds `0.7967883`, the gain is
  not explained by one target, no target materially regresses, all leakage/integrity
  checks pass, and projected inference remains safely below the nine-hour limit.
- **Repeat:** a positive gain below about `0.02` is promising but inconclusive; repeat
  the same frozen configuration with matched seeds before claiming a winner.
- **Reject/block:** integrity or purge failure, incompatible licensing, non-reproducible
  cache, material multi-target regression, or unsafe inference time.

The official architecture references are the [CoAtNet paper](https://arxiv.org/abs/2106.04803)
and the [timm model card](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k).
The public Apache-2.0 RSNA Raptor implementation informed the five-slot allocation and
target-specific attention structure, but P2 does not reuse its competition-trained
checkpoint or its expert-selected training procedure.
