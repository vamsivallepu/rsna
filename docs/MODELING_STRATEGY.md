# Modeling strategy

Snapshot date: 2026-09-01

## Evidence-based starting point

The competition has 4,407 training studies but only 58 expert-labelled rows. All
studies have multilingual reports; hidden-test reports are unavailable. The viable
pipeline is therefore report teacher to image student to image-only inference.

Current public discussion provides useful bounds, not guarantees:

- Strong single image models are reported around 0.915-0.938 public leaderboard
  AUC; five-fold systems around 0.942-0.947.
- Report-label extractors are commonly around 0.89-0.92 macro AUC on the 58 expert
  rows, while image students can outperform their report teacher.
- Multiple competitors report that slice geometry, physical scale, sequence
  selection, and label quality matter more than simply increasing backbone size.
- Small/minimal effusions and other borderline findings disagree between reports
  and expert labels. Soft labels are preferable to forcing brittle binary thresholds.
- The 58-row expert set is too small for target-specific prompt, threshold, blend,
  or hyperparameter fitting. It is an audit and OOF evaluation set.

## Weak-label contract

The first teacher uses three independent, public, CC0 report-label datasets:

1. `pilkwang/rsna-knee-llm-labels`
2. `stevenleehans/rsna-knee-llm-report-labels`
3. `lixin73/rsna-knee-llm-report-labels-sol56`

The primary target was fixed as the arithmetic mean before expert-set inspection.
The expert audit produced:

| Method | Macro AUC on 58 expert studies |
|---|---:|
| Pilkwang | 0.8700 |
| Steven Lee | 0.8927 |
| Lixin | 0.8352 |
| Three-LLM median | 0.8857 |
| Three-LLM rank mean | 0.8934 |
| **Three-LLM arithmetic mean** | **0.8948** |
| Three LLMs plus multilingual NLI rank mean | 0.8784 |

The primary method remains the predeclared arithmetic mean. NLI is retained only
as an audit because adding it weakened the fixed blend. The 95% study-bootstrap
interval for the primary macro AUC was approximately 0.859-0.925, emphasizing how
uncertain any conclusion from 58 rows remains.

`training_targets.csv` contains:

- unchanged expert labels and `is_gold=1` for 58 studies;
- three-LLM mean soft targets for 4,349 studies;
- target-specific weights from 0.50 to 1.00 for pseudo labels based on source-vote
  agreement, and weight 2.00 for expert labels.

No threshold or source weight is fitted to expert labels.

## Image baseline

The first image experiment is deliberately reusable and runtime-conscious:

- one longest series for each of six plane x sequence slots;
- physical slice ordering from DICOM orientation and position;
- 150 mm center crop using pixel spacing;
- five positions spanning 10%-90% of each volume;
- adjacent/center/adjacent slices stacked as a 2.5D input;
- frozen RadImageNet ResNet-50 features;
- a target-specific attention head over the six slots;
- five fixed expert OOF folds, with every held-out expert study excluded from that
  fold's training and all pseudo-labelled studies retained;
- one scan of the DICOM mount followed by cheap cached-feature experiments.

The local smoke run decoded all 30 windows for the representative six-slot study,
strictly loaded the public checkpoint, cached six 2,048-dimensional slot features,
and completed a forward, backward, and optimizer step.

The full T4 run completed feature extraction in about 50 minutes:

- 4,407/4,407 studies usable and zero feature failures;
- mean 4.84 populated acquisition slots per study;
- 103.3 MiB reusable feature cache;
- five-fold expert OOF macro AUC **0.6943**.

An independent local validator reloaded all five heads, checked every cached feature for
finiteness, recomputed all target AUCs from the 58 OOF rows, and byte-matched the teacher
targets. Cheap ridge ablations on the same cache reached at most **0.7053** with strongly
regularized L2-normalized mean pooling. This small gain confirms that global frozen
features are the bottleneck; additional head tuning is not the next priority.

## DINOv2 adaptation experiment

The completed first private Kaggle adaptation notebook uses the official Apache-2.0
Meta DINOv2-small model:

- 130 mm physical crops and structural/fluid slots in all three planes;
- three 2.5D positions per slot rather than one global pooled slot vector;
- class-token plus mean-patch features with target-specific token attention;
- fixed three-epoch adaptation on the 4,349 pseudo-labelled studies only;
- all 58 expert studies excluded from backbone adaptation and epoch selection;
- direct untouched-gold evaluation, followed by five fold-safe heads on the adapted
  token representation.

This design follows the strongest reproducible public pattern—many windows, physical
geometry, target-specific MIL, and trainable image features—without importing public
competition-trained checkpoints. Public source reviewed for architecture evidence:
[`RSNA Knee | DINOsaur V4`](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4).

The full T4 run completed successfully:

- all 4,407 studies cached in 128.1 minutes with zero failures and all 18 windows
  valid per study;
- training loss decreased from 0.5664 to 0.5339 to 0.5157 over the fixed three
  pseudo-only adaptation epochs;
- the adapted pseudo-only model scored **0.7852** macro AUC on the untouched 58-study
  expert gate;
- five fold-safe heads scored **0.7976** macro OOF AUC across those same 58 studies.

The fold-head result improves by **0.1034** over the frozen RadImageNet attention
baseline and by **0.0924** over the best cached-feature ridge ablation. An independent
local validator reconstructed the direct model from its saved state and token cache,
recomputed both macro AUCs, verified every feature and mask, reloaded all five heads,
and byte-matched the teacher targets. This establishes adapted DINOv2 tokens as the new
internal baseline, but the result remains below the roughly 0.9+ public-performance
range; the next experiments should improve image coverage and representation rather
than tune target-specific choices against 58 labels.

## Experiment ladder

Changes graduate only after repeated OOF evidence and runtime accounting.

1. **Complete:** frozen RadImageNet baseline establishing extraction runtime, failure
   rate, slot coverage, and expert OOF AUC.
2. **Next:** geometry and coverage ablations: 130/150 mm crops, additional positions,
   central versus broader volume sampling, and physical order versus instance-number
   fallback.
3. **Planned:** teacher ablations: mean versus rank supervision, unanimity masking,
   label smoothing, and student-teacher blending. Keep all choices global rather than
   target-tuned.
4. **In progress:** backbone comparison at identical geometry. DINOv2-small at 224
   pixels is complete; DINOv2-base and a compact convolutional encoder remain.
5. **Initial adaptation complete:** the first DINOv2 run unfroze its final two blocks.
   Repeat fine-tuning only after cheap token/head and geometry experiments identify a
   robust configuration.
6. **Planned:** five folds and 336/392 pixels only for configurations that clear the
   cheap ablation gate.
7. **Planned:** rank-average genuinely diverse backbones/geometry; do not ensemble many
   near-duplicates merely because public LB noise rewards them.
8. **Planned:** maintain a smaller slice/window count and single compact backbone for
   the separate runtime-aware prize.

## Validation guardrails

- Split only at study level; never let slices or series cross folds.
- Score OOF predictions once per expert study across all 58 rows, then compute the
  12 target AUCs and their unweighted mean.
- Never choose per-target thresholds for an AUC competition.
- Record target-wise AUC, extraction failures, slot coverage, runtime, seed, and exact
  attached dataset/model versions for every run.
- Treat public leaderboard improvements as secondary evidence because the public split
  and the 58-row expert set are both small enough for accidental selection.
- Preserve an untouched final candidate and re-read live rules/licenses before making
  a final submission or publishing weights.
