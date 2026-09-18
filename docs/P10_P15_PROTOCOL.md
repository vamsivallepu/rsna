# P10–P15 implementation and execution protocol

Frozen plan: 2026-09-17. This extends the historical P0–P9 record. The executable
recipe matrix is `experiments/next_plan.json`; live run/cost state is
`experiments/next_registry.json`. P8 submission **56077025 (0.930 public)** remains
the fallback. New implementation does not imply a new validated model.

**Latest completion check (September 17):** All three initial P10 Kaggle jobs are
COMPLETE. Teacher smoke passes artifact/membership/resource checks, but independent
image replay remains pending. The pixel audit found one exact selected-cache-pixel
pair with different reports crossing pseudo-training/shared validation. Neither row
was in the smoke subset. Resolve this image-group overlap before launching full
teachers; the existing report-only split is not yet sufficient. See
`experiments/results/P10-pixel-audit.json` and `P10-teacher-smoke.json`.

**Active correction:** `P10-report-image-v2` now unions report and image groups,
preserves all 435 original validation studies and all expert fold assignments,
and quarantines one training-side image twin. Counts are 3,909 pseudo train, 435
pseudo valid, 58 expert, four expert-report twins, and one validation-image twin.
The versioned split and its `.groups.json` sidecar live under
`outputs/experiments/P10-contract-v2/`. Full builds refuse the new split until its
raw DICOM verification succeeds. Historical artifacts remain readable as v1.

The private `rsna-knee-p10-duplicate-replay` job verifies decoded source pixels,
reconstructs both cached inputs, and replays all 12 v1 smoke predictions. ImageNet
student initialization has a separate smoke job with the corrected split and
automatic replay. The prepared full-training wave consists of teacher folds 0–2
in one job, folds 3–4 in another, and the fastMRI/report student fold-0 baseline
screen. This screen is not complete five-fold OOF evidence.

P11 is complete: all 60 saved fold heads were replayed within 1e-6. Neither
position nor hierarchical pooling passes the predefined statistical screen.
The baseline head is retained. See `experiments/results/P11-complete.json` and
`experiments/results/P11-replay.json`.

## Implementation status

| Stage | Implemented | Execution / evidence |
|---|---|---|
| P10 | Shared grouped split, teacher/student membership, immutable original validation targets, artifact validation, replay, 240-study P8 runtime job, pixel duplicate audit | All three jobs complete. Smoke artifact checks and runtime replay pass. Full training held for image-duplicate split correction and independent smoke image replay. |
| P11 | Baseline, slot/position, hierarchical heads; matched two-seed/five-fold runner; cached-head replay and group bootstrap | Both matrices complete and all 60 fold heads replayed. No variant passes; baseline retained. |
| P12 | ImageNet/fastMRI/report/distilled arms; 8/16 epochs; global/balanced 8/16 windows; expert repetition; DDP and checkpointing | Configured, not full-run validated. At most two screened recipes advance to complete confirmation. |
| P13 | Pinned offline 4-bit Qwen extractor, evidence validation, four mention states, fourth-source and mention-mask label compilers | Organizer rubric read and prompt frozen. Qwen snapshot staging and quantized throughput benchmark still required; no corpus extraction run. |
| P14 | Versioned native triplets, patient-axis ordering, physical crop/orientation, longest/fluid axial selection, 336/384 cache variants | Real representative study produced all 32 windows across five slots. Synthetic compressed-DICOM and deployment/cache agreement tests pass. Full cache QA/training not run. |
| P15 / release | Fully unfrozen official DINO-small path, fixed 80/20 rank comparison, fixed-duration full-data refits, hashed offline inference and runtime measurement | DINO and fastMRI initializers load strictly. No diversity or final refit launched; no new submission. |

## Contract and reproducibility

`scripts/rsna_next/contracts.py` preserves the historical expert KFold assignment.
The shared split contains **3,910 pseudo-training, 435 pseudo-validation, 58 expert,
and four excluded expert-report twins**. Each of five teacher holdouts contains
782 pseudo-training studies. Every teacher excludes all expert groups, all shared
validation groups, the four twins, and its own pseudo holdout. Teacher predictions
must cover each pseudo-training study exactly once out of fold. All five teachers
predict shared validation and expert studies without training on them.

The split SHA256 is
`0031102d6a8145fa6ab768e5f90817b74f87a6cf464b1b0e9899518d5718eea5`.
The original target file SHA256 is
`ca63cfff863c87286e07a8059fee88f86ff8b3f4b640a98078d62113f6b93f2a`.
The trainer rejects changed original targets. Checkpoints minimize original
report-consensus validation BCE; distilled BCE is a separate diagnostic.
Full-data refits use the frozen duration and never count as OOF evidence.

The exact P8 benchmark processed **240 studies in 8.217 minutes** on two T4s,
including 7.686 minutes of DICOM/model inference. Its conservative projection for
1,300 studies, with 25% inference headroom, is **0.876 hours**. All artifact hashes,
240 unique rows, 32 valid windows per study, fold averaging, and final clipped-rank
arithmetic were checked independently (maximum blend error 1.11e-16). This replaces
the unrepresentative three-study estimate for this sampled workload. Native volume
size and scanner-domain stratification are still missing, so this does not certify
all future pipelines or all acquisition distributions. Evidence:
[`P10-runtime.json`](../experiments/results/P10-runtime.json).

The subsequent header audit covered all **24,371 series / 4,407 studies** without
read failures. It found 47 scanner-model domains, 67–632 native slices per study,
and 6.89–383.78 million estimated pixels per study. The earlier runtime sample
covers 33 domains and 9.82–289.12 million pixels. A replacement 240-study sample
covers every domain and both volume extremes; its exact P8 benchmark is built at
`kaggle/rsna-p10-scanner-size-runtime` but has not launched. File hashes, unique
IDs, size aggregation and sample coverage were independently verified. The audit
flags 407 series in 385 studies with plane-normal agreement below 0.85, mostly
coronal (365); these are geometry review candidates, not proven incorrect labels.
One header per series was read, so within-series geometry still requires P14.
Evidence: [`P10-header-audit.json`](../experiments/results/P10-header-audit.json).

Historical audit reproduces **400/435** P8 pseudo-validation studies exposed to
P2/P6 teacher training, plus four expert report twins at representation training.
Shared report text is not proof of shared patients. Pixel audit produces candidate
pairs for review, not automatic identity merges. Its hashes describe selected cache
pixels, not all original DICOM pixels. Acquisition summaries are metadata coverage
evidence; scanner-model coverage is now audited, while site identity is not inferred.

All new artifacts use `rsna-next-v1`; manifests contain split, cache, target,
checkpoint and prediction hashes and exact training memberships. New full builds
require successful two-GPU smoke and matching full prediction-replay evidence.
Changes in initializer, head, window count or checkpointing require another smoke.
The first smoke bundle predates automatic post-training replay; validate/replay its
outputs explicitly, or use the prepared v2 bundle after inspecting v1.

## Commands

Run from the repository root; all paths below are relative to that root.

```bash
rtk proxy python3 -m pytest -q tests/test_rsna_next.py
rtk proxy ruff check scripts/rsna_next scripts/build_next_kernels.py tests/test_rsna_next.py
rtk proxy python3 -m scripts.rsna_next.audit --help
rtk proxy python3 -m scripts.rsna_next.registry status
```

Builds do not launch, provision paid compute, or submit to the competition.

```bash
rtk proxy python3 scripts/build_next_kernels.py runtime --output kaggle/rsna-p10-runtime
rtk proxy python3 scripts/build_next_kernels.py pixel-audit --output kaggle/rsna-p10-pixel-audit
rtk proxy python3 scripts/build_next_kernels.py training --recipe fastmri-report --mode teacher --fold 0 --smoke --output kaggle/rsna-p10-teacher-smoke-v2
rtk proxy kaggle kernels push -p kaggle/rsna-p10-teacher-smoke-v2
```

Do not push a second version of a running experiment simply to get new logs.
Download completed artifacts into an immutable version-specific output directory.
Current private job slugs under `vallepuvamsikrishna/`:

- `rsna-knee-next-teacher-fastmri-report-0-smoke`
- `rsna-knee-p10-representative-runtime`
- `rsna-knee-p10-pixel-audit`

The runtime slug differs from the initial local bundle ID because Kaggle derived
it from the title. The builder now uses the authoritative slug.

Full teacher build after smoke and replay pass:

```bash
rtk proxy python3 scripts/build_next_kernels.py training --recipe fastmri-report --mode teacher --fold 0 --smoke-evidence outputs/smoke/validation/validation.json --replay-evidence outputs/smoke/replay.json --output kaggle/rsna-next-teacher-0
```

Repeat for the five grouped teachers without changing the recipe/seed. Compile
their predictions with `python3 -m scripts.rsna_next.labels --help`. Each student
label file has a companion `.manifest.json` and `.provenance.csv`. The compiler
enforces teacher memberships, hashes, checkpoint selection and prediction coverage.
The 50/50 report/image loss remains fixed. Mention masking removes only the report
term for unmentioned findings. Expert labels are unchanged.

For P12, use `--mode student`, the desired recipe, and `--label-source` for distilled
arms. First screen initialization/labels. Recipes currently name fastMRI/report as
the reference; if another arm wins, freeze that control in the plan before the
one-factor screens. Keep the study batch size fixed. Screen pseudo-validation before
opening expert results; do not run the whole Cartesian product.

P11 matrix (read-only input caches):

```bash
rtk proxy python3 -m scripts.rsna_next.screen --cache outputs/kaggle/rsna-p2-coatnet-rmlp2-v2 --output outputs/experiments/P11-p2-anatomical
rtk proxy python3 -m scripts.rsna_next.screen --cache outputs/kaggle/rsna-p6-p2-transfer-v1 --output outputs/experiments/P11-p6-anatomical
```

Both initial runs retain the exact three source modules under each output's
`source/` directory. P2 began before a later fp16 empty-mask fix; its historical
cache has valid windows. Resume that run with its archived source, not changed code:

```bash
rtk proxy python3 -m outputs.experiments.P11-p2-anatomical.source.screen --cache outputs/kaggle/rsna-p2-coatnet-rmlp2-v2 --output outputs/experiments/P11-p2-anatomical --threads 4
rtk proxy python3 -m outputs.experiments.P11-p6-anatomical.source.screen --cache outputs/kaggle/rsna-p6-p2-transfer-v1 --output outputs/experiments/P11-p6-anatomical --threads 2
```

Completed fold checkpoints resume without retraining. Interrupted incomplete folds
restart deterministically; partial predictions cannot qualify for promotion.

P14 cache builder (four parts; QA all parts before training):

```bash
rtk proxy python3 scripts/build_next_kernels.py geometry --part 0 --size 336 --axial-policy longest --output kaggle/rsna-p14-native-0
```

Keep 336/longest as the first geometry comparison. Test fluid axial separately,
then native 384 separately. Do not mix signatures. Check failures, physical spacing,
native indices, orientation, and missing slots across acquisition domains. Cache
construction and inference call the same geometry implementation. Geometry remains
a combined native-adjacency/orientation comparison, not proof that either change
alone caused any eventual gain.

P13 extraction CLI requires an official Qwen3-14B snapshot, immutable 40-character
revision, and hashes for staged files. Benchmark with `--limit` in a separate output
before processing the corpus. Full extraction is offline. Preserve prompt hash and
all evidence, including uncertain/unmentioned findings. Do not revise the prompt in
response to the 58 expert outcomes. Source: [organizer rubric](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
and [official model](https://huggingface.co/Qwen/Qwen3-14B).

## Promotion and deployment

`validate.py` independently checks artifacts and aggregates exactly five student
folds. `replay.py` reconstructs image predictions from pixels/checkpoints (maximum
absolute difference 0.0005). `compare.py` takes two matched baseline/candidate seed
files, bootstraps whole report groups, and applies both-seed wins, positive mean,
bootstrap probability at least 0.90, target regression no worse than 0.02, and a
7.5-hour exact-release runtime gate. Statistical success alone never makes a
historical-cache result eligible; membership and replay must also pass.

`infer.py` accepts a release JSON with `schema_version`, `rank_blend`, and `groups`.
Each group declares fixed `weight`, `preprocessing`, and `models`. Each model entry
contains `run`, `weights`, and `manifest_sha256`, `checkpoint_sha256`,
`initialization_sha256`. Paths resolve relative to the release file. `preprocessing`
is `{ "kind": "p2" }` or P14 kind/size/axial_policy/cache_signature/contract. P14's
contract is the original cache-signature input, excluding per-part file metadata.
Probability-average within a group, rank across groups. The runner saves individual
model predictions for independent arithmetic replay. It does not automatically
submit or mark a model submission-eligible.

```bash
rtk proxy python3 -m scripts.rsna_next.infer --help
rtk proxy python3 -m scripts.rsna_next.compare --help
rtk proxy python3 -m scripts.rsna_next.replay --help
rtk proxy python3 scripts/build_next_kernels.py inference --release experiments/frozen_release.json --benchmark --output kaggle/rsna-next-release-benchmark
rtk proxy python3 scripts/build_next_kernels.py inference --release experiments/frozen_release.json --output kaggle/rsna-next-release-canary
```

The new conservative runner currently loads one model at a time and decodes each
model's studies separately. Its own 200+ study benchmark must pass; the old P8
runtime result cannot certify it. Optimize reuse/scheduling only if the measured
candidate runtime requires it, then rebenchmark that exact implementation. The
builder packages immutable offline benchmark/canary code from a frozen release
spec; include its `kernel_sources`, `dataset_sources`, `model_sources`, and absolute
Kaggle-mounted model paths. The example frozen_release.json does not exist until
a winner is selected. Full offline execution, artifact freezing and submission
selection occur after confirmation. No such winner exists yet.

## Calendar and budget

| Dates | Exit condition |
|---|---|
| Sep 17–20 | P10 full audit/runtime/smoke and corrected baseline; P11 screen |
| Sep 21–27 | Matched initialization/label screens and individual P12 changes; frozen extractor |
| Sep 28–Oct 5 | P13 students and P14 cache comparisons |
| Oct 6–13 | Two-seed, five-fold confirmation; optional one-recipe DINO branch |
| Oct 14–18 | Fixed-seed full-data refits versus folds; exact release verification |
| Oct 19–22 | Freeze two validated alternatives; recovery time |

Paid compute cap is **$500 including storage/transfer**: $60 audit/labels/benchmark,
$160 core, $100 geometry, $80 teachers/confirmation, $50 diversity, $50 contingency.
Kaggle first. No paid resources have been provisioned. Measure throughput and verify
current price before converting allocations into job-hour limits; reserve worst-case
cost before launch, settle actual cost afterward, and enforce infrastructure job
timeouts. The registry refuses commitments above the cap, but is an accounting
guardrail, not an AWS billing kill switch. Drop diversity first under pressure.

```bash
rtk proxy python3 -m scripts.rsna_next.registry reserve --run-id JOB_ID --usd WORST_CASE_USD
rtk proxy python3 -m scripts.rsna_next.registry settle --run-id JOB_ID --usd ACTUAL_USD
```

Never treat a queued/accepted job, passing unit test, full-data refit, or public
leaderboard bump as substitute evidence for the complete exclusion/replay contract.
