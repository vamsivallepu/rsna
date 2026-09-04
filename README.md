# RSNA Knee Abnormality Detection

Private working project for the Kaggle competition
[`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Large competition data stays on Kaggle; only representative samples and small
public dependencies are kept locally and are excluded from version control.

## Current pipeline

1. Audit the metadata and DICOM loader on Kaggle's full competition mount.
2. Build fixed soft targets from three independent public report-label sources.
3. Establish a six-slot frozen RadImageNet baseline, then adapt official DINOv2
   features using only the 4,349 pseudo-labelled studies on Kaggle GPU.
4. Train five-fold attention heads and score all 58 expert studies exactly once out
   of fold. The first verified DINOv2 run reached **0.7976** macro OOF AUC, up from
   **0.6943** for frozen RadImageNet. A paired normalized-report purge closed the only
   expert/pseudo twin path and established a **0.7968** honest cached-token baseline.
5. Train the frozen wide-coverage CoAtNet P2 system. Its independently validated
   report-purged OOF score is **0.8911**, a paired **+0.0943** over the honest baseline;
   its first notebook-only submission scored **0.908** on the public leaderboard.
6. P3 tested cache-native DINOv2-small final-block adaptation and hierarchical MIL,
   but was rejected: **0.8060** OOF, and its fixed P2+P3 rank blend fell to **0.8639**.
7. P4 trained five fold-safe, end-to-end CoAtNet models. P4 alone tied P2 at
   **0.8912**, while the predeclared equal-rank P2+P4 blend reached **0.8958** and
   passed its promotion gate. Only the blend is promoted; P2 remains a required
   member. Its optimized six-model submission canary passed with a projected
   **7.28-hour** hidden runtime, and submission `55990677` scored **0.916** publicly,
   improving P2 by **0.008**. P5 is the next model-research stage.
8. Promote only repeatable OOF improvements through the canonical ordered roadmap in
   [`docs/EXPERIMENT_ROADMAP.md`](docs/EXPERIMENT_ROADMAP.md), and record every run
   in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## Key artifacts

- Competition description, rules, metric, and data inventory:
  [`docs/COMPETITION_BRIEF.md`](docs/COMPETITION_BRIEF.md)
- Kaggle execution and synchronization workflow:
  [`docs/KAGGLE_NOTEBOOK_WORKFLOW.md`](docs/KAGGLE_NOTEBOOK_WORKFLOW.md)
- Evidence, validation contract, and experiment ladder:
  [`docs/MODELING_STRATEGY.md`](docs/MODELING_STRATEGY.md)
- Canonical ordered experiment queue and promotion gates:
  [`docs/EXPERIMENT_ROADMAP.md`](docs/EXPERIMENT_ROADMAP.md)
- Durable baseline, run, result, and decision history:
  [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
- Reproducible notebook builders: [`scripts/`](scripts/)
- Kaggle-ready private notebook bundles: [`kaggle/`](kaggle/)

The current notebooks and their verified execution states are listed in the
workflow document. A competition win cannot be guaranteed; this repository is
structured to maximize our odds without tuning directly to the tiny 58-study
expert set or leaking held-out studies into fold training.
