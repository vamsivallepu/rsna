# Tracked experiment results

This directory contains small, durable result records that are safe to version in
Git. Large caches, checkpoints, raw predictions, and notebook logs remain under
`outputs/` and are intentionally ignored.

Each result record points to the exact code commit and source-cache hashes used by
the run. Human-readable decisions and the ordered queue live in
[`docs/EXPERIMENT_LOG.md`](../docs/EXPERIMENT_LOG.md) and
[`docs/EXPERIMENT_ROADMAP.md`](../docs/EXPERIMENT_ROADMAP.md).
