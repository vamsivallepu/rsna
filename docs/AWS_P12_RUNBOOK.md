# P12 temporary AWS batch

## Active resources and budget

The user authorized AWS fallback training under the existing USD 500 total cap,
explicitly approved temporary encrypted Kaggle credentials and scoped role access,
and selected `us-east-1` after Mumbai capacity failures. Profile: `personal`;
account: `261513860644`.

- Active batch host: `i-0d0ec06ac8023cffe`, `g6.2xlarge`, `us-east-1c`.
- GPU: NVIDIA L4, 24 GB; host: eight vCPUs, 32 GiB RAM.
- AWS Pricing API on 2026-09-22: USD 0.9776/hour, at most USD 23.4624 for
  24 compute hours. Reserve USD 80 including all setup attempts, storage, and
  transfer; actual billing remains to be reconciled.
- Deadline: no later than 2026-09-23 06:32 UTC. Recovery calculates the remaining
  minutes against this original deadline; it does not grant a new 24 hours.
- Successful or failed batch execution also stops the host. Stopped EBS storage
  still incurs charges. Do not restart without a new bounded execution decision.
- Root storage: 200 GiB encrypted gp3, 500 MiB/s, delete on termination.
- Local NVMe: disposable image cache only. Checkpoints/logs remain on EBS and
  are uploaded to private S3 after each validated job, including failures.
- SSM-only access through `sg-05096916741b41ba2`, with no inbound rules and IMDSv2.

The unused Mumbai host `i-0186081c12610ea5f` was terminated after explicit user
approval. Its 200 GiB volume `vol-0f6340b752093c5d5` was verified deleted. No model
training occurred on it. Remaining Mumbai IAM/security-group resources predate
this batch and are not to be removed as unrelated cleanup.

## Frozen work and evidence gates

This is a fixed four-job EC2 batch, not a local dispatcher or a cloud provisioning
loop. Kaggle launches remain manual. Run fold 3 for `fastmri-report` (eight epochs)
and `duration16` (16 epochs), each with seeds 20260917 and 20260918. These four jobs
are reserved for AWS; do not duplicate them on Kaggle. Reuse exact configurations
from `experiments/P12-confirmation.json`, original report targets, initialization,
and version-2 split. Preserve the separate in-flight Kaggle fold-2 jobs.

The single GPU runs two logical workers through Gloo, retaining per-worker sample
ordering, loss reduction, accumulation, and optimizer schedule. Both members of
each matched comparison use the same backend. This is not a claim of bitwise
agreement with Kaggle's two physical T4s. Environment metadata records the change.

Before full training, require a 64-study smoke, finite losses, valid held-out
coverage, checkpoint replay within 0.0005, and successful two-worker execution.
The original recipe measured approximately 7.1 GB allocated per worker. Record GPU
utilization every five seconds; report measured performance rather than assuming
full utilization. Validate and replay each full job before advancing. AWS outputs
stay separate from Kaggle artifacts. No automatic promotion or submission.

## Software and private data

Training retains Python 3.10, torch 2.10.0/cu128, torchvision 0.25.0, timm 1.0.26,
and the pinned numerical packages. Kaggle 2.2.4 requires Python >=3.11, so its CLI
runs in a separate Python 3.12 environment. The initial incompatible install
stopped safely and was corrected before training.

The ordinary `kernels_output` client buffers whole output files and failed on the
16 GB pixel arrays. `download_kernel_stream.py` reuses the same client's streaming,
retrying downloader with identity-checked Range resume. A localhost integration
check deliberately interrupts a transfer and verifies the exact final SHA-256.
Training independently verifies all cache hashes before consuming them. Partial
or corrupt downloads cannot enter model training.

Private bucket: `rsna-p12-261513860644-ap-south-1` (Mumbai); transfer between the
Virginia host and this bucket remains within the user's account and budget.
All public access is blocked and encryption is enabled. Source archives include
licensed split rows and remain outside Git. Staging objects expire in seven days;
model results do not expire automatically.

Instance profile: `rsna-p6-ec2-profile`; temporary additional role policy:
`rsna-p12-batch`. Read access is limited to staging, write access to results.
Temporary encrypted SSM parameter: `/rsna/p6/p12-kaggle-token` in `ap-south-1`.
The token is never included in user data or logs. Delete the host token after
staging and the SSM parameter after successful downloads; remove the added role
policy after the batch. Do not delete the pre-existing P6 role/profile.

## Status

Use `experiments/results/P12-aws-batch.json` and `experiments/next_registry.json`
for the latest verified stage. A running instance or completed setup is not proof
of a trained model. Smoke, full-run validation, and promotion are separate gates.
