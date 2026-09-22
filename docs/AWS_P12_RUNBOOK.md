# P12 temporary AWS batch

Prepared 2026-09-22. Profile `personal`, account `261513860644`, region
`ap-south-1`. Kaggle launches remain manual. This is a fixed four-job EC2 batch,
not a local dispatcher or an automatic cloud provisioning loop.

## Selection and cost

Live quota: eight on-demand G/VT vCPUs. Selected `g6e.2xlarge`, one NVIDIA L40S
with 48 GB GPU memory, eight vCPUs, and 64 GiB host memory. AWS Pricing API returned
USD 2.69266/hour on 2026-09-22. A 24-hour host shutdown limit bounds compute to
USD 64.62384. Reserve USD 80 including storage and transfer from the existing
USD 500 plan cap. Actual charges must be reconciled after completion.

Use 200 GiB encrypted gp3 root storage, with deletion on instance termination.
Stopping preserves storage and its charges. Keep completed outputs until verified
locally. The host must stop after success or failure, as well as at the 24-hour
deadline. Never restart without resetting the deadline.

## Frozen work

Run fold 3 for `fastmri-report` (eight epochs) and `duration16` (16 epochs), each
with seeds 20260917 and 20260918. Reuse exact configurations from the frozen
`experiments/P12-confirmation.json` builders, original report targets, initialization,
and version-2 split. Do not duplicate the two fold-2 jobs already on Kaggle.

The single L40S runs two logical training workers through Gloo. This preserves
the existing per-worker sample sequence, weighted loss reduction, gradient
accumulation, and optimizer schedule without changing the effective study batch
size. CUDA execution and communication backend differ from Kaggle; this is not a
claim of bitwise equivalence. Both members of each matched comparison use the same
AWS execution path. The environment records this difference explicitly.

Before any full job, require a 64-study two-worker smoke, finite losses, valid
held-out coverage, checkpoint prediction replay within 0.0005, and per-worker
allocated memory below 20 GB. Record GPU utilization every five seconds. Validate
and replay each full job before proceeding. Keep AWS output separate from Kaggle
artifacts; do not mark the original queue jobs complete until independent import
and validation. No automatic promotion or competition submission.

## Private access

Reuse the existing no-inbound security group and SSM instance profile documented
in `AWS_P6_RUNBOOK.md`. No SSH key or public inbound ports. Require IMDSv2.
Stage source and licensed metadata in a new private, encrypted S3 bucket;
source archives and generated split rows remain outside Git. A temporary encrypted
SSM Kaggle token enables direct cache downloads; never include it in user data,
logs, source archives, or command output. Delete the host token after downloads
and the temporary SSM parameter after staging. Remove the added role policy when
the batch is finished. Staging objects expire after seven days; results do not.

Initial staging was blocked by automatic approval review because credential
transfer and persistent IAM access require explicit authorization. At preparation
time no instance or staging resources had been created. Await that approval before
provisioning; record the final resource IDs and launch status separately.
