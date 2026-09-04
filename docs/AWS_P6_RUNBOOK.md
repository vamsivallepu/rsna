# AWS P6 runbook

Provisioned: 2026-09-04

Lifecycle status: instance `i-0952ebd05fab9e563` was terminated on 2026-09-04 after
P6 source training and downstream checkpoint staging completed. Both EBS volumes and
the private staging bucket were permanently deleted after explicit user approval.
Post-cleanup verification found no RSNA EBS volumes, snapshots, Elastic IPs, or
staging bucket. Only non-chargeable IAM, security-group, and Standard-tier SSM
configuration remains.

This is the durable operating record for the private AWS compute used by P6. It
contains resource identifiers and public commands, but no credentials, temporary
fastMRI URLs, patient data, images, or model artifacts.

## Provisioned resources and final state

All resources use AWS CLI profile `personal`, account `261513860644`, and region
`ap-south-1`.

| Resource | Identifier / setting |
|---|---|
| EC2 | `i-0952ebd05fab9e563` (`rsna-p6-gpu`); terminated 2026-09-04 |
| Type / AZ | on-demand `g6.2xlarge`, `ap-south-1b` |
| GPU | one NVIDIA L4, 24 GB |
| Host memory / CPU | 32 GiB / 8 vCPU |
| AMI | `ami-0391c53a255869d46`, Deep Learning Base OSS NVIDIA GPU Ubuntu 22.04 |
| Access | AWS Systems Manager only; no key pair and no inbound security-group rules |
| Security group | `sg-058b303bf550878cc` (`rsna-p6-ssm-only`) |
| IAM role / profile | `rsna-p6-ec2-role` / `rsna-p6-ec2-profile` |
| Root EBS | `vol-06b9b60c02150d33f`, 100 GiB encrypted gp3; deleted |
| Data EBS | `vol-03a6067d66b1a8ca7`, 500 GiB encrypted gp3; deleted |
| Data mount | `/mnt/rsna` by filesystem UUID |
| Instance NVMe | `/opt/dlami/nvme`, about 412 GiB; disposable scratch only |
| Private staging bucket | `rsna-p6-261513860644-ap-south-1`; versions and bucket deleted |

Termination protection is enabled and instance-initiated shutdown is configured to
stop. The S3 bucket blocks all public access, uses default AES-256 encryption and
versioning, and expires `staging/` objects and noncurrent versions after seven days.
The instance role is limited to Systems Manager, read-only access to this bucket, and
decryption of parameters below `/rsna/p6/*`.

## Cost controls

The live AWS Pricing API reported `g6.2xlarge` on-demand compute at approximately
USD 1.1739/hour in Mumbai when this host was created. The 600 GiB of provisioned gp3
storage was approximately USD 54.72/month before snapshots, even while the instance
is stopped. These are observations, not a billing guarantee; check current pricing
before resizing or leaving the host running for a long period.

Stop the GPU whenever no download, extraction, training, or inference process is
active:

```bash
aws ec2 stop-instances \
  --instance-ids i-0952ebd05fab9e563 \
  --profile personal --region ap-south-1
```

Start it again with:

```bash
aws ec2 start-instances \
  --instance-ids i-0952ebd05fab9e563 \
  --profile personal --region ap-south-1
```

Stopping preserves both EBS volumes but erases the instance-store NVMe. Persistent
archives, source snapshots, manifests, caches worth retaining, and checkpoints must
therefore stay below `/mnt/rsna`.

## Data and secret contract

Official fastMRI data remains private and must not be uploaded to GitHub, a public S3
bucket, or a public Kaggle Dataset. Signed NYU links are stored only as encrypted SSM
SecureString values:

- `/rsna/p6/fastmri/knee-dicoms-batch1-url`
- `/rsna/p6/fastmri/knee-dicoms-batch2-url`
- `/rsna/p6/fastmri/knee-singlecoil-train-url`
- `/rsna/p6/fastmri/knee-singlecoil-val-url`

The supplied links expire on 2026-12-03 at 08:14:31 UTC. Scripts may retrieve them at
runtime but must never print command tracing, URLs, or query strings. The official
archive checksums are:

```text
1b8a149897326931efb868448a81038320d23be7a651a0199de98ae641a61fba  knee_DICOMs_batch1.tar.xz
91efa0626c1ce7af68915bbd0589fc02f1aab5101569abdfbe7815fa265d5e89  knee_DICOMs_batch2.tar.xz
1e8ab7f602bb0fea649736020aa9317ebf3ba9c417c211b286b85c81a28bce5f  knee_singlecoil_train.tar.xz
cd65a27f25d11369ad73fd329ff0c68ea8d2510f98e9f46aaeff7fcaf0dba43b  knee_singlecoil_val.tar.xz
```

The DICOM batch-2 archive was downloaded and verified, but it contains anonymized
clinical-series paths with no verified mapping to the `file*.h5` annotation IDs. It
is excluded from P6-B localization. The 128-volume smoke instead uses the 16.05 GB
single-coil validation archive, whose HDF5 files contain `reconstruction_rss` in the
exact identity space used by fastMRI+.

Archives live in `/mnt/rsna/data/fastmri/archives`. Verify each complete archive
before extracting it. Extract HDF5 volumes to `/opt/dlami/nvme` for fast iteration,
then write only compact, reproducible caches and manifests back to `/mnt/rsna`.

The hash-pinned public fastMRI+ annotations live in
`/mnt/rsna/data/fastmri/annotations`. Their source commit and hashes remain frozen in
[`P6_PROTOCOL.md`](P6_PROTOCOL.md).

## Software and source

The persistent Python environment is `/mnt/rsna/.venv`. It intentionally matches the
proven Kaggle core stack (`torch 2.10.0+cu128`, `torchvision 0.25.0`, and
`timm 1.0.26`) rather than changing the model and runtime simultaneously. A complete
resolved package freeze is written to `/mnt/rsna/logs/p6-environment-freeze.txt`.

Because the GitHub repository is private and no reusable GitHub credential was placed
on the host, source is deployed as a current-tree archive through the private staging
bucket. The archive must exclude `.git`, ignored datasets, caches, checkpoints, and
credentials, and its SHA-256 must be checked after download. `/mnt/rsna/repos/current`
points to the active immutable source snapshot.

## P6 execution order

1. Complete and checksum the single-coil validation HDF5 archive on persistent EBS.
2. Match its `file*.h5` paths against the 1,172 hash-pinned reviewed IDs.
3. Extract a deterministic 128-volume smoke cohort to the disposable NVMe.
4. Validate `reconstruction_rss`, slice numbering, image geometry, box sanitation,
   and random overlays after applying the labeling-time up/down flip exactly once.
5. Run a short localization overfit/speed smoke and measure GPU utilization,
   throughput, memory, and projected full-run time.
6. Only after those gates pass, download the single-coil training archive and run
   full source pretraining.
7. Transfer the selected backbone into the unchanged P2 contract and run the matched
   ImageNet-initialized control.

The version-2 localization GPU smoke on 2026-09-04 passed all five execution gates:
99.44% mean GPU utilization, 31.68 images/s, 6.89 GiB peak allocated memory, 79.21%
fixed-batch loss reduction, and a 43.66-minute projection for eight source epochs.
The projection excludes the one-time 78.02 GB training archive download and
extraction. The official single-coil training archive may now be staged; it must
match 78,016,052,604 bytes and SHA-256
`1e8ab7f602bb0fea649736020aa9317ebf3ba9c417c211b286b85c81a28bce5f`.

The experimental promotion gate remains unchanged; moving compute from Kaggle to AWS
does not permit selecting architecture, weights, or hyperparameters on leaderboard
feedback.
