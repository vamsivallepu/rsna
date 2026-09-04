# P5 diagnostic submission protocol

Frozen: 2026-09-04

This is a diagnostic leaderboard submission requested after P5-A failed its OOF
promotion and LoRA gates. It does not replace the promoted P2+P4 system, and its
public score will not be used to tune weights or override the fold-safe decision.

- Run ID: `P5-DIAGNOSTIC-SUBMISSION-v1`.
- Source P5 manifest SHA-256:
  `3b739d18f85347bbc457885bba72b1a20712d2ab56dda67ac68fc35840b3163d`.
- Verify the official OrthoFoundation checkpoint and all five P5 fold-head hashes
  before inference.
- Reproduce the frozen five-slot, 64-slice preprocessing and select 32 single slices
  with quotas `(9, 7, 6, 4, 6)`, 224 px model input, and the same normalization.
- Shard test studies across two T4s. Each rank loads one frozen ViT-L backbone and
  all five fold heads, then predicts its disjoint studies in batches of two.
- The submitted prediction is the arithmetic mean probability from the five P5
  fold heads. Preserve sample-submission rows and columns and clip only to
  `[1e-6, 1-1e-6]`.
- Run the private three-study Kaggle canary, download its artifacts, and independently
  reproduce the fold mean and geometry checks before submitting that exact version.
- Record the public score as diagnostic evidence only. Regardless of leaderboard
  score, P5 remains rejected unless new genuinely independent validation justifies a
  new protocol.
