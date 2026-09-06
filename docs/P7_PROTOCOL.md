# P7 protocol — multilingual report-image auxiliary training

## Objective

P7 tests whether semantic information in the 4,349 pseudo-study radiology reports
can improve the P6-transfer image representation without making reports available at
inference. The text encoder and report-projection head are training-only. A promoted
P7 model replaces P6 in the existing 25% ensemble slot, keeping hidden inference at
seven image backbones rather than adding an eighth model.

## Frozen text representation

- Encoder: `BAAI/bge-m3`, exposed to the offline Kaggle run as the immutable model
  source `yethukmutt/bge-m3/Transformers/m3/1`.
- Upstream model card: <https://huggingface.co/BAAI/bge-m3>.
- Paper: <https://arxiv.org/abs/2402.03216>.
- Licence: MIT. The model supports more than 100 languages, 1,024-dimensional dense
  embeddings, and sequences up to 8,192 tokens.
- Pooling: normalized first-token dense embedding from the frozen encoder.
- Tokenization: original NFKC-normalized report text, dynamic padding, no translation,
  and an extraction ceiling of 4,096 tokens. The preflight must report the observed
  untruncated lengths and stop if any report exceeds this ceiling.
- Precision: frozen BGE-M3 inference in FP16; saved embeddings are FP16 with their
  row order and all source files hash-pinned.

P7-A is a standalone two-GPU embedding preflight. It does not access DICOM pixels or
expert labels and produces only embeddings, token lengths, study order, and an
integrity manifest. The full P7 training job consumes that immutable output and does
not load BGE-M3 again.

P7-A version 1 completed and independently validated on 2026-09-05. It produced all
4,407 by 1,024 embeddings with no truncation (maximum 1,238 tokens), maximum norm
error `0.00007463`, and peak allocations of 1,260 and 1,231 MiB. The immutable
manifest SHA-256 is
`f82ddf5f5b95224a21cb83b2f1e8b3ed1bf3bad2e8495396d5c398956a9556a8`.

## Matched image-training arm

P7 starts from the exact P6 fastMRI+ localization checkpoint and otherwise preserves
the matched P6-transfer configuration:

- P2 cache geometry, normalization, stochastic eight-window training, 32-window
  evaluation, two-process T4 DDP, optimizer, learning rates, weight decay, eight
  representation epochs, augmentation, seeds, and five fold heads remain unchanged;
- checkpoint selection remains minimum report-grouped pseudo-validation weighted BCE;
- all 58 expert studies remain absent from representation gradients and checkpoint
  selection;
- expert OOF folds retain the normalized-report purge and score every expert row
  exactly once.

The only candidate change is a training-only report alignment branch:

1. Masked-mean pool the 1,024-dimensional image-window features.
2. Apply `LayerNorm(1024) -> Linear(1024, 1024) -> GELU -> Linear(1024, 1024)` and
   L2-normalize the result.
3. Compare each image embedding with a fixed bank containing only pseudo-training
   BGE-M3 report embeddings.
4. Use duplicate-aware multi-positive InfoNCE: reports with the same normalized
   report hash are positives, not negatives.
5. Use temperature `0.07`. Divide InfoNCE by `log(bank_size)` and add it to the
   unchanged classification loss at weight `0.10`.

The report projection is saved separately for audit and omitted from the deployable
image checkpoint. No report, report embedding, text model, or auxiliary head is used
for direct expert inference, OOF inference, or hidden-test inference.

## Leakage and integrity gates

- The BGE embedding preflight must contain exactly 4,407 unique study rows matching
  `train.csv`, finite unit-normalized embeddings, and no label columns.
- The report bank contains only the pseudo adaptation split. It excludes the pseudo
  validation split and all 58 expert studies.
- Normalized report hashes are group-pure between pseudo training and validation.
- Checkpoint selection uses classification BCE only, never expert AUC, report
  retrieval, or the public leaderboard.
- Every input and output artifact is SHA-256 checked. The deployable checkpoint must
  strict-load into the existing image-only CoAtNet class.

## Execution ladder

1. Build and independently validate P7-A report embeddings. **Complete.**
2. Run a 64-study, one-epoch T4 x2 speed/memory smoke with the full report bank and
   auxiliary loss. **Complete:** finite report loss `1.0142`, 8,148 MiB peak per rank,
   1.89-hour steady/full projection and 5.62-hour conservative/full projection.
3. Launch one full matched P7 candidate only if the smoke is finite, both GPUs are
   active, peak allocation is safe, and projected runtime is acceptable.
4. Independently replay the saved direct and five fold heads.
5. Compare P7 against P6 transfer and replace P6 in the frozen ensemble:
   `0.75 * rank_blend(P2, P4) + 0.25 * rank(P7)`.

## Promotion gate

Promotion requires every condition:

- pseudo-validation weighted BCE is no worse than P6 transfer (`0.4625987`);
- P7 standalone macro OOF exceeds P6 transfer (`0.9098782`);
- the fixed P7 ensemble exceeds the current P6 ensemble (`0.9054206` OOF);
- at least eight of 12 ensemble target deltas are nonnegative;
- the worst ensemble target delta is greater than `-0.02`;
- paired 5,000-sample bootstrap probability of a positive ensemble delta is at least
  `0.80`;
- the seven-backbone hidden canary retains at least 0.9 hours below Kaggle's
  nine-hour GPU limit.

If the fixed arm fails, do not tune temperature, auxiliary weight, pooling, language,
or targets against the 58 expert rows. Retain the current P6 blend and advance to P8.

## Final result

P7-B version 1 completed and independently validated on 2026-09-06. The selected
epoch was 8 with pseudo-validation weighted BCE `0.459584`, improving P6's
`0.462599`. However, standalone expert OOF was `0.904283` versus P6's `0.909878`.
The frozen 75% P2+P4 plus 25% P7 blend reached `0.904354`, below the current P6 blend
at `0.905421`; paired-bootstrap probability of a positive blend delta was `0.3724`.
The standalone, blend, and bootstrap gates failed. P7 is rejected without tuning,
P6 remains promoted, and the ordered roadmap advances to P8.
