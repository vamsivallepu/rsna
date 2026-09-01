# Local competition data

The official small metadata files were downloaded with Kaggle CLI 2.2.4 into
`data/raw/` on 2026-08-31:

- `train.csv`
- `train_series.csv`
- `test.csv`
- `test_series.csv`
- `sample_submission.csv`

The complete Kaggle archive reports a download size of 247 GB. It was not kept
because this machine had only 118 GB free. A size probe was stopped and its
partial file was removed.

Competition data must not be committed, transmitted, published, or shared with
anyone who has not formally accepted the competition rules. See the official
rules and MIRA license before using or moving it.

To resume on storage with sufficient free space:

```bash
kaggle competitions download rsna-knee-abnormality-detection -p data/raw
```

## Representative local DICOM sample

`data/sample/` contains a bounded sample selected and downloaded through Kaggle:

- 5 studies: 3 expert-labeled, 1 unlabeled English-like report, and 1 unlabeled
  Cyrillic report.
- The labeled studies jointly cover both 0 and 1 for all 12 targets.
- 6 complete MRI series cover every plane x fluid-sensitive/fat-suppression
  combination present in the training metadata.
- 4 additional representative slices cover the remaining selected studies.
- 202 DICOM files totaling 177,339,814 bytes (0.165 GiB).

Selection is recorded in `data/sample/selection.json`; downloaded files and byte
sizes are recorded in `data/sample/download_manifest.json`.

All 202 retained DICOMs use Explicit VR Little Endian and decode successfully.
A bounded probe of 120 additional random training series did not locate the other
transfer syntaxes named in Kaggle's data description. That unresolved gap is
recorded in `data/sample/transfer_syntax_samples.json`; loaders must continue to
support and explicitly test Implicit VR, JPEG Lossless, and JPEG 2000 when such
files are encountered.

Recreate or verify the sample with:

```bash
python3 scripts/select_representative_sample.py
python3 scripts/download_representative_sample.py --dry-run
python3 scripts/download_representative_sample.py
```

## Small public modeling dependencies

The ignored sample tree also holds dependencies used to execute the modeling
notebooks locally without the full MRI archive:

- `data/sample/public_weak_labels/`: 3.0 MB containing the three public LLM
  sources plus the multilingual NLI audit scores. The source Kaggle datasets are
  CC0.
- `data/sample/models/ResNet50.pt`: the 94 MB RadImageNet ResNet-50 checkpoint
  from `marwanmath/resnet-50-radimagenet-marwan` (CC BY-NC-SA 4.0).
- `data/sample/models/dinov2-small/`: the official Kaggle-hosted Meta
  DINOv2-small variation used for local smoke tests (Apache 2.0; 88 MB
  uncompressed).

These files are attached directly as Kaggle inputs for remote execution and are
not embedded in the uploaded notebooks. Re-check source licenses before publishing
derived weights or using them outside this competition.
