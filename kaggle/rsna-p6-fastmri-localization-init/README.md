# P6 fastMRI localization initialization

This private Kaggle Dataset contains only the validated CoAtNet-RMLP2 backbone,
its compact training manifest, and its epoch history. It contains no fastMRI source
images, reconstructions, k-space data, annotations, or RSNA competition data.

The backbone was selected at epoch 2 by the frozen held-out-volume localization-loss
criterion and independently validated for exact compatibility with the unchanged P2
CoAtNet backbone. Exact file sizes and SHA-256 values are pinned in `PROVENANCE.json`.

Keep this Dataset private. Its upstream image data remains governed by the fastMRI
data-use agreement; the timm backbone is Apache-2.0 and fastMRI+ annotations are MIT.
