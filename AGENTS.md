@/home/vamsik/.codex/RTK.md

## Version control

The user requests a commit and push to the configured Git remote after every
major change. Complete relevant validation, review the staged diff, commit the
change, push it, and verify the remote commit. Report any push failure explicitly.
Kaggle uploads do not replace Git commits or pushes.

Keep competition data, model weights, credentials, and generated bundles containing
embedded split rows out of Git. Commit source builders, tests, documentation,
aggregate results, and artifact hashes. Do not force-push or discard others' work.
