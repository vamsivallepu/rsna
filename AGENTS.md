@/home/vamsik/.codex/RTK.md

## Kaggle execution

Launch Kaggle jobs manually during user-requested work. The user will report when
jobs finish; check and validate them, then launch the next agreed batch. Do not
start dispatchers, queue daemons, user services, or automatic follow-up launchers.
Existing remote Kaggle jobs may continue when the local machine is off.

## Version control

The user requests a commit and push to the configured Git remote after every
major change. Complete relevant validation, review the staged diff, commit the
change, push it, and verify the remote commit. Report any push failure explicitly.
Kaggle uploads do not replace Git commits or pushes.

Keep competition data, model weights, credentials, and generated bundles containing
embedded split rows out of Git. Commit source builders, tests, documentation,
aggregate results, and artifact hashes. Do not force-push or discard others' work.
