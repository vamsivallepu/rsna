"""One-shot replay gate and distilled student launch; retain every error on disk."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ID, sha256, write_json


STATE = Path('experiments/results/P12-distilled-launch.json')
OUTPUT = Path('outputs/kaggle/p12-replay-labels-v1')
SLUG = 'vallepuvamsikrishna/rsna-knee-p12-replay-labels'


def command(args, timeout=300):
    p = subprocess.run(['rtk', 'proxy', *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr)[-6000:])
    return p.stdout


def main():
    if STATE.exists() and json.loads(STATE.read_text()).get('status') == 'accepted':
        print('Already accepted; no duplicate push', flush=True)
        return
    write_json(STATE, {'status': 'waiting_for_replay', 'paid_usd': 0})
    deadline, previous = time.monotonic() + 4 * 3600, None
    while time.monotonic() < deadline:
        status = command(['kaggle', 'kernels', 'status', SLUG], timeout=120)
        if status != previous:
            print(status.strip(), flush=True)
            previous = status
        if 'COMPLETE' in status:
            command(['kaggle', 'kernels', 'output', SLUG, '-p', str(OUTPUT)], timeout=900)
            break
        if any(x in status for x in ['ERROR', 'CANCEL', 'FAILED']):
            command(['kaggle', 'kernels', 'output', SLUG, '-p', str(OUTPUT)], timeout=900)
            raise RuntimeError('Replay job failed; inspect downloaded logs')
        time.sleep(45)
    else:
        raise TimeoutError('Replay wait exceeded four hours; no student launched')
    gate = json.loads((OUTPUT / 'replay_labels_gate.json').read_text())
    build = json.loads(Path('kaggle/rsna-p12-replay-labels/build.json').read_text())
    expected = {r['manifest_sha256'] for r in build['expected_runs']}
    replays = gate['replays']
    if (gate['status'] != 'all_replays_and_labels_passed' or len(replays) != 6 or
            {r['run_manifest_sha256'] for r in replays} != expected or
            gate['split_sha256'] != build['split_sha256']):
        raise ValueError('Replay gate provenance failed')
    roots = sorted(Path('outputs/kaggle').glob('p10-teachers-*-v1/next_run/fold-*'))
    all_roots = roots + [Path('outputs/kaggle/p12-fastmri-baseline-f0-v1/next_run')]
    originals = {sha256(p / 'manifest.json'): p for p in all_roots}
    for r in replays:
        root = originals[r['run_manifest_sha256']]
        if (r['status'] != 'replay_validated' or r['rows'] < 12 or
                not np.isfinite(r['maximum_absolute_error']) or r['maximum_absolute_error'] > 5e-4):
            raise ValueError('Incomplete or inaccurate replay')
        if (r['rows'] != len(pd.read_csv(root / 'predictions.csv')) or
                r['checkpoint_sha256'] != sha256(root / 'best.pt') or
                r['predictions_sha256'] != sha256(root / 'predictions.csv')):
            raise ValueError('Replay does not cover the complete original artifact')
    if gate['targets_sha256'] != sha256(OUTPUT / 'student_targets.csv'):
        raise ValueError('Label hash mismatch')
    local = Path('outputs/experiments/P12-distilled-label-replay/student_targets.csv')
    command([sys.executable, '-m', 'scripts.rsna_next.labels', '--train', 'data/raw/train.csv',
             '--splits', 'outputs/experiments/P10-contract-v2/splits.csv', '--targets',
             'outputs/kaggle/rsna-report-weak-supervision/training_targets.csv',
             '--teacher-roots', *map(str, roots), '--arm', 'distilled', '--output', str(local)])
    remote = pd.read_csv(OUTPUT / 'student_targets.csv').set_index(ID).sort_index()
    replay = pd.read_csv(local).set_index(ID).sort_index()
    pd.testing.assert_frame_equal(remote, replay, check_exact=False, atol=1e-12, rtol=0)
    rm = json.loads((OUTPUT / 'student_targets.manifest.json').read_text())
    lm = json.loads(local.with_suffix('.manifest.json').read_text())
    for key in ['split_sha256', 'original_targets_sha256', 'teacher_sources', 'formula', 'arm']:
        if rm[key] != lm[key]:
            raise ValueError(f'Label provenance mismatch: {key}')
    write_json('experiments/results/P12-full-replay.json', gate)
    bundle = Path('kaggle/p12-fastmri-distilled-f0')
    command([sys.executable, 'scripts/build_next_kernels.py', 'training', '--recipe', 'fastmri-distilled',
             '--mode', 'student', '--fold', '0', '--seed', '20260917', '--label-source', SLUG,
             '--splits', 'outputs/experiments/P10-contract-v2/splits.csv', '--smoke-evidence',
             'outputs/experiments/P10-teacher-smoke-validation/validation.json', '--replay-evidence',
             'outputs/kaggle/rsna-p10-duplicate-replay-v1/replay.json', '--output', str(bundle)])
    response = command(['kaggle', 'kernels', 'push', '-p', str(bundle)], timeout=180)
    if 'successfully pushed' not in response:
        raise RuntimeError('Uncertain push result; inspect remote before any retry: ' + response)
    slug = json.loads((bundle / 'kernel-metadata.json').read_text())['id']
    write_json(STATE, {'status': 'accepted', 'slug': slug, 'paid_usd': 0,
                      'build_sha256': sha256(bundle / 'build.json'), 'response': response})
    command([sys.executable, '-m', 'scripts.rsna_next.registry', 'record', '--run-id',
             'P12-FASTMRI-DISTILLED-F0', '--state', 'running', '--artifact', str(bundle / 'build.json'),
             '--note', 'Accepted after six full checkpoint replays and independent local label arithmetic; not yet trained or promoted'])
    print(response.strip(), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        # Preserve acceptance if only the subsequent registry update failed.
        prior = json.loads(STATE.read_text()) if STATE.exists() else {}
        write_json(STATE, {**prior, 'status': prior.get('status') if prior.get('status') == 'accepted' else 'failed',
                          'error': error, 'requires_remote_check_before_retry': True})
        raise
