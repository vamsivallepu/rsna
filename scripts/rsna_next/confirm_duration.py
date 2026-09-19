"""Frozen 16-job confirmation queue, at most two active free Kaggle jobs."""
import argparse
import fcntl
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from .advance_p12 import command
from .contracts import read_splits, sha256, write_json
from .validate import aggregate, validate_run

PLAN = Path('experiments/P12-confirmation.json')
STATE = Path('outputs/experiments/P12-confirmation/queue.json')
SPLIT = Path('outputs/experiments/P10-contract-v2/splits.csv')
BASES = {
    'fastmri-report-20260917': 'p12-fastmri-baseline-f0-v1',
    'duration16-20260917': 'p12-duration16-f0-v1',
    'fastmri-report-20260918': 'p12-fastmri-report-f0-seed2-v1',
    'duration16-20260918': 'p12-duration16-f0-seed2-v1',
}


def prepare():
    if PLAN.exists():
        raise ValueError('Confirmation plan already frozen; do not overwrite')
    screen = json.loads(Path('experiments/results/P12-duration-two-seed-screen.json').read_text())
    if not screen['advance_to_five_fold']:
        raise ValueError('Screening gate has not passed')
    jobs = []
    for fold in range(1, 5):
        for seed in [20260917, 20260918]:
            for recipe in ['fastmri-report', 'duration16']:
                bundle = Path(f'kaggle/p12-confirm-{recipe}-f{fold}-s{seed}')
                command([sys.executable, 'scripts/build_next_kernels.py', 'training', '--recipe', recipe,
                         '--mode', 'student', '--fold', str(fold), '--seed', str(seed), '--replay',
                         '--splits', str(SPLIT), '--smoke-evidence',
                         'outputs/experiments/P10-teacher-smoke-validation/validation.json', '--replay-evidence',
                         'outputs/kaggle/rsna-p10-duplicate-replay-v1/replay.json', '--output', str(bundle)])
                jobs.append({'recipe': recipe, 'seed': seed, 'fold': fold, 'bundle': str(bundle),
                             'build_sha256': sha256(bundle / 'build.json'),
                             'slug': json.loads((bundle / 'kernel-metadata.json').read_text())['id'],
                             'output': f'outputs/kaggle/{bundle.name}-v1'})
    bases = {k: {'root': f'outputs/kaggle/{v}/next_run',
                 'manifest_sha256': sha256(f'outputs/kaggle/{v}/next_run/manifest.json')}
             for k, v in BASES.items()}
    write_json(PLAN, {'jobs': jobs, 'bases': bases, 'split_sha256': sha256(SPLIT),
                     'maximum_active_jobs': 2, 'paid_usd': 0,
                     'purpose': 'Two recipes x two seeds x five folds; reuse four completed fold0 runs. No submission or automatic promotion.'})
    print(f'Frozen {len(jobs)} jobs', flush=True)


def verify_replay(root, manifest, rows):
    replay = json.loads((root / 'replay.json').read_text())
    if (replay['status'] != 'replay_validated' or replay['rows'] != rows or
            replay['run_manifest_sha256'] != sha256(root / 'manifest.json') or
            not 0 <= replay['maximum_absolute_error'] <= 5e-4):
        raise ValueError('Full replay failed')
    for key in ['checkpoint_sha256', 'predictions_sha256']:
        if replay[key] != manifest[key]:
            raise ValueError('Replay artifact mismatch')


def run():
    plan = json.loads(PLAN.read_text())
    meta = pd.read_csv('data/raw/train.csv')
    splits = read_splits(SPLIT, meta)
    if plan['split_sha256'] != sha256(SPLIT):
        raise ValueError('Split changed')
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'plan_sha256': sha256(PLAN), 'status': 'running', 'jobs': ['pending'] * len(plan['jobs'])}
    if state['plan_sha256'] != sha256(PLAN):
        raise ValueError('Frozen queue changed')
    if 'submitting' in state['jobs']:
        raise ValueError('Uncertain prior launch: verify remote before changing queue state')
    for base in plan['bases'].values():
        if sha256(Path(base['root']) / 'manifest.json') != base['manifest_sha256']:
            raise ValueError('Fold0 artifact changed')
        validate_run(base['root'], splits, SPLIT)
    deadline = time.monotonic() + 72 * 3600
    try:
        while not all(s == 'validated' for s in state['jobs']):
            if time.monotonic() > deadline:
                raise TimeoutError('72-hour queue limit; inspect before resuming')
            for i, job in enumerate(plan['jobs']):
                if state['jobs'][i] != 'running':
                    continue
                status = command(['kaggle', 'kernels', 'status', job['slug']], timeout=120)
                if any(x in status for x in ['ERROR', 'CANCEL', 'FAILED']):
                    command(['kaggle', 'kernels', 'output', job['slug'], '-p', job['output']], timeout=900)
                    raise RuntimeError(status)
                if 'COMPLETE' not in status:
                    continue
                command(['kaggle', 'kernels', 'output', job['slug'], '-p', job['output']], timeout=900)
                root = Path(job['output']) / 'next_run'
                manifest, pred = validate_run(root, splits, SPLIT)
                build = json.loads((Path(job['bundle']) / 'build.json').read_text())
                if manifest['config'] != build['recipe'] or manifest['fold'] != job['fold']:
                    raise ValueError('Unexpected completed recipe/fold')
                verify_replay(root, manifest, len(pred))
                state['jobs'][i] = 'validated'
                write_json(STATE, state)
                print('Validated ' + job['slug'], flush=True)
            for i, job in enumerate(plan['jobs']):
                if state['jobs'].count('running') >= 2:
                    break
                if state['jobs'][i] != 'pending':
                    continue
                bundle = Path(job['bundle'])
                if sha256(bundle / 'build.json') != job['build_sha256']:
                    raise ValueError('Build changed')
                build = json.loads((bundle / 'build.json').read_text())
                if sha256(bundle / 'run.py') != build['code_sha256']:
                    raise ValueError('Frozen source changed')
                state['jobs'][i] = 'submitting'
                write_json(STATE, state)
                response = command(['kaggle', 'kernels', 'push', '-p', str(bundle)], timeout=180)
                if 'successfully pushed' not in response:
                    raise RuntimeError(response)
                state['jobs'][i] = 'running'
                write_json(STATE, state)
                print(response.strip(), flush=True)
            if not all(s == 'validated' for s in state['jobs']):
                time.sleep(45)
        results = {}
        for key, base in plan['bases'].items():
            recipe, seed = key.rsplit('-', 1)
            roots = [base['root']] + [str(Path(j['output']) / 'next_run') for j in plan['jobs']
                                     if j['recipe'] == recipe and j['seed'] == int(seed)]
            oof, metrics = aggregate(roots, splits, SPLIT, meta)
            output = STATE.parent / key
            output.mkdir(parents=True, exist_ok=True)
            oof.to_csv(output / 'oof.csv')
            results[key] = {'metrics': metrics, 'oof_sha256': sha256(output / 'oof.csv')}
        write_json('experiments/results/P12-confirmation-oof.json', {
            'status': 'all_four_oof_sets_validated', 'results': results, 'promotion_eligible': False,
            'note': 'Paired bootstrap, per-target deployed-blend gate, remaining fold0 replays and exact final runtime still required.'})
        state['status'] = 'complete'
        write_json(STATE, state)
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        write_json(STATE, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        with (STATE.parent / 'queue.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()
