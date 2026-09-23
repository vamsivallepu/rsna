"""Run one fixed, budget-bounded P12 batch; never dispatch more cloud resources."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/opt/rsna')


def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def validate_batch(batch):
    final = batch.get('batch_id') == 'remaining-confirmation'
    expected = ({(r, s, 4) for r in ('fastmri-report', 'duration16')
                 for s in (20260917, 20260918)} |
                {(r, 20260918, 2) for r in ('fastmri-report', 'duration16')}) if final else {
                    (r, s, 3) for r in ('fastmri-report', 'duration16') for s in (20260917, 20260918)}
    if len(batch['jobs']) != len(expected) or {
            (j['recipe'], j['seed'], j['fold']) for j in batch['jobs']} != expected:
        raise ValueError('Batch does not match the fixed approved jobs')
    hours = 36 if final else 24
    ceiling = 36 * 1.212 + .001 if final else 65
    if batch['maximum_hours'] != hours or not 0 < batch['compute_ceiling_usd'] <= ceiling:
        raise ValueError('Batch exceeds its compute limit')
    return 'results/final' if final else 'results'


def main():
    import boto3
    os.chdir(ROOT)
    batch = json.loads((ROOT / 'batch.json').read_text())
    result_name = validate_batch(batch)
    bucket = os.environ['RSNA_BUCKET']
    s3 = boto3.client('s3', region_name='ap-south-1')
    result = ROOT / result_name
    result.mkdir(parents=True, exist_ok=True)
    token = Path.home() / '.kaggle/access_token'
    token.parent.mkdir(mode=0o700, exist_ok=True)
    token.write_text(boto3.client('ssm', region_name='ap-south-1').get_parameter(
        Name='/rsna/p6/p12-kaggle-token', WithDecryption=True)['Parameter']['Value'])
    token.chmod(0o600)
    sources = [('cache-a', 'vallepuvamsikrishna/rsna-knee-p2-wide-cache-a'),
               ('cache-b', 'vallepuvamsikrishna/rsna-knee-p2-wide-cache-b')]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, [str(ROOT / 'kaggle-env/bin/python'),
                       str(ROOT / 'scripts/download_kernel_stream.py'), slug, str(ROOT / name)])
                       for name, slug in sources]
            for future in futures:
                future.result()
        run(['kaggle', 'datasets', 'download', '-d',
             'vallepuvamsikrishna/rsna-p6-fastmri-localization-init',
             '-p', str(ROOT / 'weights'), '--unzip'])
    finally:
        token.unlink(missing_ok=True)
    weights = list((ROOT / 'weights').rglob('p6_localization_backbone_best.safetensors'))
    assert len(weights) == 1
    common = ['--train', str(ROOT / 'train.csv'), '--splits', str(ROOT / 'splits.csv')]
    images = ['--cache-manifests', str(ROOT / 'cache-a/p2_cache_manifest_part_a.json'),
              str(ROOT / 'cache-b/p2_cache_manifest_part_b.json'), '--weights', str(weights[0])]
    env = {**os.environ, 'RSNA_SHARED_GPU': '1', 'OMP_NUM_THREADS': '2',
           'PYTHONPATH': str(ROOT), 'GLOO_SOCKET_IFNAME': 'lo'}
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                '--format=csv', '-l', '5'], stdout=(result / 'gpu.csv').open('w'))

    def upload():
        for path in result.rglob('*'):
            if path.is_file():
                s3.upload_file(str(path), bucket, result_name + '/' + str(path.relative_to(result)))

    try:
        for index, job in enumerate([batch['jobs'][0], *batch['jobs']]):
            smoke = index == 0
            name = 'smoke' if smoke else f"{job['recipe']}-f{job['fold']}-s{job['seed']}"
            output = result / name
            config = ROOT / (name + '.json')
            config.write_text(json.dumps(job['config']))
            args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                    '-m', 'scripts.rsna_next.train', '--config', str(config), '--mode', 'student',
                    '--fold', str(job['fold']), '--targets', str(ROOT / 'training_targets.csv'),
                    '--output', str(output), *common, *images]
            if smoke:
                args.append('--smoke')
            started = time.monotonic()
            with (result / (name + '.log')).open('w') as log:
                run(args, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800 if smoke else 8*3600)
                run([sys.executable, '-m', 'scripts.rsna_next.replay', '--run', str(output),
                     '--output', str(output / 'replay.json'), *common, *images],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            manifest = json.loads((output / 'manifest.json').read_text())
            assert manifest['world_size'] == 2 and manifest['environment']['shared_gpu']
            if smoke:
                assert manifest['peak_cuda_memory_mib'] < 20000
                assert time.monotonic() - started < 1800
            upload()
        (result / 'batch-complete.json').write_text(json.dumps({'status': 'complete', 'jobs': len(batch['jobs'])}))
    finally:
        monitor.terminate()
        monitor.wait()
        upload()


if __name__ == '__main__':
    main()
