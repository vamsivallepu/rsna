"""Build a pinned, full-prediction replay of the remaining P12 student screens."""
import base64
import json
import zlib
from pathlib import Path

from rsna_next.contracts import sha256, write_json


def main():
    out = Path('kaggle/rsna-p12-student-replay')
    split = Path('outputs/experiments/P10-contract-v2/splits.csv')
    runs = []
    for name, slug, weights in [
        ('p12-imagenet-report-f0-v1', 'rsna-knee-next-student-imagenet-report-0-s20260917', 'model.safetensors'),
        ('p12-fastmri-distilled-f0-v1', 'rsna-student-fastmri-distilled-0-s20260917', 'p6_localization_backbone_best.safetensors'),
    ]:
        root = Path('outputs/kaggle') / name / 'next_run'
        m = json.loads((root / 'manifest.json').read_text())
        assert m['split_sha256'] == sha256(split)
        for source in ['train.py', 'models.py', 'data.py', 'contracts.py']:
            assert sha256(Path('scripts/rsna_next') / source) == m['source_sha256'][source]
        runs.append({'slug': slug, 'weights': weights, 'manifest_sha256': sha256(root / 'manifest.json'),
                     'checkpoint_sha256': m['checkpoint_sha256'], 'predictions_sha256': m['predictions_sha256'],
                     'rows': len(m['prediction_ids'])})
    payload = {str(p): p.read_text() for p in Path('scripts/rsna_next').glob('*.py')}
    payload['scripts/__init__.py'] = ''
    payload['splits.csv.b64z'] = base64.b64encode(zlib.compress(split.read_bytes())).decode()
    payload['splits.groups.json'] = split.with_suffix('.groups.json').read_text()
    program = '''import sys, os, json, base64, zlib, subprocess
from pathlib import Path
ROOT = Path('/kaggle/working/replay_runtime')
for name, text in __PAYLOAD__.items():
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
(ROOT / 'splits.csv').write_bytes(zlib.decompress(base64.b64decode((ROOT / 'splits.csv.b64z').read_text())))
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(ROOT)
from scripts.rsna_next.contracts import sha256, write_json
BASE = Path('/kaggle/input')
def find(name):
    found = list(BASE.glob('**/' + name))
    if len(found) != 1:
        raise ValueError((name, len(found)))
    return found[0]
runs = __RUNS__
available = {}
for path in BASE.glob('**/manifest.json'):
    digest = sha256(path)
    if digest in {r['manifest_sha256'] for r in runs}:
        if digest in available:
            raise ValueError('Duplicate mounted manifest')
        available[digest] = path.parent
if len(available) != len(runs):
    raise ValueError('Missing pinned student run')
results = []
for i, run in enumerate(runs):
    output = Path('/kaggle/working') / f'replay-{i}.json'
    subprocess.run([sys.executable, '-m', 'scripts.rsna_next.replay',
        '--run', str(available[run['manifest_sha256']]), '--train', str(find('train.csv')),
        '--splits', str(ROOT / 'splits.csv'), '--weights', str(find(run['weights'])),
        '--cache-manifests', str(find('p2_cache_manifest_part_a.json')), str(find('p2_cache_manifest_part_b.json')),
        '--output', str(output)], check=True)
    result = json.loads(output.read_text())
    if result['status'] != 'replay_validated' or result['run_manifest_sha256'] != run['manifest_sha256']:
        raise ValueError('Incomplete replay')
    for key in ['rows', 'checkpoint_sha256', 'predictions_sha256']:
        if result[key] != run[key]:
            raise ValueError(f'Replay provenance mismatch: {key}')
    results.append({'slug': run['slug'], **result})
write_json('/kaggle/working/student_replay_gate.json', {'status': 'all_student_replays_passed',
    'replays': results, 'split_sha256': sha256(ROOT / 'splits.csv')})
'''.replace('__PAYLOAD__', repr(payload)).replace('__RUNS__', repr(runs))
    compile(program, 'run.py', 'exec')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run.py').write_text(program)
    write_json(out / 'kernel-metadata.json', {
        'id': 'vallepuvamsikrishna/rsna-p12-student-replay', 'title': 'RSNA P12 Student Replay',
        'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'dataset_sources': ['vallepuvamsikrishna/rsna-p6-fastmri-localization-init',
                            'vallepuvamsikrishna/rsna-coatnet-rmlp2-imagenet-init'],
        'kernel_sources': ['vallepuvamsikrishna/' + slug for slug in
                           ['rsna-knee-p2-wide-cache-a', 'rsna-knee-p2-wide-cache-b'] + [r['slug'] for r in runs]]})
    write_json(out / 'build.json', {'code_sha256': sha256(out / 'run.py'), 'expected_runs': runs,
                                   'split_sha256': sha256(split)})
    print(out)


if __name__ == '__main__':
    main()
