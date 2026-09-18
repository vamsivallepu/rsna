"""Freeze all six completed checkpoints, replay every prediction, then build labels."""
import base64
import json
import zlib
from pathlib import Path

from rsna_next.contracts import sha256, write_json


def main():
    out = Path('kaggle/rsna-p12-replay-labels')
    split = Path('outputs/experiments/P10-contract-v2/splits.csv')
    roots = sorted(Path('outputs/kaggle').glob('p10-teachers-*-v1/next_run/fold-*'))
    assert len(roots) == 5
    roots.append(Path('outputs/kaggle/p12-fastmri-baseline-f0-v1/next_run'))
    expected = []
    for root in roots:
        m = json.loads((root / 'manifest.json').read_text())
        assert m['split_sha256'] == sha256(split)
        expected.append({'manifest_sha256': sha256(root / 'manifest.json'),
                         'mode': m['mode'], 'fold': m['fold']})
    payload = {str(p): p.read_text() for p in Path('scripts/rsna_next').glob('*.py')}
    payload['scripts/__init__.py'] = ''
    payload['splits.csv.b64z'] = base64.b64encode(zlib.compress(split.read_bytes())).decode()
    payload['splits.groups.json'] = split.with_suffix('.groups.json').read_text()
    program = '''import os, sys, json, base64, zlib, subprocess
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
    paths = list(BASE.glob('**/' + name))
    if len(paths) != 1:
        raise ValueError((name, len(paths)))
    return paths[0]
expected = __EXPECTED__
available = {}
for p in BASE.glob('**/manifest.json'):
    h = sha256(p)
    if h in {r['manifest_sha256'] for r in expected}:
        if h in available:
            raise ValueError('Ambiguous checkpoint mount')
        available[h] = p.parent
if len(available) != 6:
    raise ValueError('Missing pinned checkpoint manifests')
common = ['--train', str(find('train.csv')), '--splits', str(ROOT / 'splits.csv')]
cache = [str(find('p2_cache_manifest_part_a.json')), str(find('p2_cache_manifest_part_b.json'))]
replays, teachers = [], []
for entry in expected:
    run = available[entry['manifest_sha256']]
    output = Path('/kaggle/working') / f"replay-{entry['mode']}-{entry['fold']}.json"
    subprocess.run([sys.executable, '-m', 'scripts.rsna_next.replay', '--run', str(run),
        *common, '--cache-manifests', *cache, '--weights', str(find('p6_localization_backbone_best.safetensors')),
        '--output', str(output)], check=True)
    result = json.loads(output.read_text())
    if result['status'] != 'replay_validated' or result['run_manifest_sha256'] != entry['manifest_sha256']:
        raise ValueError('Replay gate failed')
    replays.append(result)
    if entry['mode'] == 'teacher':
        teachers.append(str(run))
subprocess.run([sys.executable, '-m', 'scripts.rsna_next.labels', *common,
    '--targets', str(find('training_targets.csv')), '--teacher-roots', *teachers,
    '--arm', 'distilled', '--output', '/kaggle/working/student_targets.csv'], check=True)
write_json('/kaggle/working/replay_labels_gate.json', {'status': 'all_replays_and_labels_passed',
    'replays': replays, 'split_sha256': sha256(ROOT / 'splits.csv'),
    'targets_sha256': sha256('/kaggle/working/student_targets.csv')})
'''.replace('__PAYLOAD__', repr(payload)).replace('__EXPECTED__', repr(expected))
    compile(program, 'run.py', 'exec')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run.py').write_text(program)
    write_json(out / 'kernel-metadata.json', {
        'id': 'vallepuvamsikrishna/rsna-knee-p12-replay-labels',
        'title': 'RSNA Knee P12 Replay Labels', 'code_file': 'run.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True,
        'enable_tpu': False, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'dataset_sources': ['vallepuvamsikrishna/rsna-p6-fastmri-localization-init'],
        'kernel_sources': ['vallepuvamsikrishna/' + s for s in [
            'rsna-knee-p2-wide-cache-a', 'rsna-knee-p2-wide-cache-b',
            'rsna-knee-report-weak-supervision', 'rsna-knee-teacher-bank-012-s20260917',
            'rsna-knee-teacher-bank-34-s20260917',
            'rsna-knee-next-student-fastmri-report-0-s20260917']]})
    write_json(out / 'build.json', {'code_sha256': sha256(out / 'run.py'),
        'expected_runs': expected, 'split_sha256': sha256(split),
        'source_sha256': {str(p): sha256(p) for p in Path('scripts/rsna_next').glob('*.py')}})
    print(out)


if __name__ == '__main__':
    main()
