"""One frozen exploratory P8/P12 release; benchmark and submit use identical models."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

IMPORT_STARTED = time.perf_counter()

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def fixed_blend(champion, candidate):
    if champion.shape != candidate.shape or champion.ndim != 2 or champion.shape[1] != 12:
        raise ValueError("Mismatched prediction arrays")
    if not np.isfinite(champion).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite predictions")
    return np.clip((.8 * rankdata(champion, axis=0) + .2 * rankdata(candidate, axis=0)) / len(champion), 0., 1.)


def main():
    started = IMPORT_STARTED
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    from scripts.rsna_next.contracts import ID, TARGETS, sha256, write_json, read_splits
    from scripts.rsna_next.train import INIT_HASHES
    from scripts.rsna_next.infer import dicom_windows
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / 'spec.json').read_text())
    def find(name):
        paths = list(Path('/kaggle/input').glob('**/' + name))
        if len(paths) != 1:
            raise ValueError(f'Expected one {name}, found {len(paths)}')
        return paths[0]
    # Kaggle expands uploaded tar archives into the read-only dataset mount.
    models = find('seed20260917-fold0/manifest.json').parent.parent
    for model in spec['models']:
        assert sha256(models/model['name']/'manifest.json') == model['manifest_sha256']
    weights = find('p6_localization_backbone_best.safetensors')
    entries = [dict(run=str(models/m['name']), weights=str(weights),
                    manifest_sha256=m['manifest_sha256'], checkpoint_sha256=m['checkpoint_sha256'],
                    initialization_sha256=INIT_HASHES['fastmri']) for m in spec['models']]
    release = dict(schema_version='rsna-next-v1', rank_blend=False,
                   groups=[dict(weight=1., preprocessing={'kind':'p2'}, models=entries)])
    write_json(root/'release.json', release)
    startup_seconds = time.perf_counter() - started
    if args.benchmark:
        # Replay the only selected model lacking a previous pinned replay.
        from scripts.rsna_next.data import PixelCache
        train = find('train.csv')
        meta = pd.read_csv(train)
        splits = read_splits(models/'splits.csv', meta)
        cache_paths = [find('p2_cache_manifest_part_a.json'),find('p2_cache_manifest_part_b.json')]
        common = ['--train',str(train),'--splits',str(models/'splits.csv'),
                  '--cache-manifests',*map(str,cache_paths),'--weights',str(weights)]
        subprocess.run([sys.executable,'-m','scripts.rsna_next.replay',
                        '--run',str(models/'seed20260918-fold0'),
                        '--output','/kaggle/working/p12_missing_replay.json',*common],check=True)
        # Compare native deployment windows with immutable training-cache windows.
        cache = PixelCache(cache_paths,meta[ID])
        series = pd.read_csv(find('train_series.csv'))
        records = {uid: rows.to_dict('records') for uid,rows in series.groupby(ID)}
        for ix in np.flatnonzero(splits.is_gold):
            uid=meta.iloc[ix][ID]
            raw=dicom_windows(uid,records[uid],train.parent/'train_series',{'kind':'p2'})
            saved=cache.windows(ix,32)
            assert np.array_equal(raw[1],saved[1]), 'Deployment mask mismatch'
            assert np.array_equal(raw[0][raw[1]],saved[0][saved[1]]), 'Deployment pixels mismatch'
        del cache
        write_json('/kaggle/working/p12_geometry_agreement.json',{'status':'passed','studies':int(splits.is_gold.sum())})
    inference_started = time.perf_counter()
    champion = root / ('champion_benchmark.py' if args.benchmark else 'champion.py')
    subprocess.run([sys.executable,str(champion)],check=True)
    champion_path = Path('/kaggle/working') / ('p10_benchmark_predictions.csv' if args.benchmark else 'submission.csv')
    baseline = pd.read_csv(champion_path,dtype={ID:str})
    baseline.to_csv('/kaggle/working/p8_preserved_predictions.csv',index=False)
    inferargs = ['--release',str(root/'release.json'),'--output','/kaggle/working/p12_inference']
    if args.benchmark:
        train=find('train.csv')
        inferargs += ['--benchmark','--studies','/kaggle/working/p10_benchmark_studies.csv',
                      '--series',str(find('train_series.csv')),'--dicom-root',str(train.parent/'train_series')]
    else:
        test=find('test.csv')
        inferargs += ['--studies',str(test),'--series',str(find('test_series.csv')),
                      '--dicom-root',str(test.parent/'test_series'),'--sample-submission',str(find('sample_submission.csv'))]
    subprocess.run([sys.executable,'-m','scripts.rsna_next.infer',*inferargs],check=True)
    candidate=pd.read_csv(Path('/kaggle/working/p12_inference')/('benchmark_predictions.csv' if args.benchmark else 'submission.csv'),dtype={ID:str})
    assert not baseline[ID].duplicated().any() and not candidate[ID].duplicated().any()
    assert set(baseline[ID])==set(candidate[ID])
    candidate=candidate.set_index(ID).loc[baseline[ID]]
    values=fixed_blend(baseline[TARGETS].to_numpy(),candidate[TARGETS].to_numpy())
    baseline[TARGETS]=values
    assert baseline.columns.tolist()==[ID,*TARGETS]
    output=Path('/kaggle/working')/('exploratory_benchmark_predictions.csv' if args.benchmark else 'submission.csv')
    baseline.to_csv(output,index=False)
    elapsed=time.perf_counter()-inference_started
    projected=1.25*(elapsed+startup_seconds)*1300/len(baseline)/3600
    result={'spec_sha256':sha256(root/'spec.json'),'benchmark':args.benchmark,'studies':len(baseline),
            'elapsed_inference_seconds':elapsed,'total_seconds':time.perf_counter()-started,
            'projected_hidden_hours':projected if args.benchmark else None,
            'runtime_gate_passed':bool(args.benchmark and len(baseline)>=200 and projected<=7.5),
            'predictions_sha256':sha256(output),'exploratory':True}
    write_json('/kaggle/working/p12_exploratory_run.json',result)
    if args.benchmark and not result['runtime_gate_passed']:
        raise ValueError('Exact combined runtime gate failed')
    print(result,flush=True)


if __name__=='__main__':
    main()
