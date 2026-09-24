"""Materialize the one frozen exploratory benchmark and submission notebook."""
import json
from pathlib import Path
from build_next_kernels import build_runtime
from rsna_next.contracts import sha256, write_json


def main():
    spec_path=Path('experiments/P12-exploratory-submission.json')
    spec=json.loads(spec_path.read_text())
    champion=Path('kaggle/rsna-p8-blend-canary/rsna-p8-blend-canary.py')
    assert sha256(champion)=='69f1d0c828d218b445f3571ec879c4ab689b93ca5c7e1e00926dbad35777a9fe'
    bench=Path('outputs/releases/p12-champion-benchmark')
    build_runtime(bench,Path('outputs/kaggle/rsna-p10-header-audit-v1/result/benchmark_studies.csv'))
    payload={str(p):p.read_text() for p in Path('scripts/rsna_next').glob('*.py')}
    payload.update({'scripts/__init__.py':'','scripts/p12_exploratory.py':Path('scripts/p12_exploratory.py').read_text(),
                    'scripts/p2_submission.py':Path('scripts/p2_submission.py').read_text(),
                    'champion.py':champion.read_text(),'champion_benchmark.py':(bench/'run.py').read_text(),
                    'spec.json':spec_path.read_text()})
    base=json.loads(Path('kaggle/rsna-p8-blend-canary/kernel-metadata.json').read_text())
    for benchmark in [True,False]:
        slug='rsna-p12-exploratory-'+('benchmark' if benchmark else 'submission')
        output=Path('kaggle')/slug;output.mkdir(exist_ok=True)
        program="""import os,sys,subprocess
from pathlib import Path
ROOT=Path('/kaggle/working/exploratory_runtime')
PAYLOAD=__PAYLOAD__
for name,text in PAYLOAD.items():
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
os.environ['PYTHONPATH']=str(ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/p12_exploratory.py'),*__ARGS__],check=True)
""".replace('__PAYLOAD__',repr(payload)).replace('__ARGS__',repr(['--benchmark'] if benchmark else []))
        compile(program,'run.py','exec');(output/'run.py').write_text(program)
        meta={**base,'id':'vallepuvamsikrishna/'+slug,'title':slug.replace('-',' ').title(),'code_file':'run.py',
              'dataset_sources':['vallepuvamsikrishna/rsna-p12-exploratory-duration-models','vallepuvamsikrishna/rsna-p6-fastmri-localization-init']}
        if benchmark:meta['kernel_sources']=base['kernel_sources']+['vallepuvamsikrishna/rsna-knee-p2-wide-cache-a','vallepuvamsikrishna/rsna-knee-p2-wide-cache-b']
        write_json(output/'kernel-metadata.json',meta)
        write_json(output/'build.json',{'spec_sha256':sha256(spec_path),'code_sha256':sha256(output/'run.py'),
                   'benchmark':benchmark,'driver_sha256':sha256('scripts/p12_exploratory.py'),
                   'champion_sha256':sha256(champion),'selection_sha256':sha256('outputs/kaggle/rsna-p10-header-audit-v1/result/benchmark_studies.csv')})
        print(output)


if __name__=='__main__':main()
