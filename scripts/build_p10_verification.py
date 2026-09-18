from pathlib import Path
import csv
from rsna_next.contracts import write_json, sha256

p = Path("kaggle/rsna-p10-duplicate-replay")
p.mkdir(parents=True, exist_ok=True)
payload = {str(x): x.read_text() for x in Path("scripts/rsna_next").glob("*.py")}
payload["scripts/__init__.py"] = ""
payload["scripts/p2_submission.py"] = Path("scripts/p2_submission.py").read_text()
with Path("outputs/kaggle/rsna-p10-pixel-audit-v1/result/exact_selected_pixel_matches.csv").open() as handle:
    duplicate_ids = [row["StudyInstanceUID"] for row in csv.DictReader(handle)]
if len(duplicate_ids) != 2 or len(set(duplicate_ids)) != 2:
    raise ValueError("Expected the reviewed two-study duplicate candidate")
code = """import os,sys,json,hashlib,subprocess
from pathlib import Path
ROOT=Path('/kaggle/working/runtime')
PAYLOAD=__PAYLOAD__
for name,text in PAYLOAD.items():
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
sys.path.insert(0,str(ROOT));os.environ['PYTHONPATH']=str(ROOT)
import numpy as np,pandas as pd,pydicom
from scripts.rsna_next.contracts import ID,write_json,sha256
from scripts.rsna_next.data import PixelCache
from scripts.p2_submission import TestStudyDataset
BASE=Path('/kaggle/input')
def find(name):
 paths=list(BASE.glob('**/'+name))
 if len(paths)!=1:raise ValueError((name,len(paths)))
 return paths[0]
train=find('train.csv');meta=pd.read_csv(train);series=pd.read_csv(find('train_series.csv'));dicom=train.parent/'train_series'
ids=__DUPLICATE_IDS__
manifests=[find('p2_cache_manifest_part_a.json'),find('p2_cache_manifest_part_b.json')]
cache=PixelCache(manifests,meta[ID],verify=False)
results=[]
for uid in ids:
 records=series[series[ID]==uid].to_dict('records');raw=[]
 for record in records:
  pixel_hashes=[]
  for file in sorted((dicom/uid/record['SeriesInstanceUID']).glob('*.dcm')):
   ds=pydicom.dcmread(file);a=ds.pixel_array
   h=hashlib.sha256();h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());pixel_hashes.append(h.hexdigest())
  raw.append({'series_uid':record['SeriesInstanceUID'],'plane':record['Anatomical_Plane'],'slices':len(pixel_hashes),'decoded_pixel_multiset_sha256':hashlib.sha256(json.dumps(sorted(pixel_hashes)).encode()).hexdigest()})
 _,x,mask,centers,_=TestStudyDataset([uid],{uid:records},dicom)[0]
 ix=meta.index[meta[ID]==uid][0];cached,cm,_,_=cache.windows(ix,32)
 results.append({ID:uid,'raw_series':raw,'raw_study_multiset_sha256':hashlib.sha256(json.dumps(sorted(r['decoded_pixel_multiset_sha256'] for r in raw)).encode()).hexdigest(),'decoded_slices':sum(r['slices'] for r in raw),'reconstructed_cache_match':bool(np.array_equal(mask.astype(bool),cm) and np.array_equal(x[cm],cached[cm])),'reconstructed_windows_sha256':hashlib.sha256(x[mask.astype(bool)].tobytes()).hexdigest()})
verified=all(r['reconstructed_cache_match'] for r in results) and results[0]['reconstructed_windows_sha256']==results[1]['reconstructed_windows_sha256']
write_json('/kaggle/working/duplicate_verification.json',{'status':'verified_image_overlap' if verified else 'requires_investigation','studies':results,'full_decoded_study_equal':results[0]['raw_study_multiset_sha256']==results[1]['raw_study_multiset_sha256'],'source_train_sha256':sha256(train),'cache_manifest_sha256':[sha256(p) for p in manifests]})
if not verified:raise ValueError('Duplicate not reproduced from raw DICOM')
run=find('best.pt').parent
split=find('splits.csv')
subprocess.run([sys.executable,'-m','scripts.rsna_next.replay','--run',str(run),'--train',str(train),'--splits',str(split),'--cache-manifests',*map(str,manifests),'--weights',str(find('p6_localization_backbone_best.safetensors')),'--output','/kaggle/working/replay.json'],check=True)
""".replace("__PAYLOAD__", repr(payload)).replace("__DUPLICATE_IDS__", repr(duplicate_ids))
(p / "run.py").write_text(code)
compile(code, str(p / "run.py"), "exec")
write_json(
    p / "kernel-metadata.json",
    {
        "id": "vallepuvamsikrishna/rsna-knee-p10-duplicate-replay",
        "title": "RSNA Knee P10 Duplicate Replay",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["vallepuvamsikrishna/rsna-p6-fastmri-localization-init"],
        "kernel_sources": [
            "vallepuvamsikrishna/rsna-knee-p2-wide-cache-a",
            "vallepuvamsikrishna/rsna-knee-p2-wide-cache-b",
            "vallepuvamsikrishna/rsna-knee-next-teacher-fastmri-report-0-smoke",
        ],
    },
)
write_json(
    p / "build.json",
    {
        "code_sha256": sha256(p / "run.py"),
        "purpose": "raw duplicate verification and full v1 smoke replay",
    },
)
