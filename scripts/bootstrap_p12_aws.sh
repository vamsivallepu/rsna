#!/bin/bash
set -euo pipefail
shutdown -h +1440
mkdir -p /opt/rsna
exec > >(tee -a /opt/rsna/bootstrap.log) 2>&1
trap 'shutdown -h now' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y python3.10-venv
python3.10 -m venv /opt/rsna/.venv
export PATH=/opt/rsna/.venv/bin:$PATH
export RSNA_BUCKET="${RSNA_BUCKET:?Private staging bucket required}"
export RSNA_SOURCE_SHA256="${RSNA_SOURCE_SHA256:?Source hash required}"
pip install boto3
python - <<'INNER'
import boto3,hashlib,tarfile,os
p='/opt/rsna/source.tar.gz'
boto3.client('s3',region_name='ap-south-1').download_file(os.environ['RSNA_BUCKET'],'staging/source.tar.gz',p)
assert hashlib.sha256(open(p,'rb').read()).hexdigest() == os.environ['RSNA_SOURCE_SHA256']
with tarfile.open(p) as t: t.extractall('/opt/rsna',filter='data')
INNER
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install timm==1.0.26 numpy==2.2.6 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.6.1 safetensors==0.5.3 kaggle==2.2.4
pip freeze > /opt/rsna/environment.txt
cd /opt/rsna
set +e
python scripts/aws_p12_batch.py
batch_status=$?
python - <<'INNER'
import boto3,pathlib,os
s=boto3.client('s3',region_name='ap-south-1')
for n in ['bootstrap.log','environment.txt']:
 p=pathlib.Path('/opt/rsna')/n
 if p.exists():s.upload_file(str(p),os.environ['RSNA_BUCKET'],'results/'+n)
INNER
exit "$batch_status"
