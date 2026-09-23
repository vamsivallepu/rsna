#!/bin/bash
set -euo pipefail
shutdown -h "+${RSNA_MAX_MINUTES:-1440}"
mkdir -p /opt/rsna
# Disposable caches use local NVMe; checkpoints and logs stay on persistent EBS.
if mountpoint -q /opt/dlami/nvme; then
  mkdir -p /opt/dlami/nvme/rsna-p12
  for part in cache-a cache-b; do
    if [[ -d /opt/rsna/$part && ! -L /opt/rsna/$part && ! -e /opt/dlami/nvme/rsna-p12/$part ]]; then
      mv "/opt/rsna/$part" "/opt/dlami/nvme/rsna-p12/$part"
    fi
    mkdir -p "/opt/dlami/nvme/rsna-p12/$part"
    if [[ ! -e /opt/rsna/$part && ! -L /opt/rsna/$part ]]; then
      ln -s "/opt/dlami/nvme/rsna-p12/$part" "/opt/rsna/$part"
    fi
  done
fi
exec > >(tee -a /opt/rsna/bootstrap.log) 2>&1
finish() {
  status=$?
  if [[ -x /opt/rsna/.venv/bin/python ]]; then
    timeout 60 /opt/rsna/.venv/bin/python - <<'INNER' || true
import boto3, pathlib, os
bucket = os.environ.get('RSNA_BUCKET')
if bucket:
    s = boto3.client('s3', region_name='ap-south-1')
    for name in ['bootstrap.log', 'environment.txt']:
        path = pathlib.Path('/opt/rsna') / name
        if path.exists():
            s.upload_file(str(path), bucket, os.environ.get('RSNA_RESULTS_PREFIX', 'results') + '/' + name)
INNER
  fi
  shutdown -h now
  exit "$status"
}
trap finish EXIT
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
boto3.client('s3',region_name='ap-south-1').download_file(os.environ['RSNA_BUCKET'],os.environ.get('RSNA_SOURCE_KEY','staging/source.tar.gz'),p)
assert hashlib.sha256(open(p,'rb').read()).hexdigest() == os.environ['RSNA_SOURCE_SHA256']
with tarfile.open(p) as t: t.extractall('/opt/rsna',filter='data')
INNER
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install timm==1.0.26 numpy==2.2.6 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.6.1 safetensors==0.5.3 uv==0.8.22
# Kaggle >=2 requires Python >=3.11; keep the validated training stack on 3.10.
if [[ ! -x /opt/rsna/kaggle-env/bin/python ]]; then
  uv venv --python 3.12 /opt/rsna/kaggle-env
fi
uv pip install --python /opt/rsna/kaggle-env/bin/python kaggle==2.2.4
ln -sf /opt/rsna/kaggle-env/bin/kaggle /opt/rsna/.venv/bin/kaggle
pip freeze > /opt/rsna/environment.txt
cd /opt/rsna
python scripts/aws_p12_batch.py
