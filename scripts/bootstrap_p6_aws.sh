#!/usr/bin/env bash
set -euo pipefail

P6_ROOT="${P6_ROOT:-/mnt/rsna}"
P6_SCRATCH="${P6_SCRATCH:-/opt/dlami/nvme}"
P6_OWNER="${P6_OWNER:-ubuntu}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/../requirements/p6-aws.txt"
VENV="${P6_ROOT}/.venv"
PIP_CACHE="${P6_SCRATCH}/pip-cache"

if ! dpkg-query --show --showformat='${Status}' python3.10-venv 2>/dev/null \
  | grep -q 'install ok installed'; then
  sudo apt-get update -y
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv
fi

sudo install -d -o "${P6_OWNER}" -g "${P6_OWNER}" \
  "${P6_ROOT}" "${P6_ROOT}/logs" "${PIP_CACHE}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  sudo -u "${P6_OWNER}" python3 -m venv --clear "${VENV}"
fi
sudo chown -R "${P6_OWNER}:${P6_OWNER}" "${VENV}"

sudo -u "${P6_OWNER}" env PIP_CACHE_DIR="${PIP_CACHE}" \
  "${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
sudo -u "${P6_OWNER}" env PIP_CACHE_DIR="${PIP_CACHE}" \
  "${VENV}/bin/python" -m pip install --requirement "${REQUIREMENTS}"

sudo -u "${P6_OWNER}" "${VENV}/bin/python" -c \
  "import json, cv2, pydicom, timm, torch; x=torch.randn((2048,2048), device='cuda', dtype=torch.float16); y=x@x; torch.cuda.synchronize(); print(json.dumps({'torch':torch.__version__,'timm':timm.__version__,'pydicom':pydicom.__version__,'opencv':cv2.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),'matmul_finite':bool(torch.isfinite(y).all().item())}))"
sudo -u "${P6_OWNER}" "${VENV}/bin/python" -m pip freeze \
  > "${P6_ROOT}/logs/p6-environment-freeze.txt"
