#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

pwd
uname -a
cat /etc/os-release
whoami
nvidia-smi
conda env list
which python
python --version
python - <<'PY'
import importlib
import torch
import torchvision

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
for module in ('fire', 'efficientnet_pytorch', 'nuscenes', 'cv2',
               'tensorboardX', 'matplotlib', 'sklearn', 'skimage'):
    imported = importlib.import_module(module)
    print(f'{module}: {getattr(imported, "__version__", "installed")}')
PY
python -m pip check
