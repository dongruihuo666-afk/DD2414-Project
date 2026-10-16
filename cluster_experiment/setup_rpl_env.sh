#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/Midgard/home/hongen/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dd2414-hybrid}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
REQUIREMENTS="${REPO_ROOT}/cluster_experiment/requirements-rpl.txt"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "Miniforge conda was not found at ${CONDA_BIN}." >&2
    echo "Set MINIFORGE_ROOT if it is installed elsewhere." >&2
    exit 1
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "Missing RPL requirements file: ${REQUIREMENTS}" >&2
    exit 1
fi

environment_exists=false
if "${CONDA_BIN}" env list | awk -v name="${CONDA_ENV_NAME}" \
    '$1 == name { found=1 } END { exit !found }'; then
    environment_exists=true
fi

if [[ "${environment_exists}" == false ]]; then
    echo "Creating Conda environment ${CONDA_ENV_NAME} with Python 3.10."
    "${CONDA_BIN}" create -y -n "${CONDA_ENV_NAME}" python=3.10 pip
else
    echo "Using existing Conda environment ${CONDA_ENV_NAME}; it will not be deleted or recreated."
fi

if ! ENV_PREFIX=$("${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" \
    python -c 'import sys; print(sys.prefix)'); then
    echo "Conda knows ${CONDA_ENV_NAME}, but its Python interpreter cannot run." >&2
    exit 1
fi
PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Environment Python is missing: ${PYTHON_BIN}" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f'Existing environment uses Python {sys.version.split()[0]}; '
        'dd2414-hybrid requires Python 3.10 and was not modified.'
    )
print(f'Python version: {sys.version.split()[0]}')
PY

echo "Updating packaging tools from ordinary PyPI."
"${PYTHON_BIN}" -m pip install --index-url "${PYPI_INDEX_URL}" --upgrade \
    pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

torch_is_correct=false
if "${PYTHON_BIN}" - <<'PY'
import torch
import torchvision

torch_base = torch.__version__.split('+', 1)[0]
vision_base = torchvision.__version__.split('+', 1)[0]
raise SystemExit(0 if (
    torch_base == '2.5.1'
    and vision_base == '0.20.1'
    and torch.version.cuda == '12.4'
    and getattr(torchvision.version, 'cuda', None) == 12040
    and torchvision.extension._HAS_OPS
) else 1)
PY
then
    torch_is_correct=true
fi

if [[ "${torch_is_correct}" == false ]]; then
    echo "Installing the official PyTorch 2.5.1 CUDA 12.4 wheels."
    # --force-reinstall is required when an environment already contains the
    # same public version number backed by a CPU-only wheel.
    "${PYTHON_BIN}" -m pip install --index-url "${PYTORCH_INDEX_URL}" \
        --upgrade --force-reinstall \
        torch==2.5.1 torchvision==0.20.1
else
    echo "PyTorch 2.5.1 / TorchVision 0.20.1 with CUDA 12.4 is already installed."
fi

echo "Installing pinned project dependencies from ordinary PyPI."
# RPL is x86_64 Linux. Refuse source builds except efficientnet-pytorch, whose
# upstream 0.7.1 release is a small pure-Python sdist with no native compiler.
# This prevents an unexpected CUDA/C++ build from starting on the login node.
"${PYTHON_BIN}" -m pip install --index-url "${PYPI_INDEX_URL}" \
    --upgrade --only-binary=:all: --no-binary=efficientnet-pytorch \
    -r "${REQUIREMENTS}"

echo "Checking the resolved dependency graph."
"${PYTHON_BIN}" -m pip check

verify_args=(--mode cpu)
if [[ -n "${BEVCAR_SOURCE_DIR:-}" ]]; then
    verify_args+=(--bevcar-source "${BEVCAR_SOURCE_DIR}")
elif [[ -f "${REPO_ROOT}/external/BEVCar/nets/voxelnet.py" ]]; then
    verify_args+=(--bevcar-source "${REPO_ROOT}/external/BEVCar")
fi

echo "Checking imports without touching a GPU."
MPLBACKEND=Agg PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/cluster_experiment/verify_rpl_env.py" \
    "${verify_args[@]}"

echo
echo "RPL Python environment is ready."
echo "Environment prefix: ${ENV_PREFIX}"
echo "Python executable: ${PYTHON_BIN}"
printf "Copy this command into the job environment:\nexport PYTHON_BIN=%q\n" "${PYTHON_BIN}"
