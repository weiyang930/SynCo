#!/usr/bin/env bash
set -euo pipefail

# Idempotent environment setup for each PyTorchJob pod.
# You can override paths/versions through env vars.

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

MINICONDA_INSTALLER="${MINICONDA_INSTALLER:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/Miniconda3-latest-Linux-x86_64.sh}"
INSTALL_STACK_SCRIPT="${INSTALL_STACK_SCRIPT:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/research_project/adaptive_joint_optim/rl/verl/scripts/install_vllm_sglang_mcore_0.7.sh}"
VERL_REPO_PATH="${VERL_REPO_PATH:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/research_project/adaptive_joint_optim/rl/verl}"

PROXY_URL="${PROXY_URL:-}"

echo "[setup] preparing conda env: ${CONDA_ENV_NAME}"
if [[ ! -d "${CONDA_ROOT}" ]]; then
  if [[ ! -f "${MINICONDA_INSTALLER}" ]]; then
    echo "[setup] miniconda installer not found: ${MINICONDA_INSTALLER}"
    exit 1
  fi
  mkdir -p "$(dirname "$CONDA_ROOT")"
  bash "${MINICONDA_INSTALLER}" -b -p "${CONDA_ROOT}"
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV_NAME}"

if [[ -n "${PROXY_URL}" ]]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
fi

if [[ -f "${INSTALL_STACK_SCRIPT}" ]]; then
  bash "${INSTALL_STACK_SCRIPT}"
else
  echo "[setup] skip stack script, file not found: ${INSTALL_STACK_SCRIPT}"
fi

if [[ ! -d "${VERL_REPO_PATH}" ]]; then
  echo "[setup] verl repo path not found: ${VERL_REPO_PATH}"
  exit 1
fi

pip install --no-deps -e "${VERL_REPO_PATH}"
pip install "numpy<2.0"

# H20 compatibility
pip uninstall -y transformers || true
pip install --no-cache-dir "transformers==4.57"

# New code compatibility
pip uninstall -y trl || true
pip install "trl==0.26.2"

# nvcc / CUDA toolchain in current env
CUDA_MM="$(python - <<'PY'
import torch
print(".".join(torch.version.cuda.split(".")[:2]))
PY
)"
conda install -y -c nvidia "cuda-toolkit=${CUDA_MM}"
conda install -y -c nvidia cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CPATH}"
export CPLUS_INCLUDE_PATH="${CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=1

nvcc --version

pip install "debugpy==1.8.0"

if [[ -n "${PROXY_URL}" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
fi

echo "[setup] environment is ready: ${CONDA_ENV_NAME}"
