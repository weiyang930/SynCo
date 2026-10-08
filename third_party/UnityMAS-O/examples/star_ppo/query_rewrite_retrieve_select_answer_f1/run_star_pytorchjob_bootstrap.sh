#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
DO_ENV_SETUP="${DO_ENV_SETUP:-false}"

if [[ "${DO_ENV_SETUP}" == "true" ]]; then
  echo "[bootstrap] DO_ENV_SETUP=true, running environment setup..."
  bash "${SCRIPT_DIR}/setup_verl_env.sh"
fi

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "[bootstrap] conda not found at ${CONDA_ROOT}"
  echo "[bootstrap] run once with DO_ENV_SETUP=true or execute setup_verl_env.sh manually."
  exit 1
fi

# Some conda deactivate.d scripts are not nounset-safe; temporarily disable `set -u`.
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  set -u
  echo "[bootstrap] conda env '${CONDA_ENV_NAME}' not found."
  echo "[bootstrap] run once with DO_ENV_SETUP=true or execute setup_verl_env.sh manually."
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"
set -u

echo "[bootstrap] using existing env '${CONDA_ENV_NAME}', starting rank-routed entry..."
bash "${SCRIPT_DIR}/run_star_pytorchjob_entry.sh"
