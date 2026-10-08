#!/usr/bin/env bash
set -euo pipefail

# Generic STAR PPO entry for Kubeflow PyTorchJob.
#
# PyTorchJob should launch this same script in every replica. Rank 0 becomes the
# Ray head and runs training; other ranks join the Ray cluster and block.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-true}"

RANK="${RANK:-${PET_NODE_RANK:-${GROUP_RANK:-0}}}"
WORLD_SIZE="${WORLD_SIZE:-${PET_NNODES:-${GROUP_WORLD_SIZE:-1}}}"
MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP:-}}"
MASTER_PORT="${MASTER_PORT:-${RAY_PORT:-6379}}"

export RANK WORLD_SIZE
export HEAD_IP="${HEAD_IP:-${MASTER_ADDR}}"
export RAY_PORT="${RAY_PORT:-${MASTER_PORT}}"
export CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"

if [[ -z "${HEAD_IP}" ]]; then
  echo "[common/run_pytorchjob] ERROR: MASTER_ADDR or HEAD_IP is required." >&2
  echo "[common/run_pytorchjob] In PyTorchJob, set MASTER_ADDR to the rank-0 pod/service address." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${ACTIVATE_CONDA}" == "true" ]]; then
  if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "[common/run_pytorchjob] ERROR: conda not found at ${CONDA_ROOT}" >&2
    exit 1
  fi
  # Some conda deactivate.d scripts are not nounset-safe; temporarily disable set -u.
  set +u
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  set -u
fi

echo "[common/run_pytorchjob] RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} HEAD_IP=${HEAD_IP} RAY_PORT=${RAY_PORT}"

exec bash "${SCRIPT_DIR}/run_per_node.sh" "$@"
