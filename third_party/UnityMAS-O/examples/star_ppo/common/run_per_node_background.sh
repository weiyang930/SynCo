#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
mkdir -p logs/star_ppo

RANK="${RANK:-0}"
HEAD_IP="${HEAD_IP:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"

if [[ -z "${HEAD_IP}" ]]; then
  echo "[common/run_per_node_background] ERROR: HEAD_IP is required"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/star_ppo/run_rank${RANK}_${TS}.log}"

echo "[common/run_per_node_background] starting RANK=${RANK} in background, log=${LOG_FILE}"
RANK="${RANK}" HEAD_IP="${HEAD_IP}" WORLD_SIZE="${WORLD_SIZE}" \
  nohup stdbuf -oL -eL bash "${SCRIPT_DIR}/run_per_node.sh" "$@" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!

echo "PID=${PID} LOG=${LOG_FILE}"
echo "  tail -f ${LOG_FILE}"
