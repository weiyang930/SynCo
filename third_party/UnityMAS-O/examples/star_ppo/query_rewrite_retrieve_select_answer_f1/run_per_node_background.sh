#!/usr/bin/env bash
# =============================================================================
# Star PPO 多节点后台运行脚本
# 在每个 node 上分别执行此脚本，以后台方式启动训练并写入日志。
#
# 使用方法：
#   Node 0:  RANK=0 HEAD_IP=<head-ip> WORLD_SIZE=4 bash run_per_node_background.sh
#   Node 1:  RANK=1 HEAD_IP=<head-ip> WORLD_SIZE=4 bash run_per_node_background.sh
#   Node 2:  RANK=2 HEAD_IP=<head-ip> WORLD_SIZE=4 bash run_per_node_background.sh
#   Node 3:  RANK=3 HEAD_IP=<head-ip> WORLD_SIZE=4 bash run_per_node_background.sh
#
# 执行完毕后会打印 PID 和日志路径，可用 tail -f <LOG> 查看输出。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
mkdir -p logs/star_ppo

RANK="${RANK:-0}"
HEAD_IP="${HEAD_IP:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"

if [[ -z "${HEAD_IP}" ]]; then
  echo "[run_per_node_background] ERROR: HEAD_IP is required"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-logs/star_ppo/run_rank${RANK}_${TS}.log}"

echo "[run_per_node_background] Starting RANK=${RANK} in background, log=${LOG_FILE}"
RANK="${RANK}" HEAD_IP="${HEAD_IP}" WORLD_SIZE="${WORLD_SIZE}" nohup stdbuf -oL -eL bash "${SCRIPT_DIR}/run_per_node.sh" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
echo "PID=${PID} LOG=${LOG_FILE}"
echo "  tail -f ${LOG_FILE}"
