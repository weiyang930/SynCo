#!/usr/bin/env bash
set -euo pipefail

HEAD_IP="${HEAD_IP:?HEAD_IP is required}"
RAY_PORT="${RAY_PORT:-6379}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

ray stop -f >/dev/null 2>&1 || true

echo "[star-worker] connecting to ${HEAD_IP}:${RAY_PORT}"
ray start \
  --address="${HEAD_IP}:${RAY_PORT}" \
  --num-cpus="${CPUS_PER_NODE}" \
  --num-gpus="${GPUS_PER_NODE}" \
  --disable-usage-stats \
  --block
