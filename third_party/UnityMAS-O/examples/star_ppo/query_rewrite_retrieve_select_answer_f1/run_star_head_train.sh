#!/usr/bin/env bash
set -euo pipefail

HEAD_IP="${HEAD_IP:?HEAD_IP is required}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

TRAIN_PARQUET="${TRAIN_PARQUET:?TRAIN_PARQUET is required}"
VAL_PARQUET="${VAL_PARQUET:?VAL_PARQUET is required}"
AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:-${AGENT_MODEL_PATH}}"
SELECT_MODEL_PATH="${SELECT_MODEL_PATH:-${REWRITE_MODEL_PATH}}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-${REWRITE_MODEL_PATH}}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:?REWRITE_MODEL_PATH (or AGENT_MODEL_PATH) is required}"
CONFIG_NAME="${CONFIG_NAME:-star_query_rewrite_retrieve_select_answer_f1_trainer}"
RETRIEVAL_API_URLS_JSON="${RETRIEVAL_API_URLS_JSON:-[\"http://retriever.example.com:8000/retrieve\"]}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-300}"

ray stop -f >/dev/null 2>&1 || true

echo "[star-head] starting ray head at ${HEAD_IP}:${RAY_PORT}"
ray start --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${DASHBOARD_PORT}" \
  --num-cpus="${CPUS_PER_NODE}" \
  --num-gpus="${GPUS_PER_NODE}" \
  --disable-usage-stats

echo "[star-head] waiting for ${WORLD_SIZE} alive nodes"
python3 - <<'PY'
import os
import time
import ray

expected = int(os.environ.get("WORLD_SIZE", "4"))
ray.init(address="auto")
for _ in range(120):
    alive = sum(1 for n in ray.nodes() if n.get("Alive", False))
    print(f"[star-head] alive nodes: {alive}/{expected}")
    if alive >= expected:
        break
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected} nodes")
ray.shutdown()
PY

echo "[star-head] launching training"
python3 -m verl.experimental.star_ppo.main_ppo \
  --config-name "${CONFIG_NAME}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  trainer.nnodes="${WORLD_SIZE}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  actor_rollout_ref.model.path="${REWRITE_MODEL_PATH}" \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  star.workflow.tools.retriever.api_urls="${RETRIEVAL_API_URLS_JSON}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger='["console","wandb"]'
