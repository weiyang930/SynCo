#!/usr/bin/env bash
set -euo pipefail

# Use PyTorchJob env vars to bootstrap a Ray cluster:
# - RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
# Rank 0 starts Ray head and launches training.
# Other ranks join as Ray workers and block.

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP:-}}"
MASTER_PORT="${MASTER_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR_FILE="${MASTER_ADDR_FILE:-}"

TRAIN_PARQUET="${TRAIN_PARQUET:-}"
VAL_PARQUET="${VAL_PARQUET:-}"
CONFIG_NAME="${CONFIG_NAME:-star_query_rewrite_retrieve_select_answer_f1_trainer}"
PROJECT_NAME="${PROJECT_NAME:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:-}"
SELECT_MODEL_PATH="${SELECT_MODEL_PATH:-${REWRITE_MODEL_PATH:-}}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-${REWRITE_MODEL_PATH:-}}"
DECOMPOSE_MODEL_PATH="${DECOMPOSE_MODEL_PATH:-${REWRITE_MODEL_PATH:-}}"
SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-${REWRITE_MODEL_PATH:-}}"
ALLOW_ACTOR_MODEL_OVERRIDE="${ALLOW_ACTOR_MODEL_OVERRIDE:-false}"
RETRIEVAL_API_URLS_JSON="${RETRIEVAL_API_URLS_JSON:-[\"http://retriever.example.com:8000/retrieve\"]}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-300}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-5}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-2}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
STAR_MAX_INFLIGHT_QUERIES="${STAR_MAX_INFLIGHT_QUERIES:-${GEN_BATCH_SIZE}}"
STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL="${STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL:-16}"
VERL_VLLM_FORCE_SHM_WEIGHT_SYNC="${VERL_VLLM_FORCE_SHM_WEIGHT_SYNC:-1}"
STAR_LLM_TIMEOUT_SECONDS="${STAR_LLM_TIMEOUT_SECONDS:-0}"
STAR_QUERY_TIMEOUT_SECONDS="${STAR_QUERY_TIMEOUT_SECONDS:-600}"
STAR_TOOL_TIMEOUT_SECONDS="${STAR_TOOL_TIMEOUT_SECONDS:-30}"
STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS="${STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS:-900}"
STAR_RAY_GET_TIMEOUT_SECONDS="${STAR_RAY_GET_TIMEOUT_SECONDS:-300}"
STAR_WORKER_CALL_TIMEOUT_SECONDS="${STAR_WORKER_CALL_TIMEOUT_SECONDS:-${STAR_RAY_GET_TIMEOUT_SECONDS}}"
STAR_WEIGHT_SYNC_TIMEOUT_SECONDS="${STAR_WEIGHT_SYNC_TIMEOUT_SECONDS:-600}"
STAR_STALL_DETECT_SECONDS="${STAR_STALL_DETECT_SECONDS:-180}"
STAR_STALL_HEARTBEAT_SECONDS="${STAR_STALL_HEARTBEAT_SECONDS:-30}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"

if [[ -z "${TRAIN_PARQUET}" ]]; then
  unset TRAIN_PARQUET
fi
if [[ -z "${VAL_PARQUET}" ]]; then
  unset VAL_PARQUET
fi

ray stop -f >/dev/null 2>&1 || true

if [[ -z "${MASTER_ADDR}" ]]; then
  if [[ "${RANK}" == "0" ]]; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -z "${MASTER_ADDR}" ]]; then
      MASTER_ADDR="$(hostname -i 2>/dev/null | awk '{print $1}')"
    fi
    if [[ -z "${MASTER_ADDR}" ]]; then
      echo "[star-pytorchjob] failed to auto-detect master IP on rank0"
      exit 1
    fi
    if [[ -n "${MASTER_ADDR_FILE}" ]]; then
      echo "${MASTER_ADDR}" > "${MASTER_ADDR_FILE}"
      echo "[star-pytorchjob] wrote MASTER_ADDR=${MASTER_ADDR} to ${MASTER_ADDR_FILE}"
    fi
  else
    if [[ -z "${MASTER_ADDR_FILE}" ]]; then
      echo "[star-pytorchjob] MASTER_ADDR is empty and MASTER_ADDR_FILE is not set on rank${RANK}"
      exit 1
    fi
    echo "[star-pytorchjob] rank${RANK} waiting for ${MASTER_ADDR_FILE}"
    for _ in $(seq 1 180); do
      if [[ -s "${MASTER_ADDR_FILE}" ]]; then
        MASTER_ADDR="$(cat "${MASTER_ADDR_FILE}")"
        break
      fi
      sleep 2
    done
    if [[ -z "${MASTER_ADDR}" ]]; then
      echo "[star-pytorchjob] timeout waiting MASTER_ADDR_FILE=${MASTER_ADDR_FILE}"
      exit 1
    fi
  fi
fi
export MASTER_ADDR
export VLLM_USE_V1
export WANDB_API_KEY
export CONFIG_NAME
export DECOMPOSE_MODEL_PATH SUMMARY_MODEL_PATH
export ROLLOUT_FREE_CACHE_ENGINE
export ROLLOUT_GPU_MEMORY_UTILIZATION
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE
export ROLLOUT_MAX_NUM_BATCHED_TOKENS
export ROLLOUT_MAX_NUM_SEQS
export STAR_MAX_INFLIGHT_QUERIES
export STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL
export VERL_VLLM_FORCE_SHM_WEIGHT_SYNC
export STAR_LLM_TIMEOUT_SECONDS
export STAR_QUERY_TIMEOUT_SECONDS STAR_TOOL_TIMEOUT_SECONDS
export STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS
export STAR_RAY_GET_TIMEOUT_SECONDS STAR_WORKER_CALL_TIMEOUT_SECONDS
export STAR_WEIGHT_SYNC_TIMEOUT_SECONDS
export STAR_STALL_DETECT_SECONDS STAR_STALL_HEARTBEAT_SECONDS
export ACTOR_PARAM_OFFLOAD
export ACTOR_OPTIMIZER_OFFLOAD
export ALLOW_ACTOR_MODEL_OVERRIDE

if [[ "${RANK}" == "0" ]]; then
  echo "[star-pytorchjob] rank0 starts Ray head at ${MASTER_ADDR}:${MASTER_PORT}"
  ray start --head \
    --node-ip-address="${MASTER_ADDR}" \
    --port="${MASTER_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-cpus="${CPUS_PER_NODE}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats

  echo "[star-pytorchjob] waiting for ${WORLD_SIZE} Ray nodes"
  if [[ "${WORLD_SIZE}" == "1" ]]; then
    echo "[star-pytorchjob] WORLD_SIZE=1, skip waiting for workers"
  else
    python3 - <<PY
import os
import time
import ray

expected = int(os.environ.get("WORLD_SIZE", "4"))
ray.init(address="auto")
timeout = int(os.environ.get("STAR_NODE_WAIT_TIMEOUT", "180"))  # 默认 15 分钟
for i in range(timeout):
    alive = sum(1 for n in ray.nodes() if n.get("Alive", False))
    print(f"[star-pytorchjob] alive nodes: {alive}/{expected}")
    if alive >= expected:
        break
    if i > 0 and i % 6 == 0:
        print("[star-pytorchjob] 提示: 若一直卡住，请确保已在所有 worker 节点执行相同命令")
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected} nodes (waited {timeout*5}s)")
ray.shutdown()
PY
  fi

  echo "[star-pytorchjob] launching training on rank0"
  hydra_overrides=(
    trainer.nnodes="${WORLD_SIZE}"
    trainer.n_gpus_per_node="${GPUS_PER_NODE}"
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"
    star.workflow.tools.retriever.api_urls="${RETRIEVAL_API_URLS_JSON}"
    data.gen_batch_size="${GEN_BATCH_SIZE}"
    data.train_batch_size="${GEN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.test_freq="${TEST_FREQ}"
    ++trainer.val_max_batches="${VAL_MAX_BATCHES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.logger='["console","wandb"]'
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  )
  if [[ -n "${TRAIN_PARQUET:-}" ]]; then
    hydra_overrides+=(data.train_files="${TRAIN_PARQUET}")
  fi
  if [[ -n "${VAL_PARQUET:-}" ]]; then
    hydra_overrides+=(data.val_files="${VAL_PARQUET}")
  fi
  if [[ "${ALLOW_ACTOR_MODEL_OVERRIDE}" == "true" && -n "${ACTOR_MODEL_PATH:-}" ]]; then
    hydra_overrides+=(actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}")
    hydra_overrides+=(actor_rollout_ref.model.tokenizer_path="${ACTOR_MODEL_PATH}")
  elif [[ -n "${ACTOR_MODEL_PATH:-}" ]]; then
    echo "[star-pytorchjob] ACTOR_MODEL_PATH is set but ignored (ALLOW_ACTOR_MODEL_OVERRIDE=false)."
  fi
  if [[ -n "${PROJECT_NAME}" ]]; then
    hydra_overrides+=(trainer.project_name="${PROJECT_NAME}")
  fi
  if [[ -n "${EXPERIMENT_NAME}" ]]; then
    hydra_overrides+=(trainer.experiment_name="${EXPERIMENT_NAME}")
  fi
  python3 -m verl.experimental.star_ppo.main_ppo \
    --config-name "${CONFIG_NAME}" \
    "${hydra_overrides[@]}"
else
  echo "[star-pytorchjob] rank${RANK} waits for Ray head ${MASTER_ADDR}:${MASTER_PORT}"
  python3 - <<'PY'
import os
import socket
import time

host = os.environ["MASTER_ADDR"]
port = int(os.environ.get("MASTER_PORT", "6379"))
for _ in range(180):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        sock.close()
        print(f"[star-pytorchjob] head reachable at {host}:{port}")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError(f"Cannot reach Ray head {host}:{port}")
PY

  echo "[star-pytorchjob] rank${RANK} starts Ray worker"
  ray start \
    --address="${MASTER_ADDR}:${MASTER_PORT}" \
    --num-cpus="${CPUS_PER_NODE}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --block
fi
