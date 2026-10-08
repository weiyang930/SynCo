#!/usr/bin/env bash
# SynCo training: Synthesizer-Reasoner co-training with GRPO on UnityMAS-O star_ppo.
#
# Topology (paper setting): 2 nodes x 8 GPUs. Each policy (Synthesizer, Reasoner; both Qwen3-8B)
# gets one full node: FSDP actor over 8 ranks + a colocated TP=8 vLLM engine.
#
# Ray cluster -- pick one:
#   (a) an existing cluster with >= 16 GPUs:           RAY_ADDRESS=<head_ip>:<port>
#   (b) start it here (this host = head), worker via SLURM:
#         WORKER_NODE_IP=<ip>  WORKER_SLURM_JOBID=<job holding the 2nd node>  [WORKER_NODELIST=<node>]
#   (c) start it here, worker via ssh:                   WORKER_NODE_IP=<ip>  WORKER_SSH=<user@host>
#
# Required:  BASE_MODEL_PATH=/path/to/Qwen3-8B
# Optional:
#   MODE=cotrain (default) | frozen    frozen = Synthesizer not updated, report never refreshed
#   SYNTH_REWARD_MODE=gated (default) | additive
#   RUN_ID=<name>                      artifacts under runs/<RUN_ID>/ (pool, ckpts, tensorboard)
#   TRAIN_QUERIES=<file>               optional custom Synthesizer query file; when omitted,
#                                      query slots are generated from an empty pool
#   TOTAL_EPOCHS=100 SAVE_FREQ=10 TEST_FREQ=50 GEN_BATCH_SIZE=32 REASONER_ROLLOUTS_PER_TASK=4
#   ROLLOUT_GPU_MEMORY_UTILIZATION=0.3 (B200; lower it on smaller GPUs)
#   RESUME_MODE=disable (default) | auto
#
# Example:
#   BASE_MODEL_PATH=/path/to/Qwen3-8B WORKER_NODE_IP=<worker-node-ip> \
#     WORKER_SLURM_JOBID=<slurm-job-id> bash scripts/train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH to the Qwen3-8B checkpoint}"
export BASE_MODEL_PATH
UNITYMAS_ROOT="${UNITYMAS_ROOT:-${ROOT}/third_party/UnityMAS-O}"
[[ -d "${UNITYMAS_ROOT}/verl" ]] || { echo "UnityMAS-O not found at ${UNITYMAS_ROOT}; run scripts/setup_env.sh" >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAY_BIN="${RAY_BIN:-ray}"
export PYTHONPATH="${ROOT}/src:${UNITYMAS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- method switches -------------------------------------------------------------------------
MODE="${MODE:-cotrain}"
case "${MODE}" in
  cotrain) export TRAIN_SYNTHESIZER=true;  export REPORT_REFRESH_EVERY="${REPORT_REFRESH_EVERY:-5}" ;;
  frozen)  export TRAIN_SYNTHESIZER=false; export REPORT_REFRESH_EVERY="${REPORT_REFRESH_EVERY:-0}" ;;
  *) echo "MODE must be cotrain or frozen" >&2; exit 1 ;;
esac
export SYNTH_REWARD_MODE="${SYNTH_REWARD_MODE:-gated}"
[[ "${SYNTH_REWARD_MODE}" == gated || "${SYNTH_REWARD_MODE}" == additive ]] || { echo "SYNTH_REWARD_MODE must be gated or additive" >&2; exit 1; }
# Synthesizer GRPO group = all Synthesizer samples of a step (batch-mean baseline); see
# third_party/unitymas_o.patch. Reasoner groups = the K rollouts of one task.
export STAR_SYNTH_BATCH_LEVEL_GRPO=1

# ---- run artifacts -----------------------------------------------------------------------------
RUN_ID="${RUN_ID:-synco-${MODE}-$(date +%Y%m%d-%H%M%S)}"
export SYNTH_REASONER_RUNS_ROOT="${ROOT}/runs"
RUN_DIR="${SYNTH_REASONER_RUNS_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export CKPTS_DIR="${CKPTS_DIR:-${RUN_DIR}/ckpts}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard_log}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_ID}}"
mkdir -p "${TENSORBOARD_DIR}"

# ---- Synthesizer query slots (generated from an empty pool by default) -------------------------
if [[ -z "${TRAIN_QUERIES:-}" ]]; then
  TRAIN_QUERIES="${RUN_DIR}/train_query.json"
  "${PYTHON_BIN}" -m synth_reasoner.cli.prepare_train_jsonl \
      --output "${TRAIN_QUERIES}" \
      --n-tasks "${GEN_BATCH_SIZE:-32}"
else
  [[ -f "${TRAIN_QUERIES}" ]] || { echo "TRAIN_QUERIES not found: ${TRAIN_QUERIES}" >&2; exit 1; }
  cp "${TRAIN_QUERIES}" "${RUN_DIR}/train_query.json" 2>/dev/null || true
fi
export SYNCO_TRAIN_QUERIES="${TRAIN_QUERIES}"

# ---- environment hygiene -----------------------------------------------------------------------
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then unset PYTORCH_CUDA_ALLOC_CONF; fi
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---- Ray cluster -------------------------------------------------------------------------------
WORKER_PID=""
if [[ -z "${RAY_ADDRESS:-}" ]]; then
  : "${WORKER_NODE_IP:?set RAY_ADDRESS (existing cluster) or WORKER_NODE_IP (+ WORKER_SLURM_JOBID or WORKER_SSH)}"
  HEAD_IP="${HEAD_IP:-$(hostname -i | awk '{print $1}')}"
  RAY_PORT="${RAY_PORT:-6379}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
  "${RAY_BIN}" stop -f >/dev/null 2>&1 || true
  "${RAY_BIN}" start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
      --num-cpus="${CPUS_PER_NODE}" --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats
  WORKER_CMD="cd ${ROOT} && export PYTHONPATH=${PYTHONPATH} && unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; \
${RAY_BIN} stop -f >/dev/null 2>&1; ${RAY_BIN} start --address=${HEAD_IP}:${RAY_PORT} \
--num-cpus=${CPUS_PER_NODE} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block"
  if [[ -n "${WORKER_SLURM_JOBID:-}" ]]; then
    NODELIST_ARG=(); [[ -n "${WORKER_NODELIST:-}" ]] && NODELIST_ARG=(--nodelist="${WORKER_NODELIST}")
    srun --jobid="${WORKER_SLURM_JOBID}" "${NODELIST_ARG[@]}" --nodes=1 --ntasks=1 --overlap \
        bash -lc "${WORKER_CMD}" > "${RUN_DIR}/ray_worker.log" 2>&1 &
  elif [[ -n "${WORKER_SSH:-}" ]]; then
    ssh "${WORKER_SSH}" "bash -lc '${WORKER_CMD}'" > "${RUN_DIR}/ray_worker.log" 2>&1 &
  else
    echo "set WORKER_SLURM_JOBID or WORKER_SSH to start the worker node" >&2; exit 1
  fi
  WORKER_PID=$!
  trap 'kill ${WORKER_PID} 2>/dev/null || true; ${RAY_BIN} stop -f >/dev/null 2>&1 || true' EXIT
  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
fi

RAY_ADDRESS="${RAY_ADDRESS}" "${PYTHON_BIN}" - <<'PY'
import os, time, ray
ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
for _ in range(120):
    gpus = int(ray.cluster_resources().get("GPU", 0))
    if gpus >= 16:
        break
    time.sleep(5)
print(f"[train] ray cluster GPUs: {gpus}", flush=True)
assert gpus >= 16, "SynCo needs >= 16 GPUs (one 8-GPU node per policy)"
ray.shutdown()
PY

# ---- train -------------------------------------------------------------------------------------
echo "[train] run_id=${RUN_ID} mode=${MODE} synth_reward=${SYNTH_REWARD_MODE} report_refresh_every=${REPORT_REFRESH_EVERY}"
"${PYTHON_BIN}" -m synth_reasoner.unitymas_adapter.trainer_entry \
    --config-path "${ROOT}/configs" \
    --config-name synco_trainer \
    "hydra.searchpath=[file://${UNITYMAS_ROOT}/verl/experimental/star_ppo/config,file://${UNITYMAS_ROOT}/verl/trainer/config]" \
    +run_id="${RUN_ID}" \
    ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    trainer.resume_mode="${RESUME_MODE:-disable}" \
    "$@"
echo "[train] done: pool=${RUN_DIR}/pool ckpts=${CKPTS_DIR} tensorboard=${TENSORBOARD_DIR}"
