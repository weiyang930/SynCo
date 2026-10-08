#!/usr/bin/env bash
set -euo pipefail

# Master-only launcher for fixed-IP STAR PPO runs.
#
# Run this script once on the head/master container. It SSHes to every IP in
# NODE_IPS, assigns RANK by list order, cleans stale processes, and starts the
# existing common/run_per_node_background.sh on each node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

NODE_IPS="${NODE_IPS:-}"
REMOTE_CWD="${REMOTE_CWD:-${PROJECT_ROOT}}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
HEAD_START_DELAY_SECONDS="${HEAD_START_DELAY_SECONDS:-10}"
CLEANUP_BEFORE_LAUNCH="${CLEANUP_BEFORE_LAUNCH:-true}"
PYTHON_KILL_PATTERN="${PYTHON_KILL_PATTERN:-/miniconda3/envs/${CONDA_ENV_NAME}/bin/python3.10}"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<'USAGE'
Usage:
  launch_ip_list_background.sh --node-ips "HEAD_IP,WORKER0_IP,..." [-- HYDRA_OVERRIDES...]

Run once on the master/head container. The first IP is rank0/Ray head.

Options:
  --node-ips IPS        Comma/space separated IP list. First IP is rank0.
  --remote-cwd DIR      Repo path on every node. Default: current repo path.
  --conda-env NAME      Conda env to activate on every node. Default: verl.
  --conda-root DIR      Conda root on every node. Default: $HOME/miniconda3.
  --ssh-user USER       Optional SSH user. Default: current SSH default.
  --no-cleanup          Skip ray stop / stale python cleanup before launch.
  --dry-run             Print remote commands without executing them.
  -h, --help            Show this help.

Environment:
  CONFIG_NAME is required. STAR/verl launch env vars from the current shell are
  forwarded to every node. Use STAR_SSH_FORWARD_ENV="VAR1 VAR2" for extra vars.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ips)
      NODE_IPS="${2:?--node-ips requires a value}"
      shift 2
      ;;
    --remote-cwd)
      REMOTE_CWD="${2:?--remote-cwd requires a value}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="${2:?--conda-env requires a value}"
      shift 2
      ;;
    --conda-root)
      CONDA_ROOT="${2:?--conda-root requires a value}"
      shift 2
      ;;
    --ssh-user)
      SSH_USER="${2:?--ssh-user requires a value}"
      shift 2
      ;;
    --no-cleanup)
      CLEANUP_BEFORE_LAUNCH="false"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "[common/launch_ip_list_background] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${NODE_IPS}" ]]; then
  echo "[common/launch_ip_list_background] ERROR: --node-ips or NODE_IPS is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "${CONFIG_NAME:-}" ]]; then
  echo "[common/launch_ip_list_background] ERROR: CONFIG_NAME is required" >&2
  exit 2
fi

IFS=$' ,\n\t' read -r -a RAW_IPS <<< "${NODE_IPS}"
IP_ARRAY=()
for ip in "${RAW_IPS[@]}"; do
  if [[ -n "${ip}" ]]; then
    IP_ARRAY+=("${ip}")
  fi
done
if [[ "${#IP_ARRAY[@]}" -eq 0 ]]; then
  echo "[common/launch_ip_list_background] ERROR: NODE_IPS did not contain any IPs" >&2
  exit 2
fi

DEFAULT_FORWARD_ENV=(
  CONFIG_NAME PROJECT_NAME EXPERIMENT_NAME
  TRAIN_JSONL VAL_JSONL TRAIN_PARQUET VAL_PARQUET
  AGENT_MODEL_PATH ACTOR_MODEL_PATH ACTOR_TOKENIZER_PATH
  PLANNER_MODEL_PATH CODER_MODEL_PATH REFLECTION_MODEL_PATH
  SOLVER_MODEL_PATH VERIFIER_MODEL_PATH REFINER_MODEL_PATH FINALIZER_MODEL_PATH
  REWRITE_MODEL_PATH SELECT_MODEL_PATH ANSWER_MODEL_PATH DECOMPOSE_MODEL_PATH SUMMARY_MODEL_PATH
  SOLVER_NNODES VERIFIER_NNODES REFINER_NNODES FINALIZER_NNODES
  PLANNER_NNODES CODER_NNODES REFLECTION_NNODES
  AGENT_GPUS_PER_NODE SOLVER_GPUS_PER_NODE VERIFIER_GPUS_PER_NODE REFINER_GPUS_PER_NODE FINALIZER_GPUS_PER_NODE
  QWEN_ENABLE_THINKING ADV_ESTIMATOR
  SAVE_FREQ TEST_FREQ GEN_BATCH_SIZE VAL_BATCH_SIZE VAL_MAX_BATCHES VAL_BEFORE_TRAIN
  STAR_MAX_INFLIGHT_QUERIES STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL
  STAR_LLM_MICROBATCH_MAX_SIZE STAR_LLM_MICROBATCH_MAX_WAIT_MS
  STAR_QUERY_TIMEOUT_SECONDS STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS STAR_RAY_GET_TIMEOUT_SECONDS
  STAR_WORKER_CALL_TIMEOUT_SECONDS STAR_LLM_TIMEOUT_SECONDS STAR_TOOL_TIMEOUT_SECONDS
  STAR_VAL_PROGRESS_EVERY STAR_WORKFLOW_DEBUG STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES
  STAR_WORKFLOW_DEBUG_SAMPLE_INDEX STAR_WORKFLOW_DEBUG_MAX_CHARS
  STAR_VAL_DEBUG STAR_VAL_DEBUG_SAMPLE_COUNT STAR_VAL_DEBUG_EVERY_N_BATCHES STAR_VAL_DEBUG_MAX_CHARS
  STAR_PER_INFER_PROMPT_MAX_TOKENS STAR_PARALLEL_POST_INIT STAR_POST_INIT_PARALLELISM
  STAR_WEIGHT_SYNC_MODE STAR_WORKER_MAX_CONCURRENCY STAR_WEIGHT_SYNC_TIMEOUT_SECONDS
  ACTOR_PPO_MINI_BATCH_SIZE ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU
  CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU
  REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU
  USE_DYNAMIC_BSZ USE_REMOVE_PADDING ULYSSES_SEQUENCE_PARALLEL_SIZE ENABLE_ACTIVATION_OFFLOAD
  FSDP_PARAM_OFFLOAD FSDP_OPTIMIZER_OFFLOAD FSDP_OFFLOAD_POLICY
  REF_FSDP_PARAM_OFFLOAD REF_FSDP_OFFLOAD_POLICY
  ROLLOUT_NAME ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE ROLLOUT_GPU_MEMORY_UTILIZATION
  ROLLOUT_PROMPT_LENGTH ROLLOUT_RESPONSE_LENGTH ROLLOUT_MAX_MODEL_LEN
  ROLLOUT_MAX_NUM_SEQS ROLLOUT_MAX_NUM_BATCHED_TOKENS ROLLOUT_FREE_CACHE_ENGINE
  DATA_MAX_PROMPT_LENGTH FILTER_OVERLONG_PROMPTS
  MATH_EXPOSE_GROUND_TRUTH_TO_PROMPTS MATH_MAX_STATE_CHARS
  MATH_SOLVER_FORMAT_WEIGHT MATH_VERIFIER_FORMAT_WEIGHT MATH_REFINER_FORMAT_WEIGHT MATH_FINALIZER_FORMAT_WEIGHT
  CODE_MAX_TURNS CODE_STOP_ON_ALL_PASSED CODE_VERIFY_TIMEOUT_SECONDS
  CODE_VERIFY_DEFAULT_CHECKER_TYPE CODE_VERIFY_MAX_TESTS_PER_EXAMPLE
  CODE_VERIFY_MAX_CASE_INPUT_CHARS CODE_VERIFY_MAX_CASE_OUTPUT_CHARS
  CODE_VERIFY_MAX_TOTAL_TEST_CHARS CODE_VERIFIER_FAIL_OPEN
  RAY_PORT DASHBOARD_PORT CPUS_PER_NODE GPUS_PER_NODE
  WANDB_API_KEY WANDB_ENTITY
  PYTHONUNBUFFERED RAY_DEDUP_LOGS VLLM_USE_V1 VERL_VLLM_FORCE_SHM_WEIGHT_SYNC
)

FORWARD_ENV=("${DEFAULT_FORWARD_ENV[@]}")
if [[ -n "${STAR_SSH_FORWARD_ENV:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FORWARD_ENV=(${STAR_SSH_FORWARD_ENV})
  FORWARD_ENV+=("${EXTRA_FORWARD_ENV[@]}")
fi

quote() {
  printf "%q" "$1"
}

append_export_if_set() {
  local name="$1"
  if [[ -n "${!name+x}" ]]; then
    printf 'export %s=%q\n' "${name}" "${!name}"
  fi
}

build_env_exports() {
  local name
  printf 'export HEAD_IP=%q\n' "${IP_ARRAY[0]}"
  printf 'export WORLD_SIZE=%q\n' "${#IP_ARRAY[@]}"
  printf 'export CONDA_ROOT=%q\n' "${CONDA_ROOT}"
  printf 'export CONDA_ENV_NAME=%q\n' "${CONDA_ENV_NAME}"
  for name in "${FORWARD_ENV[@]}"; do
    append_export_if_set "${name}"
  done
}

build_remote_command() {
  local rank="$1"
  shift
  local cmd=""

  cmd+="set -euo pipefail"$'\n'
  cmd+="cd $(quote "${REMOTE_CWD}")"$'\n'
  cmd+="export RANK=$(quote "${rank}")"$'\n'
  cmd+="$(build_env_exports)"$'\n'
  cmd+="if [[ ! -f \"\${CONDA_ROOT}/etc/profile.d/conda.sh\" ]]; then echo \"conda not found at \${CONDA_ROOT}\" >&2; exit 1; fi"$'\n'
  cmd+="set +u; source \"\${CONDA_ROOT}/etc/profile.d/conda.sh\"; conda activate \"\${CONDA_ENV_NAME}\"; set -u"$'\n'
  if [[ "${CLEANUP_BEFORE_LAUNCH}" == "true" ]]; then
    cmd+="ray stop --force >/dev/null 2>&1 || true"$'\n'
    cmd+="pkill -9 -f $(quote "${PYTHON_KILL_PATTERN}") >/dev/null 2>&1 || true"$'\n'
  fi
  cmd+="mkdir -p logs/star_ppo"$'\n'
  cmd+="bash examples/star_ppo/common/run_per_node_background.sh"
  local arg
  for arg in "$@"; do
    cmd+=" $(quote "${arg}")"
  done
  cmd+=$'\n'

  printf '%s' "${cmd}"
}

ssh_target() {
  local ip="$1"
  if [[ -n "${SSH_USER}" ]]; then
    printf '%s@%s' "${SSH_USER}" "${ip}"
  else
    printf '%s' "${ip}"
  fi
}

detect_local_ips() {
  {
    hostname -I 2>/dev/null || true
    hostname -i 2>/dev/null || true
    ip -o -4 addr show 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true
  } | tr ' ' '\n' | awk 'NF && $1 != "127.0.0.1" {print $1}' | awk '!seen[$0]++'
}

is_local_ip() {
  local ip="$1"
  local local_candidate
  while IFS= read -r local_candidate; do
    if [[ "${local_candidate}" == "${ip}" ]]; then
      return 0
    fi
  done < <(detect_local_ips)
  return 1
}

run_remote() {
  local ip="$1"
  local remote_command="$2"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "----- ${ip} -----"
    printf '%s\n' "${remote_command}"
  elif is_local_ip "${ip}"; then
    bash -s <<< "${remote_command}"
  else
    # Feed the script over stdin so `pkill -f` cannot match the launcher shell's
    # command line and kill the shell that is still executing this launch.
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "$(ssh_target "${ip}")" "bash -s" <<< "${remote_command}"
  fi
}

echo "[common/launch_ip_list_background] nodes=${#IP_ARRAY[@]} head=${IP_ARRAY[0]} remote_cwd=${REMOTE_CWD}"

echo "[common/launch_ip_list_background] launching rank0/head ${IP_ARRAY[0]}"
run_remote "${IP_ARRAY[0]}" "$(build_remote_command 0 "$@")"

if [[ "${#IP_ARRAY[@]}" -gt 1 ]]; then
  echo "[common/launch_ip_list_background] waiting ${HEAD_START_DELAY_SECONDS}s before workers"
  if [[ "${DRY_RUN}" != "true" ]]; then
    sleep "${HEAD_START_DELAY_SECONDS}"
  fi
fi

for rank in "${!IP_ARRAY[@]}"; do
  if [[ "${rank}" -eq 0 ]]; then
    continue
  fi
  ip="${IP_ARRAY[$rank]}"
  echo "[common/launch_ip_list_background] launching rank=${rank} ip=${ip}"
  run_remote "${ip}" "$(build_remote_command "${rank}" "$@")"
done

echo "[common/launch_ip_list_background] submitted all nodes."
echo "[common/launch_ip_list_background] logs: ${REMOTE_CWD}/logs/star_ppo/run_rank*_*.log"
