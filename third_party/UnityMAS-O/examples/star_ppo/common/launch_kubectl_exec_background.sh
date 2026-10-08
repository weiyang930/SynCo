#!/usr/bin/env bash
set -euo pipefail

# Master-only launcher for already-created PyTorchJob pods.
#
# Run once from any pod/machine that has kubectl access to the PyTorchJob pods.
# It maps NODE_IPS to pod names via `kubectl get pods -o wide`, then kubectl-execs
# into every pod, assigns RANK by IP-list order, cleans stale processes, and
# starts common/run_per_node_background.sh on each pod.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

NODE_IPS="${NODE_IPS:-}"
REMOTE_CWD="${REMOTE_CWD:-${PROJECT_ROOT}}"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-}"
KUBE_CONTEXT="${KUBE_CONTEXT:-}"
KUBE_CONTAINER="${KUBE_CONTAINER:-}"
POD_NAMES="${POD_NAMES:-}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CLEANUP_BEFORE_LAUNCH="${CLEANUP_BEFORE_LAUNCH:-true}"
PYTHON_KILL_PATTERN="${PYTHON_KILL_PATTERN:-/miniconda3/envs/${CONDA_ENV_NAME}/bin/python3.10}"
HEAD_START_DELAY_SECONDS="${HEAD_START_DELAY_SECONDS:-10}"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<'USAGE'
Usage:
  launch_kubectl_exec_background.sh --node-ips "HEAD_IP,WORKER0_IP,..." [-- HYDRA_OVERRIDES...]

Run once from the master pod or any shell with kubectl permissions.
The first IP is rank0/Ray head. Logs are written in each pod under logs/star_ppo.

Options:
  --node-ips IPS        Comma/space separated pod IP list. First IP is rank0.
  --pod-names NAMES     Optional comma/space separated pod names, same order as IPs.
                        If omitted, names are discovered from kubectl get pods -o wide.
  --remote-cwd DIR      Repo path inside every pod. Default: current repo path.
  --namespace NS        Kubernetes namespace. Default: kubectl current namespace.
  --context CTX         Kubernetes context. Default: kubectl current context.
  --container NAME      Container name for kubectl exec, if pod has multiple containers.
  --conda-env NAME      Conda env to activate inside every pod. Default: verl.
  --conda-root DIR      Conda root inside every pod. Default: $HOME/miniconda3.
  --no-cleanup          Skip ray stop / stale python cleanup before launch.
  --dry-run             Print resolved commands without executing.
  -h, --help            Show this help.

Environment:
  CONFIG_NAME is required. STAR/verl launch env vars from the current shell are
  forwarded to every pod. Use STAR_KUBECTL_FORWARD_ENV="VAR1 VAR2" for extra vars.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ips)
      NODE_IPS="${2:?--node-ips requires a value}"
      shift 2
      ;;
    --pod-names)
      POD_NAMES="${2:?--pod-names requires a value}"
      shift 2
      ;;
    --remote-cwd)
      REMOTE_CWD="${2:?--remote-cwd requires a value}"
      shift 2
      ;;
    --namespace)
      KUBE_NAMESPACE="${2:?--namespace requires a value}"
      shift 2
      ;;
    --context)
      KUBE_CONTEXT="${2:?--context requires a value}"
      shift 2
      ;;
    --container)
      KUBE_CONTAINER="${2:?--container requires a value}"
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
      echo "[common/launch_kubectl_exec_background] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${NODE_IPS}" ]]; then
  echo "[common/launch_kubectl_exec_background] ERROR: --node-ips or NODE_IPS is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "${CONFIG_NAME:-}" ]]; then
  echo "[common/launch_kubectl_exec_background] ERROR: CONFIG_NAME is required" >&2
  exit 2
fi

split_words_to_stdout() {
  local raw="$1"
  local item
  # shellcheck disable=SC2206
  local values=(${raw//,/ })
  for item in "${values[@]}"; do
    if [[ -n "${item}" ]]; then
      printf '%s\n' "${item}"
    fi
  done
}

IP_ARRAY=()
while IFS= read -r item; do
  IP_ARRAY+=("${item}")
done < <(split_words_to_stdout "${NODE_IPS}")
if [[ "${#IP_ARRAY[@]}" -eq 0 ]]; then
  echo "[common/launch_kubectl_exec_background] ERROR: NODE_IPS did not contain any IPs" >&2
  exit 2
fi

KUBECTL_BASE=(kubectl)
if [[ -n "${KUBE_CONTEXT}" ]]; then
  KUBECTL_BASE+=(--context "${KUBE_CONTEXT}")
fi
if [[ -n "${KUBE_NAMESPACE}" ]]; then
  KUBECTL_BASE+=(--namespace "${KUBE_NAMESPACE}")
fi

resolve_pods_from_ips() {
  local ip pod
  POD_ARRAY=()
  for ip in "${IP_ARRAY[@]}"; do
    pod="$("${KUBECTL_BASE[@]}" get pods -o wide --no-headers | awk -v ip="${ip}" '$6 == ip {print $1; exit}')"
    if [[ -z "${pod}" ]]; then
      echo "[common/launch_kubectl_exec_background] ERROR: cannot find pod for IP ${ip}" >&2
      echo "[common/launch_kubectl_exec_background] kubectl get pods -o wide:" >&2
      "${KUBECTL_BASE[@]}" get pods -o wide >&2 || true
      exit 1
    fi
    POD_ARRAY+=("${pod}")
  done
}

if [[ -n "${POD_NAMES}" ]]; then
  POD_ARRAY=()
  while IFS= read -r item; do
    POD_ARRAY+=("${item}")
  done < <(split_words_to_stdout "${POD_NAMES}")
  if [[ "${#POD_ARRAY[@]}" -ne "${#IP_ARRAY[@]}" ]]; then
    echo "[common/launch_kubectl_exec_background] ERROR: pod count ${#POD_ARRAY[@]} != IP count ${#IP_ARRAY[@]}" >&2
    exit 2
  fi
else
  resolve_pods_from_ips
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
if [[ -n "${STAR_KUBECTL_FORWARD_ENV:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FORWARD_ENV=(${STAR_KUBECTL_FORWARD_ENV})
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

build_pod_command() {
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

kubectl_exec() {
  local pod="$1"
  local pod_command="$2"
  local exec_cmd=("${KUBECTL_BASE[@]}" exec "${pod}")
  if [[ -n "${KUBE_CONTAINER}" ]]; then
    exec_cmd+=(-c "${KUBE_CONTAINER}")
  fi
  exec_cmd+=(-- bash -s)

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "----- ${pod} -----"
    printf '%s\n' "${pod_command}"
  else
    "${exec_cmd[@]}" <<< "${pod_command}"
  fi
}

echo "[common/launch_kubectl_exec_background] nodes=${#IP_ARRAY[@]} head=${IP_ARRAY[0]}"
for rank in "${!IP_ARRAY[@]}"; do
  echo "[common/launch_kubectl_exec_background] rank=${rank} ip=${IP_ARRAY[$rank]} pod=${POD_ARRAY[$rank]}"
done

echo "[common/launch_kubectl_exec_background] launching rank0/head"
kubectl_exec "${POD_ARRAY[0]}" "$(build_pod_command 0 "$@")"

if [[ "${#IP_ARRAY[@]}" -gt 1 ]]; then
  echo "[common/launch_kubectl_exec_background] waiting ${HEAD_START_DELAY_SECONDS}s before workers"
  if [[ "${DRY_RUN}" != "true" ]]; then
    sleep "${HEAD_START_DELAY_SECONDS}"
  fi
fi

for rank in "${!IP_ARRAY[@]}"; do
  if [[ "${rank}" -eq 0 ]]; then
    continue
  fi
  echo "[common/launch_kubectl_exec_background] launching rank=${rank}"
  kubectl_exec "${POD_ARRAY[$rank]}" "$(build_pod_command "${rank}" "$@")"
done

echo "[common/launch_kubectl_exec_background] submitted all pods."
echo "[common/launch_kubectl_exec_background] logs: ${REMOTE_CWD}/logs/star_ppo/run_rank*_*.log"
