#!/usr/bin/env bash
set -euo pipefail

# Launch STAR PPO on a fixed set of already-created containers/nodes.
#
# Put the same command in every container:
#   bash examples/star_ppo/common/run_ip_list.sh --node-ips "ip0,ip1,ip2,ip3" -- ...
#
# The first IP is rank 0 / Ray head. Each container detects its own IP, maps it
# to a rank, then forwards to common/run_per_node.sh in the foreground.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

NODE_IPS="${NODE_IPS:-}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-true}"
LOCAL_IP="${LOCAL_IP:-}"
CLEANUP_BEFORE_LAUNCH="${CLEANUP_BEFORE_LAUNCH:-true}"
PYTHON_KILL_PATTERN="${PYTHON_KILL_PATTERN:-/miniconda3/envs/${CONDA_ENV_NAME}/bin/python3.10}"
LOG_TO_FILE="${LOG_TO_FILE:-true}"

usage() {
  cat <<'USAGE'
Usage:
  run_ip_list.sh --node-ips "HEAD_IP,WORKER0_IP,WORKER1_IP,..." [-- HYDRA_OVERRIDES...]

Options:
  --node-ips IPS        Comma/space separated IP list. First IP is rank0 Ray head.
  --local-ip IP         Override local IP detection if a container has multiple IPs.
  --conda-env NAME      Conda env to activate. Default: verl.
  --conda-root DIR      Conda root. Default: $HOME/miniconda3.
  --no-cleanup          Skip ray stop / stale python cleanup before launch.
  --no-log-file         Do not tee stdout/stderr to logs/star_ppo/run_rank*.log.
  -h, --help            Show this help.

Environment:
  NODE_IPS can be used instead of --node-ips.
  CONFIG_NAME is required by common/run_per_node.sh.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ips)
      NODE_IPS="${2:?--node-ips requires a value}"
      shift 2
      ;;
    --local-ip)
      LOCAL_IP="${2:?--local-ip requires a value}"
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
    --no-log-file)
      LOG_TO_FILE="false"
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
      echo "[common/run_ip_list] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${NODE_IPS}" ]]; then
  echo "[common/run_ip_list] ERROR: --node-ips or NODE_IPS is required" >&2
  usage >&2
  exit 2
fi

IFS=$' ,\n\t' read -r -a IP_ARRAY <<< "${NODE_IPS}"
FILTERED_IPS=()
for ip in "${IP_ARRAY[@]}"; do
  if [[ -n "${ip}" ]]; then
    FILTERED_IPS+=("${ip}")
  fi
done
IP_ARRAY=("${FILTERED_IPS[@]}")

if [[ "${#IP_ARRAY[@]}" -eq 0 ]]; then
  echo "[common/run_ip_list] ERROR: NODE_IPS did not contain any IPs" >&2
  exit 2
fi

detect_local_ips() {
  {
    hostname -I 2>/dev/null || true
    hostname -i 2>/dev/null || true
    ip -o -4 addr show 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true
  } | tr ' ' '\n' | awk 'NF && $1 != "127.0.0.1" {print $1}' | awk '!seen[$0]++'
}

if [[ -z "${LOCAL_IP}" ]]; then
  LOCAL_IPS="$(detect_local_ips)"
else
  LOCAL_IPS="${LOCAL_IP}"
fi

RANK_MATCH=""
LOCAL_IP_MATCH=""
for idx in "${!IP_ARRAY[@]}"; do
  ip="${IP_ARRAY[$idx]}"
  while IFS= read -r local_candidate; do
    if [[ "${local_candidate}" == "${ip}" ]]; then
      RANK_MATCH="${idx}"
      LOCAL_IP_MATCH="${local_candidate}"
      break
    fi
  done <<< "${LOCAL_IPS}"
  if [[ -n "${RANK_MATCH}" ]]; then
    break
  fi
done

if [[ -z "${RANK_MATCH}" ]]; then
  echo "[common/run_ip_list] ERROR: this node IP is not in NODE_IPS." >&2
  echo "[common/run_ip_list] NODE_IPS=${IP_ARRAY[*]}" >&2
  echo "[common/run_ip_list] detected local IPs:" >&2
  printf '  %s\n' ${LOCAL_IPS:-"<none>"} >&2
  echo "[common/run_ip_list] If detection picked the wrong interface, pass --local-ip <this-container-ip>." >&2
  exit 1
fi

export RANK="${RANK_MATCH}"
export WORLD_SIZE="${#IP_ARRAY[@]}"
export HEAD_IP="${IP_ARRAY[0]}"

cd "${PROJECT_ROOT}"

if [[ "${ACTIVATE_CONDA}" == "true" ]]; then
  if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "[common/run_ip_list] ERROR: conda not found at ${CONDA_ROOT}" >&2
    exit 1
  fi
  set +u
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  set -u
fi

echo "[common/run_ip_list] local_ip=${LOCAL_IP_MATCH} rank=${RANK}/${WORLD_SIZE} head=${HEAD_IP}"

if [[ "${CLEANUP_BEFORE_LAUNCH}" == "true" ]]; then
  echo "[common/run_ip_list] cleanup stale Ray/Python processes on local rank ${RANK}"
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f "${PYTHON_KILL_PATTERN}" >/dev/null 2>&1 || true
fi

if [[ "${LOG_TO_FILE}" == "true" ]]; then
  mkdir -p logs/star_ppo
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_FILE:-logs/star_ppo/run_rank${RANK}_${TS}.log}"
  echo "[common/run_ip_list] writing log to ${LOG_FILE}"
  stdbuf -oL -eL bash "${SCRIPT_DIR}/run_per_node.sh" "$@" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

exec bash "${SCRIPT_DIR}/run_per_node.sh" "$@"
