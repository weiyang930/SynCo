#!/usr/bin/env bash
# Evaluate one Hugging Face model directory on all GPUs of this node (one shard per GPU), then
# aggregate.  bash scripts/eval_checkpoint.sh <model_dir> <out_dir> [benchmark ...]
# NUM_GPUS (default: all visible) sets the number of shards.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?model dir}"; OUT="${2:?output dir}"; shift 2
BENCH_ARGS=(); [[ $# -gt 0 ]] && BENCH_ARGS=(--benchmarks "$@")
N="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
mkdir -p "${OUT}/logs"
for g in $(seq 0 $((N-1))); do
  # separate vLLM compile cache per process: concurrent writers to one cache dir collide
  VLLM_CACHE_ROOT="${VLLM_CACHE_BASE:-${TMPDIR:-/tmp}}/synco_vllm_cache_$g" \
  CUDA_VISIBLE_DEVICES=$g VLLM_WORKER_MULTIPROC_METHOD=spawn "${PYTHON_BIN:-python3}" "${ROOT}/scripts/evaluate.py" run \
      --model-path "${MODEL}" --out-dir "${OUT}" --shard-id $g --num-shards $N "${BENCH_ARGS[@]}" \
      > "${OUT}/logs/shard_$g.log" 2>&1 &
done
wait
"${PYTHON_BIN:-python3}" "${ROOT}/scripts/evaluate.py" aggregate --out-dir "${OUT}" "${BENCH_ARGS[@]}"
