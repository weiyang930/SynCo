#!/usr/bin/env bash
# Merge a star_ppo FSDP checkpoint of one policy into a Hugging Face model directory.
#   bash scripts/merge_checkpoint.sh <global_step_dir> <out_dir> [model_id]
# model_id: reasoner_qwen3_8b (default) or synthesizer_qwen3_8b.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP_DIR="${1:?global_step_N dir}"; OUT="${2:?output dir}"; MODEL_ID="${3:-reasoner_qwen3_8b}"
UNITYMAS_ROOT="${UNITYMAS_ROOT:-${ROOT}/third_party/UnityMAS-O}"
mkdir -p "${OUT}"
PYTHONPATH="${UNITYMAS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN:-python3}" -m verl.model_merger merge \
    --backend fsdp --local_dir "${STEP_DIR}/${MODEL_ID}/actor" --target_dir "${OUT}"
echo "merged ${STEP_DIR}/${MODEL_ID} -> ${OUT}"
