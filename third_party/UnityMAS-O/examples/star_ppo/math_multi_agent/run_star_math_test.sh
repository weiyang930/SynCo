#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-star_math_solver_verifier_refiner_finalizer_trainer}"
PROJECT_NAME="${PROJECT_NAME:-star_math_multi_agent_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-solver_verifier_refiner_finalizer_eval}"

AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/base_models/Qwen/Qwen2.5-7B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/datasets/Math/DAPO-Math-17k/data/dapo-math-17k.question.jsonl}"

MATH500_FILE="${MATH500_FILE:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/datasets/Math/MATH-500/test.jsonl}"
AIME24_FILE="${AIME24_FILE:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/datasets/Math/aime24/test.jsonl}"
AIME25_FILE="${AIME25_FILE:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/datasets/Math/aime25/test.jsonl}"
AIME26_FILE="${AIME26_FILE:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/datasets/Math/aime26/test.jsonl}"
AMC23_FILE="${AMC23_FILE:-${UNITYMAS_ROOT:?Set UNITYMAS_ROOT}/datasets/Math/amc23/test.jsonl}"
VAL_FILES="${VAL_FILES:-[${MATH500_FILE},${AIME24_FILE},${AIME25_FILE},${AIME26_FILE},${AMC23_FILE}]}"

NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
AGENT_GPUS_PER_NODE="${AGENT_GPUS_PER_NODE:-8}"
SHARED_LLM_NNODES="${SHARED_LLM_NNODES:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-false}"

python3 -m verl.experimental.star_ppo.main_ppo \
  --config-name "${CONFIG_NAME}" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "trainer.nnodes=${NNODES}" \
  "trainer.n_gpus_per_node=${GPUS_PER_NODE}" \
  "trainer.logger=[console,wandb]" \
  "trainer.val_before_train=true" \
  "trainer.val_only=true" \
  "trainer.test_freq=-1" \
  "trainer.save_freq=-1" \
  "trainer.total_epochs=1" \
  "trainer.llm_engines.0.model_path=${SOLVER_VERIFIER_MODEL_PATH:-${SOLVER_MODEL_PATH:-${AGENT_MODEL_PATH}}}" \
  "trainer.llm_engines.1.model_path=${REFINER_FINALIZER_MODEL_PATH:-${REFINER_MODEL_PATH:-${AGENT_MODEL_PATH}}}" \
  "trainer.llm_engines.0.nnodes=${SOLVER_VERIFIER_NNODES:-${SHARED_LLM_NNODES}}" \
  "trainer.llm_engines.1.nnodes=${REFINER_FINALIZER_NNODES:-${SHARED_LLM_NNODES}}" \
  "trainer.llm_engines.0.n_gpus_per_node=${SOLVER_VERIFIER_GPUS_PER_NODE:-${SOLVER_GPUS_PER_NODE:-${AGENT_GPUS_PER_NODE}}}" \
  "trainer.llm_engines.1.n_gpus_per_node=${REFINER_FINALIZER_GPUS_PER_NODE:-${REFINER_GPUS_PER_NODE:-${AGENT_GPUS_PER_NODE}}}" \
  "actor_rollout_ref.model.path=${SOLVER_VERIFIER_MODEL_PATH:-${SOLVER_MODEL_PATH:-${AGENT_MODEL_PATH}}}" \
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}" \
  "critic.model.use_remove_padding=${USE_REMOVE_PADDING}" \
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  "critic.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  "actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  "critic.model.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  "actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD}" \
  "critic.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD}" \
  "data.train_files=${TRAIN_JSONL}" \
  "data.val_files=${VAL_FILES}" \
  "data.val_batch_size=${VAL_BATCH_SIZE}" \
  "$@"
