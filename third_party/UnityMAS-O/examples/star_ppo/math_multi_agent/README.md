# STAR PPO Math Multi-Agent Workflow

This workflow trains four math agents with two shared LLM parameter groups:

- `solver_agent` and `verifier_agent` share `solver_verifier_llm`
- `refiner_agent` and `finalizer_agent` share `refiner_finalizer_llm`

The default model path for both shared LLM groups is:

```bash
${UNITYMAS_ROOT}/base_models/Qwen/Qwen2.5-7B-Instruct
```

By default the launch scripts use 4 total nodes and place each shared LLM group
on 2 nodes:

- `solver_verifier_llm`: 2 nodes
- `refiner_finalizer_llm`: 2 nodes

The workflow is:

```text
Problem -> Solver -> Verifier -> Refiner -> Finalizer -> Final Answer
```

Reward for every trainable agent is:

```text
final answer accuracy in {0, 1} + format penalty in {0, -1}
```

Validation logs overall accuracy at:

```text
validation/workflow/math/final_acc
```

and per-dataset accuracy at paths like:

```text
validation/workflow/math/dataset/math-500/acc
validation/workflow/math/dataset/aime24/acc
validation/workflow/math/dataset/aime25/acc
validation/workflow/math/dataset/aime26/acc
validation/workflow/math/dataset/amc23/acc
```

Run training:

```bash
bash examples/star_ppo/math_multi_agent/run_star_math_train.sh
```

Run validation-only evaluation over multiple datasets:

```bash
VAL_FILES='[/path/to/MATH-500.jsonl,/path/to/aime24.jsonl,/path/to/aime25.jsonl,/path/to/aime26.jsonl,/path/to/amc23.jsonl]' \
bash examples/star_ppo/math_multi_agent/run_star_math_test.sh
```

Useful overrides:

- `AGENT_MODEL_PATH`
- `SOLVER_VERIFIER_MODEL_PATH`, `REFINER_FINALIZER_MODEL_PATH`
- `SOLVER_MODEL_PATH`, `REFINER_MODEL_PATH` as backward-compatible fallbacks
- `TRAIN_JSONL`
- `VAL_FILES`
- `NNODES`, `SHARED_LLM_NNODES`, `SOLVER_VERIFIER_NNODES`, `REFINER_FINALIZER_NNODES`
- `GPUS_PER_NODE`, `AGENT_GPUS_PER_NODE`
- `GEN_BATCH_SIZE`, `VAL_BATCH_SIZE`
- `TEST_FREQ`, `SAVE_FREQ`
- `ULYSSES_SEQUENCE_PARALLEL_SIZE` for long-response PPO memory reduction
- `ENABLE_ACTIVATION_OFFLOAD`, `FSDP_PARAM_OFFLOAD`, `FSDP_OPTIMIZER_OFFLOAD`

Print complete validation examples:

```bash
STAR_VAL_DEBUG=true \
STAR_VAL_DEBUG_SAMPLE_COUNT=2 \
STAR_VAL_DEBUG_EVERY_N_BATCHES=1 \
STAR_VAL_DEBUG_MAX_CHARS=0 \
bash examples/star_ppo/math_multi_agent/run_star_math_test.sh
```

`STAR_VAL_DEBUG_MAX_CHARS=0` means no truncation. The printed block includes the
problem, ground truth, every agent's raw/parsed output, final answer, and final acc.
