# STAR PPO Code Iterative Workflow

This workflow trains three non-shared LLM agents for code generation:

- `planner_agent` on `planner_llm`
- `coder_agent` on `coder_llm`
- `reflection_agent` on `reflection_llm`

Each sample should provide `problem` and `tests`. `tests` may be a JSON string:

```json
{"uid":"example/0","source":"codeforces","problem":"...","starter_code":"","tests":"[{\"input\":\"1\\n2 1\",\"output\":\"1 2\"}]"}
```

Run with:

```bash
python3 -m verl.experimental.star_ppo.main_ppo \
  --config-path verl/experimental/star_ppo/config \
  --config-name star_code_iterative_plan_code_reflect_trainer \
  data.train_files=/path/to/train.jsonl \
  data.val_files=/path/to/test.jsonl \
  trainer.llm_engines.0.model_path=/path/to/planner_model \
  trainer.llm_engines.1.model_path=/path/to/coder_model \
  trainer.llm_engines.2.model_path=/path/to/reflection_model
```

Useful environment overrides:

- `CODE_MAX_TURNS=3`
- `CODE_VERIFY_TIMEOUT_SECONDS=5`
- `PLANNER_MODEL_PATH`, `CODER_MODEL_PATH`, `REFLECTION_MODEL_PATH`
- `TRAIN_JSONL`, `VAL_JSONL`
