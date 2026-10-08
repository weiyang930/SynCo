# Star PPO RAG Two-Agent Training Example

This example now uses a **pluginized graph workflow runner** and trains multiple LLM agents in one PPO loop:

- `query_rewriter` agent: rewrites user question for retrieval.
- `document_selector` agent: selects useful docs.
- `summary_agent` agent: summarizes stable evidence.
- `answer_generator` agent: generates final answer from retrieved docs.

Retrieval is a tool call through the retriever interface in:

- `verl/experimental/star_ppo/tools/retriever.py`

## Workflow

1. `rewrite` -> retrieval round 1
2. `doc_selector` decides path:
   - if empty selection: trigger retrieval round 2
   - otherwise: go to summary directly
3. `summary` -> `answer`
4. Rewards:
   - each LLM node has per-node format reward
   - final EM is shared outcome reward

Workflow is not hard-coded in `fit`; it is executed by `GraphWorkflowRunner` from config graph.
Each query is processed asynchronously, and each LLM node rollout is committed into worker-local buffers.

## Minimal Data Format

Each sample should include at least:

- `prompt` (kept for RL dataset compatibility)
- `question` (used by the RAG workflow)
- `ground_truth` (used for EM)

See:

- `examples/star_ppo/rag_two_agent/train.json`
- `examples/star_ppo/rag_two_agent/val.json`

## Config

Use:

- `verl/experimental/star_ppo/config/star_rag_two_agent_trainer.yaml`

Key settings:

- `star.workflow.runner.path/name` (workflow plugin)
- `star.workflow.tools` (tool aliases)
- `star.workflow.graph.start_nodes/end_nodes/max_steps`
- `star.workflow.graph.nodes.*` (llm/tool node definitions)
- `star.workflow.graph.outcome_reward`
- `star.workflow.max_inflight_queries`
- `star.workflow.max_parallel_rollouts_per_model`
- `trainer.llm_engines[*].model_path` (optional per-engine model override)

For PPO/FSDP safety, trainer enforces that each model-ready batch is truncated to an integer
multiple of actor DP size before update (metric: `model/*/star/drop_divisor`).

`trainer.llm_engines[*].accelerator_type` is optional. Keep it `null` unless your Ray cluster uses custom
resource labels for hard node pinning.

To add new agents (for example `critic_selector`, `planner`, `verifier`), add new nodes in graph config and connect
edges with optional `when` conditions, without editing trainer `fit`.

## Run

```bash
python3 -m verl.experimental.star_ppo.main_ppo \
  --config-name star_rag_two_agent_trainer \
  data.train_files=$PWD/examples/star_ppo/rag_two_agent/train.json \
  data.val_files=$PWD/examples/star_ppo/rag_two_agent/val.json \
  actor_rollout_ref.model.path=/path/to/rewrite_or_shared_base_model \
  trainer.llm_engines.0.model_path=/path/to/rewrite_model \
  trainer.llm_engines.1.model_path=/path/to/answer_model \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.logger='["console"]'
```

If a per-engine `model_path` is null, trainer falls back to `actor_rollout_ref.model.path`.
