# M-ASK Iterative Workflow

This workflow now uses a dedicated iterative runner plus a dedicated turn-level reward allocator.

- config: `verl/experimental/star_ppo/config/star_iterative_plan_search_summary_update_answer_f1_trainer.yaml`
- runner: `verl.experimental.star_ppo.workflows.mask_iterative_workflow.MAskIterativeWorkflowRunner`
- reward allocator: `verl.experimental.star_ppo.reward_allocators.mask_turn_level.MAskTurnLevelRewardAllocator`

## Topology

This workflow has 5 agents but only 4 LLMs:

- `planning_agent` and `answer_agent` share one LLM: `reasoning_agent_llm`
- `search_agent` uses one LLM
- `summary_agent` uses one LLM
- `update_agent` uses one LLM

With the current config, each LLM occupies 1 node, so the training topology is:

- 5 agents
- 4 LLMs
- 4 nodes

## Execution flow

For each query, the runner executes:

1. `plan`
2. `answer_0`
3. repeat up to `max_turns`
4. `search_t`
5. if `search_t` outputs `end`, terminate the workflow immediately
6. `retrieve_t`
7. `summary_t`
8. `update_t`
9. `answer_t`
10. `final_answer`

The runner stores a full query trace and hands that trace to the reward allocator.

## Reward flow

The M-ASK allocator implements the paper-style reward split:

- `plan`: absolute F1 of the planning state's predicted answer
- `answer_t`: absolute F1 of the current answer
- `search_t`, `summary_t`, `update_t`: shared delta reward `F1(a_t) - F1(a_{t-1})`
- `search_t` with `action=end`: zero task reward
- optional parser-format shaping is added on top through each node's `format_weight`

The tracked metrics include:

- planning-stage F1
- final-round F1
- final F1 minus planning F1
- per-agent format penalty
- per-agent task reward
- per-agent total reward

## Generic extension pattern

To add a new trainable workflow in this framework:

1. add a config
2. implement a `WorkflowRunner` that emits a full trace
3. implement a `RewardAllocator` that maps the trace to reward assignments

The PPO training loop does not need to change as long as the allocator finally returns rewards tied to rollout `traj_id`s.

## Notes

- Tool nodes can appear in the trace, but only LLM nodes with rollout trajectories are trainable.
- Reward allocation is code-defined, so turn-level, node-level, sparse, dense, and cross-turn reward rules are all supported.

## Train

Use the existing STAR launch scripts and point `CONFIG_NAME` at this workflow config:

```bash
CONFIG_NAME=star_iterative_plan_search_summary_update_answer_f1_trainer \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_head_train.sh
```

Recommended model env vars for this workflow:

- `REASONING_MODEL_PATH`
- `SEARCH_MODEL_PATH`
- `SUMMARY_MODEL_PATH`
- `UPDATE_MODEL_PATH`

`REASONING_MODEL_PATH` is shared by both `planning_agent` and `answer_agent`.
