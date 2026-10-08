# Modifications made for SynCo

This directory is [UnityMAS-O](https://github.com/chenyiqun/UnityMAS-O) at commit
`7f1616ab812a8413f8baa9fe3738e7de9f487aad` (Apache License 2.0, see `LICENSE` and `Notice.txt`),
vendored into the SynCo repository with the changes below. The same changes are provided as a
diff in `../unitymas_o.patch`. Each modified file carries a "Modified by the SynCo authors" notice.

1. `verl/experimental/star_ppo/ray_trainer.py` — GRPO grouping for multi-agent training.
   `compute_grpo_outcome_advantage` groups samples by `uid`, which star_ppo does not populate.
   Reasoner samples are grouped by `query_id` (the K rollouts of one task). Synthesizer samples
   (one per task) of a training step form one group (`uid = synth_batch_<global_step>`), giving
   a batch-mean baseline; toggle with `STAR_SYNTH_BATCH_LEVEL_GRPO` (default `1`).
2. `verl/workers/rollout/vllm_rollout/utils.py` — `build_cli_args_from_config` dropped boolean
   `False` values, so e.g. `enable_prefix_caching=False` never reached vLLM (whose default
   enables it). Negatable vLLM flags are now emitted as `--no-<flag>`.
3. `tests/workers/rollout/test_vllm_cli_args_on_cpu.py` — tests for change 2.
