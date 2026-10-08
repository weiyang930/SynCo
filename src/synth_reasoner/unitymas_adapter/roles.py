"""Agent / model / node identifiers for the UnityMAS-O adapter.

These constants stay small and import-safe so other modules (including the
training entrypoint) can refer to them without pulling in ``verl``.

* ``agent_id`` is what UnityMAS-O uses to attribute trainable trajectories
  to a logical role inside the workflow.
* ``model_id`` matches an entry in ``trainer.llm_engines`` and decides which
  policy network produces the rollout.
* ``node_id`` is the workflow-step name; reward allocation matches on it.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Node identifiers (workflow step names)
# ---------------------------------------------------------------------------

SYNTHESIZER_NODE_ID = "synthesizer"
REASONER_NODE_ID = "reasoner"
TASK_VERIFIER_NODE_ID = "task_verifier"
ANSWER_CHECKER_NODE_ID = "answer_checker"

# ---------------------------------------------------------------------------
# Agent identifiers (logical roles)
# ---------------------------------------------------------------------------

SYNTHESIZER_AGENT_ID = "synthesizer_agent"
REASONER_AGENT_ID = "reasoner_agent"

# ---------------------------------------------------------------------------
# Model-group identifiers — must match ``trainer.llm_engines.*.model_id``.
# ---------------------------------------------------------------------------

# Mapping type: full_separation (default per spec §15.1).
SYNTHESIZER_MODEL_ID_FULL_SEPARATION = "synthesizer_qwen3_8b"
REASONER_MODEL_ID_FULL_SEPARATION = "reasoner_qwen3_8b"

# Mapping type: full_sharing.
SHARED_MODEL_ID_FULL_SHARING = "shared_qwen3_8b"
SYNTHESIZER_MODEL_ID_FULL_SHARING = SHARED_MODEL_ID_FULL_SHARING
REASONER_MODEL_ID_FULL_SHARING = SHARED_MODEL_ID_FULL_SHARING


__all__ = [
    "ANSWER_CHECKER_NODE_ID",
    "REASONER_AGENT_ID",
    "REASONER_MODEL_ID_FULL_SEPARATION",
    "REASONER_MODEL_ID_FULL_SHARING",
    "REASONER_NODE_ID",
    "SHARED_MODEL_ID_FULL_SHARING",
    "SYNTHESIZER_AGENT_ID",
    "SYNTHESIZER_MODEL_ID_FULL_SEPARATION",
    "SYNTHESIZER_MODEL_ID_FULL_SHARING",
    "SYNTHESIZER_NODE_ID",
    "TASK_VERIFIER_NODE_ID",
]
