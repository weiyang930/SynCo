"""UnityMAS-O ``star_ppo`` adapter for the Synthesizer-Reasoner co-training method.

``workflow_adapter`` (the per-query workflow) and ``reward_adapter`` (the per-role reward
allocator) are loaded by UnityMAS-O through the Hydra config; they import ``verl`` at module
load. ``roles`` holds the agent / model / node identifiers and is import-safe without ``verl``.
"""
from .roles import (
    ANSWER_CHECKER_NODE_ID,
    REASONER_AGENT_ID,
    REASONER_MODEL_ID_FULL_SEPARATION,
    REASONER_NODE_ID,
    SYNTHESIZER_AGENT_ID,
    SYNTHESIZER_MODEL_ID_FULL_SEPARATION,
    SYNTHESIZER_NODE_ID,
    TASK_VERIFIER_NODE_ID,
)

__all__ = [
    "ANSWER_CHECKER_NODE_ID",
    "REASONER_AGENT_ID",
    "REASONER_MODEL_ID_FULL_SEPARATION",
    "REASONER_NODE_ID",
    "SYNTHESIZER_AGENT_ID",
    "SYNTHESIZER_MODEL_ID_FULL_SEPARATION",
    "SYNTHESIZER_NODE_ID",
    "TASK_VERIFIER_NODE_ID",
]
