"""``RewardAllocator`` subclass that splits trace rewards across roles.

* The Synthesizer record gets the Synthesizer reward of its task (``invalid_synth_reward``
  if the task failed verification).
* Each Reasoner rollout record gets its own reward (answer correctness).
* Tool nodes (TaskVerifier, AnswerChecker) are non-trainable and skipped.

Both roles additionally get ``format_weight * format_reward`` (UnityMAS-O
``RewardAllocator.compose_reward``; format_reward = 1 for any non-empty output).
The trace already carries the reward breakdown in ``trace.state``; we only consume it here.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from verl.experimental.star_ppo.reward_allocators.base import RewardAllocator  # type: ignore[import]
from verl.experimental.star_ppo.workflows.schema import (  # type: ignore[import]
    RewardAssignment,
    WorkflowTrace,
)

from . import roles


class SynthReasonerRewardAllocator(RewardAllocator):
    """Per-role reward allocation for the Synthesizer-Reasoner workflow."""

    def __init__(self, trainer, config, runner=None, **kwargs):
        super().__init__(trainer=trainer, config=config, runner=runner)
        synth_cfg = dict(self.config.star.get("workflow", {}).get("synth_reasoner", {}))
        self._invalid_synth_reward = float(
            synth_cfg.get("invalid_synth_reward", -0.5)
        )
        self._train_synthesizer = bool(synth_cfg.get("train_synthesizer", True))
        self._train_reasoner = bool(synth_cfg.get("train_reasoner", True))
        # ``kwargs`` is accepted for forward compatibility with extra YAML flags.
        for k, v in kwargs.items():
            setattr(self, f"_extra_{k}", v)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, trace: WorkflowTrace) -> tuple[list[RewardAssignment], dict[str, float]]:
        assignments: list[RewardAssignment] = []
        agent_totals: dict[str, list[float]] = defaultdict(list)

        is_valid = bool(trace.state.get("is_valid", False))
        breakdown = dict(trace.state.get("synth_immediate_breakdown", {}) or {})
        immediate_total = float(breakdown.get("immediate_total", 0.0))
        synth_total = immediate_total if is_valid else self._invalid_synth_reward

        for record in trace.records:
            if not record.trainable:
                continue
            node_id = str(record.node_id or "")

            # Synthesizer step.
            if node_id == roles.SYNTHESIZER_NODE_ID:
                if not self._train_synthesizer:
                    continue
                fmt_reward = float(record.meta.get("format_reward", 0.0))
                fmt_weight = float(record.meta.get("format_weight", 0.0))
                total = RewardAllocator.compose_reward(
                    synth_total, format_reward=fmt_reward, format_weight=fmt_weight
                )
                assignments.append(
                    RewardAssignment(
                        record=record,
                        reward=float(total),
                        reward_type="synthesizer_composite",
                        meta={
                            "task_reward": float(synth_total),
                            "immediate_total": float(immediate_total),
                            "is_valid": is_valid,
                            **{f"breakdown.{k}": float(v) for k, v in breakdown.items()},
                        },
                    )
                )
                agent_totals[record.agent_id].append(float(total))
                continue

            # Reasoner step: per-rollout reward already stored on the record.
            if node_id.startswith(f"{roles.REASONER_NODE_ID}_"):
                if not self._train_reasoner:
                    continue
                per_reward = float(record.meta.get("reasoner_reward", 0.0))
                fmt_reward = float(record.meta.get("format_reward", 0.0))
                fmt_weight = float(record.meta.get("format_weight", 0.0))
                total = RewardAllocator.compose_reward(
                    per_reward, format_reward=fmt_reward, format_weight=fmt_weight
                )
                assignments.append(
                    RewardAssignment(
                        record=record,
                        reward=float(total),
                        reward_type="reasoner_per_rollout",
                        meta={
                            "task_reward": float(per_reward),
                            "is_correct": bool(record.meta.get("is_correct", False)),
                        },
                    )
                )
                agent_totals[record.agent_id].append(float(total))
                continue
            # Other trainable nodes (none today) are ignored — be loud about it.
            if node_id:
                continue

        metrics: dict[str, Any] = {
            "workflow/synth_reasoner/synth_total_reward": float(synth_total),
            "workflow/synth_reasoner/synth_immediate_total": float(immediate_total),
            "workflow/synth_reasoner/reward_assignments": float(len(assignments)),
        }
        for agent_id, values in agent_totals.items():
            if not values:
                continue
            metrics[f"workflow/synth_reasoner/agent/{agent_id}/reward_mean"] = float(
                sum(values) / len(values)
            )
            metrics[f"workflow/synth_reasoner/agent/{agent_id}/reward_count"] = float(
                len(values)
            )
        return assignments, metrics


__all__ = ["SynthReasonerRewardAllocator"]
