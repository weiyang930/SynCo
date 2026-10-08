from __future__ import annotations

import re
from collections import defaultdict

from verl.experimental.star_ppo.reward_allocators.base import RewardAllocator
from verl.experimental.star_ppo.workflows.schema import RewardAssignment, WorkflowExecutionRecord, WorkflowTrace


class MathFinalAnswerRewardAllocator(RewardAllocator):
    """Assign the final answer accuracy reward to every math agent.

    Each trainable agent receives:
      final correctness score in {0, 1} + its own format penalty in {0, -1}.
    """

    _NODE_RE = re.compile(r"^(solver|verifier|refiner|finalizer)$")

    @staticmethod
    def _record_format_info(record: WorkflowExecutionRecord) -> tuple[float, float]:
        return float(record.meta.get("format_reward", 0.0)), float(record.meta.get("format_weight", 0.0))

    @classmethod
    def _build_assignment(cls, record: WorkflowExecutionRecord, task_reward: float) -> RewardAssignment:
        format_reward, format_weight = cls._record_format_info(record)
        format_penalty = float(format_weight) * float(format_reward)
        total_reward = RewardAllocator.compose_reward(
            task_reward,
            format_reward=format_reward,
            format_weight=format_weight,
        )
        return RewardAssignment(
            record=record,
            reward=float(total_reward),
            reward_type="math_final_answer_acc_plus_format",
            meta={
                "task_reward": float(task_reward),
                "format_penalty": float(format_penalty),
                "total_reward": float(total_reward),
            },
        )

    @staticmethod
    def _final_acc(trace: WorkflowTrace) -> float:
        if "workflow/math/final_acc" in trace.metrics:
            return float(trace.metrics.get("workflow/math/final_acc", 0.0) or 0.0)
        final_grade = dict(trace.state.get("final_grade", {})) if isinstance(trace.state, dict) else {}
        return float(1.0 if final_grade.get("acc", False) else 0.0)

    def allocate(self, trace: WorkflowTrace) -> tuple[list[RewardAssignment], dict[str, float]]:
        task_reward = self._final_acc(trace)
        assignments: list[RewardAssignment] = []
        agent_task_acc: dict[str, list[float]] = defaultdict(list)
        agent_format_acc: dict[str, list[float]] = defaultdict(list)
        agent_total_acc: dict[str, list[float]] = defaultdict(list)

        for record in trace.records:
            if not record.trainable or self._NODE_RE.match(str(record.node_id)) is None:
                continue
            assignment = self._build_assignment(record, task_reward)
            assignments.append(assignment)
            agent_id = str(record.agent_id or "")
            if agent_id:
                agent_task_acc[agent_id].append(float(assignment.meta.get("task_reward", 0.0)))
                agent_format_acc[agent_id].append(float(assignment.meta.get("format_penalty", 0.0)))
                agent_total_acc[agent_id].append(float(assignment.meta.get("total_reward", assignment.reward)))

        metrics: dict[str, float] = {
            "workflow/math/final_acc": float(task_reward),
            "workflow/math/reward_assignments": float(len(assignments)),
        }
        for agent_id, values in agent_task_acc.items():
            metrics[f"workflow/math/agent/{agent_id}/task_reward_mean"] = float(sum(values) / max(1, len(values)))
        for agent_id, values in agent_format_acc.items():
            metrics[f"workflow/math/agent/{agent_id}/format_penalty_mean"] = float(sum(values) / max(1, len(values)))
        for agent_id, values in agent_total_acc.items():
            metrics[f"workflow/math/agent/{agent_id}/total_reward_mean"] = float(sum(values) / max(1, len(values)))
        return assignments, metrics
