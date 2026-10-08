from __future__ import annotations

import re
from collections import defaultdict

from verl.experimental.star_ppo.reward_allocators.base import RewardAllocator
from verl.experimental.star_ppo.workflows.schema import RewardAssignment, WorkflowExecutionRecord, WorkflowTrace


class CodeTurnLevelRewardAllocator(RewardAllocator):
    """Turn-level credit assignment for iterative code generation.

    Planner/Coder at turn i receive R_i - R_{i-1} (or R_0 for i=0).
    Reflection at turn i receives R_{i+1} - R_i.
    Each LLM node also receives its own format penalty.
    """

    _NODE_RE = re.compile(r"^(planner|coder|reflection|verifier)_(\d+)$")

    @staticmethod
    def _record_format_info(record: WorkflowExecutionRecord) -> tuple[float, float]:
        return float(record.meta.get("format_reward", 0.0)), float(record.meta.get("format_weight", 0.0))

    @classmethod
    def _build_assignment(
        cls,
        record: WorkflowExecutionRecord,
        task_reward: float,
        reward_type: str,
    ) -> RewardAssignment:
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
            reward_type=reward_type,
            meta={
                "task_reward": float(task_reward),
                "format_penalty": float(format_penalty),
                "total_reward": float(total_reward),
            },
        )

    @staticmethod
    def _verifier_pass_rate(record: WorkflowExecutionRecord) -> float:
        parsed = record.parsed_output if isinstance(record.parsed_output, dict) else {}
        value = float(parsed.get("pass_rate", 0.0) or 0.0)
        return max(0.0, min(1.0, value))

    @staticmethod
    def _verifier_all_passed(record: WorkflowExecutionRecord) -> float:
        parsed = record.parsed_output if isinstance(record.parsed_output, dict) else {}
        return float(1.0 if int(parsed.get("all_passed", 0) or 0) else 0.0)

    def allocate(self, trace: WorkflowTrace) -> tuple[list[RewardAssignment], dict[str, float]]:
        planner_by_turn: dict[int, WorkflowExecutionRecord] = {}
        coder_by_turn: dict[int, WorkflowExecutionRecord] = {}
        reflection_by_turn: dict[int, WorkflowExecutionRecord] = {}
        verifier_by_turn: dict[int, WorkflowExecutionRecord] = {}

        for record in trace.records:
            match = self._NODE_RE.match(str(record.node_id))
            if match is None:
                continue
            kind, turn_s = match.groups()
            turn = int(turn_s)
            if kind == "planner":
                planner_by_turn[turn] = record
            elif kind == "coder":
                coder_by_turn[turn] = record
            elif kind == "reflection":
                reflection_by_turn[turn] = record
            elif kind == "verifier":
                verifier_by_turn[turn] = record

        pass_rate_by_turn = {turn: self._verifier_pass_rate(record) for turn, record in verifier_by_turn.items()}
        all_passed_by_turn = {turn: self._verifier_all_passed(record) for turn, record in verifier_by_turn.items()}

        assignments: list[RewardAssignment] = []
        agent_task_acc: dict[str, list[float]] = defaultdict(list)
        agent_format_acc: dict[str, list[float]] = defaultdict(list)
        agent_total_acc: dict[str, list[float]] = defaultdict(list)

        def track(assignment: RewardAssignment) -> None:
            agent_id = str(assignment.record.agent_id or "")
            if not agent_id:
                return
            agent_task_acc[agent_id].append(float(assignment.meta.get("task_reward", 0.0)))
            agent_format_acc[agent_id].append(float(assignment.meta.get("format_penalty", 0.0)))
            agent_total_acc[agent_id].append(float(assignment.meta.get("total_reward", assignment.reward)))

        prev_rate = 0.0
        turn_deltas: list[float] = []
        planner_coder_delta_by_turn: dict[int, float] = {}
        for turn in sorted(pass_rate_by_turn):
            rate = float(pass_rate_by_turn[turn])
            delta = rate if turn == 0 else rate - prev_rate
            planner_coder_delta_by_turn[turn] = float(delta)
            turn_deltas.append(float(delta))
            if turn in planner_by_turn:
                assignment = self._build_assignment(planner_by_turn[turn], delta, "planner_pass_rate_delta")
                assignments.append(assignment)
                track(assignment)
            if turn in coder_by_turn:
                assignment = self._build_assignment(coder_by_turn[turn], delta, "coder_pass_rate_delta")
                assignments.append(assignment)
                track(assignment)
            prev_rate = rate

        reflection_deltas: list[float] = []
        reflection_delta_by_turn: dict[int, float] = {}
        for turn, record in sorted(reflection_by_turn.items()):
            if turn not in pass_rate_by_turn or (turn + 1) not in pass_rate_by_turn:
                task_reward = 0.0
            else:
                task_reward = float(pass_rate_by_turn[turn + 1] - pass_rate_by_turn[turn])
            reflection_delta_by_turn[turn] = float(task_reward)
            reflection_deltas.append(float(task_reward))
            assignment = self._build_assignment(record, task_reward, "reflection_next_pass_rate_delta")
            assignments.append(assignment)
            track(assignment)

        final_turn = max(pass_rate_by_turn.keys()) if pass_rate_by_turn else -1
        final_pass_rate = float(pass_rate_by_turn.get(final_turn, 0.0))
        final_all_passed = float(all_passed_by_turn.get(final_turn, 0.0))

        metrics: dict[str, float] = {
            "workflow/code/final_pass_rate": final_pass_rate,
            "workflow/code/final_all_passed": final_all_passed,
            "workflow/code/reward_assignments": float(len(assignments)),
            "workflow/code/turn_delta_mean": float(sum(turn_deltas) / max(1, len(turn_deltas))) if turn_deltas else 0.0,
            "workflow/code/reflection_delta_mean": (
                float(sum(reflection_deltas) / max(1, len(reflection_deltas))) if reflection_deltas else 0.0
            ),
        }
        for turn, rate in pass_rate_by_turn.items():
            metrics[f"workflow/code/turn_{turn}/pass_rate"] = float(rate)
            metrics[f"workflow/code/turn_{turn}/all_passed"] = float(all_passed_by_turn.get(turn, 0.0))
            metrics[f"workflow/code/turn_{turn}/planner_coder_delta"] = float(
                planner_coder_delta_by_turn.get(turn, 0.0)
            )
        for turn, delta in reflection_delta_by_turn.items():
            metrics[f"workflow/code/turn_{turn}/reflection_delta"] = float(delta)
        for agent_id, values in agent_format_acc.items():
            metrics[f"workflow/code/agent/{agent_id}/format_penalty_mean"] = float(sum(values) / max(1, len(values)))
        for agent_id, values in agent_task_acc.items():
            metrics[f"workflow/code/agent/{agent_id}/task_reward_mean"] = float(sum(values) / max(1, len(values)))
        for agent_id, values in agent_total_acc.items():
            metrics[f"workflow/code/agent/{agent_id}/total_reward_mean"] = float(sum(values) / max(1, len(values)))
        return assignments, metrics
