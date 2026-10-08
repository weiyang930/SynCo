from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from verl.experimental.star_ppo.reward_allocators.base import RewardAllocator
from verl.experimental.star_ppo.workflows.schema import RewardAssignment, WorkflowExecutionRecord, WorkflowTrace


class MAskTurnLevelRewardAllocator(RewardAllocator):
    """Paper-style turn-level reward allocation for M-ASK."""

    _TURN_NODE_RE = re.compile(r"^(search|summary|update|answer)_(\d+)$")

    def __init__(self, trainer, config, runner=None):
        super().__init__(trainer=trainer, config=config, runner=runner)
        self.workflow_cfg = self.config.star.get("workflow", {})
        self.mask_cfg = dict(self.workflow_cfg.get("mask", {}))

    @staticmethod
    def _record_predicted_answer(record: WorkflowExecutionRecord) -> str:
        parsed = record.parsed_output
        if isinstance(parsed, dict):
            for key in ("predicted_answer", "answer"):
                if key in parsed:
                    return str(parsed.get(key, "")).strip()
        if record.node_id == "plan" and isinstance(parsed, dict):
            state = parsed
            if "predicted_answer" in state:
                return str(state.get("predicted_answer", "")).strip()
        return str(parsed or "").strip()

    @staticmethod
    def _record_format_info(record: WorkflowExecutionRecord) -> tuple[float, float]:
        return float(record.meta.get("format_reward", 0.0)), float(record.meta.get("format_weight", 0.0))

    @staticmethod
    def _reward_components(record: WorkflowExecutionRecord, task_reward: float) -> tuple[float, float, float]:
        format_reward, format_weight = MAskTurnLevelRewardAllocator._record_format_info(record)
        format_penalty = float(format_weight) * float(format_reward)
        total_reward = RewardAllocator.compose_reward(task_reward, format_reward=format_reward, format_weight=format_weight)
        return float(task_reward), float(format_penalty), float(total_reward)

    @classmethod
    def _build_assignment(
        cls,
        record: WorkflowExecutionRecord,
        task_reward: float,
        reward_type: str,
    ) -> RewardAssignment:
        task_component, format_penalty, total_reward = cls._reward_components(record, task_reward)
        return RewardAssignment(
            record=record,
            reward=total_reward,
            reward_type=reward_type,
            meta={
                "task_reward": float(task_component),
                "format_penalty": float(format_penalty),
                "total_reward": float(total_reward),
            },
        )

    @staticmethod
    def _extract_update_op(record: WorkflowExecutionRecord) -> str:
        parsed = record.parsed_output
        if isinstance(parsed, dict):
            return str(parsed.get("operation", "")).strip().lower()
        return ""

    def allocate(self, trace: WorkflowTrace) -> tuple[list[RewardAssignment], dict[str, float]]:
        records = trace.records
        plan_record = next((rec for rec in records if rec.node_id == "plan"), None)

        answers_by_turn: dict[int, WorkflowExecutionRecord] = {}
        search_by_turn: dict[int, WorkflowExecutionRecord] = {}
        summary_by_turn: dict[int, WorkflowExecutionRecord] = {}
        update_by_turn: dict[int, WorkflowExecutionRecord] = {}
        for rec in records:
            match = self._TURN_NODE_RE.match(rec.node_id)
            if match is None:
                continue
            kind, turn_str = match.groups()
            turn = int(turn_str)
            if kind == "answer":
                answers_by_turn[turn] = rec
            elif kind == "search":
                search_by_turn[turn] = rec
            elif kind == "summary":
                summary_by_turn[turn] = rec
            elif kind == "update":
                update_by_turn[turn] = rec

        assignments: list[RewardAssignment] = []
        metrics: dict[str, float] = {}
        agent_task_acc: dict[str, list[float]] = defaultdict(list)
        agent_format_acc: dict[str, list[float]] = defaultdict(list)
        agent_total_acc: dict[str, list[float]] = defaultdict(list)

        def _track_agent_components(assignment: RewardAssignment) -> None:
            agent_id = str(assignment.record.agent_id or "")
            if not agent_id:
                return
            agent_task_acc[agent_id].append(float(assignment.meta.get("task_reward", 0.0)))
            agent_format_acc[agent_id].append(float(assignment.meta.get("format_penalty", 0.0)))
            agent_total_acc[agent_id].append(float(assignment.meta.get("total_reward", assignment.reward)))

        plan_f1 = 0.0
        if plan_record is not None:
            plan_answer = self._record_predicted_answer(plan_record)
            plan_f1 = self.max_f1_score(plan_answer, trace.ground_truth)
            assignment = self._build_assignment(plan_record, plan_f1, "plan_absolute_f1")
            assignments.append(assignment)
            _track_agent_components(assignment)

        prev_answer_f1 = None
        if 0 in answers_by_turn:
            answer_record = answers_by_turn[0]
            answer_f1 = self.max_f1_score(self._record_predicted_answer(answer_record), trace.ground_truth)
            assignment = self._build_assignment(answer_record, answer_f1, "answer_absolute_f1")
            assignments.append(assignment)
            _track_agent_components(assignment)
            prev_answer_f1 = answer_f1
            metrics["workflow/mask/answer0_f1"] = float(answer_f1)

        iter_deltas: list[float] = []
        search_end_count = 0
        update_ops: dict[str, int] = defaultdict(int)
        max_turn = 0

        for turn in sorted(set(search_by_turn.keys()) | set(answers_by_turn.keys()) | set(update_by_turn.keys())):
            max_turn = max(max_turn, turn)
            if turn == 0:
                continue
            search_record = search_by_turn.get(turn)
            answer_record = answers_by_turn.get(turn)
            summary_record = summary_by_turn.get(turn)
            update_record = update_by_turn.get(turn)

            if search_record is not None:
                parsed = search_record.parsed_output if isinstance(search_record.parsed_output, dict) else {}
                action = str(parsed.get("action", "")).strip().lower()
                if action == "end":
                    search_end_count += 1
                    assignment = self._build_assignment(search_record, 0.0, "search_end_zero")
                    assignments.append(assignment)
                    _track_agent_components(assignment)
                    continue

            if answer_record is None:
                continue

            answer_f1 = self.max_f1_score(self._record_predicted_answer(answer_record), trace.ground_truth)
            if prev_answer_f1 is None:
                delta = answer_f1
            else:
                delta = answer_f1 - prev_answer_f1
            iter_deltas.append(float(delta))

            if search_record is not None:
                assignment = self._build_assignment(search_record, delta, "iter_delta")
                assignments.append(assignment)
                _track_agent_components(assignment)
            if summary_record is not None:
                assignment = self._build_assignment(summary_record, delta, "iter_delta")
                assignments.append(assignment)
                _track_agent_components(assignment)
            if update_record is not None:
                assignment = self._build_assignment(update_record, delta, "iter_delta")
                assignments.append(assignment)
                _track_agent_components(assignment)
                op = self._extract_update_op(update_record)
                if op:
                    update_ops[op] += 1
            assignment = self._build_assignment(answer_record, answer_f1, "answer_absolute_f1")
            assignments.append(assignment)
            _track_agent_components(assignment)
            prev_answer_f1 = answer_f1

        final_f1 = float(prev_answer_f1) if prev_answer_f1 is not None else 0.0

        used_turns = float(trace.metrics.get("workflow/mask/used_turns", max_turn))
        completed_turns = float(trace.metrics.get("workflow/mask/completed_turns", len(iter_deltas)))

        metrics["workflow/mask/plan_f1"] = float(plan_f1)
        metrics["workflow/mask/final_f1"] = float(final_f1)
        metrics["workflow/mask/f1_gain_over_plan"] = float(final_f1 - plan_f1)
        metrics["workflow/mask/used_turns"] = float(used_turns)
        metrics["workflow/mask/turns"] = float(used_turns)
        metrics["workflow/mask/completed_turns"] = float(completed_turns)
        metrics["workflow/mask/search_end_count"] = float(search_end_count)
        metrics["workflow/mask/iter_delta_mean"] = float(sum(iter_deltas) / max(1, len(iter_deltas))) if iter_deltas else 0.0
        metrics["workflow/mask/reward_assignments"] = float(len(assignments))
        metrics["workflow/mask/update_add_count"] = float(update_ops.get("add", 0))
        metrics["workflow/mask/update_update_count"] = float(update_ops.get("update", 0))
        metrics["workflow/mask/update_other_count"] = float(
            sum(v for k, v in update_ops.items() if k not in {"add", "update"})
        )
        for agent_id, values in agent_format_acc.items():
            metrics[f"workflow/mask/agent/{agent_id}/format_penalty_mean"] = float(sum(values) / max(1, len(values)))
        for agent_id, values in agent_task_acc.items():
            metrics[f"workflow/mask/agent/{agent_id}/task_reward_mean"] = float(sum(values) / max(1, len(values)))
        for agent_id, values in agent_total_acc.items():
            metrics[f"workflow/mask/agent/{agent_id}/total_reward_mean"] = float(sum(values) / max(1, len(values)))
        return assignments, metrics
