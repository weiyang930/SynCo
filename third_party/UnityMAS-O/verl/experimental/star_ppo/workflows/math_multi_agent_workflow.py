from __future__ import annotations

import copy
import json
import re
from typing import Any

from verl import DataProto
from verl.experimental.star_ppo.tools.math_answer import grade_math_answer
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord, WorkflowTrace
from verl.experimental.star_ppo.workflows.trace_workflow import TraceWorkflowRunner


class MathMultiAgentWorkflowRunner(TraceWorkflowRunner):
    """Solver -> Verifier -> Refiner -> Finalizer workflow for math tasks."""

    AGENT_ORDER = ("solver", "verifier", "refiner", "finalizer")
    AGENT_TAGS = {
        "solver": ("SOLUTION", "ANSWER"),
        "verifier": ("STATUS", "ERROR_TYPE", "FEEDBACK"),
        "refiner": ("REFINED_SOLUTION", "ANSWER"),
        "finalizer": ("FINAL_SOLUTION", "FINAL_ANSWER"),
    }
    STATUS_VALUES = {"correct", "incorrect", "uncertain"}
    ERROR_TYPE_VALUES = {
        "none",
        "calculation",
        "reasoning",
        "missing_case",
        "misread_problem",
        "format",
        "unclear",
    }

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        self.math_cfg = dict(self.workflow_cfg.get("math", {}))
        self.agent_cfgs = {name: dict(self.math_cfg.get(name, {})) for name in self.AGENT_ORDER}
        self.expose_ground_truth_to_prompts = bool(self.math_cfg.get("expose_ground_truth_to_prompts", False))
        self.max_state_chars = int(self.math_cfg.get("max_state_chars", 12000))

    @staticmethod
    def _safe_json(value: Any, max_chars: int = 12000) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            text = str(value)
        if max_chars > 0 and len(text) > max_chars:
            keep = max_chars // 2
            return text[:keep] + "\n...(truncated)...\n" + text[-keep:]
        return text

    @staticmethod
    def _safe_metric_name(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
        return text.strip("_") or "unknown"

    def _extract_first(self, query_batch: DataProto, candidates: list[str], default: Any = "") -> Any:
        for key in candidates:
            value = self._extract_from_batch(query_batch, key)
            if value is not None:
                if isinstance(value, str) and not value.strip():
                    continue
                return value
        return default

    def _extract_query_id(self, query_batch: DataProto) -> str:
        for key in ("query_id", "uid", "id", "extra_info.index"):
            value = self._extract_from_batch(query_batch, key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @classmethod
    def _parse_required_tags(cls, response_text: str, tags: tuple[str, ...]) -> tuple[dict[str, str], bool]:
        raw = str(response_text or "").strip()
        parsed: dict[str, str] = {}
        if not raw:
            return {tag.lower(): "" for tag in tags}, False

        cleaned = raw
        legal = True
        for tag in tags:
            pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(raw)
            if len(matches) != 1 or not str(matches[0]).strip():
                legal = False
                parsed[tag.lower()] = str(matches[-1]).strip() if matches else ""
            else:
                parsed[tag.lower()] = str(matches[0]).strip()
            cleaned = pattern.sub("", cleaned)

        if cleaned.strip():
            legal = False

        status = parsed.get("status", "").strip().lower()
        if "STATUS" in tags and status not in cls.STATUS_VALUES:
            legal = False
        error_type = parsed.get("error_type", "").strip().lower()
        if "ERROR_TYPE" in tags and error_type not in cls.ERROR_TYPE_VALUES:
            legal = False
        return parsed, legal

    @staticmethod
    def _set_record_format_reward(record: WorkflowExecutionRecord, is_legal: bool) -> None:
        record.meta["format_reward"] = 0.0 if is_legal else -1.0
        record.meta["is_legal_format"] = bool(is_legal)

    def _new_global_state(self, problem: str, ground_truth: list[str], data_source: str) -> dict[str, Any]:
        return {
            "problem": str(problem or ""),
            "ground_truth": list(ground_truth or []),
            "data_source": str(data_source or "unknown"),
            "outputs": {},
        }

    def _format_global_state(self, state: dict[str, Any]) -> str:
        visible_state = copy.deepcopy(state)
        if not self.expose_ground_truth_to_prompts:
            visible_state["ground_truth"] = "[hidden from agents; used only for reward]"
        return self._safe_json(visible_state, max_chars=self.max_state_chars)

    def _build_prompt_context(
        self,
        *,
        problem: str,
        state: dict[str, Any],
        solver_output: str = "",
        verifier_output: str = "",
        refiner_output: str = "",
    ) -> dict[str, Any]:
        return {
            "problem": problem,
            "question": problem,
            "global_state_text": self._format_global_state(state),
            "solver_output": solver_output,
            "initial_solution": solver_output,
            "verifier_output": verifier_output,
            "verification_feedback": verifier_output,
            "refiner_output": refiner_output,
            "refined_solution": refiner_output,
        }

    async def _run_agent(
        self,
        *,
        query_batch: DataProto,
        agent_name: str,
        step_id: int,
        query_id: str,
        prompt_context: dict[str, Any],
        state_before: dict[str, Any],
    ) -> WorkflowExecutionRecord:
        node_cfg = self.agent_cfgs[agent_name]
        record = await self._execute_llm_step(
            query_batch=query_batch,
            node_id=agent_name,
            turn_id=0,
            step_id=step_id,
            node_cfg=node_cfg,
            prompt_context=prompt_context,
            state_before=copy.deepcopy(state_before),
        )
        record.query_id = query_id
        parsed, is_legal = self._parse_required_tags(record.raw_output, self.AGENT_TAGS[agent_name])
        self._set_record_format_reward(record, is_legal)
        record.parsed_output = parsed
        return record

    def _build_debug_dump(
        self,
        query_local_idx: int,
        problem: str,
        records: list[WorkflowExecutionRecord],
        final_state: dict[str, Any],
        debug_max_chars: int | None = None,
    ) -> str:
        max_chars = self.debug_max_chars if debug_max_chars is None else int(debug_max_chars)

        def compact(value: Any) -> str:
            if isinstance(value, dict):
                text = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                text = str(value)
            return text if max_chars <= 0 else text[:max_chars]

        final_grade = final_state.get("final_grade", {}) if isinstance(final_state, dict) else {}
        lines = [
            "[math-example] ===== full example begin =====",
            f"[math-example] query_idx={query_local_idx}",
            f"[math-example] problem:\n{compact(problem)}",
            f"[math-example] ground_truth:\n{compact(final_state.get('ground_truth', []))}",
            f"[math-example] final_answer:\n{compact(final_state.get('final_answer', ''))}",
            f"[math-example] final_acc={float(1.0 if final_grade.get('acc', False) else 0.0)}",
            f"[math-example] final_grade:\n{compact(final_grade)}",
        ]
        for record in records:
            meta = record.meta if isinstance(record.meta, dict) else {}
            lines.append(
                "[math-example] --- agent output ---\n"
                f"step={record.step_id} node={record.node_id} agent={record.agent_id} model={record.model_id}\n"
                f"format_legal={meta.get('is_legal_format', 'n/a')} "
                f"format_reward={meta.get('format_reward', 'n/a')} "
                f"prompt_tokens={meta.get('prompt_tokens', 'n/a')} "
                f"output_tokens={meta.get('output_tokens', 'n/a')}\n"
                f"raw_output:\n{compact(record.raw_output)}\n"
                f"parsed_output:\n{compact(record.parsed_output)}"
            )
        lines.append(f"[math-example] final_state:\n{compact(final_state)}")
        lines.append("[math-example] ===== full example end =====")
        return "\n".join(lines)

    async def run_query(
        self,
        query_batch: DataProto,
        query_local_idx: int,
        debug: bool,
        debug_max_chars: int | None = None,
    ) -> WorkflowTrace:
        problem = self._extract_question(query_batch)
        ground_truth = self._extract_gt_list(query_batch)
        data_source = str(
            self._extract_first(query_batch, ["data_source", "source", "extra_info.data_source"], "math")
        )
        query_id = self._extract_query_id(query_batch)
        state = self._new_global_state(problem, ground_truth, data_source)

        records: list[WorkflowExecutionRecord] = []
        solver_output = ""
        verifier_output = ""
        refiner_output = ""

        solver_record = await self._run_agent(
            query_batch=query_batch,
            agent_name="solver",
            step_id=0,
            query_id=query_id,
            prompt_context=self._build_prompt_context(problem=problem, state=state),
            state_before=state,
        )
        solver_record.state_after = copy.deepcopy(state)
        state["outputs"]["solver"] = copy.deepcopy(solver_record.parsed_output)
        solver_record.state_after["outputs"]["solver"] = copy.deepcopy(solver_record.parsed_output)
        records.append(solver_record)
        solver_output = solver_record.raw_output

        verifier_record = await self._run_agent(
            query_batch=query_batch,
            agent_name="verifier",
            step_id=1,
            query_id=query_id,
            prompt_context=self._build_prompt_context(problem=problem, state=state, solver_output=solver_output),
            state_before=state,
        )
        verifier_record.state_after = copy.deepcopy(state)
        state["outputs"]["verifier"] = copy.deepcopy(verifier_record.parsed_output)
        verifier_record.state_after["outputs"]["verifier"] = copy.deepcopy(verifier_record.parsed_output)
        records.append(verifier_record)
        verifier_output = verifier_record.raw_output

        refiner_record = await self._run_agent(
            query_batch=query_batch,
            agent_name="refiner",
            step_id=2,
            query_id=query_id,
            prompt_context=self._build_prompt_context(
                problem=problem,
                state=state,
                solver_output=solver_output,
                verifier_output=verifier_output,
            ),
            state_before=state,
        )
        refiner_record.state_after = copy.deepcopy(state)
        state["outputs"]["refiner"] = copy.deepcopy(refiner_record.parsed_output)
        refiner_record.state_after["outputs"]["refiner"] = copy.deepcopy(refiner_record.parsed_output)
        records.append(refiner_record)
        refiner_output = refiner_record.raw_output

        finalizer_record = await self._run_agent(
            query_batch=query_batch,
            agent_name="finalizer",
            step_id=3,
            query_id=query_id,
            prompt_context=self._build_prompt_context(
                problem=problem,
                state=state,
                solver_output=solver_output,
                verifier_output=verifier_output,
                refiner_output=refiner_output,
            ),
            state_before=state,
        )
        finalizer_record.state_after = copy.deepcopy(state)
        state["outputs"]["finalizer"] = copy.deepcopy(finalizer_record.parsed_output)
        finalizer_record.state_after["outputs"]["finalizer"] = copy.deepcopy(finalizer_record.parsed_output)
        records.append(finalizer_record)

        final_answer = str(
            finalizer_record.parsed_output.get("final_answer", "")
            if isinstance(finalizer_record.parsed_output, dict)
            else ""
        )
        grades = [grade_math_answer(final_answer, gt) for gt in ground_truth]
        grade = next(
            (item for item in grades if item.get("acc", False)),
            grades[0] if grades else grade_math_answer(final_answer, ""),
        )
        state["final_answer"] = final_answer
        state["final_grade"] = grade

        agent_call_counts: dict[str, int] = {}
        agent_legal_counts: dict[str, int] = {}
        agent_prompt_tokens: dict[str, list[float]] = {}
        agent_output_tokens: dict[str, list[float]] = {}
        for record in records:
            if not record.trainable:
                continue
            agent_id = str(record.agent_id or "")
            agent_call_counts[agent_id] = agent_call_counts.get(agent_id, 0) + 1
            agent_legal_counts[agent_id] = agent_legal_counts.get(agent_id, 0) + int(
                bool(record.meta.get("is_legal_format", False))
            )
            agent_prompt_tokens.setdefault(agent_id, []).append(float(record.meta.get("prompt_tokens", 0.0) or 0.0))
            agent_output_tokens.setdefault(agent_id, []).append(float(record.meta.get("output_tokens", 0.0) or 0.0))

        safe_source = self._safe_metric_name(data_source)
        metrics: dict[str, float] = {
            "workflow/math/final_acc": float(1.0 if grade["acc"] else 0.0),
            "workflow/math/final_score": float(grade["score"]),
            "workflow/math/global_state_chars": float(len(self._format_global_state(state))),
            f"workflow/math/dataset/{safe_source}/acc": float(1.0 if grade["acc"] else 0.0),
        }
        for agent_id, count in agent_call_counts.items():
            legal_count = agent_legal_counts.get(agent_id, 0)
            prompt_values = agent_prompt_tokens.get(agent_id, [])
            output_values = agent_output_tokens.get(agent_id, [])
            metrics[f"workflow/math/agent/{agent_id}/call_count"] = float(count)
            metrics[f"workflow/math/agent/{agent_id}/format_legal_rate"] = float(legal_count / max(1, count))
            metrics[f"workflow/math/agent/{agent_id}/prompt_tokens_mean"] = (
                float(sum(prompt_values) / max(1, len(prompt_values))) if prompt_values else 0.0
            )
            metrics[f"workflow/math/agent/{agent_id}/output_tokens_mean"] = (
                float(sum(output_values) / max(1, len(output_values))) if output_values else 0.0
            )

        trace = WorkflowTrace(
            query_id=query_id,
            question=problem,
            ground_truth=ground_truth,
            records=records,
            state={
                "global_state": state,
                "final_answer": final_answer,
                "final_grade": grade,
                "data_source": data_source,
            },
            metrics=metrics,
        )
        if debug:
            trace.debug_dump = self._build_debug_dump(
                query_local_idx,
                problem,
                records,
                state,
                debug_max_chars=debug_max_chars,
            )
        return trace
