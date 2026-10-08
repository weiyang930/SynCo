from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from verl import DataProto
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord, WorkflowTrace
from verl.experimental.star_ppo.workflows.trace_workflow import TraceWorkflowRunner


class CodeIterativeWorkflowRunner(TraceWorkflowRunner):
    """Planner/Coder/Verifier/Reflection iterative workflow for code tasks."""

    GLOBAL_STATE_SCHEMA = {
        "problem": "Original coding task text.",
        "starter_code": "Optional starter/template code from the dataset.",
        "iterations": [
            {
                "turn": "0-based iteration id.",
                "pseudocode": "Planner output for this turn.",
                "code": "Coder output for this turn.",
                "verification": {
                    "pass_rate": "Verifier pass ratio in [0, 1].",
                    "all_passed": "1 if all visible tests pass, otherwise 0.",
                    "passed": "Number of passed tests.",
                    "total": "Number of tests.",
                    "error": "Verifier error/wrong-answer/timeout details.",
                },
                "reflection": "Reflection output. Missing on final turn or early stop.",
            }
        ],
    }

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        self.code_cfg = dict(self.workflow_cfg.get("code", {}))
        self.max_turns = int(self.code_cfg.get("max_turns", 3))
        self.stop_on_all_passed = bool(self.code_cfg.get("stop_on_all_passed", True))
        self.planner_cfg = dict(self.code_cfg.get("planner", {}))
        self.coder_cfg = dict(self.code_cfg.get("coder", {}))
        self.reflection_cfg = dict(self.code_cfg.get("reflection", {}))
        self.verifier_cfg = dict(self.code_cfg.get("verifier", {}))
        self.tests_candidates = list(
            self.workflow_cfg.get(
                "tests_candidates",
                [
                    "tests",
                    "test_cases",
                    "answer",
                    "reward_model.ground_truth",
                    "reward_model",
                    "extra.tests",
                    "extra_info.tests",
                    "extra_info.public_test_cases",
                ],
            )
        )
        self.starter_code_candidates = list(
            self.workflow_cfg.get("starter_code_candidates", ["starter_code", "extra.starter_code"])
        )

    @staticmethod
    def _set_record_format_reward(record: WorkflowExecutionRecord, is_legal: bool) -> None:
        record.meta["format_reward"] = 0.0 if is_legal else -1.0
        record.meta["is_legal_format"] = bool(is_legal)

    @staticmethod
    def _parse_tagged_text(response_text: str, tag: str, fallback: str = "") -> tuple[str, bool]:
        raw = str(response_text or "").strip()
        if not raw:
            return fallback, False
        pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(raw)
        cleaned = pattern.sub("", raw).strip()
        is_legal = len(matches) == 1 and cleaned == "" and bool(str(matches[0]).strip())
        if matches:
            return str(matches[0]).strip(), is_legal
        return raw.strip() or fallback, False

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

    @staticmethod
    def _safe_metric_name(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
        return text.strip("_") or "unknown"

    @staticmethod
    def _loads_maybe(value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return value
        try:
            return json.loads(raw)
        except Exception:
            return value

    @classmethod
    def _find_fn_name_in_value(cls, value: Any) -> str:
        value = cls._loads_maybe(value)
        if isinstance(value, dict):
            for key in ("fn_name", "function_name", "func_name"):
                fn_name = value.get(key)
                if fn_name is not None and str(fn_name).strip():
                    return str(fn_name).strip()
            for key in ("metadata", "extra_info", "label", "tests", "public_tests", "private_tests"):
                fn_name = cls._find_fn_name_in_value(value.get(key))
                if fn_name:
                    return fn_name
        if isinstance(value, list):
            for item in value:
                fn_name = cls._find_fn_name_in_value(item)
                if fn_name:
                    return fn_name
        return ""

    def _extract_fn_name(self, query_batch: DataProto, tests: Any) -> str:
        for value in (
            tests,
            self._extract_first(query_batch, ["metadata", "extra_info.metadata", "extra.metadata"], ""),
            self._extract_first(query_batch, ["extra_info", "extra"], ""),
            self._extract_first(query_batch, ["label"], ""),
        ):
            fn_name = self._find_fn_name_in_value(value)
            if fn_name:
                return fn_name
        return ""

    @staticmethod
    def _execution_instruction(fn_name: str) -> str:
        fn_name = str(fn_name or "").strip()
        if fn_name:
            return (
                f'This is a call-based function task. Implement the function "{fn_name}" with exactly this name. '
                "Do not read from stdin, do not write to stdout, and do not wrap the solution in an input loop. "
                "Return the answer from the function. You may define helper functions/classes if needed."
            )
        return (
            "This is a standard input/output task. Write a complete Python 3 program that reads from stdin "
            "and writes the answer to stdout."
        )

    @staticmethod
    def _coder_task_instruction(fn_name: str) -> str:
        fn_name = str(fn_name or "").strip()
        if fn_name:
            return (
                f'Write a Python 3 function solution. The submitted code must define "{fn_name}" exactly. '
                "The verifier will call this function directly with the provided arguments. "
                "Do not parse stdin, do not print the answer, and do not add an input loop. "
                "Return the result using Python objects or scalar values matching the expected output."
            )
        return (
            "Write a complete Python 3 stdin/stdout program. The submitted code must read all required input "
            "from standard input and print the required output to standard output. "
            "Do not define only a function unless the program also calls it from main."
        )

    @staticmethod
    def _format_example_code(fn_name: str) -> str:
        fn_name = str(fn_name or "").strip()
        if fn_name:
            return (
                f"def {fn_name}(*args):\n"
                "    # Replace this example logic with the actual solution.\n"
                "    return None"
            )
        return (
            "import sys\n\n"
            "def solve():\n"
            "    data = sys.stdin.read().strip().split()\n"
            "    # Replace this example logic with the actual solution.\n"
            "    if not data:\n"
            "        return\n"
            "    print(data[0])\n\n"
            "if __name__ == \"__main__\":\n"
            "    solve()"
        )

    @staticmethod
    def _format_visible_tests(tests: Any, max_examples: int = 3, max_chars: int = 3000) -> str:
        try:
            from verl.experimental.star_ppo.tools.code_verifier import CodeVerifierTool

            cases = CodeVerifierTool.normalize_tests(tests)
        except Exception:
            cases = []

        if not cases:
            return "No visible tests were provided."

        lines: list[str] = []
        total = len(cases)
        for idx, case in enumerate(cases[: max(0, int(max_examples))]):
            fn_name = str(case.get("fn_name") or "").strip()
            mode = f"call fn_name={fn_name}" if fn_name else "stdin/stdout"
            lines.append(f"Test {idx + 1} of {total} ({mode})")
            lines.append("Input:")
            lines.append(str(case.get("input", "")))
            lines.append("Expected output:")
            lines.append(str(case.get("output", "")))
            lines.append("")
        if total > max_examples:
            lines.append(f"... {total - max_examples} more visible tests omitted from the prompt.")

        text = "\n".join(lines).strip()
        if max_chars > 0 and len(text) > max_chars:
            keep = max_chars // 2
            text = text[:keep] + "\n...(visible tests truncated)...\n" + text[-keep:]
        return text

    @staticmethod
    def _new_global_state(
        problem: str,
        starter_code: str = "",
        fn_name: str = "",
        visible_tests_summary: str = "",
    ) -> dict[str, Any]:
        return {
            "problem": str(problem or ""),
            "starter_code": str(starter_code or ""),
            "fn_name": str(fn_name or ""),
            "execution_instruction": CodeIterativeWorkflowRunner._execution_instruction(fn_name),
            "visible_tests": str(visible_tests_summary or ""),
            "iterations": [],
        }

    def _format_global_state(self, state: dict[str, Any]) -> str:
        visible_state = {
            "problem": state.get("problem", ""),
            "starter_code": state.get("starter_code", ""),
            "fn_name": state.get("fn_name", ""),
            "execution_instruction": state.get("execution_instruction", ""),
            "visible_tests": state.get("visible_tests", ""),
            "iterations": state.get("iterations", []),
        }
        return self._safe_json(visible_state, max_chars=int(self.code_cfg.get("max_state_chars", 12000)))

    def _build_prompt_context(
        self,
        *,
        problem: str,
        state: dict[str, Any],
        turn_id: int,
        current_pseudocode: str = "",
        current_code: str = "",
        current_error: str = "",
        current_pass_rate: float = 0.0,
        current_all_passed: int = 0,
    ) -> dict[str, Any]:
        return {
            "problem": problem,
            "original_problem": problem,
            "turn_id": int(turn_id),
            "turn_number": int(turn_id) + 1,
            "max_turns": int(self.max_turns),
            "global_state_text": self._format_global_state(state),
            "current_pseudocode": current_pseudocode,
            "current_code": current_code,
            "current_error": current_error,
            "current_pass_rate": float(current_pass_rate),
            "current_all_passed": int(current_all_passed),
            "starter_code": str(state.get("starter_code", "")),
            "fn_name": str(state.get("fn_name", "")),
            "execution_instruction": str(state.get("execution_instruction", "")),
            "coder_task_instruction": self._coder_task_instruction(str(state.get("fn_name", ""))),
            "visible_tests": str(state.get("visible_tests", "")),
            "code_format_example": self._format_example_code(str(state.get("fn_name", ""))),
        }

    async def _run_verifier_record(
        self,
        *,
        query_batch: DataProto,
        query_id: str,
        turn_id: int,
        step_id: int,
        problem: str,
        code: str,
        tests: Any,
        state_before: Any,
    ) -> WorkflowExecutionRecord:
        tool_name = str(self.verifier_cfg.get("tool", "code_verifier"))
        payload = {
            "problem": problem,
            "code": code,
            "tests": tests,
            "starter_code": self._extract_first(query_batch, self.starter_code_candidates, ""),
            "metadata": self._extract_first(
                query_batch,
                ["metadata", "extra_info.metadata", "extra_info", "extra.metadata", "extra"],
                {},
            ),
            "extra_info": self._extract_first(query_batch, ["extra_info", "extra"], {}),
        }
        start = time.perf_counter()
        output = await self._run_tool(
            tool_name,
            input_value=payload,
            top_k=1,
            max_attempts=1,
            fail_open=bool(self.verifier_cfg.get("fail_open", True)),
        )
        duration_s = float(time.perf_counter() - start)
        if not isinstance(output, dict):
            output = {
                "pass_rate": 0.0,
                "all_passed": 0,
                "passed": 0,
                "total": 0,
                "error": str(output),
                "error_code": -5,
                "error_message": "Verifier Tool Error",
                "failed_test_index": -1,
            }
        output["pass_rate"] = float(output.get("pass_rate", 0.0) or 0.0)
        output["all_passed"] = int(output.get("all_passed", 0) or 0)
        output["duration_s"] = float(output.get("duration_s", duration_s) or duration_s)
        return WorkflowExecutionRecord(
            query_id=query_id,
            node_id=f"verifier_{turn_id}",
            agent_id=tool_name,
            model_id="tool",
            turn_id=int(turn_id),
            step_id=int(step_id),
            node_type="tool",
            raw_output=json.dumps(output, ensure_ascii=False),
            parsed_output=output,
            thin=None,
            trainable=False,
            state_before=state_before,
            state_after=state_before,
            meta={"duration_s": duration_s},
        )

    def _build_debug_dump(
        self,
        query_local_idx: int,
        problem: str,
        records: list[WorkflowExecutionRecord],
        final_state: dict[str, Any],
        debug_max_chars: int | None = None,
    ) -> str:
        max_chars = self.debug_max_chars if debug_max_chars is None else int(debug_max_chars)

        def maybe_truncate(value: str) -> str:
            if max_chars <= 0:
                return value
            return value[:max_chars]

        def compact(value: Any) -> str:
            if isinstance(value, dict):
                return maybe_truncate(json.dumps(value, ensure_ascii=False, indent=2))
            return maybe_truncate(str(value))

        lines = [
            "[code-debug] ===== trace begin =====",
            f"[code-debug] query_idx={query_local_idx}",
            f"[code-debug] problem={maybe_truncate(problem)}",
            f"[code-debug] global_state_schema={json.dumps(self.GLOBAL_STATE_SCHEMA, ensure_ascii=False)}",
        ]
        for record in records:
            meta = record.meta if isinstance(record.meta, dict) else {}
            lines.append(
                "[code-debug] --- node ---\n"
                f"turn={record.turn_id} step={record.step_id} node={record.node_id} "
                f"agent={record.agent_id} model={record.model_id} type={record.node_type}\n"
                f"format_legal={meta.get('is_legal_format', 'n/a')} "
                f"format_reward={meta.get('format_reward', 'n/a')} "
                f"prompt_tokens={meta.get('prompt_tokens', 'n/a')} "
                f"output_tokens={meta.get('output_tokens', 'n/a')}\n"
                f"parsed_output:\n{compact(record.parsed_output)}\n"
                f"raw_output:\n{maybe_truncate(str(record.raw_output or ''))}"
            )
        lines.append(f"[code-debug] final_state={maybe_truncate(json.dumps(final_state, ensure_ascii=False))}")
        lines.append("[code-debug] ===== trace end =====")
        return "\n".join(lines)

    async def run_query(
        self,
        query_batch: DataProto,
        query_local_idx: int,
        debug: bool,
        debug_max_chars: int | None = None,
    ) -> WorkflowTrace:
        problem = self._extract_question(query_batch)
        tests = self._extract_first(query_batch, self.tests_candidates, "")
        starter_code = str(self._extract_first(query_batch, self.starter_code_candidates, "") or "")
        fn_name = self._extract_fn_name(query_batch, tests)
        query_id = self._extract_query_id(query_batch)
        tests_source = str(
            self._extract_first(query_batch, ["tests_source", "extra_info.tests_source"], "unknown") or "unknown"
        )
        visible_tests_summary = self._format_visible_tests(
            tests,
            max_examples=int(self.code_cfg.get("visible_tests_max_examples", 3)),
            max_chars=int(self.code_cfg.get("visible_tests_max_chars", 3000)),
        )
        try:
            tests_count = float(self._extract_first(query_batch, ["tests_count", "extra_info.tests_count"], 0) or 0)
            tests_count_raw = float(
                self._extract_first(query_batch, ["tests_count_raw", "extra_info.tests_count_raw"], tests_count) or 0
            )
        except Exception:
            tests_count = 0.0
            tests_count_raw = 0.0

        state: dict[str, Any] = self._new_global_state(
            problem,
            starter_code,
            fn_name=fn_name,
            visible_tests_summary=visible_tests_summary,
        )
        records: list[WorkflowExecutionRecord] = []
        step_id = 0
        stopped_by_all_passed = False

        for turn_id in range(self.max_turns):
            planner_ctx = self._build_prompt_context(problem=problem, state=state, turn_id=turn_id)
            planner_record = await self._execute_llm_step(
                query_batch=query_batch,
                node_id=f"planner_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                node_cfg=self.planner_cfg,
                prompt_context=planner_ctx,
                state_before=copy.deepcopy(state),
            )
            planner_record.query_id = query_id
            pseudocode, planner_legal = self._parse_tagged_text(planner_record.raw_output, "pseudocode")
            self._set_record_format_reward(planner_record, planner_legal)
            planner_record.parsed_output = {"pseudocode": pseudocode}
            planner_record.state_after = copy.deepcopy(state)
            records.append(planner_record)
            step_id += 1

            coder_ctx = self._build_prompt_context(
                problem=problem,
                state=state,
                turn_id=turn_id,
                current_pseudocode=pseudocode,
            )
            coder_record = await self._execute_llm_step(
                query_batch=query_batch,
                node_id=f"coder_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                node_cfg=self.coder_cfg,
                prompt_context=coder_ctx,
                state_before=copy.deepcopy(state),
            )
            coder_record.query_id = query_id
            code, coder_legal = self._parse_tagged_text(coder_record.raw_output, "code")
            self._set_record_format_reward(coder_record, coder_legal)
            coder_record.parsed_output = {"code": code}
            coder_record.state_after = copy.deepcopy(state)
            records.append(coder_record)
            step_id += 1

            verifier_record = await self._run_verifier_record(
                query_batch=query_batch,
                query_id=query_id,
                turn_id=turn_id,
                step_id=step_id,
                problem=problem,
                code=code,
                tests=tests,
                state_before=copy.deepcopy(state),
            )
            records.append(verifier_record)
            step_id += 1

            verification = verifier_record.parsed_output if isinstance(verifier_record.parsed_output, dict) else {}
            pass_rate = float(verification.get("pass_rate", 0.0) or 0.0)
            all_passed = int(verification.get("all_passed", 0) or 0)
            current_error = str(verification.get("error", "") or "")
            iteration = {
                "turn": int(turn_id),
                "pseudocode": pseudocode,
                "code": code,
                "verification": copy.deepcopy(verification),
            }

            if all_passed and self.stop_on_all_passed:
                state["iterations"].append(iteration)
                stopped_by_all_passed = True
                break

            if turn_id < self.max_turns - 1:
                state_with_current = copy.deepcopy(state)
                state_with_current["iterations"].append(copy.deepcopy(iteration))
                reflection_ctx = self._build_prompt_context(
                    problem=problem,
                    state=state_with_current,
                    turn_id=turn_id,
                    current_pseudocode=pseudocode,
                    current_code=code,
                    current_error=current_error,
                    current_pass_rate=pass_rate,
                    current_all_passed=all_passed,
                )
                reflection_record = await self._execute_llm_step(
                    query_batch=query_batch,
                    node_id=f"reflection_{turn_id}",
                    turn_id=turn_id,
                    step_id=step_id,
                    node_cfg=self.reflection_cfg,
                    prompt_context=reflection_ctx,
                    state_before=copy.deepcopy(state_with_current),
                )
                reflection_record.query_id = query_id
                reflection, reflection_legal = self._parse_tagged_text(reflection_record.raw_output, "reflection")
                self._set_record_format_reward(reflection_record, reflection_legal)
                reflection_record.parsed_output = {"reflection": reflection}
                iteration["reflection"] = reflection
                reflection_record.state_after = copy.deepcopy(state_with_current)
                reflection_record.state_after["iterations"][-1]["reflection"] = reflection
                records.append(reflection_record)
                step_id += 1

            state["iterations"].append(iteration)

        verifier_records = [rec for rec in records if rec.node_id.startswith("verifier_")]
        final_verification = verifier_records[-1].parsed_output if verifier_records else {}
        if not isinstance(final_verification, dict):
            final_verification = {}

        agent_call_counts: dict[str, int] = {}
        agent_legal_counts: dict[str, int] = {}
        agent_prompt_tokens: dict[str, list[float]] = {}
        agent_output_tokens: dict[str, list[float]] = {}
        verifier_pass_rates: list[float] = []
        verifier_all_passed: list[float] = []
        verifier_total_tests: list[float] = []
        verifier_total_available_tests: list[float] = []
        verifier_skipped_by_limit: list[float] = []
        verifier_timeouts: list[float] = []
        verifier_durations: list[float] = []
        verifier_error_codes: list[float] = []
        verifier_runner_counts: list[float] = []
        for record in records:
            if record.node_id.startswith("verifier_"):
                parsed = record.parsed_output if isinstance(record.parsed_output, dict) else {}
                verifier_pass_rates.append(float(parsed.get("pass_rate", 0.0) or 0.0))
                verifier_all_passed.append(float(parsed.get("all_passed", 0) or 0))
                verifier_total_tests.append(float(parsed.get("total", 0) or 0))
                verifier_total_available_tests.append(float(parsed.get("total_available", parsed.get("total", 0)) or 0))
                verifier_skipped_by_limit.append(float(parsed.get("skipped_by_limit", 0) or 0))
                verifier_timeouts.append(float(parsed.get("timeout_seconds", 0.0) or 0.0))
                verifier_durations.append(float(parsed.get("duration_s", record.meta.get("duration_s", 0.0)) or 0.0))
                verifier_error_codes.append(float(parsed.get("error_code", 0) or 0))
                verifier_runner_counts.append(float(parsed.get("runner_count", 0) or 0))
            if not record.agent_id or not record.trainable:
                continue
            agent_id = str(record.agent_id)
            agent_call_counts[agent_id] = agent_call_counts.get(agent_id, 0) + 1
            agent_legal_counts[agent_id] = agent_legal_counts.get(agent_id, 0) + int(
                bool(record.meta.get("is_legal_format", False))
            )
            agent_prompt_tokens.setdefault(agent_id, []).append(float(record.meta.get("prompt_tokens", 0.0) or 0.0))
            agent_output_tokens.setdefault(agent_id, []).append(float(record.meta.get("output_tokens", 0.0) or 0.0))

        metrics = {
            "workflow/code/used_turns": float(len(verifier_records)),
            "workflow/code/final_pass_rate": float(final_verification.get("pass_rate", 0.0) or 0.0),
            "workflow/code/final_all_passed": float(final_verification.get("all_passed", 0) or 0),
            "workflow/code/stopped_by_all_passed": float(1.0 if stopped_by_all_passed else 0.0),
            "workflow/code/global_state_iterations": float(len(state.get("iterations", []))),
            "workflow/code/global_state_chars": float(len(self._format_global_state(state))),
            "workflow/code/tests_count_raw": float(tests_count_raw),
            "workflow/code/tests_count": float(tests_count),
            f"workflow/code/tests_source/{self._safe_metric_name(tests_source)}": 1.0,
            "workflow/code/verifier/pass_rate_mean": (
                float(sum(verifier_pass_rates) / max(1, len(verifier_pass_rates))) if verifier_pass_rates else 0.0
            ),
            "workflow/code/verifier/all_passed_mean": (
                float(sum(verifier_all_passed) / max(1, len(verifier_all_passed))) if verifier_all_passed else 0.0
            ),
            "workflow/code/verifier/total_tests_mean": (
                float(sum(verifier_total_tests) / max(1, len(verifier_total_tests))) if verifier_total_tests else 0.0
            ),
            "workflow/code/verifier/total_available_tests_mean": (
                float(sum(verifier_total_available_tests) / max(1, len(verifier_total_available_tests)))
                if verifier_total_available_tests
                else 0.0
            ),
            "workflow/code/verifier/skipped_by_limit_mean": (
                float(sum(verifier_skipped_by_limit) / max(1, len(verifier_skipped_by_limit)))
                if verifier_skipped_by_limit
                else 0.0
            ),
            "workflow/code/verifier/timeout_seconds_mean": (
                float(sum(verifier_timeouts) / max(1, len(verifier_timeouts))) if verifier_timeouts else 0.0
            ),
            "workflow/code/verifier/duration_s_mean": (
                float(sum(verifier_durations) / max(1, len(verifier_durations))) if verifier_durations else 0.0
            ),
            "workflow/code/verifier/error_code_mean": (
                float(sum(verifier_error_codes) / max(1, len(verifier_error_codes))) if verifier_error_codes else 0.0
            ),
            "workflow/code/verifier/runner_count_mean": (
                float(sum(verifier_runner_counts) / max(1, len(verifier_runner_counts)))
                if verifier_runner_counts
                else 0.0
            ),
            "workflow/code/agent/planner_agent/call_count": float(agent_call_counts.get("planner_agent", 0)),
            "workflow/code/agent/coder_agent/call_count": float(agent_call_counts.get("coder_agent", 0)),
            "workflow/code/agent/reflection_agent/call_count": float(agent_call_counts.get("reflection_agent", 0)),
        }
        for agent_id, count in agent_call_counts.items():
            legal_count = agent_legal_counts.get(agent_id, 0)
            prompt_values = agent_prompt_tokens.get(agent_id, [])
            output_values = agent_output_tokens.get(agent_id, [])
            metrics[f"workflow/code/agent/{agent_id}/call_count"] = float(count)
            metrics[f"workflow/code/agent/{agent_id}/format_legal_rate"] = float(legal_count / max(1, count))
            metrics[f"workflow/code/agent/{agent_id}/prompt_tokens_mean"] = (
                float(sum(prompt_values) / max(1, len(prompt_values))) if prompt_values else 0.0
            )
            metrics[f"workflow/code/agent/{agent_id}/prompt_tokens_min"] = (
                float(min(prompt_values)) if prompt_values else 0.0
            )
            metrics[f"workflow/code/agent/{agent_id}/prompt_tokens_max"] = (
                float(max(prompt_values)) if prompt_values else 0.0
            )
            metrics[f"workflow/code/agent/{agent_id}/output_tokens_mean"] = (
                float(sum(output_values) / max(1, len(output_values))) if output_values else 0.0
            )
            metrics[f"workflow/code/agent/{agent_id}/output_tokens_min"] = (
                float(min(output_values)) if output_values else 0.0
            )
            metrics[f"workflow/code/agent/{agent_id}/output_tokens_max"] = (
                float(max(output_values)) if output_values else 0.0
            )
        for verifier_record in verifier_records:
            turn = int(verifier_record.turn_id)
            parsed = verifier_record.parsed_output if isinstance(verifier_record.parsed_output, dict) else {}
            metrics[f"workflow/code/turn_{turn}/pass_rate"] = float(parsed.get("pass_rate", 0.0) or 0.0)
            metrics[f"workflow/code/turn_{turn}/all_passed"] = float(parsed.get("all_passed", 0) or 0)

        trace = WorkflowTrace(
            query_id=query_id,
            question=problem,
            ground_truth=[],
            records=records,
            state={
                "global_state": state,
                "final_code": state.get("iterations", [{}])[-1].get("code", "") if state.get("iterations") else "",
                "final_verification": final_verification,
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
