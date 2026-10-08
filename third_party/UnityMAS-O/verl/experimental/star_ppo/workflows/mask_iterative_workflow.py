from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from verl import DataProto
from verl.experimental.star_ppo.tools.prompt_builders import MAskIterativeContextBuilder
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord, WorkflowTrace
from verl.experimental.star_ppo.workflows.trace_workflow import TraceWorkflowRunner


class MAskIterativeWorkflowRunner(TraceWorkflowRunner):
    """Iterative Planning/Search/Summary/Update/Answer workflow with turn-level rewards."""

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        self.mask_cfg = dict(self.workflow_cfg.get("mask", {}))
        self.max_turns = int(self.mask_cfg.get("max_turns", 5))
        self.stop_on_search_end = bool(self.mask_cfg.get("stop_on_search_end", True))
        self.plan_cfg = dict(self.mask_cfg.get("plan", {}))
        self.search_cfg = dict(self.mask_cfg.get("search", {}))
        self.summary_cfg = dict(self.mask_cfg.get("summary", {}))
        self.update_cfg = dict(self.mask_cfg.get("update", {}))
        self.answer_cfg = dict(self.mask_cfg.get("answer", {}))
        self.retriever_cfg = dict(self.mask_cfg.get("retriever", {}))
        self.context_builder = MAskIterativeContextBuilder()

    @staticmethod
    def _set_record_format_reward(record: WorkflowExecutionRecord, is_legal: bool) -> None:
        record.meta["format_reward"] = 0.0 if is_legal else -1.0
        record.meta["is_legal_format"] = bool(is_legal)

    @staticmethod
    def _normalize_knowledge_state(value: Any, question: str, fallback_answer: str = "Unknown") -> dict[str, Any]:
        if isinstance(value, dict) and "knowledge_state" in value and isinstance(value["knowledge_state"], dict):
            value = value["knowledge_state"]
        if not isinstance(value, dict):
            value = {}
        state = dict(value)
        state["question"] = str(state.get("question") or question).strip()
        trajectory = state.get("thinking_trajectory", [])
        if not isinstance(trajectory, list):
            trajectory = []
        normalized_traj = []
        for idx, item in enumerate(trajectory, start=1):
            if not isinstance(item, dict):
                continue
            normalized_traj.append(
                {
                    "step_id": str(item.get("step_id") or f"tau{idx}").strip(),
                    "sub_question": str(item.get("sub_question", item.get("question", ""))).strip(),
                    "sub_answer": str(item.get("sub_answer", item.get("answer", fallback_answer))).strip() or fallback_answer,
                }
            )
        state["thinking_trajectory"] = normalized_traj
        state["predicted_answer"] = str(state.get("predicted_answer") or fallback_answer).strip() or fallback_answer
        return state

    @classmethod
    def _parse_plan_output(cls, response_text: str, question: str) -> tuple[dict[str, Any], bool]:
        raw = str(response_text or "").strip()
        if not raw:
            state = {
                "question": question,
                "thinking_trajectory": [{"step_id": "tau1", "sub_question": question, "sub_answer": "unkown"}],
                "predicted_answer": "unkown",
            }
            return cls._normalize_knowledge_state(state, question=question, fallback_answer="unkown"), False

        q_matches = re.findall(r"<q(\d+)>(.*?)</q\1>", raw, re.DOTALL | re.IGNORECASE)
        a_matches = re.findall(r"<a(\d+)>(.*?)</a\1>", raw, re.DOTALL | re.IGNORECASE)
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", raw, re.DOTALL | re.IGNORECASE)

        q_dict = {int(idx): str(text).strip() for idx, text in q_matches}
        a_dict = {int(idx): str(text).strip() for idx, text in a_matches}
        indices = sorted(set(q_dict.keys()) | set(a_dict.keys()))

        thinking_trajectory = []
        for pos, idx in enumerate(indices, start=1):
            thinking_trajectory.append(
                {
                    "step_id": f"tau{pos}",
                    "sub_question": q_dict.get(idx, question if pos == 1 else ""),
                    "sub_answer": a_dict.get(idx, "unkown") or "unkown",
                }
            )
        if not thinking_trajectory:
            thinking_trajectory = [{"step_id": "tau1", "sub_question": question, "sub_answer": "unkown"}]

        final_answer = str(final_match.group(1)).strip() if final_match else ""
        if not final_answer:
            final_answer = str(thinking_trajectory[-1]["sub_answer"] or "unkown").strip() or "unkown"

        allowed_segments = re.findall(
            r"<q\d+>.*?</q\d+>|<a\d+>.*?</a\d+>|<final_answer>.*?</final_answer>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        reconstructed = "".join(allowed_segments)
        is_legal = bool(q_matches or a_matches) and bool(final_match)
        if re.sub(r"\s+", "", raw) != re.sub(r"\s+", "", reconstructed):
            is_legal = False
        if len(thinking_trajectory) > 5:
            is_legal = False
        questions = [str(item.get("sub_question", "")).strip() for item in thinking_trajectory]
        if len(set(questions)) < len(questions):
            is_legal = False
        expected_tags = []
        for idx in indices:
            expected_tags.extend([("q", str(idx)), ("a", str(idx))])
        actual_tags = re.findall(r"<\s*(q|a)(\d+)\s*>", raw, re.IGNORECASE)
        if actual_tags and expected_tags and actual_tags[: len(expected_tags)] != expected_tags:
            is_legal = False

        state = {
            "question": question,
            "thinking_trajectory": thinking_trajectory,
            "predicted_answer": final_answer or "unkown",
        }
        return cls._normalize_knowledge_state(state, question=question, fallback_answer="unkown"), is_legal

    @staticmethod
    def _parse_search_output(response_text: str) -> tuple[dict[str, Any], bool]:
        raw = str(response_text or "").strip()
        if not raw:
            return {"action": "end", "query": ""}, False

        if re.fullmatch(r"<end\s*/?>", raw, re.IGNORECASE):
            return {"action": "end", "query": ""}, True

        search_tags = re.findall(r"<search>(.*?)</search>", raw, re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<search>.*?</search>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<end\s*/?>", "", cleaned, flags=re.IGNORECASE)
        is_legal = len(search_tags) == 1 and cleaned.strip() == ""
        if search_tags:
            query = str(search_tags[0]).strip()
            if query:
                return {"action": "search", "query": query}, is_legal

        snippet = raw.split("\n")[0][:200]
        snippet = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", " ", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if snippet:
            return {"action": "search", "query": snippet}, False
        return {"action": "end", "query": ""}, False

    @staticmethod
    def _parse_summary_output(response_text: str, query: str) -> tuple[dict[str, Any], bool]:
        raw = str(response_text or "").strip()
        if not raw:
            return {"query": query, "summary": "No useful information", "salient_facts": []}, False

        if raw.lower() == "no useful information":
            return {"query": query, "summary": "No useful information", "salient_facts": []}, True

        evidence_tags = re.findall(r"<evidence>(.*?)</evidence>", raw, re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<evidence>.*?</evidence>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        is_legal = len(evidence_tags) == 1 and cleaned.strip() == ""
        if evidence_tags:
            summary = str(evidence_tags[0]).strip() or "No useful information"
            if len(summary.split()) > 75:
                is_legal = False
            return {"query": query, "summary": summary, "salient_facts": []}, is_legal

        snippet = raw.split("\n")[0][:200].strip() or "No useful information"
        return {"query": query, "summary": snippet, "salient_facts": []}, False

    def _apply_update_action(
        self,
        prev_state: dict[str, Any],
        query: str,
        evidence: str,
        action: str,
        target_step: str,
    ) -> dict[str, Any]:
        state = copy.deepcopy(prev_state)
        trajectory = list(state.get("thinking_trajectory", []))
        if action == "update":
            updated = False
            for item in trajectory:
                step_id = str(item.get("step_id", "")).strip().lower()
                step_num = step_id.replace("tau", "t") if step_id.startswith("tau") else step_id
                if step_id == target_step.lower().replace("t", "tau", 1) or step_num == target_step.lower():
                    item["sub_question"] = str(query).strip()
                    item["sub_answer"] = str(evidence).strip()
                    updated = True
                    break
            if not updated:
                trajectory.append(
                    {
                        "step_id": f"tau{len(trajectory) + 1}",
                        "sub_question": str(query).strip(),
                        "sub_answer": str(evidence).strip(),
                    }
                )
        else:
            trajectory.append(
                {
                    "step_id": f"tau{len(trajectory) + 1}",
                    "sub_question": str(query).strip(),
                    "sub_answer": str(evidence).strip(),
                }
            )
        state["thinking_trajectory"] = trajectory
        return self._normalize_knowledge_state(
            state,
            question=str(prev_state.get("question", "")),
            fallback_answer=str(prev_state.get("predicted_answer", "Unknown")),
        )

    def _parse_update_output(
        self,
        response_text: str,
        *,
        prev_state: dict[str, Any],
        query: str,
        evidence: str,
    ) -> tuple[dict[str, Any], bool]:
        raw = str(response_text or "").strip()
        trajectory = prev_state.get("thinking_trajectory", [])
        next_target = f"t{len(trajectory) + 1}"
        if not raw:
            updated_state = self._apply_update_action(prev_state, query, evidence, "add", next_target)
            return {"operation": "add", "target_step": next_target, "knowledge_state": updated_state}, False

        update_tags = re.findall(r"<update>(.*?)</update>", raw, re.DOTALL | re.IGNORECASE)
        add_tags = re.findall(r"<add>(.*?)</add>", raw, re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<update>.*?</update>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<add>.*?</add>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        is_legal = cleaned.strip() == "" and len(update_tags) + len(add_tags) == 1

        if update_tags:
            target = str(update_tags[0]).strip()
            valid_targets = {f"t{i + 1}" for i in range(len(trajectory))}
            if target not in valid_targets:
                is_legal = False
            updated_state = self._apply_update_action(prev_state, query, evidence, "update", target if target else next_target)
            return {"operation": "update", "target_step": target or next_target, "knowledge_state": updated_state}, is_legal

        if add_tags:
            target = str(add_tags[0]).strip()
            if target != next_target:
                is_legal = False
            updated_state = self._apply_update_action(prev_state, query, evidence, "add", next_target)
            return {"operation": "add", "target_step": next_target, "knowledge_state": updated_state}, is_legal

        updated_state = self._apply_update_action(prev_state, query, evidence, "add", next_target)
        return {"operation": "add", "target_step": next_target, "knowledge_state": updated_state}, False

    @staticmethod
    def _parse_answer_output(response_text: str, output_key: str) -> tuple[dict[str, Any], bool]:
        raw = str(response_text or "").strip()
        if not raw:
            return {output_key: "Unknown"}, False

        match = re.search(r"<final_answer>(.*?)</final_answer>", raw, re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<final_answer>.*?</final_answer>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            answer = str(match.group(1)).strip() or "Unknown"
            return {output_key: answer}, cleaned.strip() == ""

        first_line = raw.split("\n")[0].strip()
        return {output_key: first_line or "Unknown"}, False

    @staticmethod
    def _docs_to_text(output: Any) -> str:
        if isinstance(output, list):
            return "\n\n".join(str(item) for item in output)
        return str(output)

    def _build_prompt_context(self, question: str, nodes: dict[str, Any], current_state: dict[str, Any], turn_id: int, **extra) -> dict[str, Any]:
        context = self.context_builder(nodes)
        out = {
            "question": question,
            "turn_id": int(turn_id),
            "current_state": current_state,
            "knowledge_state_text": context.get("knowledge_state_text", ""),
            "thinking_trajectory_text": context.get("thinking_trajectory_text", ""),
            "search_history_text": context.get("search_history_text", ""),
            "search_history_queries_text": context.get("search_history_queries_text", ""),
            "latest_predicted_answer": context.get("latest_predicted_answer", "Unknown"),
        }
        out.update(extra)
        return out

    async def _run_tool_record(
        self,
        *,
        query_batch: DataProto,
        node_id: str,
        turn_id: int,
        step_id: int,
        tool_name: str,
        input_value: Any,
        top_k: int,
        max_attempts: int,
        fail_open: bool,
        state_before: Any,
        state_after: Any,
    ) -> WorkflowExecutionRecord:
        start = time.perf_counter()
        output = await self._run_tool(
            tool_name,
            input_value=input_value,
            top_k=top_k,
            max_attempts=max_attempts,
            fail_open=fail_open,
        )
        duration_s = float(time.perf_counter() - start)
        return WorkflowExecutionRecord(
            query_id=str(self._extract_from_batch(query_batch, "query_id") or ""),
            node_id=node_id,
            agent_id=tool_name,
            model_id="tool",
            turn_id=int(turn_id),
            step_id=int(step_id),
            node_type="tool",
            raw_output=self._docs_to_text(output),
            parsed_output=output,
            thin=None,
            trainable=False,
            state_before=state_before,
            state_after=state_after,
            meta={"duration_s": duration_s},
        )

    def _build_debug_dump(
        self,
        query_local_idx: int,
        question: str,
        records: list[WorkflowExecutionRecord],
        final_state: dict[str, Any],
        debug_max_chars: int | None = None,
    ) -> str:
        max_chars = self.debug_max_chars if debug_max_chars is None else int(debug_max_chars)

        def maybe_truncate(value: str) -> str:
            if max_chars <= 0:
                return value
            return value[:max_chars]

        lines = [
            "[mask-debug] ===== trace begin =====",
            f"[mask-debug] query_idx={query_local_idx}",
            f"[mask-debug] question={question}",
        ]
        for record in records:
            payload = record.parsed_output
            if isinstance(payload, dict):
                payload = json.dumps(payload, ensure_ascii=False)
            lines.append(
                f"[mask-debug] turn={record.turn_id} step={record.step_id} "
                f"node={record.node_id} agent={record.agent_id} out={maybe_truncate(str(payload))}"
            )
        lines.append(f"[mask-debug] final_state={maybe_truncate(json.dumps(final_state, ensure_ascii=False))}")
        lines.append("[mask-debug] ===== trace end =====")
        return "\n".join(lines)

    async def run_query(
        self,
        query_batch: DataProto,
        query_local_idx: int,
        debug: bool,
        debug_max_chars: int | None = None,
    ) -> WorkflowTrace:
        question = self._extract_question(query_batch)
        ground_truth = self._extract_gt_list(query_batch)
        query_id = str(self._extract_from_batch(query_batch, "query_id") or "")
        nodes: dict[str, Any] = {}
        records: list[WorkflowExecutionRecord] = []
        step_id = 0
        terminated_by_search_end = False

        plan_prompt_ctx = self._build_prompt_context(question, nodes, current_state={}, turn_id=0)
        plan_record = await self._execute_llm_step(
            query_batch=query_batch,
            node_id="plan",
            turn_id=0,
            step_id=step_id,
            node_cfg=self.plan_cfg,
            prompt_context=plan_prompt_ctx,
            state_before={},
        )
        plan_state, plan_legal = self._parse_plan_output(plan_record.raw_output, question=question)
        self._set_record_format_reward(plan_record, plan_legal)
        plan_record.parsed_output = plan_state
        plan_record.state_after = copy.deepcopy(plan_state)
        nodes["plan"] = {"knowledge_state": copy.deepcopy(plan_state)}
        records.append(plan_record)
        current_state = copy.deepcopy(plan_state)
        step_id += 1

        answer0_ctx = self._build_prompt_context(question, nodes, current_state=current_state, turn_id=0)
        answer0_record = await self._execute_llm_step(
            query_batch=query_batch,
            node_id="answer_0",
            turn_id=0,
            step_id=step_id,
            node_cfg=self.answer_cfg,
            prompt_context=answer0_ctx,
            state_before=copy.deepcopy(current_state),
            state_after=copy.deepcopy(current_state),
        )
        answer0_record.parsed_output, answer0_legal = self._parse_answer_output(
            answer0_record.raw_output, "predicted_answer"
        )
        self._set_record_format_reward(answer0_record, answer0_legal)
        current_state["predicted_answer"] = str(answer0_record.parsed_output.get("predicted_answer", "Unknown")).strip() or "Unknown"
        nodes["plan"]["knowledge_state"]["predicted_answer"] = current_state["predicted_answer"]
        answer0_record.state_after = copy.deepcopy(current_state)
        nodes["answer_0"] = copy.deepcopy(answer0_record.parsed_output)
        records.append(answer0_record)
        step_id += 1

        for turn_id in range(1, self.max_turns + 1):
            search_ctx = self._build_prompt_context(question, nodes, current_state=current_state, turn_id=turn_id)
            search_record = await self._execute_llm_step(
                query_batch=query_batch,
                node_id=f"search_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                node_cfg=self.search_cfg,
                prompt_context=search_ctx,
                state_before=copy.deepcopy(current_state),
                state_after=copy.deepcopy(current_state),
            )
            search_record.parsed_output, search_legal = self._parse_search_output(search_record.raw_output)
            self._set_record_format_reward(search_record, search_legal)
            nodes[f"search_{turn_id}"] = {"search_decision": copy.deepcopy(search_record.parsed_output)}
            records.append(search_record)
            step_id += 1

            if str(search_record.parsed_output.get("action", "")).lower() == "end" and self.stop_on_search_end:
                terminated_by_search_end = True
                break

            retrieve_record = await self._run_tool_record(
                query_batch=query_batch,
                node_id=f"retrieve_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                tool_name=str(self.retriever_cfg.get("tool", "retriever")),
                input_value=search_record.parsed_output.get("query", ""),
                top_k=int(self.retriever_cfg.get("top_k", 5)),
                max_attempts=int(self.retriever_cfg.get("max_attempts", 5)),
                fail_open=bool(self.retriever_cfg.get("fail_open", True)),
                state_before=copy.deepcopy(current_state),
                state_after=copy.deepcopy(current_state),
            )
            nodes[f"retrieve_{turn_id}"] = {"retrieved_docs": copy.deepcopy(retrieve_record.parsed_output)}
            records.append(retrieve_record)
            step_id += 1

            summary_ctx = self._build_prompt_context(
                question,
                nodes,
                current_state=current_state,
                turn_id=turn_id,
                search_query=search_record.parsed_output.get("query", ""),
                retrieved_docs=retrieve_record.parsed_output,
            )
            summary_record = await self._execute_llm_step(
                query_batch=query_batch,
                node_id=f"summary_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                node_cfg=self.summary_cfg,
                prompt_context=summary_ctx,
                state_before=copy.deepcopy(current_state),
                state_after=copy.deepcopy(current_state),
            )
            summary_record.parsed_output, summary_legal = self._parse_summary_output(
                summary_record.raw_output,
                query=str(search_record.parsed_output.get("query", "")),
            )
            self._set_record_format_reward(summary_record, summary_legal)
            nodes[f"summary_{turn_id}"] = {"evidence_summary": copy.deepcopy(summary_record.parsed_output)}
            records.append(summary_record)
            step_id += 1

            prev_state = copy.deepcopy(current_state)
            update_ctx = self._build_prompt_context(
                question,
                nodes,
                current_state=current_state,
                turn_id=turn_id,
                search_query=search_record.parsed_output.get("query", ""),
                evidence_summary=summary_record.parsed_output,
                next_step_target=f"t{len(prev_state.get('thinking_trajectory', [])) + 1}",
            )
            update_record = await self._execute_llm_step(
                query_batch=query_batch,
                node_id=f"update_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                node_cfg=self.update_cfg,
                prompt_context=update_ctx,
                state_before=copy.deepcopy(prev_state),
            )
            update_record.parsed_output, update_legal = self._parse_update_output(
                update_record.raw_output,
                prev_state=prev_state,
                query=str(search_record.parsed_output.get("query", "")),
                evidence=str(summary_record.parsed_output.get("summary", "No useful information")),
            )
            self._set_record_format_reward(update_record, update_legal)
            current_state = copy.deepcopy(update_record.parsed_output["knowledge_state"])
            update_record.state_after = copy.deepcopy(current_state)
            nodes[f"update_{turn_id}"] = {"update_result": copy.deepcopy(update_record.parsed_output)}
            records.append(update_record)
            step_id += 1

            answer_ctx = self._build_prompt_context(question, nodes, current_state=current_state, turn_id=turn_id)
            answer_record = await self._execute_llm_step(
                query_batch=query_batch,
                node_id=f"answer_{turn_id}",
                turn_id=turn_id,
                step_id=step_id,
                node_cfg=self.answer_cfg,
                prompt_context=answer_ctx,
                state_before=copy.deepcopy(current_state),
                state_after=copy.deepcopy(current_state),
            )
            answer_record.parsed_output, answer_legal = self._parse_answer_output(
                answer_record.raw_output, "predicted_answer"
            )
            self._set_record_format_reward(answer_record, answer_legal)
            current_state["predicted_answer"] = str(answer_record.parsed_output.get("predicted_answer", "Unknown")).strip() or "Unknown"
            nodes[f"update_{turn_id}"]["update_result"]["knowledge_state"]["predicted_answer"] = current_state["predicted_answer"]
            answer_record.state_after = copy.deepcopy(current_state)
            nodes[f"answer_{turn_id}"] = copy.deepcopy(answer_record.parsed_output)
            records.append(answer_record)
            step_id += 1

        predicted_answer_value = str(current_state.get("predicted_answer", "Unknown")).strip() or "Unknown"

        used_turns = len([rec for rec in records if rec.node_id.startswith("search_")])
        completed_turns = len(
            [
                rec
                for rec in records
                if rec.node_id.startswith("answer_") and rec.node_id not in {"answer_0"}
            ]
        )
        agent_call_counts: dict[str, int] = {}
        for record in records:
            if not record.agent_id or not record.trainable:
                continue
            agent_call_counts[record.agent_id] = agent_call_counts.get(record.agent_id, 0) + 1

        trace = WorkflowTrace(
            query_id=query_id,
            question=question,
            ground_truth=ground_truth,
            records=records,
            state={
                "nodes": nodes,
                "knowledge_state": current_state,
                "predicted_answer": predicted_answer_value,
            },
            metrics={
                "workflow/mask/used_turns": float(used_turns),
                "workflow/mask/query_elapsed_turns": float(used_turns),
                "workflow/mask/completed_turns": float(completed_turns),
                "workflow/mask/stopped_by_search_end": float(1.0 if terminated_by_search_end else 0.0),
                "workflow/mask/agent/planning_agent/call_count": float(agent_call_counts.get("planning_agent", 0)),
                "workflow/mask/agent/search_agent/call_count": float(agent_call_counts.get("search_agent", 0)),
                "workflow/mask/agent/summary_agent/call_count": float(agent_call_counts.get("summary_agent", 0)),
                "workflow/mask/agent/update_agent/call_count": float(agent_call_counts.get("update_agent", 0)),
                "workflow/mask/agent/answer_agent/call_count": float(agent_call_counts.get("answer_agent", 0)),
            },
        )
        if debug:
            trace.debug_dump = self._build_debug_dump(
                query_local_idx,
                question,
                records,
                current_state,
                debug_max_chars=debug_max_chars,
            )
        return trace
