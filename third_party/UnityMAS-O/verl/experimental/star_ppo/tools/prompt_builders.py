from __future__ import annotations

import re
from typing import Any


class SubQEvidenceContextBuilder:
    """Build dynamic sub-question/evidence context text from workflow node outputs."""

    @staticmethod
    def _to_clean_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text

    def __call__(self, value: Any) -> str:
        nodes = value if isinstance(value, dict) else {}
        decompose_node = nodes.get("decompose", {}) if isinstance(nodes.get("decompose", {}), dict) else {}
        sub_queries_raw = decompose_node.get("sub_queries", [])

        if isinstance(sub_queries_raw, (list, tuple)):
            sub_queries = [self._to_clean_text(x) for x in sub_queries_raw]
        elif isinstance(sub_queries_raw, str):
            cleaned = self._to_clean_text(sub_queries_raw)
            sub_queries = [cleaned] if cleaned else []
        else:
            sub_queries = []

        blocks: list[str] = []
        for idx, sub_q in enumerate(sub_queries, start=1):
            if not sub_q:
                continue
            evidence_item_node = nodes.get(f"evidence_{idx}", {})
            if isinstance(evidence_item_node, dict):
                sub_ev = self._to_clean_text(evidence_item_node.get("sub_evidence", ""))
            else:
                sub_ev = ""
            if not sub_ev:
                sub_ev = "Unknown"
            blocks.append(f"Sub-question {idx}:\n{sub_q}\nSub-evidence {idx}:\n{sub_ev}")

        if len(blocks) == 0:
            return "No valid sub-question/evidence is available."

        return "\n\n".join(blocks)


class MAskIterativeContextBuilder:
    """Build readable knowledge-state and search-history text for M-ASK-style prompts."""

    _UPDATE_NODE_RE = re.compile(r"^update_(\d+)$")
    _SEARCH_NODE_RE = re.compile(r"^search_(\d+)$")
    @staticmethod
    def _clean_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    @classmethod
    def _extract_state_from_node(cls, node_value: Any) -> dict[str, Any] | None:
        if not isinstance(node_value, dict):
            return None
        direct_state = node_value.get("knowledge_state")
        if isinstance(direct_state, dict):
            return direct_state
        update_result = node_value.get("update_result")
        if isinstance(update_result, dict):
            nested_state = update_result.get("knowledge_state")
            if isinstance(nested_state, dict):
                return nested_state
        return None

    @classmethod
    def _latest_knowledge_state(cls, nodes: dict[str, Any]) -> dict[str, Any]:
        latest_state = {}
        latest_turn = -1

        for plan_key in ("plan", "planning"):
            plan_state = cls._extract_state_from_node(nodes.get(plan_key))
            if isinstance(plan_state, dict):
                latest_state = plan_state
                latest_turn = 0
                break

        for node_id, node_value in nodes.items():
            match = cls._UPDATE_NODE_RE.match(str(node_id))
            if match is None:
                continue
            state = cls._extract_state_from_node(node_value)
            if not isinstance(state, dict):
                continue
            turn = int(match.group(1))
            if turn >= latest_turn:
                latest_state = state
                latest_turn = turn

        return latest_state

    @classmethod
    def _iter_trajectory_steps(cls, trajectory: Any) -> list[tuple[str, str, str]]:
        steps: list[tuple[str, str, str]] = []
        if isinstance(trajectory, list):
            for idx, item in enumerate(trajectory, start=1):
                if not isinstance(item, dict):
                    continue
                step_id = cls._clean_text(item.get("step_id"), default=f"tau{idx}")
                sub_question = cls._clean_text(item.get("sub_question", item.get("question", "")))
                sub_answer = cls._clean_text(item.get("sub_answer", item.get("answer", "")), default="Unknown")
                steps.append((step_id, sub_question, sub_answer))
            return steps

        if isinstance(trajectory, dict):
            keyed_items: list[tuple[int, str, Any]] = []
            for raw_key, raw_value in trajectory.items():
                order_match = re.search(r"(\d+)", str(raw_key))
                order = int(order_match.group(1)) if order_match else len(keyed_items) + 1
                keyed_items.append((order, str(raw_key), raw_value))
            keyed_items.sort(key=lambda item: item[0])
            for idx, raw_key, raw_value in keyed_items:
                if not isinstance(raw_value, dict):
                    continue
                step_id = cls._clean_text(raw_value.get("step_id"), default=raw_key or f"tau{idx}")
                sub_question = cls._clean_text(raw_value.get("sub_question", raw_value.get("question", "")))
                sub_answer = cls._clean_text(raw_value.get("sub_answer", raw_value.get("answer", "")), default="Unknown")
                steps.append((step_id, sub_question, sub_answer))
        return steps

    @classmethod
    def _format_knowledge_state(cls, state: dict[str, Any]) -> str:
        if not isinstance(state, dict) or len(state) == 0:
            return "Question: \nPredicted Answer: Unknown\nThinking Trajectory:\n(No trajectory yet)"

        question = cls._clean_text(state.get("question"))
        predicted_answer = cls._clean_text(state.get("predicted_answer"), default="Unknown")
        trajectory = cls._iter_trajectory_steps(state.get("thinking_trajectory", []))

        lines = [
            f"Question: {question}",
            f"Predicted Answer: {predicted_answer}",
            "Thinking Trajectory:",
        ]
        if not trajectory:
            lines.append("(No trajectory yet)")
            return "\n".join(lines)

        for step_id, sub_question, sub_answer in trajectory:
            lines.append(f"{step_id}:")
            lines.append(f"  Sub-question: {sub_question}")
            lines.append(f"  Sub-answer: {sub_answer}")
        return "\n".join(lines)

    @classmethod
    def _format_thinking_trajectory(cls, state: dict[str, Any]) -> str:
        if not isinstance(state, dict) or len(state) == 0:
            return "(No trajectory yet)"

        trajectory = cls._iter_trajectory_steps(state.get("thinking_trajectory", []))
        if not trajectory:
            return "(No trajectory yet)"

        lines = []
        for step_id, sub_question, sub_answer in trajectory:
            lines.append(f"{step_id}: Q: {sub_question} | A: {sub_answer}")
        return "\n".join(lines)

    @classmethod
    def _extract_summary_text(cls, node_value: Any) -> str:
        if not isinstance(node_value, dict):
            return ""
        evidence = node_value.get("evidence_summary")
        if isinstance(evidence, dict):
            summary = cls._clean_text(evidence.get("summary"))
            if summary:
                return summary
            salient_facts = evidence.get("salient_facts")
            if isinstance(salient_facts, list):
                joined = "; ".join(cls._clean_text(item) for item in salient_facts if cls._clean_text(item))
                if joined:
                    return joined
        return cls._clean_text(evidence)

    @classmethod
    def _format_search_history(cls, nodes: dict[str, Any]) -> str:
        history_rows: list[tuple[int, str, str, str]] = []
        for node_id, node_value in nodes.items():
            match = cls._SEARCH_NODE_RE.match(str(node_id))
            if match is None or not isinstance(node_value, dict):
                continue
            turn = int(match.group(1))
            decision = node_value.get("search_decision")
            if not isinstance(decision, dict):
                continue
            action = cls._clean_text(decision.get("action")).lower()
            query = cls._clean_text(decision.get("query"))
            if action != "search" or not query:
                continue
            summary_node = nodes.get(f"summarize_{turn}")
            summary_text = cls._extract_summary_text(summary_node)
            history_rows.append((turn, action, query, summary_text))

        if not history_rows:
            return "(No previous searches)"

        history_rows.sort(key=lambda item: item[0])
        lines = []
        for turn, _, query, summary_text in history_rows:
            lines.append(f"Turn {turn}: Query: {query}")
            if summary_text:
                lines.append(f"Turn {turn}: Summary: {summary_text}")
            else:
                lines.append(f"Turn {turn}: Summary: (No summary yet)")
        return "\n".join(lines)

    @classmethod
    def _format_search_history_queries(cls, nodes: dict[str, Any]) -> str:
        query_rows: list[tuple[int, str]] = []
        for node_id, node_value in nodes.items():
            match = cls._SEARCH_NODE_RE.match(str(node_id))
            if match is None or not isinstance(node_value, dict):
                continue
            turn = int(match.group(1))
            decision = node_value.get("search_decision")
            if not isinstance(decision, dict):
                continue
            if cls._clean_text(decision.get("action")).lower() != "search":
                continue
            query = cls._clean_text(decision.get("query"))
            if query:
                query_rows.append((turn, query))

        if not query_rows:
            return "(No previous searches)"

        query_rows.sort(key=lambda item: item[0])
        return "\n".join(f"{turn}. Query: {query}" for turn, query in query_rows)

    def __call__(self, value: Any) -> dict[str, str]:
        nodes = value if isinstance(value, dict) else {}
        latest_state = self._latest_knowledge_state(nodes)
        predicted_answer = self._clean_text(latest_state.get("predicted_answer"), default="Unknown")
        return {
            "knowledge_state_text": self._format_knowledge_state(latest_state),
            "thinking_trajectory_text": self._format_thinking_trajectory(latest_state),
            "search_history_text": self._format_search_history(nodes),
            "search_history_queries_text": self._format_search_history_queries(nodes),
            "latest_predicted_answer": predicted_answer,
        }
