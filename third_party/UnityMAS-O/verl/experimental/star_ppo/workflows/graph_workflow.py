from __future__ import annotations

import asyncio
import logging
import os
import re
import string
import time
from collections.abc import Mapping
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from verl import DataProto
from verl.experimental.star_ppo.tools import build_retriever_tool
from verl.experimental.star_ppo.workflows.base import WorkflowRunner
from verl.utils.import_utils import load_extern_object
from verl.utils.reward_score.search_r1_like_qa_em import em_check

logger = logging.getLogger(__name__)


class ToolNodeTimeoutError(TimeoutError):
    """Raised when a tool node exceeds configured timeout."""


class GraphWorkflowRunner(WorkflowRunner):
    """Configurable query-level workflow graph runner.

    Graph format (under `star.workflow.graph`):
    - `start_nodes`: list[str]
    - `end_nodes`: list[str]
    - `max_steps`: int
    - `nodes`: dict[node_id, node_cfg]
      - llm node:
        - `type: llm`
        - `model_id`, `agent_id`, `prompt_template`
        - optional `timing_group` (for reusable timing aggregation labels)
        - `parser`: {`type`: `json_key|raw`, `key`: str}
        - `output_key`: str
        - `reward`: {`format_weight`: float, `share_outcome`: bool}
        - `next`: list[str|{to: str, when: str}]
      - tool node:
        - `type: tool`
        - `tool`: tool alias in `star.workflow.tools`
        - optional `timing_group` (for reusable timing aggregation labels)
        - optional `fail_open`: bool (on tool error, return empty output instead of raising)
        - `input_template`: str
        - `top_k`: int (for retriever-like tools)
        - `output_key`: str
        - `next`: list[str|{to: str, when: str}]
    - `outcome_reward`: {`type`: `em`, `source`: "node_id.output_key", `weight`: float}
    """

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        self.workflow_cfg = self.config.star.get("workflow", {})
        self.graph_cfg = self.workflow_cfg.get("graph", {})
        self.nodes = dict(self.graph_cfg.get("nodes", {}))
        self.start_nodes = list(self.graph_cfg.get("start_nodes", []))
        self.end_nodes = set(self.graph_cfg.get("end_nodes", []))
        self.max_steps = int(self.graph_cfg.get("max_steps", 16))
        self.stop_on_end = bool(self.graph_cfg.get("stop_on_end", True))
        self.outcome_share_mode = str(self.workflow_cfg.get("outcome_share_mode", "full")).strip().lower()
        if self.outcome_share_mode not in {"full", "split"}:
            self.outcome_share_mode = "full"
        self.max_inflight_queries = int(self.workflow_cfg.get("max_inflight_queries", 32))
        self.llm_timeout_seconds = float(
            self.workflow_cfg.get("llm_timeout_seconds", os.environ.get("STAR_LLM_TIMEOUT_SECONDS", 0))
        )
        self.query_timeout_seconds = float(
            self.workflow_cfg.get("query_timeout_seconds", os.environ.get("STAR_QUERY_TIMEOUT_SECONDS", 0))
        )
        self.tool_timeout_seconds = float(
            self.workflow_cfg.get("tool_timeout_seconds", os.environ.get("STAR_TOOL_TIMEOUT_SECONDS", 0))
        )
        self.question_candidates = list(
            self.workflow_cfg.get("question_candidates", ["question", "query", "problem", "extra_info.question"])
        )
        self.gt_candidates = list(
            self.workflow_cfg.get(
                "ground_truth_candidates",
                [
                    "ground_truth",
                    "answer",
                    "target",
                    "golden_answers",
                    "extra_info.answer",
                    "reward_model.ground_truth",
                    "reward_model",
                ],
            )
        )
        self.outcome_cfg = self.graph_cfg.get("outcome_reward", {"type": "em", "source": "", "weight": 1.0})
        self.tools = self._build_tools()
        self._validate_graph()
        rollout_cfg = self.config.actor_rollout_ref.rollout
        prompt_len_cfg = int(rollout_cfg.get("prompt_length", 4096))
        trunc_margin = int(self.workflow_cfg.get("prompt_truncation_margin", 128))
        self.per_infer_prompt_max_tokens = max(256, prompt_len_cfg - trunc_margin)
        self.per_infer_prompt_max_tokens = int(
            self.workflow_cfg.get("per_infer_prompt_max_tokens", self.per_infer_prompt_max_tokens)
        )
        debug_cfg = dict(self.workflow_cfg.get("debug", {}))
        env_debug = str(os.environ.get("STAR_WORKFLOW_DEBUG", "")).strip().lower()
        self.debug_enabled = bool(debug_cfg.get("enabled", False)) or env_debug in {"1", "true", "yes", "on"}
        self.debug_sample_index = int(
            debug_cfg.get("sample_index", os.environ.get("STAR_WORKFLOW_DEBUG_SAMPLE_INDEX", 0))
        )
        self.debug_max_chars = int(debug_cfg.get("max_chars", os.environ.get("STAR_WORKFLOW_DEBUG_MAX_CHARS", 160)))
        self.debug_every_n_batches = max(
            1, int(debug_cfg.get("every_n_batches", os.environ.get("STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES", 20)))
        )
        self._debug_batch_counter = 0
        self._last_dropped_query_ids: list[str] = []

    def pop_dropped_query_ids(self) -> list[str]:
        dropped = self._last_dropped_query_ids
        self._last_dropped_query_ids = []
        return dropped

    def _build_tools(self) -> dict[str, Any]:
        tools = {}
        for alias, tool_cfg in dict(self.workflow_cfg.get("tools", {})).items():
            # Built-in retriever adapters.
            if str(tool_cfg.get("type", "")) in {"simple_keyword", "http", "query_api_pool", "retrieval_api_pool"}:
                tools[alias] = build_retriever_tool(tool_cfg)
                continue

            # External plugin tool object.
            if "path" in tool_cfg and "name" in tool_cfg:
                obj = load_extern_object(str(tool_cfg.get("path")), str(tool_cfg.get("name")))
                kwargs = dict(tool_cfg.get("kwargs", {}))
                tools[alias] = obj(**kwargs) if isinstance(obj, type) else obj
        return tools

    def _validate_graph(self):
        if len(self.start_nodes) == 0:
            raise ValueError("star.workflow.graph.start_nodes must be non-empty")
        for node_id in self.start_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"start node not found: {node_id}")
        for node_id in self.end_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"end node not found: {node_id}")

        for node_id, cfg in self.nodes.items():
            node_type = str(cfg.get("type", "llm"))
            if node_type not in {"llm", "tool"}:
                raise ValueError(f"node {node_id} has unsupported type={node_type}")
            if node_type == "llm":
                model_id = str(cfg.get("model_id", ""))
                if model_id not in self.trainer.model_ids:
                    raise ValueError(
                        f"node {node_id} uses unknown model_id={model_id}, available={self.trainer.model_ids}"
                    )
            if node_type == "tool":
                tool_alias = str(cfg.get("tool", ""))
                if tool_alias not in self.tools:
                    raise ValueError(
                        f"node {node_id} uses unknown tool alias={tool_alias}, "
                        f"available={sorted(self.tools.keys())}"
                    )
            for edge in self._normalize_edges(cfg.get("next", [])):
                to_node = edge["to"]
                if to_node not in self.nodes:
                    raise ValueError(f"node {node_id} has edge to missing node {to_node}")

    @staticmethod
    def _normalize_edges(edges_cfg) -> list[dict[str, Any]]:
        edges = []
        for item in list(edges_cfg):
            if isinstance(item, str):
                edges.append({"to": item, "when": None})
            # OmegaConf DictConfig is Mapping-like but not a plain dict.
            elif isinstance(item, Mapping) and "to" in item:
                edges.append({"to": str(item["to"]), "when": item.get("when", None)})
        return edges

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _clip_debug_text(self, value: Any) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.debug_max_chars:
            return text
        return text[: max(0, self.debug_max_chars - 3)] + "..."

    def _summarize_debug_item(self, value: Any, depth: int = 0) -> str:
        if value is None:
            return ""
        if isinstance(value, str | int | float | bool):
            return self._clip_debug_text(value)
        if isinstance(value, dict):
            if "text" in value and isinstance(value["text"], str | int | float | bool):
                extra_parts = []
                for k in ("title", "source", "url", "id", "score"):
                    if k in value and isinstance(value[k], str | int | float | bool):
                        extra_parts.append(f"{k}={self._clip_debug_text(value[k])}")
                    if len(extra_parts) >= 2:
                        break
                extra = (", " + ", ".join(extra_parts)) if extra_parts else ""
                return "{text=" + self._clip_debug_text(value["text"]) + extra + "}"
            keys = list(value.keys())
            preview = []
            for k in keys[:3]:
                v = value[k]
                if isinstance(v, str | int | float | bool):
                    preview.append(f"{k}={self._clip_debug_text(v)}")
            if preview:
                return "{" + ", ".join(preview) + "}"
            return f"dict(keys={keys[:5]})"
        if isinstance(value, list):
            if depth >= 1:
                return f"list(len={len(value)})"
            limit = min(len(value), 5)
            items = [self._summarize_debug_item(v, depth + 1) for v in value[:limit]]
            suffix = f", ...+{len(value) - limit}" if len(value) > limit else ""
            return f"list(len={len(value)}, items=[{', '.join(items)}{suffix}])"
        return self._clip_debug_text(value)

    def _summarize_debug_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            if len(value) == 0:
                return "list(len=0)"
            all_primitive = all(isinstance(v, str | int | float | bool) for v in value)
            limit = len(value) if all_primitive and len(value) <= 10 else min(len(value), 5)
            items = [self._summarize_debug_item(v) for v in value[:limit]]
            suffix = f", ...+{len(value) - limit}" if len(value) > limit else ""
            return f"list(len={len(value)}, items=[{', '.join(items)}{suffix}])"
        if isinstance(value, dict):
            return self._summarize_debug_item(value)
        return self._clip_debug_text(value)

    def _truncate_prompt_for_inference(self, prompt_text: str) -> tuple[str, int, int, int]:
        """Token-level truncation before each LLM node inference.

        Returns:
            tuple[str, int, int, int]:
                (possibly truncated prompt text, removed_token_count, before_tokens, after_tokens)
        """
        tokenizer = getattr(self.trainer, "tokenizer", None)
        max_tokens = int(self.per_infer_prompt_max_tokens)
        if tokenizer is None or max_tokens <= 0:
            text_len = len(str(prompt_text or ""))
            return prompt_text, 0, text_len, text_len

        chat_overhead_tokens = 0
        if hasattr(tokenizer, "apply_chat_template"):
            # Reserve chat-template overhead so the final model input respects max_tokens.
            empty_messages = [{"role": "user", "content": ""}]
            try:
                empty_ids = tokenizer.apply_chat_template(
                    empty_messages, tokenize=True, add_generation_prompt=True
                )
            except TypeError:
                try:
                    empty_ids = tokenizer.apply_chat_template(empty_messages, tokenize=True)
                except Exception:
                    empty_ids = None
            except Exception:
                empty_ids = None

            if isinstance(empty_ids, list):
                chat_overhead_tokens = len(empty_ids)
            elif hasattr(empty_ids, "tolist"):
                try:
                    chat_overhead_tokens = len(empty_ids.tolist())
                except Exception:
                    chat_overhead_tokens = 0

        try:
            token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(prompt_text)
        except Exception:
            text_len = len(str(prompt_text or ""))
            return prompt_text, 0, text_len, text_len

        if not isinstance(token_ids, list):
            text_len = len(str(prompt_text or ""))
            return prompt_text, 0, text_len, text_len

        content_total = len(token_ids)
        allowed_content_tokens = max(1, max_tokens - int(chat_overhead_tokens))
        if content_total <= allowed_content_tokens:
            total_with_overhead = content_total + int(chat_overhead_tokens)
            return prompt_text, 0, total_with_overhead, total_with_overhead

        # Direct truncation to the max allowed length.
        kept_ids = token_ids[:allowed_content_tokens]

        try:
            new_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
        except TypeError:
            new_text = tokenizer.decode(kept_ids)
        except Exception:
            total_with_overhead = content_total + int(chat_overhead_tokens)
            return prompt_text, 0, total_with_overhead, total_with_overhead

        before_tokens = content_total + int(chat_overhead_tokens)
        after_tokens = len(kept_ids) + int(chat_overhead_tokens)
        removed_tokens = max(0, content_total - len(kept_ids))
        return new_text, removed_tokens, before_tokens, after_tokens

    @staticmethod
    def _sanitize_metric_key(text: str) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"[^a-z0-9_.-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_.-")
        return s or "unknown"

    def _resolve_timing_group(self, node_id: str, node_type: str, node_cfg: Mapping[str, Any]) -> str:
        custom = str(node_cfg.get("timing_group", "") or "").strip()
        if custom:
            return self._sanitize_metric_key(custom)
        if node_type == "llm":
            agent_id = str(node_cfg.get("agent_id", "") or "").strip()
            if agent_id:
                return self._sanitize_metric_key(f"llm.{agent_id}")
            return "llm"
        if node_type == "tool":
            tool_name = str(node_cfg.get("tool", "") or "").strip()
            if tool_name:
                return self._sanitize_metric_key(f"tool.{tool_name}")
            return "tool"
        nid = str(node_id or "").strip()
        if "_" in nid:
            return self._sanitize_metric_key(f"node.{nid.split('_', 1)[0]}")
        return self._sanitize_metric_key(f"node.{nid}")

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self.trainer, "tokenizer", None)
        if tokenizer is None:
            return len(str(text or ""))
        try:
            token_ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(str(text or ""))
        except Exception:
            return len(str(text or ""))
        if isinstance(token_ids, list):
            return int(len(token_ids))
        return len(str(text or ""))

    @staticmethod
    def _as_template_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list | tuple):
            return "\n".join([GraphWorkflowRunner._as_template_value(x) for x in v])
        if isinstance(v, dict):
            if "text" in v:
                return str(v["text"])
            if "document" in v:
                return str(v["document"])
            if "content" in v:
                return str(v["content"])
            if "title" in v and "snippet" in v:
                return f"{v['title']}: {v['snippet']}"
            return str(v)
        return str(v)

    def _lookup_path(self, context: dict[str, Any], dotted: str, default: Any = "") -> Any:
        cur: Any = context
        for key in str(dotted).split("."):
            if isinstance(cur, dict):
                if key in cur:
                    cur = cur[key]
                    continue
                return default
            if isinstance(cur, list | tuple):
                if not str(key).isdigit():
                    return default
                idx = int(key)
                if idx < 0 or idx >= len(cur):
                    return default
                cur = cur[idx]
                continue
            if isinstance(cur, np.ndarray):
                if not str(key).isdigit():
                    return default
                idx = int(key)
                if idx < 0 or idx >= int(cur.shape[0]):
                    return default
                cur = cur[idx]
                continue
            return default
        return cur

    def _render_template(self, template: str, context: dict[str, Any]) -> str:
        # Respect escaped braces while supporting dotted-path placeholders.
        s = str(template).replace("{{", "\0L").replace("}}", "\0R")

        def repl(match):
            key = match.group(1).strip()
            value = self._lookup_path(context, key, default="")
            return self._as_template_value(value)

        s = re.sub(r"\{([a-zA-Z0-9_.]+)\}", repl, s)
        return s.replace("\0L", "{").replace("\0R", "}")

    @staticmethod
    def _dict_lookup(value: Any, dotted: str) -> Any:
        cur = value
        for key in str(dotted).split("."):
            if isinstance(cur, dict):
                if key in cur:
                    cur = cur[key]
                    continue
                return None
            if isinstance(cur, list | tuple):
                if not str(key).isdigit():
                    return None
                idx = int(key)
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
                continue
            if isinstance(cur, np.ndarray):
                if not str(key).isdigit():
                    return None
                idx = int(key)
                if idx < 0 or idx >= int(cur.shape[0]):
                    return None
                cur = cur[idx]
                continue
            return None
        return cur

    @staticmethod
    def _has_content(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return len(value.strip()) > 0
        if isinstance(value, np.ndarray):
            return int(value.size) > 0
        if isinstance(value, list | tuple | set):
            return any(GraphWorkflowRunner._has_content(x) for x in value)
        if isinstance(value, dict):
            return len(value) > 0
        return True

    def _extract_from_batch(self, query_batch: DataProto, key_or_path: str) -> Any:
        key_or_path = str(key_or_path)
        if key_or_path in query_batch.non_tensor_batch:
            vec = query_batch.non_tensor_batch[key_or_path]
            return vec[0] if len(vec) > 0 else None

        parts = key_or_path.split(".")
        if len(parts) <= 1:
            return None
        root = parts[0]
        if root not in query_batch.non_tensor_batch:
            return None
        root_vec = query_batch.non_tensor_batch[root]
        if len(root_vec) == 0:
            return None
        return self._dict_lookup(root_vec[0], ".".join(parts[1:]))

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, dict):
            for key in ("ground_truth", "answer", "target", "golden_answers"):
                if key in value:
                    return GraphWorkflowRunner._to_str_list(value[key])
            return [str(value)]
        if isinstance(value, np.ndarray):
            return [str(x).strip() for x in value.tolist() if str(x).strip()]
        if isinstance(value, list | tuple):
            return [str(x).strip() for x in value if str(x).strip()]
        v = str(value).strip()
        return [v] if v else []

    @staticmethod
    def _extract_question_from_messages(messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = []
                for x in content:
                    if isinstance(x, dict):
                        if x.get("type") == "text":
                            texts.append(str(x.get("text", "")))
                        elif "text" in x:
                            texts.append(str(x.get("text", "")))
                    else:
                        texts.append(str(x))
                merged = "".join(texts).strip()
                if merged:
                    return merged
        return ""

    def _eval_when(self, when_expr: str | None, context: dict[str, Any]) -> bool:
        if when_expr is None or str(when_expr).strip() == "":
            return True
        try:
            env = {
                "step": context.get("step", 0),
                "question": context.get("question", ""),
                "ground_truth": context.get("ground_truth", []),
                "nodes": context.get("nodes", {}),
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "any": any,
                "all": all,
                "min": min,
                "max": max,
            }
            return bool(eval(str(when_expr), {"__builtins__": {}}, env))
        except Exception:
            return False

    def _parse_llm_output(self, raw_text: str, node_cfg) -> tuple[Any, float]:
        parser = node_cfg.get("parser", {})
        parser_type = str(parser.get("type", "raw"))
        output_key = str(node_cfg.get("output_key", parser.get("key", "output")))
        if parser_type == "json_key":
            valid_reward = float(parser.get("valid_reward", 1.0))
            invalid_reward = float(parser.get("invalid_reward", 0.0))
            obj = self.trainer._extract_first_json_object(raw_text)
            if isinstance(obj, dict):
                key = str(parser.get("key", output_key))
                if key in obj:
                    value = obj.get(key)
                else:
                    value = parser.get("default", "")
                if isinstance(value, str):
                    value = value.strip()
                elif isinstance(value, list):
                    cleaned = []
                    for item in value:
                        if isinstance(item, str):
                            item = item.strip()
                            if not item:
                                continue
                        if item is None:
                            continue
                        cleaned.append(item)
                    value = cleaned
                    max_items = int(parser.get("max_items", 0))
                    if max_items > 0:
                        value = value[:max_items]
                return value, valid_reward if self._has_content(value) else invalid_reward
            return raw_text.strip(), invalid_reward
        if parser_type in {"tag", "tag_between"}:
            tag = str(parser.get("tag", "")).strip()
            open_tag = str(parser.get("open_tag", f"<{tag}>" if tag else "")).strip()
            close_tag = str(parser.get("close_tag", f"</{tag}>" if tag else "")).strip()
            valid_reward = float(parser.get("valid_reward", 0.0))
            invalid_reward = float(parser.get("invalid_reward", -1.0))
            if not open_tag or not close_tag:
                return raw_text.strip(), invalid_reward
            start_idx = raw_text.find(open_tag)
            if start_idx < 0:
                return raw_text.strip(), invalid_reward
            start_idx += len(open_tag)
            end_idx = raw_text.find(close_tag, start_idx)
            if end_idx < 0:
                return raw_text.strip(), invalid_reward
            value = raw_text[start_idx:end_idx].strip()
            if not value:
                return raw_text.strip(), invalid_reward
            return value, valid_reward
        value = raw_text.strip()
        return value, 1.0 if value else 0.0

    def _extract_question(self, query_batch: DataProto) -> str:
        for key in self.question_candidates:
            value = self._extract_from_batch(query_batch, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        prompt = self._extract_from_batch(query_batch, "prompt")
        parsed = self._extract_question_from_messages(prompt)
        if parsed:
            return parsed
        raw_prompt = self._extract_from_batch(query_batch, "raw_prompt")
        parsed = self._extract_question_from_messages(raw_prompt)
        if parsed:
            return parsed
        return ""

    def _extract_gt_list(self, query_batch: DataProto) -> list[str]:
        for key in self.gt_candidates:
            value = self._extract_from_batch(query_batch, key)
            gts = self._to_str_list(value)
            if gts:
                return gts
        return []

    async def _execute_node(self, node_id: str, query_batch: DataProto, context: dict[str, Any]) -> dict[str, Any]:
        node_cfg = self.nodes[node_id]
        node_type = str(node_cfg.get("type", "llm"))
        timing_group = self._resolve_timing_group(node_id, node_type, node_cfg)
        node_start = time.perf_counter()
        if node_type == "llm":
            model_id = str(node_cfg["model_id"])
            agent_id = str(node_cfg.get("agent_id", node_id))
            prompt_text = self._render_template(str(node_cfg.get("prompt_template", "{question}")), context)
            query_id = self._extract_from_batch(query_batch, "query_id")
            prompt_text, prompt_trimmed, prompt_before_tokens, prompt_after_tokens = self._truncate_prompt_for_inference(
                prompt_text
            )
            prompt_tokens = self._count_tokens(prompt_text)
            if prompt_trimmed > 0:
                msg = (
                    "[star-trunc] Prompt truncated before inference: "
                    f"node={node_id} query_id={query_id} "
                    f"before_tokens={int(prompt_before_tokens)} after_tokens={int(prompt_after_tokens)} "
                    f"removed_tokens={int(prompt_trimmed)} max_prompt_tokens={int(self.per_infer_prompt_max_tokens)}"
                )
                print(msg, flush=True)
                logger.warning(
                    "Prompt truncated before inference: node=%s query_id=%s before_tokens=%d after_tokens=%d removed_tokens=%d max_prompt_tokens=%d",
                    node_id,
                    query_id,
                    int(prompt_before_tokens),
                    int(prompt_after_tokens),
                    int(prompt_trimmed),
                    int(self.per_infer_prompt_max_tokens),
                )
            prompt_batch = self.trainer._build_workflow_prompt_batch(
                query_batch,
                [[{"role": "user", "content": prompt_text}]],
                agent_id,
            )
            rollout_timing_state: dict[str, Any] = {}
            rollout_coro = self.trainer._rollout_model_async_batched(
                model_id,
                prompt_batch,
                timing_state=rollout_timing_state,
            )
            if self.llm_timeout_seconds > 0:
                try:
                    _, thin, _, rollout_timing = await asyncio.wait_for(rollout_coro, timeout=self.llm_timeout_seconds)
                except asyncio.TimeoutError as e:
                    query_id = self._extract_from_batch(query_batch, "query_id")
                    queue_wait_s = float(rollout_timing_state.get("queue_wait_s", 0.0))
                    rollout_exec_s = float(rollout_timing_state.get("rollout_exec_s", 0.0))
                    rollout_total_s = float(rollout_timing_state.get("rollout_total_s", queue_wait_s + rollout_exec_s))
                    queue_acquired = bool(rollout_timing_state.get("queue_acquired", False))
                    raise TimeoutError(
                        f"LLM node timeout: node={node_id} model_id={model_id} "
                        f"query_id={query_id} timeout_s={self.llm_timeout_seconds} "
                        f"queue_acquired={queue_acquired} queue_wait_s={queue_wait_s:.3f} "
                        f"rollout_exec_s={rollout_exec_s:.3f} rollout_total_s={rollout_total_s:.3f}"
                    ) from e
            else:
                _, thin, _, rollout_timing = await rollout_coro
            action_text_vec = thin.non_tensor_batch.get("action_text", np.array([], dtype=object))
            raw_text = str(action_text_vec[0]) if len(action_text_vec) > 0 else ""
            output_tokens = self._count_tokens(raw_text)
            parsed_value, format_reward = self._parse_llm_output(raw_text, node_cfg)
            output_key = str(node_cfg.get("output_key", "output"))
            context["nodes"][node_id] = {
                "raw_text": raw_text,
                output_key: parsed_value,
                "format_reward": float(format_reward),
            }
            context[node_id] = context["nodes"][node_id]
            return {
                "node_id": node_id,
                "node_type": "llm",
                "agent_id": agent_id,
                "thin": thin,
                "format_reward": float(format_reward),
                "output_key": output_key,
                "output_value": parsed_value,
                "prompt_trimmed_tokens": int(prompt_trimmed),
                "prompt_tokens": int(prompt_tokens),
                "output_tokens": int(output_tokens),
                "queue_wait_s": float(rollout_timing.get("queue_wait_s", 0.0)),
                "rollout_exec_s": float(rollout_timing.get("rollout_exec_s", 0.0)),
                "rollout_total_s": float(rollout_timing.get("rollout_total_s", 0.0)),
                "timing_metrics": {
                    str(key): float(value)
                    for key, value in rollout_timing.items()
                    if isinstance(value, int | float | np.integer | np.floating)
                },
                "timing_group": timing_group,
                "duration_s": float(time.perf_counter() - node_start),
            }

        if node_type == "tool":
            tool_name = str(node_cfg["tool"])
            if tool_name not in self.tools:
                raise ValueError(f"tool node {node_id} references missing tool alias={tool_name}")
            tool = self.tools[tool_name]
            fail_open = bool(node_cfg.get("fail_open", False))
            input_source = str(node_cfg.get("input_source", "") or "").strip()
            batch_queries = bool(node_cfg.get("batch_queries", False))
            if input_source:
                input_value = self._lookup_path(context, input_source, default=[])
            else:
                input_value = self._render_template(str(node_cfg.get("input_template", "{question}")), context)
            top_k = int(node_cfg.get("top_k", 3))
            queries: list[str] = []
            async def _run_tool_call(func, *args, **kwargs):
                if self.tool_timeout_seconds > 0:
                    try:
                        return await asyncio.wait_for(
                            asyncio.to_thread(func, *args, **kwargs),
                            timeout=self.tool_timeout_seconds,
                        )
                    except asyncio.TimeoutError as exc:
                        raise ToolNodeTimeoutError(
                            f"tool node timeout: node={node_id} tool={tool_name} timeout_s={self.tool_timeout_seconds}"
                        ) from exc
                return await asyncio.to_thread(func, *args, **kwargs)
            try:
                if batch_queries:
                    if isinstance(input_value, np.ndarray):
                        raw_queries = input_value.tolist()
                    elif isinstance(input_value, list | tuple):
                        raw_queries = list(input_value)
                    else:
                        raw_queries = [input_value]
                    queries = [str(item).strip() for item in raw_queries if str(item).strip()]
                    max_attempts = int(node_cfg.get("max_attempts", 5))
                    if hasattr(tool, "retrieve_many"):
                        output = await _run_tool_call(tool.retrieve_many, queries, top_k)
                    elif hasattr(tool, "query_many"):
                        try:
                            output = await _run_tool_call(
                                tool.query_many,
                                questions=queries,
                                N=top_k,
                                max_attempts=max_attempts,
                            )
                        except TypeError:
                            output = await _run_tool_call(tool.query_many, queries, top_k, max_attempts)
                    else:
                        output = await _run_tool_call(
                            lambda: [tool.retrieve(query=q, top_k=top_k) for q in queries]
                        )
                elif hasattr(tool, "query"):
                    input_text = str(input_value)
                    max_attempts = int(node_cfg.get("max_attempts", 5))
                    # Prefer legacy named args used by RetrievalTool(question, N, max_attempts),
                    # then fallback to positional for custom tool implementations.
                    try:
                        output = await _run_tool_call(
                            tool.query,
                            question=input_text,
                            N=top_k,
                            max_attempts=max_attempts,
                        )
                    except TypeError:
                        output = await _run_tool_call(tool.query, input_text, top_k, max_attempts)
                elif hasattr(tool, "retrieve"):
                    input_text = str(input_value)
                    output = await _run_tool_call(tool.retrieve, input_text, top_k)
                elif callable(tool):
                    output = await _run_tool_call(tool, input_value)
                else:
                    raise TypeError(f"tool {tool_name} is not callable and has no query()/retrieve()")
            except Exception as exc:
                drop_on_timeout = bool(node_cfg.get("drop_on_timeout", True))
                if isinstance(exc, ToolNodeTimeoutError) and drop_on_timeout:
                    raise
                if not fail_open:
                    raise
                query_id = self._extract_from_batch(query_batch, "query_id")
                output = [[] for _ in queries] if batch_queries else []
                logger.warning(
                    "[star-tool-fail-open] tool failure ignored: node=%s tool=%s query_id=%s err=%s",
                    node_id,
                    tool_name,
                    query_id,
                    repr(exc),
                )
            output_key = str(node_cfg.get("output_key", "output"))
            context["nodes"][node_id] = {
                "input": input_value,
                output_key: output,
            }
            context[node_id] = context["nodes"][node_id]
            return {
                "node_id": node_id,
                "node_type": "tool",
                "output_key": output_key,
                "output_value": output,
                "timing_group": timing_group,
                "duration_s": float(time.perf_counter() - node_start),
            }

        raise ValueError(f"Unsupported node type for {node_id}: {node_type}")

    def _compute_outcome_reward(self, context: dict[str, Any]) -> float:
        outcome_type = str(self.outcome_cfg.get("type", "em")).lower()
        source = str(self.outcome_cfg.get("source", ""))
        weight = float(self.outcome_cfg.get("weight", 1.0))
        pred = str(self._lookup_path(context, source, default=""))
        gt = context.get("ground_truth", [])
        if len(gt) <= 0:
            return 0.0
        if outcome_type == "em":
            return weight * float(em_check(pred, gt))
        if outcome_type == "f1":
            return weight * float(max(self._f1_score(pred, str(ans)) for ans in gt))
        return 0.0

    @staticmethod
    def _normalize_text(text: str) -> str:
        s = str(text or "").lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @classmethod
    def _f1_score(cls, pred: str, gt: str) -> float:
        pred_tokens = cls._normalize_text(pred).split()
        gt_tokens = cls._normalize_text(gt).split()
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gt_tokens)
        overlap = sum(common.values())
        if overlap <= 0:
            return 0.0
        precision = overlap / float(len(pred_tokens))
        recall = overlap / float(len(gt_tokens))
        return 2.0 * precision * recall / (precision + recall)

    async def _run_one_query(
        self,
        query_batch: DataProto,
        query_sem: asyncio.Semaphore,
        query_local_idx: int,
        debug_query_idx: int | None,
        debug_batch_idx: int,
    ) -> dict[str, Any]:
        async with query_sem:
            query_start = time.perf_counter()
            debug_lines: list[str] = []
            debug_on = self.debug_enabled and debug_query_idx is not None and query_local_idx == debug_query_idx

            context = {
                "question": self._extract_question(query_batch),
                "ground_truth": self._extract_gt_list(query_batch),
                "nodes": {},
                "step": 0,
            }

            if debug_on:
                query_id = self._extract_from_batch(query_batch, "query_id")
                debug_lines.append(
                    f"[star-debug] batch={debug_batch_idx} query_idx={query_local_idx} query_id={query_id}"
                )
                debug_lines.append(f"[star-debug] question={self._clip_debug_text(context['question'])}")
                debug_lines.append(
                    f"[star-debug] ground_truth={self._summarize_debug_value(context['ground_truth'])}"
                )

            frontier = list(self.start_nodes)
            llm_exec_records: list[dict[str, Any]] = []
            node_timing_records: list[dict[str, Any]] = []

            for step in range(self.max_steps):
                if len(frontier) == 0:
                    break
                context["step"] = step
                node_tasks = [self._execute_node(node_id, query_batch, context) for node_id in frontier]
                node_results = await asyncio.gather(*node_tasks)

                next_frontier: list[str] = []
                hit_end = False
                for result in node_results:
                    node_id = result["node_id"]
                    if debug_on:
                        if result["node_type"] == "llm":
                            trim_note = ""
                            trimmed = int(result.get("prompt_trimmed_tokens", 0))
                            if trimmed > 0:
                                trim_note = f" trimmed={trimmed}tok"
                            debug_lines.append(
                                f"[star-debug] step={step} node={node_id} "
                                f"out={result['output_key']}:{self._summarize_debug_value(result['output_value'])} "
                                f"format={float(result['format_reward']):.2f}{trim_note}"
                            )
                        else:
                            debug_lines.append(
                                f"[star-debug] step={step} node={node_id} "
                                f"out={result['output_key']}:{self._summarize_debug_value(result['output_value'])}"
                            )
                    if result["node_type"] == "llm":
                        llm_exec_records.append(result)
                    node_timing_records.append(
                        {
                            "node_id": str(node_id),
                            "node_type": str(result.get("node_type", "unknown")),
                            "timing_group": str(result.get("timing_group", "")),
                            "duration_s": float(result.get("duration_s", 0.0)),
                            "queue_wait_s": float(result.get("queue_wait_s", 0.0)),
                            "rollout_exec_s": float(result.get("rollout_exec_s", 0.0)),
                            "rollout_total_s": float(result.get("rollout_total_s", 0.0)),
                            **{
                                str(key): float(value)
                                for key, value in dict(result.get("timing_metrics", {})).items()
                                if isinstance(value, int | float | np.integer | np.floating)
                            },
                        }
                    )
                    if node_id in self.end_nodes:
                        hit_end = True
                    for edge in self._normalize_edges(self.nodes[node_id].get("next", [])):
                        if self._eval_when(edge["when"], context):
                            next_frontier.append(edge["to"])

                if self.stop_on_end and hit_end:
                    frontier = []
                else:
                    frontier = self._dedupe_keep_order(next_frontier)

            outcome_reward = self._compute_outcome_reward(context)
            debug_dump = None
            if debug_on:
                outcome_source = str(self.outcome_cfg.get("source", "") or "")
                if outcome_source:
                    pred = self._lookup_path(context, outcome_source, default="")
                    debug_lines.append(
                        f"[star-debug] final={outcome_source}:{self._summarize_debug_value(pred)}"
                    )
                debug_lines.append(
                    f"[star-debug] outcome_reward={float(outcome_reward):.4f} llm_nodes={len(llm_exec_records)}"
                )
                debug_dump = "\n".join(
                    [
                        "[star-debug] ===== trace begin =====",
                        *debug_lines,
                        "[star-debug] ===== trace end =====",
                    ]
                )

            reward_parts = []
            node_format = {}
            shared_outcome_nodes = []
            for rec in llm_exec_records:
                node_id = rec["node_id"]
                reward_cfg = dict(self.nodes[node_id].get("reward", {}))
                if bool(reward_cfg.get("share_outcome", True)):
                    shared_outcome_nodes.append(node_id)
            denom = max(1, len(shared_outcome_nodes))

            for rec in llm_exec_records:
                node_id = rec["node_id"]
                reward_cfg = dict(self.nodes[node_id].get("reward", {}))
                format_weight = float(reward_cfg.get("format_weight", 0.0))
                share_outcome = bool(reward_cfg.get("share_outcome", True))
                if share_outcome and self.outcome_share_mode == "split":
                    shared_outcome = float(outcome_reward) / float(denom)
                else:
                    shared_outcome = float(outcome_reward) if share_outcome else 0.0
                total = format_weight * float(rec["format_reward"]) + shared_outcome
                thin_len = len(rec["thin"])
                if thin_len > 0:
                    reward_parts.append(
                        self.trainer._build_commit_rewards_from_thin(
                            rec["thin"], np.full((thin_len,), total, dtype=np.float32)
                        )
                    )
                node_format[node_id] = float(rec["format_reward"])

            return {
                "reward_parts": reward_parts,
                "outcome_reward": float(outcome_reward),
                "node_format": node_format,
                "llm_node_count": float(len(llm_exec_records)),
                "debug_dump": debug_dump,
                "llm_length_records": [
                    {
                        "node_id": rec["node_id"],
                        "agent_id": str(rec.get("agent_id", rec["node_id"])),
                        "prompt_tokens": int(rec.get("prompt_tokens", 0)),
                        "output_tokens": int(rec.get("output_tokens", 0)),
                    }
                    for rec in llm_exec_records
                ],
                "node_timing_records": node_timing_records,
                "query_elapsed_s": float(time.perf_counter() - query_start),
            }

    async def _run_one_query_safe(
        self,
        query_batch: DataProto,
        query_sem: asyncio.Semaphore,
        query_local_idx: int,
        debug_query_idx: int | None,
        debug_batch_idx: int,
    ) -> dict[str, Any]:
        query_id = str(self._extract_from_batch(query_batch, "query_id") or "")
        query_start = time.perf_counter()
        try:
            if self.query_timeout_seconds > 0:
                return await asyncio.wait_for(
                    self._run_one_query(
                        query_batch,
                        query_sem,
                        query_local_idx=query_local_idx,
                        debug_query_idx=debug_query_idx,
                        debug_batch_idx=debug_batch_idx,
                    ),
                    timeout=self.query_timeout_seconds,
                )
            return await self._run_one_query(
                query_batch,
                query_sem,
                query_local_idx=query_local_idx,
                debug_query_idx=debug_query_idx,
                debug_batch_idx=debug_batch_idx,
            )
        except asyncio.TimeoutError as exc:
            if isinstance(exc, ToolNodeTimeoutError):
                logger.warning(
                    "[star-query-drop] node timeout dropped: query_id=%s idx=%s err=%r",
                    query_id,
                    query_local_idx,
                    exc,
                )
                return {
                    "reward_parts": [],
                    "outcome_reward": 0.0,
                    "node_format": {},
                    "llm_node_count": 0.0,
                    "debug_dump": None,
                    "llm_length_records": [],
                    "node_timing_records": [],
                    "query_elapsed_s": float(time.perf_counter() - query_start),
                    "dropped": True,
                    "drop_reason": "node_timeout",
                    "drop_error": str(exc),
                    "drop_query_id": query_id,
                }
            logger.warning(
                "[star-query-drop] query timeout dropped: query_id=%s idx=%s timeout_s=%.1f",
                query_id,
                query_local_idx,
                float(self.query_timeout_seconds),
            )
            return {
                "reward_parts": [],
                "outcome_reward": 0.0,
                "node_format": {},
                "llm_node_count": 0.0,
                "debug_dump": None,
                "llm_length_records": [],
                "node_timing_records": [],
                "query_elapsed_s": float(time.perf_counter() - query_start),
                "dropped": True,
                "drop_reason": "query_timeout",
                "drop_error": str(exc),
                "drop_query_id": query_id,
            }
        except Exception as exc:
            logger.warning(
                "[star-query-drop] query error dropped: query_id=%s idx=%s err=%r",
                query_id,
                query_local_idx,
                exc,
            )
            return {
                "reward_parts": [],
                "outcome_reward": 0.0,
                "node_format": {},
                "llm_node_count": 0.0,
                "debug_dump": None,
                "llm_length_records": [],
                "node_timing_records": [],
                "query_elapsed_s": float(time.perf_counter() - query_start),
                "dropped": True,
                "drop_reason": "query_error",
                "drop_error": f"{type(exc).__name__}: {exc}",
                "drop_query_id": query_id,
            }

    async def run_batch(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        del epoch
        batch_start = time.perf_counter()
        self._debug_batch_counter += 1
        debug_this_batch = self.debug_enabled and (
            self._debug_batch_counter % max(1, self.debug_every_n_batches) == 0
        )
        debug_query_idx = None
        if debug_this_batch and len(batch) > 0:
            debug_query_idx = int(self.debug_sample_index) % len(batch)

        query_sem = asyncio.Semaphore(max(1, self.max_inflight_queries))
        tasks = [
            self._run_one_query_safe(
                batch.select_idxs([i]),
                query_sem,
                query_local_idx=i,
                debug_query_idx=debug_query_idx,
                debug_batch_idx=self._debug_batch_counter,
            )
            for i in range(len(batch))
        ]
        query_results = await asyncio.gather(*tasks)

        if debug_this_batch:
            for item in query_results:
                dump = item.get("debug_dump", None)
                if dump:
                    print(dump, flush=True)
                    break

        reward_parts = []
        outcome_rewards = []
        llm_node_counts = []
        dropped_query_ids: list[str] = []
        drop_reason_acc = defaultdict(int)
        node_format_acc = defaultdict(list)
        node_prompt_len_acc = defaultdict(list)
        node_output_len_acc = defaultdict(list)
        agent_prompt_len_acc = defaultdict(list)
        agent_output_len_acc = defaultdict(list)
        query_elapsed_acc: list[float] = []
        node_timing_by_id = defaultdict(list)
        node_timing_by_type = defaultdict(list)
        node_timing_by_group = defaultdict(list)
        node_timing_all: list[float] = []
        llm_timing_fields = (
            "queue_wait_s",
            "rollout_exec_s",
            "rollout_total_s",
            "rpc_roundtrip_s",
            "rpc_overhead_s",
            "worker_total_s",
            "worker_generate_call_s",
            "worker_generate_overhead_s",
            "worker_thin_build_s",
            "worker_decode_action_text_s",
            "worker_buffer_put_s",
            "worker_build_overhead_s",
            "engine_generate_s",
            "engine_generate_max_s",
            "agent_loop_tool_calls_s",
            "agent_loop_tool_calls_max_s",
            "agent_server_rpc_roundtrip_s",
            "agent_server_rpc_roundtrip_max_s",
            "agent_server_total_s",
            "agent_server_total_max_s",
            "agent_server_rpc_overhead_s",
            "agent_server_rpc_overhead_max_s",
            "agent_server_first_token_s",
            "agent_server_first_token_max_s",
            "agent_server_decode_tail_s",
            "agent_server_decode_tail_max_s",
            "agent_worker_start_lag_s",
            "agent_worker_start_lag_max_s",
            "agent_worker_prep_s",
            "agent_worker_prep_max_s",
            "agent_worker_run_loops_s",
            "agent_worker_run_loops_max_s",
            "agent_worker_postprocess_s",
            "agent_worker_postprocess_max_s",
            "agent_worker_total_s",
            "agent_worker_total_max_s",
            "agent_worker_non_loop_overhead_s",
            "agent_worker_non_loop_overhead_max_s",
            "agent_loop_manager_prep_s",
            "agent_loop_manager_worker_rpc_wait_s",
            "agent_loop_manager_worker_rpc_mean_s",
            "agent_loop_manager_worker_rpc_max_s",
            "agent_loop_manager_concat_s",
            "agent_loop_manager_metrics_reduce_s",
            "agent_loop_manager_total_s",
            "agent_loop_manager_overhead_s",
            "microbatch_enabled",
            "microbatch_size",
            "microbatch_request_count",
            "microbatch_wait_s",
            "microbatch_wait_max_s",
            "microbatch_batch_exec_s",
            "microbatch_flush_timeout",
            "microbatch_flush_size",
            "microbatch_saved_calls",
            "microbatch_saved_call_ratio",
        )
        llm_timing_all = {field: [] for field in llm_timing_fields}
        llm_timing_by_id = {field: defaultdict(list) for field in llm_timing_fields}
        llm_timing_by_group = {field: defaultdict(list) for field in llm_timing_fields}
        for item in query_results:
            if bool(item.get("dropped", False)):
                query_id = str(item.get("drop_query_id", "") or "")
                if query_id:
                    dropped_query_ids.append(query_id)
                reason = str(item.get("drop_reason", "unknown") or "unknown")
                drop_reason_acc[reason] += 1
                query_elapsed_acc.append(float(item.get("query_elapsed_s", 0.0)))
                continue
            reward_parts.extend(item["reward_parts"])
            outcome_rewards.append(item["outcome_reward"])
            llm_node_counts.append(item["llm_node_count"])
            for node_id, val in item["node_format"].items():
                node_format_acc[node_id].append(float(val))
            for rec in item.get("llm_length_records", []):
                node_id = str(rec.get("node_id", ""))
                agent_id = str(rec.get("agent_id", ""))
                prompt_tokens = int(rec.get("prompt_tokens", 0))
                output_tokens = int(rec.get("output_tokens", 0))
                if node_id:
                    node_prompt_len_acc[node_id].append(float(prompt_tokens))
                    node_output_len_acc[node_id].append(float(output_tokens))
                if agent_id:
                    agent_prompt_len_acc[agent_id].append(float(prompt_tokens))
                    agent_output_len_acc[agent_id].append(float(output_tokens))
            query_elapsed_acc.append(float(item.get("query_elapsed_s", 0.0)))
            for timing_rec in item.get("node_timing_records", []):
                node_id = str(timing_rec.get("node_id", ""))
                node_type = str(timing_rec.get("node_type", "unknown"))
                timing_group = self._sanitize_metric_key(str(timing_rec.get("timing_group", "")))
                duration_s = float(timing_rec.get("duration_s", 0.0))
                if duration_s <= 0:
                    continue
                node_timing_all.append(duration_s)
                if node_id:
                    node_timing_by_id[node_id].append(duration_s)
                if timing_group:
                    node_timing_by_group[timing_group].append(duration_s)
                node_timing_by_type[node_type].append(duration_s)
                if node_type == "llm":
                    for field in llm_timing_fields:
                        raw_value = timing_rec.get(field, None)
                        if not isinstance(raw_value, int | float | np.integer | np.floating):
                            continue
                        value = float(raw_value)
                        llm_timing_all[field].append(value)
                        if node_id:
                            llm_timing_by_id[field][node_id].append(value)
                        if timing_group:
                            llm_timing_by_group[field][timing_group].append(value)

        if len(reward_parts) == 0:
            rewards = self.trainer._empty_rewards()
        else:
            rewards = self.trainer._concat_data_proto_safe(reward_parts)

        metrics = {
            "workflow/samples": float(len(query_results)),
            "workflow/query_dropped": float(len(dropped_query_ids)),
            "workflow/query_drop_ratio": float(len(dropped_query_ids) / max(1, len(query_results))),
            "workflow/outcome_reward_mean": float(np.mean(outcome_rewards)) if outcome_rewards else 0.0,
            "workflow/llm_nodes_per_query_mean": float(np.mean(llm_node_counts)) if llm_node_counts else 0.0,
        }
        for reason, count in drop_reason_acc.items():
            safe_reason = self._sanitize_metric_key(reason)
            metrics[f"workflow/query_drop/{safe_reason}"] = float(count)
        for node_id, values in node_format_acc.items():
            metrics[f"workflow/node/{node_id}/format_reward_mean"] = float(np.mean(values)) if values else 0.0
        for node_id, values in node_prompt_len_acc.items():
            if values:
                metrics[f"workflow/node/{node_id}/prompt_tokens_min"] = float(np.min(values))
                metrics[f"workflow/node/{node_id}/prompt_tokens_max"] = float(np.max(values))
                metrics[f"workflow/node/{node_id}/prompt_tokens_mean"] = float(np.mean(values))
        for node_id, values in node_output_len_acc.items():
            if values:
                metrics[f"workflow/node/{node_id}/output_tokens_min"] = float(np.min(values))
                metrics[f"workflow/node/{node_id}/output_tokens_max"] = float(np.max(values))
                metrics[f"workflow/node/{node_id}/output_tokens_mean"] = float(np.mean(values))
        for agent_id, values in agent_prompt_len_acc.items():
            if values:
                metrics[f"workflow/agent/{agent_id}/prompt_tokens_min"] = float(np.min(values))
                metrics[f"workflow/agent/{agent_id}/prompt_tokens_max"] = float(np.max(values))
                metrics[f"workflow/agent/{agent_id}/prompt_tokens_mean"] = float(np.mean(values))
        for agent_id, values in agent_output_len_acc.items():
            if values:
                metrics[f"workflow/agent/{agent_id}/output_tokens_min"] = float(np.min(values))
                metrics[f"workflow/agent/{agent_id}/output_tokens_max"] = float(np.max(values))
                metrics[f"workflow/agent/{agent_id}/output_tokens_mean"] = float(np.mean(values))
        if query_elapsed_acc:
            metrics["workflow/timing/query_s_mean"] = float(np.mean(query_elapsed_acc))
            metrics["workflow/timing/query_s_max"] = float(np.max(query_elapsed_acc))
        if node_timing_all:
            metrics["workflow/timing/node_s_mean"] = float(np.mean(node_timing_all))
            metrics["workflow/timing/node_s_max"] = float(np.max(node_timing_all))
            metrics["workflow/timing/node_invocations"] = float(len(node_timing_all))
        if node_timing_by_type.get("llm"):
            metrics["workflow/timing/llm_node_s_mean"] = float(np.mean(node_timing_by_type["llm"]))
        for field, values in llm_timing_all.items():
            if values:
                metrics[f"workflow/timing/llm_{field}_mean"] = float(np.mean(values))
                metrics[f"workflow/timing/llm_{field}_max"] = float(np.max(values))
        if node_timing_by_type.get("tool"):
            metrics["workflow/timing/tool_node_s_mean"] = float(np.mean(node_timing_by_type["tool"]))
        for node_id, values in node_timing_by_id.items():
            if values:
                metrics[f"workflow/timing/node/{node_id}_s_mean"] = float(np.mean(values))
        for field, per_node in llm_timing_by_id.items():
            for node_id, values in per_node.items():
                if values:
                    metrics[f"workflow/timing/node/{node_id}_{field}_mean"] = float(np.mean(values))
                    metrics[f"workflow/timing/node/{node_id}_{field}_max"] = float(np.max(values))
        for group, values in node_timing_by_group.items():
            if values:
                metrics[f"workflow/timing/group/{group}_s_mean"] = float(np.mean(values))
                metrics[f"workflow/timing/group/{group}_s_max"] = float(np.max(values))
                metrics[f"workflow/timing/group/{group}_count"] = float(len(values))
        for field, per_group in llm_timing_by_group.items():
            for group, values in per_group.items():
                if values:
                    metrics[f"workflow/timing/group/{group}_{field}_mean"] = float(np.mean(values))
                    metrics[f"workflow/timing/group/{group}_{field}_max"] = float(np.max(values))
        metrics["workflow/timing/batch_total_s"] = float(time.perf_counter() - batch_start)
        self._last_dropped_query_ids = dropped_query_ids
        return rewards, metrics
