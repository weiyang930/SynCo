from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import time
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from verl import DataProto
from verl.experimental.star_ppo.tools import build_retriever_tool
from verl.experimental.star_ppo.workflows.base import WorkflowRunner
from verl.experimental.star_ppo.workflows.schema import RewardAssignment, WorkflowExecutionRecord, WorkflowTrace
from verl.utils.import_utils import load_extern_object

logger = logging.getLogger(__name__)


class TraceWorkflowRunner(WorkflowRunner):
    """Generic runner that executes one query trace and delegates reward allocation."""

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        self.workflow_cfg = self.config.star.get("workflow", {})
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
        self.debug_sample_count = max(
            1, int(debug_cfg.get("sample_count", os.environ.get("STAR_WORKFLOW_DEBUG_SAMPLE_COUNT", 1)))
        )
        self.debug_every_n_batches = max(
            1, int(debug_cfg.get("every_n_batches", os.environ.get("STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES", 20)))
        )
        self.validation_debug_enabled = bool(
            debug_cfg.get(
                "validation_enabled",
                str(os.environ.get("STAR_VAL_DEBUG", "true")).strip().lower() in {"1", "true", "yes", "on"},
            )
        )
        self.validation_debug_sample_count = max(
            1, int(debug_cfg.get("validation_sample_count", os.environ.get("STAR_VAL_DEBUG_SAMPLE_COUNT", 1)))
        )
        self.validation_debug_every_n_batches = max(
            1,
            int(debug_cfg.get("validation_every_n_batches", os.environ.get("STAR_VAL_DEBUG_EVERY_N_BATCHES", 1))),
        )
        self.validation_debug_max_chars = int(
            debug_cfg.get("validation_max_chars", os.environ.get("STAR_VAL_DEBUG_MAX_CHARS", 0))
        )
        self._debug_batch_counter_by_stage: dict[str, int] = defaultdict(int)
        self._last_dropped_query_ids: list[str] = []
        self.tools = self._build_tools()
        self.reward_allocator = self._create_reward_allocator()

    def pop_dropped_query_ids(self) -> list[str]:
        dropped = self._last_dropped_query_ids
        self._last_dropped_query_ids = []
        return dropped

    def _create_reward_allocator(self):
        alloc_cfg = dict(self.workflow_cfg.get("reward_allocator", {}))
        if "path" not in alloc_cfg or "name" not in alloc_cfg:
            raise ValueError("star.workflow.reward_allocator.path/name must be configured for TraceWorkflowRunner")
        alloc_cls = load_extern_object(str(alloc_cfg.get("path")), str(alloc_cfg.get("name")))
        kwargs = dict(alloc_cfg.get("kwargs", {}))
        return alloc_cls(trainer=self.trainer, config=self.config, runner=self, **kwargs)

    def _build_tools(self) -> dict[str, Any]:
        tools = {}
        for alias, tool_cfg in dict(self.workflow_cfg.get("tools", {})).items():
            if str(tool_cfg.get("type", "")) in {"simple_keyword", "http", "query_api_pool", "retrieval_api_pool"}:
                tools[alias] = build_retriever_tool(tool_cfg)
                continue
            if "path" in tool_cfg and "name" in tool_cfg:
                obj = load_extern_object(str(tool_cfg.get("path")), str(tool_cfg.get("name")))
                kwargs = dict(tool_cfg.get("kwargs", {}))
                tools[alias] = obj(**kwargs) if isinstance(obj, type) else obj
        return tools

    @staticmethod
    def _as_template_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list | tuple):
            return "\n".join([TraceWorkflowRunner._as_template_value(x) for x in v])
        if isinstance(v, dict):
            if "text" in v:
                return str(v["text"])
            if "document" in v:
                return str(v["document"])
            if "content" in v:
                return str(v["content"])
            if "title" in v and "snippet" in v:
                return f"{v['title']}: {v['snippet']}"
            return json.dumps(v, ensure_ascii=False, indent=2)
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
                    return TraceWorkflowRunner._to_str_list(value[key])
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

    def _count_chat_tokens(self, messages: list[dict]) -> int:
        tokenizer = getattr(self.trainer, "tokenizer", None)
        if tokenizer is None:
            return self._count_tokens(self._as_template_value(messages))
        apply_chat_template_kwargs = {}
        try:
            apply_chat_template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})
        except Exception:
            apply_chat_template_kwargs = {}
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
            return int(len(token_ids)) if isinstance(token_ids, list) else self._count_tokens(str(messages))
        except Exception:
            return self._count_tokens(self._as_template_value(messages))

    def _truncate_prompt_for_inference(self, prompt_text: str) -> tuple[str, int]:
        tokenizer = getattr(self.trainer, "tokenizer", None)
        max_tokens = int(self.per_infer_prompt_max_tokens)
        if tokenizer is None or max_tokens <= 0:
            return prompt_text, 0
        try:
            token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(prompt_text)
        except Exception:
            return prompt_text, 0
        if not isinstance(token_ids, list) or len(token_ids) <= max_tokens:
            return prompt_text, 0
        kept_ids = token_ids[:max_tokens]
        try:
            return tokenizer.decode(kept_ids, skip_special_tokens=True), int(len(token_ids) - len(kept_ids))
        except TypeError:
            return tokenizer.decode(kept_ids), int(len(token_ids) - len(kept_ids))
        except Exception:
            return prompt_text, 0

    def _build_truncated_chat_prompt(self, prompt_text: str) -> tuple[list[dict], int, int]:
        """Build a one-message chat prompt whose templated token length fits rollout.prompt_length."""

        tokenizer = getattr(self.trainer, "tokenizer", None)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_tokens = min(int(self.per_infer_prompt_max_tokens), int(rollout_cfg.get("prompt_length", 4096)))
        messages = [{"role": "user", "content": str(prompt_text or "")}]
        chat_tokens = self._count_chat_tokens(messages)
        if tokenizer is None or max_tokens <= 0 or chat_tokens <= max_tokens:
            return messages, 0, chat_tokens

        try:
            content_ids = tokenizer.encode(str(prompt_text or ""), add_special_tokens=False)
        except TypeError:
            content_ids = tokenizer.encode(str(prompt_text or ""))
        except Exception:
            truncated, trimmed = self._truncate_prompt_for_inference(prompt_text)
            messages = [{"role": "user", "content": truncated}]
            return messages, trimmed, self._count_chat_tokens(messages)

        if not isinstance(content_ids, list) or not content_ids:
            return messages, 0, chat_tokens

        lo, hi = 0, len(content_ids)
        best_text = ""
        best_keep = 0
        best_len = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                candidate = tokenizer.decode(content_ids[:mid], skip_special_tokens=True)
            except TypeError:
                candidate = tokenizer.decode(content_ids[:mid])
            except Exception:
                candidate = str(prompt_text or "")
            candidate_messages = [{"role": "user", "content": candidate}]
            candidate_len = self._count_chat_tokens(candidate_messages)
            if candidate_len <= max_tokens:
                best_text = candidate
                best_keep = mid
                best_len = candidate_len
                lo = mid + 1
            else:
                hi = mid - 1

        if not best_text and hi <= 0:
            best_text = ""
            best_len = self._count_chat_tokens([{"role": "user", "content": best_text}])
        trimmed_tokens = max(0, len(content_ids) - max(0, best_keep))
        return [{"role": "user", "content": best_text}], int(trimmed_tokens), int(best_len)

    @staticmethod
    def _has_content(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return len(value.strip()) > 0
        if isinstance(value, np.ndarray):
            return int(value.size) > 0
        if isinstance(value, list | tuple | set):
            return any(TraceWorkflowRunner._has_content(x) for x in value)
        if isinstance(value, dict):
            return len(value) > 0
        return True

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
                value = obj.get(key, parser.get("default", ""))
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

    async def _run_tool(self, tool_name: str, input_value: Any, top_k: int = 3, max_attempts: int = 5, fail_open: bool = False):
        if tool_name not in self.tools:
            raise ValueError(f"tool alias not found: {tool_name}")
        tool = self.tools[tool_name]

        async def _run_tool_call(func, *args, **kwargs):
            if self.tool_timeout_seconds > 0:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=self.tool_timeout_seconds,
                )
            return await asyncio.to_thread(func, *args, **kwargs)

        try:
            if hasattr(tool, "acall"):
                coro = tool.acall(input_value)
                if self.tool_timeout_seconds > 0:
                    return await asyncio.wait_for(coro, timeout=self.tool_timeout_seconds)
                return await coro
            if hasattr(tool, "aquery"):
                coro = tool.aquery(question=str(input_value), N=top_k, max_attempts=max_attempts)
                if self.tool_timeout_seconds > 0:
                    return await asyncio.wait_for(coro, timeout=self.tool_timeout_seconds)
                return await coro
            if hasattr(tool, "query"):
                try:
                    return await _run_tool_call(tool.query, question=str(input_value), N=top_k, max_attempts=max_attempts)
                except TypeError:
                    return await _run_tool_call(tool.query, str(input_value), top_k, max_attempts)
            if hasattr(tool, "retrieve"):
                return await _run_tool_call(tool.retrieve, str(input_value), top_k)
            if callable(tool):
                if inspect.iscoroutinefunction(tool):
                    coro = tool(input_value)
                    if self.tool_timeout_seconds > 0:
                        return await asyncio.wait_for(coro, timeout=self.tool_timeout_seconds)
                    return await coro
                return await _run_tool_call(tool, input_value)
            raise TypeError(f"tool {tool_name} is not callable and has no query()/retrieve()")
        except Exception:
            if fail_open:
                return []
            raise

    async def _execute_llm_step(
        self,
        *,
        query_batch: DataProto,
        node_id: str,
        turn_id: int,
        step_id: int,
        node_cfg,
        prompt_context: dict[str, Any],
        state_before: Any = None,
        state_after: Any = None,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> WorkflowExecutionRecord:
        prompt_text = self._render_template(str(node_cfg.get("prompt_template", "{question}")), prompt_context)
        raw_messages, trimmed_tokens, prompt_tokens = self._build_truncated_chat_prompt(prompt_text)
        model_id = str(node_cfg["model_id"])
        agent_id = str(node_cfg.get("agent_id", node_id))
        prompt_batch = self.trainer._build_workflow_prompt_batch(
            query_batch,
            [raw_messages],
            agent_id,
        )
        timing_state: dict[str, Any] = {}
        rollout_coro = self.trainer._rollout_model_async_batched(model_id, prompt_batch, timing_state=timing_state)
        if self.llm_timeout_seconds > 0:
            _, thin, _, timing_info = await asyncio.wait_for(rollout_coro, timeout=self.llm_timeout_seconds)
        else:
            _, thin, _, timing_info = await rollout_coro
        action_text_vec = thin.non_tensor_batch.get("action_text", np.array([], dtype=object))
        raw_text = str(action_text_vec[0]) if len(action_text_vec) > 0 else ""
        parsed_value, format_reward = self._parse_llm_output(raw_text, node_cfg)
        meta = dict(extra_meta or {})
        meta["format_reward"] = float(format_reward)
        meta["format_weight"] = float(dict(node_cfg.get("reward", {})).get("format_weight", 0.0))
        meta["prompt_tokens"] = int(prompt_tokens)
        meta["output_tokens"] = int(self._count_tokens(raw_text))
        meta["prompt_trimmed_tokens"] = int(trimmed_tokens)
        meta["timing"] = {
            str(k): float(v)
            for k, v in dict(timing_info or {}).items()
            if isinstance(v, int | float | np.integer | np.floating)
        }
        return WorkflowExecutionRecord(
            query_id=str(self._extract_from_batch(query_batch, "query_id") or ""),
            node_id=node_id,
            agent_id=agent_id,
            model_id=model_id,
            turn_id=int(turn_id),
            step_id=int(step_id),
            node_type="llm",
            raw_output=raw_text,
            parsed_output=parsed_value,
            thin=thin,
            trainable=True,
            state_before=state_before,
            state_after=state_after,
            meta=meta,
        )

    def _empty_trace(self, query_batch: DataProto, reason: str = "", error: str = "") -> WorkflowTrace:
        return WorkflowTrace(
            query_id=str(self._extract_from_batch(query_batch, "query_id") or ""),
            question=self._extract_question(query_batch),
            ground_truth=self._extract_gt_list(query_batch),
            dropped=True,
            drop_reason=reason,
            drop_error=error,
        )

    def _build_reward_parts(self, assignments: list[RewardAssignment]) -> tuple[list[DataProto], dict[str, float]]:
        grouped: dict[str, dict[str, Any]] = {}
        skipped = 0
        node_rewards: dict[str, list[float]] = defaultdict(list)
        reward_types: dict[str, list[float]] = defaultdict(list)
        for assignment in assignments:
            record = assignment.record
            if record.thin is None or len(record.thin) == 0:
                skipped += 1
                continue
            traj_ids = record.thin.non_tensor_batch.get("traj_id", np.array([], dtype=object))
            if len(traj_ids) == 0:
                skipped += 1
                continue
            traj_id = str(traj_ids[0])
            if traj_id not in grouped:
                grouped[traj_id] = {"thin": record.thin, "reward": 0.0}
            grouped[traj_id]["reward"] += float(assignment.reward)
            node_rewards[record.node_id].append(float(assignment.reward))
            reward_types[assignment.reward_type].append(float(assignment.reward))

        reward_parts = []
        for item in grouped.values():
            thin = item["thin"]
            reward = np.full((len(thin),), float(item["reward"]), dtype=np.float32)
            reward_parts.append(self.trainer._build_commit_rewards_from_thin(thin, reward))

        metrics: dict[str, float] = {
            "workflow/reward_parts": float(len(reward_parts)),
            "workflow/reward_skipped_non_trainable": float(skipped),
        }
        for node_id, values in node_rewards.items():
            metrics[f"workflow/node/{node_id}/assigned_reward_mean"] = float(np.mean(values))
        for reward_type, values in reward_types.items():
            metrics[f"workflow/reward_type/{reward_type}_mean"] = float(np.mean(values))
        return reward_parts, metrics

    @staticmethod
    def _merge_scalar_metrics(metric_acc: dict[str, list[float]], metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if isinstance(value, int | float | np.integer | np.floating):
                metric_acc[str(key)].append(float(value))

    @staticmethod
    def _safe_metric_component(value: Any) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "unknown")).strip("_.-")
        return safe or "unknown"

    @classmethod
    def _merge_trace_timing_metrics(cls, metric_acc: dict[str, list[float]], trace: WorkflowTrace) -> None:
        node_durations: list[float] = []
        llm_durations: list[float] = []
        tool_durations: list[float] = []
        llm_timing_fields = {
            "queue_wait_s": "llm_queue_wait_s",
            "rollout_exec_s": "llm_rollout_exec_s",
            "rollout_total_s": "llm_rollout_total_s",
            "rpc_roundtrip_s": "llm_rpc_roundtrip_s",
            "rpc_overhead_s": "llm_rpc_overhead_s",
            "engine_generate_s": "llm_engine_generate_s",
            "engine_generate_max_s": "llm_engine_generate_max_s",
            "agent_loop_tool_calls_s": "llm_agent_loop_tool_calls_s",
            "agent_loop_tool_calls_max_s": "llm_agent_loop_tool_calls_max_s",
            "agent_server_rpc_roundtrip_s": "llm_agent_server_rpc_roundtrip_s",
            "agent_server_rpc_roundtrip_max_s": "llm_agent_server_rpc_roundtrip_max_s",
            "agent_server_total_s": "llm_agent_server_total_s",
            "agent_server_total_max_s": "llm_agent_server_total_max_s",
            "agent_server_rpc_overhead_s": "llm_agent_server_rpc_overhead_s",
            "agent_server_rpc_overhead_max_s": "llm_agent_server_rpc_overhead_max_s",
            "agent_server_first_token_s": "llm_agent_server_first_token_s",
            "agent_server_first_token_max_s": "llm_agent_server_first_token_max_s",
            "agent_server_decode_tail_s": "llm_agent_server_decode_tail_s",
            "agent_server_decode_tail_max_s": "llm_agent_server_decode_tail_max_s",
            "agent_worker_start_lag_s": "llm_agent_worker_start_lag_s",
            "agent_worker_start_lag_max_s": "llm_agent_worker_start_lag_max_s",
            "agent_worker_prep_s": "llm_agent_worker_prep_s",
            "agent_worker_prep_max_s": "llm_agent_worker_prep_max_s",
            "agent_worker_run_loops_s": "llm_agent_worker_run_loops_s",
            "agent_worker_run_loops_max_s": "llm_agent_worker_run_loops_max_s",
            "agent_worker_postprocess_s": "llm_agent_worker_postprocess_s",
            "agent_worker_postprocess_max_s": "llm_agent_worker_postprocess_max_s",
            "agent_worker_total_s": "llm_agent_worker_total_s",
            "agent_worker_total_max_s": "llm_agent_worker_total_max_s",
            "agent_worker_non_loop_overhead_s": "llm_agent_worker_non_loop_overhead_s",
            "agent_worker_non_loop_overhead_max_s": "llm_agent_worker_non_loop_overhead_max_s",
            "agent_loop_manager_prep_s": "llm_agent_loop_manager_prep_s",
            "agent_loop_manager_worker_rpc_wait_s": "llm_agent_loop_manager_worker_rpc_wait_s",
            "agent_loop_manager_worker_rpc_mean_s": "llm_agent_loop_manager_worker_rpc_mean_s",
            "agent_loop_manager_worker_rpc_max_s": "llm_agent_loop_manager_worker_rpc_max_s",
            "agent_loop_manager_concat_s": "llm_agent_loop_manager_concat_s",
            "agent_loop_manager_metrics_reduce_s": "llm_agent_loop_manager_metrics_reduce_s",
            "agent_loop_manager_total_s": "llm_agent_loop_manager_total_s",
            "agent_loop_manager_overhead_s": "llm_agent_loop_manager_overhead_s",
            "worker_total_s": "llm_worker_total_s",
            "manager_generate_s": "llm_manager_generate_s",
            "build_thin_s": "llm_build_thin_s",
            "data_proto_pad_select_s": "llm_data_proto_pad_select_s",
            "request_bsz": "llm_request_bsz",
            "rollout_dp_size": "llm_rollout_dp_size",
            "dp_padding_applied": "llm_dp_padding_applied_ratio",
            "dp_padding_added": "llm_dp_padding_added",
            "dp_padding_factor": "llm_dp_padding_factor",
            "local_build_thin_used": "llm_local_build_thin_used_ratio",
            "microbatch_enabled": "llm_microbatch_enabled_ratio",
            "microbatch_size": "llm_microbatch_size",
            "microbatch_request_count": "llm_microbatch_request_count",
            "microbatch_wait_s": "llm_microbatch_wait_s",
            "microbatch_wait_max_s": "llm_microbatch_wait_max_s",
            "microbatch_batch_exec_s": "llm_microbatch_batch_exec_s",
            "microbatch_flush_timeout": "llm_microbatch_flush_timeout_ratio",
            "microbatch_flush_size": "llm_microbatch_flush_size_ratio",
            "microbatch_saved_calls": "llm_microbatch_saved_calls",
            "microbatch_saved_call_ratio": "llm_microbatch_saved_call_ratio",
        }
        for record in trace.records:
            meta = record.meta if isinstance(record.meta, dict) else {}
            timing = meta.get("timing", {})
            if not isinstance(timing, dict):
                timing = {}
            duration = timing.get("rollout_total_s", None) if record.node_type == "llm" else meta.get("duration_s", None)
            if isinstance(duration, int | float | np.integer | np.floating) and float(duration) >= 0:
                node_durations.append(float(duration))
                if record.node_type == "llm":
                    llm_durations.append(float(duration))
                elif record.node_type == "tool":
                    tool_durations.append(float(duration))

            if record.node_type != "llm":
                continue
            model_id = cls._safe_metric_component(record.model_id)
            agent_id = cls._safe_metric_component(record.agent_id)
            node_id = cls._safe_metric_component(record.node_id)
            metric_acc[f"workflow/timing/group/model_{model_id}_s_mean"].append(float(timing.get("rollout_total_s", 0.0)))
            metric_acc[f"workflow/timing/group/model_{model_id}_count"].append(1.0)
            metric_acc[f"workflow/timing/group/agent_{agent_id}_s_mean"].append(float(timing.get("rollout_total_s", 0.0)))
            metric_acc[f"workflow/timing/group/agent_{agent_id}_count"].append(1.0)
            for field, metric_suffix in llm_timing_fields.items():
                value = timing.get(field, None)
                if not isinstance(value, int | float | np.integer | np.floating):
                    continue
                value_f = float(value)
                metric_acc[f"workflow/timing/{metric_suffix}_mean"].append(value_f)
                metric_acc[f"workflow/timing/model/{model_id}/{metric_suffix}_mean"].append(value_f)
                metric_acc[f"workflow/timing/agent/{agent_id}/{metric_suffix}_mean"].append(value_f)
                metric_acc[f"workflow/timing/node/{node_id}/{metric_suffix}_mean"].append(value_f)

        if node_durations:
            metric_acc["workflow/timing/node_s_mean"].append(float(np.mean(node_durations)))
            metric_acc["workflow/timing/node_invocations"].append(float(len(node_durations)))
        if llm_durations:
            metric_acc["workflow/timing/llm_node_s_mean"].append(float(np.mean(llm_durations)))
            metric_acc["workflow/timing/llm_node_calls"].append(float(len(llm_durations)))
        if tool_durations:
            metric_acc["workflow/timing/tool_node_s_mean"].append(float(np.mean(tool_durations)))
            metric_acc["workflow/timing/tool_node_calls"].append(float(len(tool_durations)))

    @abstractmethod
    async def run_query(
        self,
        query_batch: DataProto,
        query_local_idx: int,
        debug: bool,
        debug_max_chars: int | None = None,
    ) -> WorkflowTrace:
        raise NotImplementedError

    async def _run_one_query_safe(
        self,
        query_batch: DataProto,
        query_sem: asyncio.Semaphore,
        query_local_idx: int,
        debug: bool,
        debug_max_chars: int | None = None,
    ) -> WorkflowTrace:
        async with query_sem:
            query_start = time.perf_counter()
            try:
                if self.query_timeout_seconds > 0:
                    trace = await asyncio.wait_for(
                        self.run_query(
                            query_batch,
                            query_local_idx=query_local_idx,
                            debug=debug,
                            debug_max_chars=debug_max_chars,
                        ),
                        timeout=self.query_timeout_seconds,
                    )
                else:
                    trace = await self.run_query(
                        query_batch,
                        query_local_idx=query_local_idx,
                        debug=debug,
                        debug_max_chars=debug_max_chars,
                    )
                trace.metrics["workflow/timing/query_s_mean"] = float(time.perf_counter() - query_start)
                return trace
            except asyncio.TimeoutError as exc:
                logger.warning("[trace-workflow] query timeout dropped idx=%s err=%r", query_local_idx, exc)
                trace = self._empty_trace(query_batch, reason="query_timeout", error=str(exc))
                trace.metrics["workflow/timing/query_s_mean"] = float(time.perf_counter() - query_start)
                return trace
            except Exception as exc:
                logger.warning("[trace-workflow] query error dropped idx=%s err=%r", query_local_idx, exc)
                trace = self._empty_trace(
                    query_batch,
                    reason="query_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                trace.metrics["workflow/timing/query_s_mean"] = float(time.perf_counter() - query_start)
                return trace

    async def run_batch(self, batch: DataProto, epoch: int, stage: str = "train") -> tuple[DataProto, dict[str, float]]:
        del epoch
        batch_start = time.perf_counter()
        stage = str(stage or "train")
        self._debug_batch_counter_by_stage[stage] += 1
        if stage == "validation":
            debug_enabled = self.validation_debug_enabled
            debug_every_n_batches = self.validation_debug_every_n_batches
            debug_sample_count = self.validation_debug_sample_count
            debug_max_chars = self.validation_debug_max_chars
        else:
            debug_enabled = self.debug_enabled
            debug_every_n_batches = self.debug_every_n_batches
            debug_sample_count = self.debug_sample_count
            debug_max_chars = self.debug_max_chars

        debug_this_batch = debug_enabled and (
            self._debug_batch_counter_by_stage[stage] % max(1, debug_every_n_batches) == 0
        )
        debug_query_indices: set[int] = set()
        if debug_this_batch and len(batch) > 0:
            start_idx = int(self.debug_sample_index) % len(batch)
            sample_count = min(len(batch), max(1, int(debug_sample_count)))
            debug_query_indices = {(start_idx + offset) % len(batch) for offset in range(sample_count)}

        query_sem = asyncio.Semaphore(max(1, self.max_inflight_queries))
        traces = await asyncio.gather(
            *[
                self._run_one_query_safe(
                    batch.select_idxs([i]),
                    query_sem,
                    query_local_idx=i,
                    debug=bool(i in debug_query_indices),
                    debug_max_chars=debug_max_chars,
                )
                for i in range(len(batch))
            ]
        )

        reward_parts: list[DataProto] = []
        metric_acc: dict[str, list[float]] = defaultdict(list)
        dropped_query_ids: list[str] = []
        drop_reason_acc: dict[str, int] = defaultdict(int)
        for trace in traces:
            if trace.dropped:
                if trace.query_id:
                    dropped_query_ids.append(trace.query_id)
                drop_reason_acc[str(trace.drop_reason or "unknown")] += 1
                continue
            assignments, alloc_metrics = self.reward_allocator.allocate(trace)
            parts, assign_metrics = self._build_reward_parts(assignments)
            reward_parts.extend(parts)
            self._merge_scalar_metrics(metric_acc, trace.metrics)
            self._merge_scalar_metrics(metric_acc, alloc_metrics)
            self._merge_scalar_metrics(metric_acc, assign_metrics)
            self._merge_trace_timing_metrics(metric_acc, trace)
            metric_acc["workflow/trace_records"].append(float(len(trace.records)))
            metric_acc["workflow/trace_assignments"].append(float(len(assignments)))
            metric_acc["workflow/trace_reward_sum"].append(float(sum(a.reward for a in assignments)))
            if trace.debug_dump:
                print(trace.debug_dump, flush=True)

        rewards = self.trainer._empty_rewards() if len(reward_parts) == 0 else (
            self.trainer._concat_data_proto_safe(reward_parts)
        )

        metrics = {
            "workflow/samples": float(len(traces)),
            "workflow/query_dropped": float(len(dropped_query_ids)),
            "workflow/query_drop_ratio": float(len(dropped_query_ids) / max(1, len(traces))),
            "workflow/timing/batch_total_s": float(time.perf_counter() - batch_start),
        }
        for reason, count in drop_reason_acc.items():
            safe_reason = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(reason or "unknown")).strip("_.-") or "unknown"
            metrics[f"workflow/query_drop/{safe_reason}"] = float(count)
        for key, values in metric_acc.items():
            if not values:
                continue
            metrics[key] = float(np.mean(values))
        self._last_dropped_query_ids = dropped_query_ids
        return rewards, metrics
